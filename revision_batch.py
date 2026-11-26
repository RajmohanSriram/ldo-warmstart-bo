#!/usr/bin/env python3
"""Experiments for the paper revision (reviewer comments, Nov 2026). Builds on overnight_batch.py.

Same objective, search box, budget (250) and BO batch size (14) as the original batch, and the SAME
hold-out models in runs_batch/models, so new seeds add to the existing seeds 0-2 without mixing setups.
Every job writes runs_batch/<spec>/<method>_s<seed>/done.json and is skipped once that exists, so you can
stop (Ctrl-C) and restart any time. Several jobs run side by side: BO time is mostly the GP, which
uses one core, so 3 jobs x 4 workers on a 12-thread machine finishes about 3x sooner than 1 x 12.

Parts (pick with --parts; default runs them in this order):
  core    random / de / bo_cold / bo_warm on 3.3:4, 5.0:4, 3.3:6.8, seeds 3-9          (comment 1)
  extrap  same four methods on 3.3:15 (15 uF is in no training data), seeds 0-4      (comment 2)
  lookup  bo_lookup: BO seeded with the best STORED designs at the nearest spec, no MLP,
          on the 3 original specs, seeds 0-4                                         (comment 13)
  ablate  [INVALID - do not use] Sobol-only priors: every Sobol row is at 4 uF, so these models never saw
          Cout vary and blow up at 6.8 uF. Kept only so old folders are recognised.
  runsonly bo_warm with a prior trained ONLY on logged BO runs (no dataset), on 3.3:4 and 3.3:6.8,
          seeds 0-4: is the 64k-row dataset needed at all?                         (comment 5)
  ens     no SPICE: trains 2 extra hold-out MLPs per spec and writes ens_decisions.json,
          the extrapolation-fallback decision for every warm run                     (comments 3, 4)
  screen  no SPICE: random-forest and gradient-boosting pass/fail baselines on the same
          1997-row test set as the warm-start MLP -> screen_baselines.json             (Table IV)

  python revision_batch.py plan                         # list pending jobs, rough hours
  python revision_batch.py smoke                        # 3-5 min plumbing test (runs_rev_smoke/)
  python revision_batch.py queue --jobs 3 --workers 4   # run everything pending
  python revision_batch.py queue --parts lookup,ablate  # e.g. on a second machine
  python revision_batch.py report                       # tables for the paper
"""
import argparse, glob, json, math, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.join(HERE, "runs_batch")
BATCH = 14                       # BO ask size used by the original runs (14 workers on the laptop)
BUDGET = 250
ORIG = ["random", "de", "bo_cold", "bo_warm"]
ALL_METHODS = ORIG + ["bo_warm_static", "bo_lookup", "bo_warm_sobol4k", "bo_warm_sobol16k", "bo_warm_sobolall",
               "bo_warm_runsonly"]
ABLATE = {"bo_warm_sobol4k": 4000, "bo_warm_sobol16k": 16000, "bo_warm_sobolall": None}
SPECS3 = [(3.3, 4.0), (5.0, 4.0), (3.3, 6.8)]
EST_MIN = {"random": 1.5, "de": 1.5, "bo_cold": 10, "bo_warm": 11, "bo_lookup": 10}   # minutes per job, 4 workers


def tag(v, c):
    return f"gf180_{v:g}V_{c:g}uF"


def seed_of(method, seed):
    """Same formula as overnight_batch.py. Lookup and ablation runs reuse bo_warm's BO seed, so each of them
    is a paired comparison with the warm run of the same seed (only the initial points differ)."""
    m = "bo_warm" if method in ("bo_lookup", "bo_warm_runsonly") or method in ABLATE else method
    return 1000 * seed + 17 + 101 * ALL_METHODS.index(m)


def jobs_for(parts):
    J = []
    for p in parts:
        if p == "core":
            J += [(s, m, sd) for sd in range(3, 10) for s in SPECS3 for m in ORIG]
        elif p == "extrap":
            J += [((3.3, 15.0), m, sd) for sd in range(0, 5) for m in ORIG]
        elif p == "lookup":
            J += [(s, "bo_lookup", sd) for sd in range(0, 5) for s in SPECS3]
        elif p == "runsonly":
            J += [(s, "bo_warm_runsonly", sd) for sd in range(0, 5) for s in [(3.3, 4.0), (3.3, 6.8)]]
        elif p == "ablate":
            J += [((3.3, 6.8), m, sd) for sd in range(0, 5) for m in ABLATE]
    return J


def done_path(root, spec, method, seed):
    return os.path.join(root, tag(*spec), f"{method}_s{seed}", "done.json")


# ----------------------------------------------------------------------------- models
def holdout_path(root, v, c, member=0):
    return os.path.join(root, "models", f"holdout_{v:g}V_{c:g}uF" + (f"_m{member}" if member else "") + ".pt")


def ensure_holdout(root, v, c, member=0):
    import surrogate_ws as W
    p = holdout_path(root, v, c, member)
    if not os.path.exists(p):
        print(f"--- training hold-out MLP {os.path.basename(p)} (no data at {v:g} V / {c:g} uF) ---", flush=True)
        W.train("gf180", [os.path.join(HERE, "runs")], p, exclude=[(v, c)], seed=member)
    return p


def ensure_runsonly(root, v, c):
    """Prior trained ONLY on logged evaluations of earlier BO runs (runs/), no dataset at all: the data a
    designer has anyway after sizing earlier specifications. Same exclusion of the target spec."""
    import surrogate_ws as W
    p = os.path.join(root, "models", f"runsonly_{v:g}V_{c:g}uF.pt")
    if not os.path.exists(p):
        print(f"--- training runs-only prior {os.path.basename(p)} ---", flush=True)
        W.train("gf180", [os.path.join(HERE, "runs")], p, exclude=[(v, c)], seed=0, dataset=None)
    return p


def ensure_ablation(root, method):
    """Prior trained ONLY on uniform (Sobol) dataset rows: no BO runs, no local perturbations, no active
    sampling. n rows drawn at random (seed 0) from both voltages; None = all Sobol rows."""
    import surrogate_ws as W
    n = ABLATE[method]
    p = os.path.join(root, "models", f"ablate_{method.split('_')[-1]}.pt")
    if os.path.exists(p):
        return p
    print(f"--- training ablation prior {os.path.basename(p)} (Sobol rows only, n={n or 'all'}) ---", flush=True)
    (X, Y, S, G), (Xt, Yt, _, _) = W.load_dataset(W.default_dataset_dir(), "gf180", exclude=[(3.3, 6.8)])
    keep = np.array([":sobol:" in g for g in G])
    X, Y, G = X[keep], Y[keep], [g for g, k in zip(G, keep) if k]
    if n and n < len(X):
        idx = np.random.default_rng(0).choice(len(X), n, replace=False)
        X, Y, G = X[idx], Y[idx], [G[i] for i in idx]
    near = W.near_mask(X, Xt)
    X, Y, G = X[~near], Y[~near], [g for g, k in zip(G, near) if not k]
    m = W.Surrogate(seed=0).fit(X, Y, seed=0, groups=G)
    m.meta.update(pdk="gf180", ablation=method, n_rows=int(len(X)), sources="dataset sobol rows only",
                  coverage={f"{v:g}V": int(np.isclose(X[:, 8], v).sum()) for v in (3.3, 5.0)})
    Pt = m.predict(Xt)
    m.meta.update(test_r2=W.r2_scores(Yt, Pt), n_test=int(len(Xt)))
    m.save(p)
    print(f"saved {p}: {len(X)} rows, test R2 {m.meta['test_r2']}", flush=True)
    return p


# ----------------------------------------------------------------------------- lookup baseline
def lookup_points(root, v, c, k=8, dmin=0.08):
    """Best stored designs for (v, c) from exactly the data the hold-out MLP was trained on (same exclusion).
    Nearest stored spec by |dV|/1.7 V + |ln(C/C0)|; designs ranked by their STORED worst-case margin
    against the target spec's limits; diverse top-k like surrogate_ws.propose."""
    import surrogate_ws as W
    cache = os.path.join(root, "models", f"lookup_{tag(v, c)}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    X, Y, S, _, _ = W.load_runs([os.path.join(HERE, "runs")], "gf180", exclude=[(v, c)])
    (Xd, Yd, Sd, _), _ = W.load_dataset(W.default_dataset_dir(), "gf180", exclude=[(v, c)])
    X, Y, S = np.vstack([X, Xd]), np.vstack([Y, Yd]), S + Sd
    specs = sorted(set(S))
    dist = {s: abs(s[0] - v) / 1.7 + abs(math.log(s[1] / c)) for s in specs}
    near = min(specs, key=lambda s: dist[s])
    rows = np.array([s == near for s in S])
    Xn, Yn = X[rows], Y[rows].copy()
    Yn[:, W.TARGETS.index("pm")] = np.nan_to_num(Yn[:, W.TARGETS.index("pm")], nan=-1e9)
    Yn[:, W.TARGETS.index("psrr")] = np.nan_to_num(Yn[:, W.TARGETS.index("psrr")], nan=-1e9)
    Yn[:, W.TARGETS.index("minsat")] = np.nan_to_num(Yn[:, W.TARGETS.index("minsat")], nan=-1e9)
    Yn[:, W.TARGETS.index("ov")] = np.nan_to_num(Yn[:, W.TARGETS.index("ov")], nan=1e9)
    spec = dict(vout=v, cout=c, psrr=41.0, line_pct=1.0, tol_pct=1.5, pm_floor=45.0, dropout=0.3, vin_over=0.4)
    mg = W.margin(Yn, spec)
    dims = W.BROAD["gf180"]
    pts, taken, mgs = [], [], []
    for i in np.argsort(-mg):
        u = W.unit(Xn[i:i + 1, :8], dims)[0]
        if taken and min(np.linalg.norm(u - t) / math.sqrt(len(u)) for t in taken) < dmin:
            continue
        taken.append(u)
        x = [float(q) for q in Xn[i, :8]]; x[3] = int(round(x[3]))
        pts.append(x); mgs.append(float(mg[i]))
        if len(pts) == k:
            break
    out = dict(target=[v, c], nearest_stored_spec=list(near), rows_at_nearest=int(rows.sum()),
               stored_passing_at_nearest=int((mg > 0).sum()), points=pts, stored_margins=mgs)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"), indent=1)
    return out


class Lookup:
    def __init__(self, info, rundir):
        self.info, self.rundir = info, rundir

    def initial(self, run, dims, seed):
        json.dump(self.info, open(os.path.join(self.rundir, "lookup_seeds.json"), "w"), indent=1)
        return [list(p) for p in self.info["points"]]

    def hook(self, run, dims):
        return []


# ----------------------------------------------------------------------------- one job
def run_job(root, v, c, method, seed, workers, budget=BUDGET, batch=BATCH):
    import ldo_flow as F
    import overnight_batch as OB
    rd = os.path.join(root, tag(v, c), f"{method}_s{seed}")
    if os.path.exists(os.path.join(rd, "done.json")):
        print(f"skip (done): {rd}"); return
    if os.path.isdir(rd):
        for fn in ("evals.jsonl", "log.txt", "warm_seeds.json", "progress.json", "lookup_seeds.json"):
            if os.path.exists(os.path.join(rd, fn)):
                os.remove(os.path.join(rd, fn))
    os.makedirs(rd, exist_ok=True)
    s = seed_of(method, seed)
    args = OB.make_args(v, c, workers)
    json.dump(dict(pdk="gf180", vout=v, cout_uF=c, psrr_min_dB=args.psrr, method=method, seed=seed, bo_seed=s,
                   budget=budget, batch=batch, workers=workers), open(os.path.join(rd, "spec.json"), "w"), indent=1)
    warm = None
    if method == "bo_warm" or method in ABLATE or method == "bo_warm_runsonly":
        mp = (holdout_path(root, v, c) if method == "bo_warm" else
              ensure_runsonly(root, v, c) if method == "bo_warm_runsonly" else ensure_ablation(root, method))
        if method == "bo_warm":
            ensure_holdout(root, v, c)
        wa = OB.make_args(v, c, workers, ["--warm", "mlp", "--surrogate", mp, "--retrain-every", "40"])
        wa.seed = s
        warm = F.Warm(wa, rd)
        if not warm.on:
            raise SystemExit("warm start could not load its model: " + warm.status)
    elif method == "bo_lookup":
        warm = Lookup(lookup_points(root, v, c), rd)
    pool, sp = OB.make_pool(args, os.path.join(rd, "scratch"))
    print(f"=== [{tag(v, c)}] {method} seed {seed} (bo seed {s}, batch {batch}, {workers} workers) ===", flush=True)
    try:
        run, t0 = F.Run(args, sp, rd), time.time()
        run.planned = budget
        if method == "random":
            OB.m_random(pool, run, budget, s, batch)
        elif method == "de":
            OB.m_de(pool, run, budget, s, batch)
        elif method.startswith("bo"):
            OB.m_bo(pool, run, budget, s, batch, warm)
        else:
            raise SystemExit(f"unknown method {method}")
        out = dict(spec=tag(v, c), method=method, seed=seed, bo_seed=s, batch=batch, workers=workers,
                   **OB.summarise(run, t0))
        if warm is not None and hasattr(warm, "summary"):
            out["warm"] = warm.summary(run)
        if method == "bo_lookup":
            ini = {tuple(round(q, 4) for q in p) for p in warm.info["points"]}
            hits = [e for e in run.evals if tuple(round(q, 4) for q in e["x"]) in ini]
            out["lookup"] = dict(nearest_stored_spec=warm.info["nearest_stored_spec"], simulated=len(hits),
                                 passed=sum(e["screen"] for e in hits))
        json.dump(out, open(os.path.join(rd, "done.json"), "w"), indent=1)
        print(f"done: first pass @{out['first_pass']}, {out['n_screen']} passing, {out['runtime_s']} s", flush=True)
    finally:
        pool.shutdown()
        import shutil
        shutil.rmtree(os.path.join(rd, "scratch"), ignore_errors=True)


# ----------------------------------------------------------------------------- queue
def prepare(root, jobs):
    """Train every model the jobs need BEFORE starting parallel jobs (avoids two jobs training the same file)."""
    for (v, c), m, _ in jobs:
        if m == "bo_warm":
            ensure_holdout(root, v, c)
        elif m in ABLATE:
            ensure_ablation(root, m)
        elif m == "bo_warm_runsonly":
            ensure_runsonly(root, v, c)
        elif m == "bo_lookup":
            lookup_points(root, v, c)


def queue(root, parts, njobs, workers, logdir):
    jobs = [j for j in jobs_for(parts) if not os.path.exists(done_path(root, *j))]
    print(f"{len(jobs)} pending jobs in parts {parts}; {njobs} at a time x {workers} workers", flush=True)
    os.makedirs(logdir, exist_ok=True)
    running, t0, finished = [], time.time(), 0
    while jobs or running:
        while jobs and len(running) < njobs:
            (v, c), m, sd = jobs.pop(0)
            prepare(root, [((v, c), m, sd)])        # trains a missing model here, once; running jobs carry on
            lf = os.path.join(logdir, f"{tag(v, c)}_{m}_s{sd}.log")
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "job", "--spec", f"{v:g}:{c:g}", "--method", m,
                   "--seed", str(sd), "--workers", str(workers), "--root", root]
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1")          # one core per job's GP; the SPICE workers do the rest
            running.append((subprocess.Popen(cmd, cwd=HERE, env=env, stdout=open(lf, "w"), stderr=subprocess.STDOUT),
                            (v, c, m, sd), lf))
        time.sleep(5)
        for item in list(running):
            p, (v, c, m, sd), lf = item
            if p.poll() is not None:
                running.remove(item); finished += 1
                ok = os.path.exists(done_path(root, (v, c), m, sd))
                tail = open(lf).read().strip().splitlines()[-1:] if os.path.exists(lf) else [""]
                el = (time.time() - t0) / 3600
                print(f"[{time.strftime('%H:%M')}] {'OK  ' if ok else 'FAIL'} {tag(v, c)} {m} s{sd} | {tail[0][:90]} | "
                      f"{finished} done, {len(jobs) + len(running)} left, {el:.1f} h elapsed", flush=True)


def plan(root, parts, njobs):
    jobs = [j for j in jobs_for(parts) if not os.path.exists(done_path(root, *j))]
    by = {}
    for (v, c), m, sd in jobs:
        by.setdefault(m, 0); by[m] += 1
    mins = sum(EST_MIN.get(m if m in EST_MIN else "bo_warm", 11) for _, m, _ in jobs)
    print(f"pending jobs: {len(jobs)}  " + ", ".join(f"{m}={n}" for m, n in by.items()))
    print(f"rough wall time with {njobs} parallel jobs: {mins / njobs / 60:.1f} h (estimate; first jobs give the real rate)")


# ----------------------------------------------------------------------------- no-SPICE parts
def ens_decisions(root, members=(1, 2)):
    """Fallback rule: run warm only if at least one initial proposal is predicted to pass by EVERY ensemble
    member (the original hold-out model plus `members`); otherwise fall back to a cold start. Evaluated on the
    exact initial proposals of every bo_warm run already on disk, so no SPICE is needed."""
    import surrogate_ws as W
    out = []
    for d in sorted(glob.glob(os.path.join(root, "gf180_*", "bo_warm_s*"))):
        sp = json.load(open(os.path.join(d, "spec.json")))
        v, c = sp["vout"], sp["cout_uF"]
        if not os.path.exists(os.path.join(d, "warm_seeds.json")):
            continue
        paths = [ensure_holdout(root, v, c)] + [ensure_holdout(root, v, c, m) for m in members]
        models = [W.Surrogate.load(p) for p in paths]
        rows = [r for r in json.load(open(os.path.join(d, "warm_seeds.json")))["rows"] if r["kind"] == "init"]
        if not rows:
            continue
        X = np.array([r["x"] + [v, c] for r in rows], float)
        spec = dict(vout=v, cout=c, psrr=40.0, line_pct=1.0, tol_pct=1.5, pm_floor=45.0, dropout=0.3, vin_over=0.4)
        M = np.array([W.margin(m.predict(X), spec) for m in models])            # members x proposals
        P = np.array([m.predict(X)[:, W.TARGETS.index("psrr")] for m in models])
        ev = {tuple(round(q, 4) for q in e["x"]): e for e in map(json.loads, open(os.path.join(d, "evals.jsonl")))}
        sim_pass = [bool(ev.get(tuple(round(q, 4) for q in r["x"]), {}).get("screen")) for r in rows]
        unanimous = (M > 0).all(0)
        out.append(dict(run=os.path.relpath(d, root), vout=v, cout=c, n_init=len(rows),
                        single_model_pred_pass=int((M[0] > 0).sum()), unanimous_pred_pass=int(unanimous.sum()),
                        fallback=bool(not unanimous.any()), sim_pass=int(sum(sim_pass)),
                        mean_psrr_std_dB=float(P.std(0).mean()), mean_margin_std=float(M.std(0).mean())))
        print(out[-1], flush=True)
    json.dump(out, open(os.path.join(root, "ens_decisions.json"), "w"), indent=1)


def screen_baselines(root):
    """RF and gradient-boosting regressors on the warm-start model's training data, judged as pass/fail screens
    on the same 1997 dataset test rows (Table IV)."""
    import surrogate_ws as W
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    X, Y, S, _, _ = W.load_runs([os.path.join(HERE, "runs")], "gf180")
    (Xd, Yd, _, _), (Xt, Yt, _, _) = W.load_dataset(W.default_dataset_dir(), "gf180")
    X, Y = np.vstack([X, Xd]), np.vstack([Y, Yd])
    near = W.near_mask(X, Xt); X, Y = X[~near], Y[~near]
    F, Ft = W.feat(X), W.feat(Xt)

    def fill(Yv):
        Z = Yv.copy()
        for t, val in (("pm", -1e3), ("psrr", -1e3), ("minsat", -1e3), ("ov", 1e3)):
            j = W.TARGETS.index(t); Z[:, j] = np.nan_to_num(Z[:, j], nan=val)
        return Z

    def screen(P):
        res = {}
        mt = np.zeros(len(Xt)); mp = np.zeros(len(Xt))
        for v in (3.3, 5.0):
            i = np.isclose(Xt[:, 8], v)
            spec = dict(vout=v, cout=4.0, psrr=40.0, line_pct=1.0, tol_pct=1.5, pm_floor=45.0, dropout=0.3, vin_over=0.4)
            mt[i] = W.margin(fill(Yt[i]), spec); mp[i] = W.margin(P[i], spec)
        t, p = mt > 0, mp > 0
        n, tp = int(t.sum()), int((t & p).sum())
        top = np.argsort(-mp)[:n]
        res.update(true_pass=n, pred_pass=int(p.sum()), accuracy=float((t == p).mean()),
                   precision=tp / max(1, p.sum()), recall=tp / max(1, n), precision_at_n=float(t[top].mean()),
                   psrr_mae=float(np.nanmean(np.abs(Yt[:, 5] - P[:, 5]))), pm_mae=float(np.nanmean(np.abs(Yt[:, 4] - P[:, 4]))),
                   iq_mae=float(np.nanmean(np.abs(Yt[:, 2] - P[:, 2]))))
        return res

    # Like the MLP: each metric is regressed only on rows where it was measured, and a separate classifier
    # predicts whether PM is measurable at all; "not measurable" counts as a PM failure.
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    jov, jpm = W.TARGETS.index("ov"), W.TARGETS.index("pm")
    out = {}
    for name, mk, mkc in (
            ("random_forest", lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0),
             lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0)),
            ("hist_gradient_boosting", lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0),
             lambda: HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=0))):
        t0 = time.time()
        P = np.zeros((len(Xt), len(W.TARGETS)))
        for j, t in enumerate(W.TARGETS):
            y = np.nan_to_num(Y[:, j], nan=1.0) if j == jov else Y[:, j]
            ok = ~np.isnan(y)
            yt = W.ty(np.where(np.isnan(Y), 0.0, Y))[:, j] if t in W.LOGT and j != jov else y
            P[:, j] = mk().fit(F[ok], yt[ok]).predict(Ft)
            if t in W.LOGT and j != jov:
                P[:, j] = np.expm1(P[:, j])
        pm_ok = mkc().fit(F, ~np.isnan(Y[:, jpm])).predict_proba(Ft)[:, 1]
        P[:, jpm] = np.where(pm_ok < 0.5, -1e3, P[:, jpm])
        out[name] = dict(screen(P), train_s=round(time.time() - t0))
        print(name, out[name], flush=True)
    mlp = W.Surrogate.load(os.path.join(HERE, "surrogate_ws", "gf180.pt"))
    Pm = mlp.predict(Xt)
    Pm8 = Pm[:, :len(W.TARGETS)].copy()
    Pm8[:, jpm] = np.where(Pm[:, -1] < 0.5, -1e3, Pm8[:, jpm])
    out["mlp_warm_start_model"] = screen(Pm8)
    out["note"] = ("same 1997 dataset test rows as surrogate_ws/gf180.pt; regressors trained on raw targets "
                   "(failed PM/PSRR filled as failing values)")
    json.dump(out, open(os.path.join(root, "screen_baselines.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- report
def report(root):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(root, "gf180_*", "*", "done.json"))]
    print(f"\n{'spec':<20}{'method':<20}{'seeds':>6}{'passed':>8}  first pass: median [min-max]   passing designs: median [min-max]")
    for spec in sorted({r["spec"] for r in rows}):
        for m in ALL_METHODS:
            rr = sorted([r for r in rows if r["spec"] == spec and r["method"] == m], key=lambda r: r["seed"])
            if not rr:
                continue
            fp = [r["first_pass"] for r in rr if r["first_pass"]]
            ns = [r["n_screen"] for r in rr]
            fps = f"{int(np.median(fp))} [{min(fp)}-{max(fp)}]" if fp else "-"
            print(f"{spec:<20}{m:<20}{len(rr):>6}{f'{len(fp)}/{len(rr)}':>8}  {fps:<28}{int(np.median(ns))} [{min(ns)}-{max(ns)}]")
    per_seed = {f"{r['spec']}|{r['method']}|{r['seed']}": dict(first_pass=r["first_pass"], n_screen=r["n_screen"])
                for r in rows}
    json.dump(per_seed, open(os.path.join(root, "per_seed.json"), "w"), indent=1, sort_keys=True)
    print(f"\nper-seed results -> {os.path.join(root, 'per_seed.json')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "smoke", "queue", "job", "ens", "screen", "report"])
    ap.add_argument("--parts", default="runsonly")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--spec"); ap.add_argument("--method"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    if a.cmd == "plan":
        plan(a.root, parts, a.jobs)
    elif a.cmd == "job":
        v, c = (float(q) for q in a.spec.split(":"))
        run_job(a.root, v, c, a.method, a.seed, a.workers)
    elif a.cmd == "smoke":
        sroot = os.path.join(HERE, "runs_rev_smoke")
        os.makedirs(os.path.join(sroot, "models"), exist_ok=True)
        for f in glob.glob(os.path.join(ROOT, "models", "holdout_5V_4uF.pt*")):      # reuse, don't retrain
            dst = os.path.join(sroot, "models", os.path.basename(f))
            if not os.path.exists(dst):
                import shutil; shutil.copy(f, dst)
        for m in ("random", "de", "bo_cold", "bo_warm", "bo_lookup"):
            run_job(sroot, 5.0, 4.0, m, 0, a.workers, budget=16, batch=8)
        print("\nSMOKE OK: all five job types ran. Delete runs_rev_smoke/ when you like.")
    elif a.cmd == "queue":
        queue(a.root, parts, a.jobs, a.workers, os.path.join(a.root, "logs_rev"))
    elif a.cmd == "ens":
        ens_decisions(a.root)
    elif a.cmd == "screen":
        screen_baselines(a.root)
    elif a.cmd == "report":
        report(a.root)


if __name__ == "__main__":
    main()
