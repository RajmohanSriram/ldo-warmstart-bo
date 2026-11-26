#!/usr/bin/env python3
"""Overnight comparison: random search vs differential evolution vs cold BO vs MLP-warm-started BO.

Every method gets the same objective (ldo_opt.make_objective at all five corners, via the flow's
worker pool), the same search box and the same SPICE budget. For each test spec the MLP is
re-trained WITHOUT any data at that (VOUT, Cout), so warm start is always tested on a spec it has
not seen. Results go to runs_batch/<spec>/<method>_s<seed>/ (evals.jsonl, warm_seeds.json, done.json).
It resumes: finished jobs (done.json present) are skipped, so you can stop and restart it.

  python overnight_batch.py --smoke          # 5-10 min plumbing test, writes runs_batch_smoke/
  python overnight_batch.py                  # full batch (prints an ETA after the first job)
  python overnight_batch.py --report         # summary table of whatever has finished
"""
import argparse, concurrent.futures as cf, json, math, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ldo_pdk as L
import ldo_flow as F
import surrogate_ws as W

METHODS = ["random", "de", "bo_cold", "bo_warm"]
DEFAULT_SPECS = "3.3:4,5.0:4,3.3:6.8"      # VOUT:Cout(uF). Project outputs are 3.3 V and 5.0 V only.


def tag(v, c):
    return f"gf180_{v:g}V_{c:g}uF"


def make_args(vout, cout, workers, extra=()):
    a = F.build_parser().parse_args(["--pdk", "gf180", "--vout", f"{vout:g}", "--cout", f"{cout:g}",
                                     "--workers", str(workers), "--warm", "off", *extra])
    a.seed = 0
    return a


def make_pool(a, scratch):
    sp = L.make_spec(a.vout, dropout=a.dropout, vin_over=a.vin_over)
    sp["COUT"], sp["PSRR_MIN"] = a.cout * 1e-6, a.psrr
    sp_s = dict(sp); sp_s["PSRR_MIN"] = a.psrr + a.psrr_margin
    os.makedirs(scratch, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")
    pool = cf.ProcessPoolExecutor(max_workers=a.workers, initializer=F._init_worker,
                                  initargs=(vars(a), sp, sp_s, scratch))
    return pool, sp


def evaluate(pool, run, pts, src):
    pts = [F.to_point(p, BOX) for p in pts]
    outs = list(pool.map(F._eval, pts))
    for x, (tot, text) in zip(pts, outs):
        run.record(x, tot, text, src)
    run.progress(echo=False)
    return [o[0] for o in outs]


BOX = W.BROAD["gf180"]


def m_random(pool, run, budget, seed, batch):
    rng = np.random.default_rng(seed)
    while len(run.evals) < budget:
        evaluate(pool, run, W.sample_box(rng, BOX, min(batch, budget - len(run.evals))), "random")


def m_de(pool, run, budget, seed, batch, NP=20, Fw=0.6, CR=0.9):
    """DE/rand/1/bin in log-scaled unit space; one generation = one parallel batch."""
    rng = np.random.default_rng(seed)
    lo, hi = np.log([b[0] for b in BOX]), np.log([b[1] for b in BOX])
    to_x = lambda U: np.exp(lo + np.clip(U, 0, 1) * (hi - lo))
    U = rng.random((NP, len(BOX)))
    fit = np.array(evaluate(pool, run, to_x(U), "de"))
    while len(run.evals) < budget:
        n = min(NP, budget - len(run.evals))
        trials = []
        for i in range(n):
            a, b, c = rng.choice([j for j in range(NP) if j != i], 3, replace=False)
            v = U[a] + Fw * (U[b] - U[c])
            mask = rng.random(len(BOX)) < CR
            mask[rng.integers(len(BOX))] = True
            trials.append(np.where(mask, v, U[i]))
        trials = np.clip(np.array(trials), 0, 1)
        ft = evaluate(pool, run, to_x(trials), "de")
        for i in range(n):
            if ft[i] <= fit[i]:
                U[i], fit[i] = trials[i], ft[i]


def m_bo(pool, run, budget, seed, batch, warm=None):
    if warm is None:
        F.search(pool, run, BOX, budget, max(10, budget // 5), seed, batch, float("inf"))
    else:
        x0 = warm.initial(run, BOX, seed)
        F.search(pool, run, BOX, budget, len(x0), seed, batch, float("inf"), x0=x0, x0_src="mlp-init",
                 hook=warm.hook)


def holdout_model(root, v, c):
    path = os.path.join(root, "models", f"holdout_{v:g}V_{c:g}uF.pt")
    if not os.path.exists(path):
        print(f"\n--- training hold-out MLP for {v:g} V / {c:g} uF (no data at this spec) ---", flush=True)
        W.train("gf180", [os.path.join(HERE, "runs")], path, exclude=[(v, c)])
    return path


def summarise(run, t0):
    tot = [e["total"] for e in run.evals]
    best = np.minimum.accumulate(tot) if tot else []
    return dict(n_evals=len(tot), first_pass=F.first_pass_eval(run.evals),
                n_screen=sum(e["screen"] for e in run.evals),
                best_total=float(best[-1]) if len(best) else None,
                best_at={str(k): float(best[min(k, len(best)) - 1]) for k in (25, 50, 100, 150, 200, 250, 400)
                         if len(best)}, runtime_s=round(time.time() - t0))


def report(root):
    rows = []
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not d.startswith("gf180_"):
            continue
        for j in sorted(os.listdir(os.path.join(root, d))):
            f = os.path.join(root, d, j, "done.json")
            if os.path.exists(f):
                rows.append(json.load(open(f)))
    if not rows:
        print("nothing finished yet"); return
    print(f"\n{'spec':<20}{'method':<10}{'seeds':>6}{'reached pass':>14}{'evals to 1st pass (median)':>28}"
          f"{'best score (median)':>22}{'passing designs':>17}")
    for spec in sorted({r["spec"] for r in rows}):
        for m in METHODS + ["bo_warm_static"]:
            rr = [r for r in rows if r["spec"] == spec and r["method"] == m]
            if not rr:
                continue
            fp = [r["first_pass"] for r in rr if r["first_pass"]]
            print(f"{spec:<20}{m:<10}{len(rr):>6}{f'{len(fp)}/{len(rr)}':>14}"
                  f"{(str(int(np.median(fp))) if fp else '-'):>28}{np.median([r['best_total'] for r in rr]):>22.4f}"
                  f"{int(np.median([r['n_screen'] for r in rr])):>17}")
    fl = os.path.join(root, "flow_summary.json")
    if os.path.exists(fl):
        print("\nEnd-to-end flow (single attempt each):")
        for r in json.load(open(fl)):
            print(f"  {r['mode']:<5} seed {r['seed']}: {r['status']:<20} evals={r['evaluations']} "
                  f"first pass @{r['first_pass']} runtime {r['runtime_s']}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", default=DEFAULT_SPECS)
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="from random,de,bo_cold,bo_warm,bo_warm_static (warm start without online retraining)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=250, help="SPICE evaluations per method run")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--flow-seeds", type=int, default=3, help="end-to-end ldo_flow runs per mode at 3.3 V/4 uF (0=skip)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.specs, a.seeds, a.budget, a.flow_seeds = "5.0:4", 1, 16, 0
    root = os.path.abspath(a.out or os.path.join(HERE, "runs_batch_smoke" if a.smoke else "runs_batch"))
    if a.report:
        report(root); return
    os.makedirs(root, exist_ok=True)
    specs = [(float(s.split(":")[0]), float(s.split(":")[1])) for s in a.specs.split(",")]
    bad = [v for v, _ in specs if not any(abs(v - ok) < 1e-9 for ok in F.VALID["gf180"])]
    if bad:
        raise SystemExit(f"VOUT {bad} not allowed: gf180 supports {F.VALID['gf180']} only")
    methods = [m.strip() for m in a.methods.split(",")]
    print(f"batch: {len(specs)} specs x {len(methods)} methods x {a.seeds} seeds x {a.budget} evals, "
          f"{a.workers} workers -> {root}", flush=True)
    models = {s: holdout_model(root, *s) for s in specs if any(m.startswith("bo_warm") for m in methods)}

    t_batch, n_done_here = time.time(), 0
    n_jobs = len(specs) * len(methods) * a.seeds
    for seed in range(a.seeds):                    # seed-major: a partial night still gives complete seeds
        for (v, c) in specs:
            args = make_args(v, c, a.workers)
            todo = [m for m in methods
                    if not os.path.exists(os.path.join(root, tag(v, c), f"{m}_s{seed}", "done.json"))]
            if not todo:
                continue
            pool, sp = make_pool(args, os.path.join(root, "scratch", tag(v, c)))
            try:
                for m in todo:
                    rd = os.path.join(root, tag(v, c), f"{m}_s{seed}")
                    if os.path.exists(rd):          # unfinished earlier attempt: start this job again
                        for fn in ("evals.jsonl", "log.txt", "warm_seeds.json", "progress.json"):
                            if os.path.exists(os.path.join(rd, fn)):
                                os.remove(os.path.join(rd, fn))
                    os.makedirs(rd, exist_ok=True)
                    json.dump(dict(pdk="gf180", vout=v, cout_uF=c, psrr_min_dB=args.psrr, method=m, seed=seed,
                                   budget=a.budget), open(os.path.join(rd, "spec.json"), "w"), indent=1)
                    print(f"\n=== [{tag(v, c)}] {m} seed {seed} ===", flush=True)
                    run, t0 = F.Run(args, sp, rd), time.time()
                    run.planned = a.budget
                    s = 1000 * seed + 17 + 101 * (METHODS + ["bo_warm_static"]).index(m)
                    warm = None
                    if m.startswith("bo_warm"):
                        wa = make_args(v, c, a.workers, ["--warm", "mlp", "--surrogate", models[(v, c)],
                                                         "--retrain-every", "0" if m == "bo_warm_static" else "40"])
                        wa.seed = s
                        warm = F.Warm(wa, rd)
                        if not warm.on:
                            raise SystemExit("warm start could not load its model: " + warm.status)
                    if m == "random":
                        m_random(pool, run, a.budget, s, a.workers)
                    elif m == "de":
                        m_de(pool, run, a.budget, s, a.workers)
                    elif m.startswith("bo"):
                        m_bo(pool, run, a.budget, s, a.workers, warm)
                    else:
                        raise SystemExit(f"unknown method {m}")
                    out = dict(spec=tag(v, c), method=m, seed=seed, **summarise(run, t0))
                    if warm:
                        out["warm"] = warm.summary(run)
                    json.dump(out, open(os.path.join(rd, "done.json"), "w"), indent=1)
                    n_done_here += 1
                    el = time.time() - t_batch
                    left = sum(1 for sd in range(a.seeds) for sv in specs for mm in methods
                               if not os.path.exists(os.path.join(root, tag(*sv), f"{mm}_s{sd}", "done.json")))
                    print(f"done: first pass @{out['first_pass']}, best {out['best_total']:.4f}, "
                          f"{out['n_screen']} passing, {out['runtime_s']}s | {n_jobs - left}/{n_jobs} jobs, "
                          f"ETA {left * el / n_done_here / 3600:.1f} h", flush=True)
            finally:
                pool.shutdown()

    if a.flow_seeds > 0:                            # end-to-end: the real flow, cold vs warm, one attempt each
        fl_path = os.path.join(root, "flow_summary.json")
        res = json.load(open(fl_path)) if os.path.exists(fl_path) else []
        mdl = models.get((3.3, 4.0)) or holdout_model(root, 3.3, 4.0)
        for seed in range(a.flow_seeds):
            for mode in ("cold", "warm"):
                if any(r["seed"] == seed and r["mode"] == mode for r in res):
                    continue
                print(f"\n=== end-to-end flow 3.3 V / 4 uF, {mode}, seed {seed} ===", flush=True)
                env = dict(os.environ, LDO_WARM="1" if mode == "warm" else "0", LDO_SURROGATE=mdl,
                           PYTHONUNBUFFERED="1")
                cmd = [sys.executable, "-u", os.path.join(HERE, "ldo_flow.py"), "--pdk", "gf180", "--vout", "3.3",
                       "--cout", "4", "--seed", str(5000 + seed), "--workers", str(a.workers),
                       "--runs-dir", os.path.join(root, "flow")]
                p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
                rj = [l.split("RESULT_JSON: ", 1)[1].strip() for l in p.stdout.splitlines() if l.startswith("RESULT_JSON: ")]
                if not rj or not os.path.exists(rj[-1]):
                    print("flow run failed:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-15:])); continue
                r = json.load(open(rj[-1]))
                res.append(dict(mode=mode, seed=seed, status=r["status"], evaluations=r["evaluations"],
                                first_pass=r.get("first_screen_pass_eval"), runtime_s=r["runtime_s"], result=rj[-1],
                                warm=r.get("warm_start")))
                json.dump(res, open(fl_path, "w"), indent=1)
                print(f"{mode}: {r['status']}, {r['evaluations']} evals, first pass @{r.get('first_screen_pass_eval')}, "
                      f"{r['runtime_s']} s", flush=True)
    report(root)


if __name__ == "__main__":
    main()
