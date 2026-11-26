#!/usr/bin/env python3
"""LDO flow: one cold optimisation run -> verified design, or an honest 'no design met spec'.
Never reads saved results; every run searches from scratch in its own folder."""
import argparse, concurrent.futures as cf, contextlib, io, json, math, os, re, sys, time
import numpy as np
import ldo_pdk as L
import ldo_opt as O

CORN = ["tt", "ss", "ff", "sf", "fs"]
PM_LOADS = [0.0, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
COUT_SWEEP = [2.2, 3.0, 4.0, 6.8, 10.0]
VALID = {"gf180": (3.3, 5.0), "sky130": (3.3,)}
BROAD = {  # Win, Lin, Wtail, n_pass, Wfol, Wbn2, Wmirror, Wvb (same order as ldo_opt.NAMES)
    "gf180":  [(4, 100), (0.7, 3), (4, 48), (20, 200), (4, 100), (1, 12), (4, 100), (1, 6)],
    "sky130": [(4, 100), (0.7, 3), (4, 48), (150, 2500), (4, 100), (1, 12), (4, 100), (1, 6)],
}
RE_M = re.compile(r"line=(-?[\d.]+)mV\(<=([\d.]+)\) load=(-?[\d.]+)mV iq=(-?[\d.]+)uA "
                  r"vout_err=(-?[\d.]+)mV PM=(-?[\d.]+) PSRR=(-?[\d.]+)dB "
                  r"minsat=(-?[\d.]+)mV\((\S+?)\) area=(\d+)")


def parse_metrics(text):
    line = next((l for l in reversed(text.splitlines()) if "total=" in l), "")
    m = RE_M.search(line)
    if not m:
        return None
    g = m.groups()
    ov = re.search(r"ov=([\d.]+) area", line)
    mf = re.search(r"meas_failed=(\S+)", line)
    return dict(meas_failed=mf.group(1).split(",") if mf else [], line=float(g[0]), load=float(g[2]), iq=float(g[3]), err=float(g[4]), pm=float(g[5]),
                psrr=float(g[6]), minsat=float(g[7]), minsat_dev=g[8], area=float(g[9]),
                ov=float(ov.group(1)) if ov else 0.0)


# ---- Pareto helpers. Direction: +1 = larger is better, -1 = smaller is better.
OBJ = {"area": -1, "iq": -1, "psrr": 1, "pm": 1}


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def nondominated(vecs):
    return [i for i, a in enumerate(vecs)
            if not any(j != i and dominates(b, a) for j, b in enumerate(vecs))]


def pick_spread(vecs, idxs, k):
    """Choose up to k members of the front, spread out: the extreme of each objective first,
    then repeatedly the point farthest (normalised) from those already chosen."""
    if len(idxs) <= k:
        return list(idxs)
    dims = len(vecs[0])
    lo = [min(vecs[i][d] for i in idxs) for d in range(dims)]
    hi = [max(vecs[i][d] for i in idxs) for d in range(dims)]
    nv = {i: [(vecs[i][d] - lo[d]) / ((hi[d] - lo[d]) or 1.0) for d in range(dims)] for i in idxs}
    chosen = []
    for d in range(dims):
        best = max(idxs, key=lambda i: nv[i][d])
        if best not in chosen and len(chosen) < k:
            chosen.append(best)
    while len(chosen) < k:
        far = max((i for i in idxs if i not in chosen),
                  key=lambda i: min(sum((nv[i][d] - nv[c][d]) ** 2 for d in range(dims)) for c in chosen))
        chosen.append(far)
    return chosen


def vobj(v, name):
    """Objective value of a verified design (worst case over corners)."""
    if name == "area":
        return float(v["area_um2"])
    key = {"iq": "Quiescent", "psrr": "PSRR", "pm": "Phase"}[name]
    return next(float(c["value"]) for c in v["checks"] if c["name"].startswith(key))


class Run:
    """Bookkeeping in the main process: logs every evaluation, progress file, screening."""
    def __init__(self, a, sp, rundir):
        self.a, self.sp, self.rundir = a, sp, rundir
        self.t0 = time.time()
        self.evals, self.stage, self.planned = [], 1, 0
        self.line_lim = a.line_pct / 100 * sp["VOUT"] * (sp["VIN_MAX"] - sp["VIN_MIN"]) * 1e3
        self.load_lim = 0.01 * sp["VOUT"] * 1e3
        self.tol_mv = a.tol_pct / 100 * sp["VOUT"] * 1e3
        self.log = open(os.path.join(rundir, "log.txt"), "a")
        self.jl = open(os.path.join(rundir, "evals.jsonl"), "a")

    def screen(self, m):
        a = self.a
        return bool(m and m["line"] <= self.line_lim and m["load"] <= self.load_lim and m["iq"] <= 100
                    and m["err"] <= self.tol_mv and m["pm"] >= a.pm_floor and m["psrr"] >= a.psrr
                    and m["minsat"] >= 50 and m["ov"] == 0.0 and not m.get("meas_failed"))

    def record(self, x, tot, text, src="bo"):
        m = parse_metrics(text)
        rec = dict(i=len(self.evals) + 1, stage=self.stage, x=[float(v) for v in x], total=tot,
                   m=m, screen=self.screen(m), src=src, t=round(time.time() - self.t0, 1))
        self.evals.append(rec)
        self.log.write(text); self.log.flush()
        self.jl.write(json.dumps(rec) + "\n"); self.jl.flush()

    def progress(self, note="", echo=True):
        best = min((e["total"] for e in self.evals), default=None)
        n = len(self.evals)
        npass = sum(e["screen"] for e in self.evals)
        json.dump(dict(stage=self.stage, evals=n, planned=self.planned, best_total=best,
                       screened_passing=npass, elapsed_s=round(time.time() - self.t0), note=note),
                  open(os.path.join(self.rundir, "progress.json"), "w"))
        if echo and best is not None:
            print(f"[stage {self.stage}] {n}/{self.planned}  best={best:.4f}  "
                  f"screened-passing={npass}  {round(time.time()-self.t0)}s", flush=True)


# ---- worker processes: each owns a private scratch dir, so parallel ngspice runs never share files
_W = {}


def configure_opt(a):
    O.PM_TT, O.PM_CORNER = a.pm_tt, a.pm_floor
    O.AC_AT = {(c, 27) for c in CORN}
    O.PSRR_LOADS = {(c, 27): [0.1] for c in CORN}
    O.PSRR_LOADS[("tt", 27)] = [0.002, 0.02, 0.1]


def _init_worker(a_dict, sp, sp_s, scratch_root):
    os.environ.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")
    a = argparse.Namespace(**a_dict)
    d = os.path.join(scratch_root, f"w{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    L.OPT = d
    configure_opt(a)
    obj, _ = O.make_objective(a.pdk, sp_s, a.line_pct, a.tol_pct)
    _W.update(a=a, sp=sp, obj=obj)


def _eval(x):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            tot = float(_W["obj"](list(x)))
        except Exception as e:
            tot = 1e6
            print(f"[debug] exception {e!r}")
    return tot, buf.getvalue()


def _verify(x):
    try:
        return verify(_W["a"], _W["sp"], x)
    except Exception as e:
        print(f"verify exception {e!r}", flush=True)
        return None


def search(pool, run, dims, n_calls, n_init, seed, batch, deadline, x0=None, x0_src="seed", hook=None):
    """Batch Bayesian optimisation: ask for `batch` points, simulate them in parallel, tell results.
    x0: points simulated first (hand seeds, or MLP warm-start proposals; they count toward n_calls).
    hook(run, dims) -> list of extra points for the next batch (online MLP proposals), or []."""
    from skopt import Optimizer
    from skopt.space import Real, Integer
    sp_dims = [Integer(int(lo), int(hi)) if j == 3 else Real(lo, hi) for j, (lo, hi) in enumerate(dims)]
    opt = Optimizer(sp_dims, base_estimator="GP", n_initial_points=max(1, n_init), random_state=seed)
    done = 0
    pts = [to_point(p, dims) for p in (x0 or [])]
    srcs = [x0_src] * len(pts)
    while done < n_calls and time.time() < deadline:
        if not pts:
            extra = [to_point(p, dims) for p in (hook(run, dims) if hook else [])][:max(0, batch - 1)]
            nask = min(batch, n_calls - done) - len(extra)
            asked = [[int(v) if j == 3 else float(v) for j, v in enumerate(p)]
                     for p in (opt.ask(n_points=nask) if nask > 0 else [])]
            pts, srcs = asked + extra, ["bo"] * len(asked) + ["mlp-online"] * len(extra)
            if not pts:
                break
        outs = list(pool.map(_eval, pts))
        ys = []
        for x, s, (tot, text) in zip(pts, srcs, outs):
            run.record(x, tot, text, s); ys.append(tot)
        opt.tell(pts, ys)
        done += len(pts); pts, srcs = [], []
        run.progress()


def box_from(designs, broad):
    dims = []
    for j, (blo, bhi) in enumerate(broad):
        vals = [d[j] for d in designs]
        lo, hi = min(vals), max(vals)
        pad = 0.2 * (hi - lo) + 0.04 * (bhi - blo)
        lo, hi = max(blo, lo - pad), min(bhi, hi + pad)
        if j == 3:
            lo, hi = int(math.floor(lo)), int(math.ceil(hi))
            if hi <= lo:
                hi = lo + 1
        elif hi - lo < 1e-3:
            hi = lo + 1e-3
        dims.append((lo, hi))
    return dims


def to_point(x, dims):
    out = []
    for j, v in enumerate(x):
        lo, hi = dims[j]
        v = min(max(v, lo), hi)
        out.append(int(round(v)) if j == 3 else float(v))
    return out


class Warm:
    """MLP warm start + online retraining, run in the main process (simulations stay in the pool).
    Writes warm_seeds.json in the run folder: every MLP-proposed point with its prediction, so the
    GUI and the paper can show predicted vs simulated."""
    def __init__(self, a, rundir):
        self.on, self.status, self.rows, self.rundir = False, "off", [], rundir
        self.k, self.every = a.warm_k, a.retrain_every
        if a.warm != "mlp":
            return
        try:
            import surrogate_ws as W
        except ImportError as e:
            self.status = f"unavailable: {e}"; return
        path = a.surrogate or W.default_path(a.pdk)
        model = W.load_or_none(path)
        if model is None:
            self.status = f"unavailable: no surrogate model at {path} (run: python surrogate_ws.py train)"
            print("WARM START: " + self.status + " -> running COLD", flush=True)
            return
        self.W, self.model, self.path = W, model, path
        self.spec = dict(vout=a.vout, cout=a.cout, psrr=a.psrr + a.psrr_margin, line_pct=a.line_pct,
                         tol_pct=a.tol_pct, pm_floor=a.pm_floor, dropout=a.dropout, vin_over=a.vin_over)
        self.spec_real = dict(self.spec, psrr=a.psrr)      # margin to the actual spec (display)
        self.online = W.Online(model, self.spec, seed=a.seed or 0) if self.every > 0 else None
        self.last_n, self.on, self.status = 0, True, "mlp"
        print(f"WARM START: MLP surrogate {path} (trained on {model.meta.get('n_train', '?')} evaluations); "
              f"{self.k} proposals per slice, online retrain every {self.every} evaluations", flush=True)

    def _log(self, pts, preds, mgs, kind, run):
        for x, p, g in zip(pts, preds, mgs):
            ps = float(self.W.margin(self.W.pred_row(p), self.spec_real)[0])
            self.rows.append(dict(x=[float(v) for v in x], pred=p, pred_margin=g, pred_margin_spec=ps,
                                  kind=kind, after_eval=len(run.evals)))
        json.dump(dict(status=self.status, model=getattr(self, "path", None),
                       model_meta={k: v for k, v in getattr(self, "model", None).meta.items()
                                   if k != "val_idx_sample"} if self.on else None,
                       rows=self.rows), open(os.path.join(self.rundir, "warm_seeds.json"), "w"), indent=1)

    def initial(self, run, dims, seed):
        if not self.on:
            return []
        t0 = time.time()
        pts, preds, mgs = self.W.propose(self.model, self.spec, dims, self.k, seed=seed,
                                         avoid=[e["x"] for e in run.evals])
        self._log(pts, preds, mgs, "init", run)
        print(f"WARM START: {len(pts)} MLP proposals in {time.time()-t0:.1f}s, predicted margins "
              + ", ".join(f"{g:+.2f}" for g in mgs), flush=True)
        return pts

    def hook(self, run, dims):
        if not self.on or self.online is None or len(run.evals) - self.last_n < self.every:
            return []
        self.last_n = len(run.evals)
        t0 = time.time()
        if not self.online.update(run.evals):
            return []
        pts, preds, mgs = self.online.propose(dims, 1, [e["x"] for e in run.evals], seed=len(run.evals))
        self._log(pts, preds, mgs, "online", run)
        print(f"ONLINE: MLP retrained on {len(run.evals)} run evaluations ({time.time()-t0:.1f}s), "
              f"proposing {len(pts)} point(s)", flush=True)
        return pts

    def summary(self, run):
        if self.status == "off":
            return dict(mode="cold")
        out = dict(mode="mlp" if self.on else "cold", status=self.status, proposals=len(self.rows),
                   online_retrains=self.online.n_retrains if self.on and self.online else 0)
        sim = {}
        for e in run.evals:
            if e.get("src", "").startswith("mlp"):
                sim.setdefault(tuple(round(v, 4) for v in e["x"]), e)
        hits = [sim.get(tuple(round(v, 4) for v in r["x"])) for r in self.rows]
        out["proposals_simulated"] = sum(h is not None for h in hits)
        out["proposals_passing_screen"] = sum(bool(h and h["screen"]) for h in hits)
        return out


def top_verified(verified, best, k):
    """The k best verified candidates, full detail, whether or not they meet the spec.
    Order: designs meeting every spec first (largest worst-case margin, then smaller area), then the rest by
    number of checks passed and worst margin. The reported design (best) is always rank 1."""
    rest = sorted((v for v in verified if v is not best),
                  key=lambda v: (not v["all_ok"], -sum(c["ok"] for c in v["checks"]), -v["worst_margin"],
                                 v["area_um2"]))
    return [dict(v, rank=i + 1) for i, v in enumerate(([best] if best else []) + rest)][:k]


def first_pass_eval(evals):
    return next((e["i"] for e in evals if e["screen"]), None)


def trimmed(pdk, sp, x):
    d = dict(zip(O.NAMES, [float(v) for v in x]))
    n = int(round(d.pop("n_pass")))
    sz = dict(O.FIXED); sz.update(d)
    p = L.base_params(sp, sz)
    R1 = sp["R1"]
    for _ in range(3):
        p["R1_val"] = f"{R1:.1f}"
        res, _c = L.op_run(pdk, p, n, sp)
        if res is None:
            return None
        v = res["c0"]["v(vout)"]
        if abs(v - sp["VOUT"]) < 1e-3:
            break
        R1 = sp["R2"] * (sp["VOUT"] / (v / (1 + R1 / sp["R2"])) - 1)
    p["R1_val"] = f"{R1:.1f}"
    return p, n, sz, R1


def verify(a, sp, x):
    pdk = a.pdk
    P = L.PDKS[pdk]
    t = trimmed(pdk, sp, x)
    if t is None:
        return None
    p, n, sz, R1 = t
    line_lim = a.line_pct / 100 * sp["VOUT"] * (sp["VIN_MAX"] - sp["VIN_MIN"]) * 1e3
    load_lim = 0.01 * sp["VOUT"] * 1e3
    tol_mv = a.tol_pct / 100 * sp["VOUT"] * 1e3
    rows, worst = [], dict(line=0, load=0, iq=0, err=0, sat=1e9, mv=0)
    mfail = []                                      # failed measurements: scored as failing, and listed
    for c in CORN:
        r, _c = L.op_run(pdk, p, n, sp, c, 27)
        if r is None:
            return None
        v0, v1, v2 = (r[f"c{i}"]["v(vout)"] for i in range(3))
        if "-i(vin)" in r["c0"]:
            iq = (r["c0"]["-i(vin)"] - sp["IOUT"]) * 1e6
        else:
            iq = float("inf"); mfail.append(f"iq@{c}")
        dev = [d for i in range(3) for d in L.devices(pdk, r[f"c{i}"]) if not d["err"]]
        sat = min((d["margin"] for d in dev if d["region"] != "subVt"), default=0.0) * 1e3
        mv = max(d["maxv"] for d in dev)
        err = max(abs(vv - sp["VOUT"]) for vv in (v0, v1, v2)) * 1e3
        row = dict(corner=c, vout_full=v0, vout_light=v1, vout_vinmin=v2, load_mV=abs(v1 - v0) * 1e3,
                   line_mV=abs(v2 - v0) * 1e3, iq_uA=iq, min_sat_mV=sat, max_dev_V=mv, vout_err_mV=err)
        rows.append(row)
        worst["line"] = max(worst["line"], row["line_mV"]); worst["load"] = max(worst["load"], row["load_mV"])
        worst["iq"] = max(worst["iq"], iq); worst["err"] = max(worst["err"], err)
        worst["sat"] = min(worst["sat"], sat); worst["mv"] = max(worst["mv"], mv)
    couts = sorted(set(COUT_SWEEP + [a.cout]))
    pm_tab, min_cout = {}, None
    for co in couts:
        entry = {}
        for c in CORN:
            acs = L.ac_run(pdk, p, n, PM_LOADS, co * 1e-6, sp["ESR"], c, 27)
            mfail += [f"pm@{c}/{il*1e3:g}mA/{co:g}uF" for il, q in zip(PM_LOADS, acs) if q[0] is None]
            pms = [(q[0] if q[0] is not None else 0.0) for q in acs]
            k = min(range(len(pms)), key=lambda i: pms[i])
            entry[c] = dict(pm=pms[k], at_load_mA=PM_LOADS[k] * 1e3)
        pm_tab[co] = entry
        if min_cout is None and min(e["pm"] for e in entry.values()) >= a.pm_floor:
            min_cout = co
    psrr_tab = {}
    for c in CORN:
        vals = {}
        for il in (0.1, 0.002):
            q = L.psrr_run(pdk, p, n, [il], a.cout * 1e-6, sp["ESR"], c, 27)[0][1]
            if q is None:
                mfail.append(f"psrr@{c}/{il*1e3:g}mA")
            vals[f"{il*1e3:g}mA"] = q if q is not None else 0.0
        psrr_tab[c] = vals
    psrr_w = min(v for d in psrr_tab.values() for v in d.values())
    pm_w = min(e["pm"] for e in pm_tab[a.cout].values())
    checks = [("PSRR at 10 kHz (dB)", psrr_w, ">=", a.psrr),
              ("Phase margin at spec Cout (deg)", pm_w, ">=", a.pm_floor),
              ("Line regulation (mV)", worst["line"], "<=", line_lim),
              ("Load regulation (mV)", worst["load"], "<=", load_lim),
              ("Quiescent current (uA)", worst["iq"], "<=", 100.0),
              ("VOUT accuracy (mV)", worst["err"], "<=", tol_mv),
              ("Saturation margin (mV)", worst["sat"], ">=", 50.0),
              ("Max device voltage (V)", worst["mv"], "<=", P["vrating"] - 0.2)]
    out, margins = [], []
    for name, val, op, lim in checks:
        ok = val >= lim if op == ">=" else val <= lim
        mg = (val - lim) / abs(lim) if op == ">=" else (lim - val) / abs(lim)
        margins.append(min(mg, 0.5))
        out.append(dict(name=name, value=round(val, 3), op=op, limit=round(lim, 3), ok=bool(ok)))
    area = sum(sz[w] * k * sz[l] for _, _, _, _, _, _, w, l, k in L.AMP) + sz["Wbias"] * sz["Lbias"] \
        + n * P["pass_w"] * P["pass_l"]
    return dict(x=[float(v) for v in x], sizes={k: round(float(v), 4) for k, v in sz.items()},
                n_pass=n, pass_unit_um=[P["pass_w"], P["pass_l"]], R1_ohm=round(R1, 1),
                R2_ohm=sp["R2"], checks=out, meas_failed=mfail,
                all_ok=all(c["ok"] for c in out) and not [f for f in mfail if "uF" not in f or f.endswith(f"/{a.cout:g}uF")],
                worst_margin=round(min(margins), 3), min_cout_uF=min_cout, area_um2=round(area),
                corners=rows, phase_margin_by_cout={str(k): v for k, v in pm_tab.items()},
                psrr_dB=psrr_tab)


def build_parser():
    ap = argparse.ArgumentParser(description="LDO optimisation run (parallel), cold or MLP warm start")
    ap.add_argument("--pdk", choices=["gf180", "sky130"], default="gf180")
    ap.add_argument("--vout", type=float, default=3.3)
    ap.add_argument("--cout", type=float, default=4.0, help="effective output cap, uF")
    ap.add_argument("--psrr", type=float, default=40.0)
    ap.add_argument("--psrr-margin", type=float, default=1.0, help="search aims this many dB above spec")
    ap.add_argument("--line-pct", type=float, default=1.0)
    ap.add_argument("--tol-pct", type=float, default=1.5)
    ap.add_argument("--pm-floor", type=float, default=45.0)
    ap.add_argument("--pm-tt", type=float, default=50.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--vin-over", type=float, default=0.4)
    ap.add_argument("--calls1", type=int, default=250)
    ap.add_argument("--calls2", type=int, default=120)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--verify-top", type=int, default=5)
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)),
                    help="parallel ngspice processes (each uses ~0.4 GB RAM)")
    ap.add_argument("--pareto", default="area,iq,psrr",
                    help="objectives for the trade-off front, from: area,iq,psrr,pm")
    ap.add_argument("--pareto-points", type=int, default=6,
                    help="how many front designs to verify (0 disables the front)")
    ap.add_argument("--strata", type=int, default=3, help="stage-1 slices of the pass-FET range")
    ap.add_argument("--seed", type=int, default=None, help="default: new random seed each run")
    ap.add_argument("--quick", action="store_true", help="80 + 40 calls; may miss the spec")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the plumbing")
    ap.add_argument("--warm", choices=["off", "mlp"],
                    default="mlp" if os.environ.get("LDO_WARM", "0") in ("1", "mlp") else "off",
                    help="initialise BO with MLP-proposed designs (env LDO_WARM=1 sets this; the GUI uses it)")
    ap.add_argument("--warm-k", type=int, default=int(os.environ.get("LDO_WARM_K", "8")),
                    help="MLP proposals per stage-1 slice (replace the random initial points)")
    ap.add_argument("--retrain-every", type=int, default=int(os.environ.get("LDO_RETRAIN_EVERY", "40")),
                    help="online: retrain the MLP on this run's simulations every N evaluations (0 = off)")
    ap.add_argument("--surrogate", default=None, help="model path (default surrogate_ws/<pdk>.pt or env LDO_SURROGATE)")
    ap.add_argument("--runs-dir", default=os.environ.get("LDO_RUNS_DIR", ""),
                    help="where run folders go (default <optimizer>/runs)")
    return ap


def main():
    a = build_parser().parse_args()
    if not any(abs(a.vout - v) < 1e-9 for v in VALID[a.pdk]):
        print("RESULT_JSON: " + json.dumps(dict(status="refused", message=
              f"{a.pdk} supports VOUT {VALID[a.pdk]} only")))
        sys.exit(2)
    if a.quick:
        a.calls1, a.calls2 = 80, 40
    if a.smoke:
        a.calls1, a.calls2, a.verify_top = 16, 16, 2
    seed = a.seed if a.seed is not None else int(time.time()) % (2 ** 31)
    workers = max(1, a.workers)

    a.seed = seed
    base = os.path.abspath(a.runs_dir) if a.runs_dir else os.path.join(L.OPT, "runs")
    rundir = os.path.join(base, time.strftime("%Y%m%d_%H%M%S") + f"_{a.pdk}_{a.vout:g}V"
                          + ("_warm" if a.warm == "mlp" else ""))
    while os.path.exists(rundir):
        rundir += "x"
    scratch = os.path.join(rundir, "scratch")
    os.makedirs(scratch, exist_ok=True)
    json.dump(dict(pdk=a.pdk, vout=a.vout, cout_uF=a.cout, psrr_min_dB=a.psrr, line_pct_per_V=a.line_pct,
                   vout_tol_pct=a.tol_pct, pm_floor_deg=a.pm_floor, dropout=a.dropout, vin_over=a.vin_over,
                   seed=seed, warm=a.warm), open(os.path.join(rundir, "spec.json"), "w"), indent=1)

    sp = L.make_spec(a.vout, dropout=a.dropout, vin_over=a.vin_over)
    sp["COUT"], sp["PSRR_MIN"] = a.cout * 1e-6, a.psrr
    sp_s = dict(sp); sp_s["PSRR_MIN"] = a.psrr + a.psrr_margin

    print(f"LDO flow: {a.pdk} VOUT {a.vout} V, VIN {sp['VIN_MIN']}-{sp['VIN_MAX']} V, {sp['IOUT']*1e3:g} mA, "
          f"Cout {a.cout} uF, PSRR>={a.psrr} dB, seed {seed}, {workers} workers\nrun folder: {rundir}", flush=True)

    run = Run(a, sp, rundir)
    warm = Warm(a, rundir)
    deadline = run.t0 + a.max_minutes * 60
    broad = BROAD[a.pdk]
    os.environ.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")
    pool = cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                  initargs=(vars(a), sp, sp_s, scratch))

    # Stage 1 is split into slices of the pass-FET count (log-spaced). A single global model tends to
    # settle in one pass-FET basin early; searching each slice separately keeps every basin represented.
    run.stage, run.planned = 1, a.calls1
    k = max(1, a.strata)
    lo, hi = broad[3]
    edges = [int(round(lo * (hi / lo) ** (i / k))) for i in range(k + 1)]
    edges[0], edges[-1] = int(lo), int(hi)
    per = max(8, a.calls1 // k)
    for i in range(k):
        sl = list(broad)
        sl[3] = (edges[i], max(edges[i + 1], edges[i] + 1))
        print(f"stage 1 slice {i + 1}/{k}: pass units {sl[3][0]}-{sl[3][1]}, {per} calls", flush=True)
        sdl = run.t0 + 0.6 * a.max_minutes * 60 * (i + 1) / k
        if warm.on:   # MLP proposals replace the random initial points; same total budget per slice
            x0 = warm.initial(run, sl, seed + 10 * i)
            search(pool, run, sl, per, len(x0), seed + 10 * i, workers, sdl, x0=x0, x0_src="mlp-init",
                   hook=warm.hook)
        else:
            search(pool, run, sl, per, max(4, per // 5), seed + 10 * i, workers, sdl)

    ranked = sorted((e for e in run.evals if e["total"] < 1e5), key=lambda e: e["total"])
    seen, top = set(), []
    for e in ranked:
        key = tuple(round(v, 2) for v in e["x"])
        if key not in seen:
            seen.add(key); top.append(e)
        if len(top) == 8:
            break
    if top and time.time() < deadline:
        dims = box_from([e["x"] for e in top], broad)
        n_init2 = len(top) + 3
        calls2 = max(a.calls2, n_init2)
        run.stage, run.planned = 2, len(run.evals) + calls2
        search(pool, run, dims, calls2, n_init2, seed + 1, workers, deadline,
               x0=[to_point(e["x"], dims) for e in top], hook=warm.hook if warm.on else None)

    run.stage = 3
    run.progress("verifying")
    pool_s = sorted((e for e in run.evals if e["screen"]), key=lambda e: e["total"])
    if not pool_s:
        pool_s = sorted((e for e in run.evals if e["total"] < 1e5), key=lambda e: e["total"])
    cands, seen = [], set()
    for e in pool_s:
        key = tuple(round(v, 1) for v in e["x"])
        if key not in seen:
            seen.add(key); cands.append(e)
        if len(cands) == a.verify_top:
            break
    names = [n.strip() for n in a.pareto.split(",") if n.strip()]
    bad = [n for n in names if n not in OBJ]
    if bad:
        sys.exit(f"unknown Pareto objective(s) {bad}; choose from {sorted(OBJ)}")
    front_xs = []
    if a.pareto_points > 0 and names:
        uniq, seen2 = [], set()
        for e in run.evals:
            if e["screen"] and e["m"]:
                key = tuple(round(v, 1) for v in e["x"])
                if key not in seen2:
                    seen2.add(key); uniq.append(e)
        if uniq:
            vecs = [[OBJ[n] * e["m"][n] for n in names] for e in uniq]
            front = pick_spread(vecs, nondominated(vecs), a.pareto_points)
            front_xs = [uniq[i]["x"] for i in front]
            print(f"Pareto: {len(uniq)} designs passed the screen; verifying {len(front_xs)} from the front",
                  flush=True)
    verify_xs = [e["x"] for e in cands]
    have = {tuple(round(v, 1) for v in x) for x in verify_xs}
    for x in front_xs:
        if tuple(round(v, 1) for v in x) not in have:
            have.add(tuple(round(v, 1) for v in x)); verify_xs.append(x)
    verified = []
    for v in pool.map(_verify, verify_xs):
        if v:
            verified.append(v)
            print(f"verified candidate: all_ok={v['all_ok']}  worst margin={v['worst_margin']}  "
                  f"area={v['area_um2']}", flush=True)
    pool.shutdown()
    passing = sorted([v for v in verified if v["all_ok"]], key=lambda v: (-v["worst_margin"], v["area_um2"]))
    pareto = []
    if passing and names and a.pareto_points > 0:
        vvecs = [[OBJ[n] * vobj(v, n) for n in names] for v in passing]
        seen3 = set()
        for i in sorted(nondominated(vvecs), key=lambda i: passing[i]["area_um2"]):
            key = tuple(round(x, 3) for x in vvecs[i])
            if key in seen3:
                continue
            seen3.add(key)
            d = dict(passing[i])
            d["objectives"] = {n: round(vobj(passing[i], n), 3) for n in names}
            pareto.append(d)
    if passing:
        best, status = passing[0], "met_spec"
        msg = "Design verified against all specs at all five process corners (27 C only)."
    elif verified:
        best = max(verified, key=lambda v: (sum(c["ok"] for c in v["checks"]), v["worst_margin"]))
        status = "no_design_met_spec"
        msg = "No design met every spec. Closest design shown with its failing items."
    else:
        best, status, msg = None, "error", "No candidate could be simulated."
    result = dict(status=status, message=msg,
                  spec=dict(pdk=a.pdk, vout=a.vout, vin_min=sp["VIN_MIN"], vin_max=sp["VIN_MAX"],
                            iout_A=sp["IOUT"], cout_uF=a.cout, psrr_min_dB=a.psrr, line_pct_per_V=a.line_pct,
                            vout_tol_pct=a.tol_pct, pm_floor_deg=a.pm_floor, temperature="27 C only"),
                  seed=seed, workers=workers, evaluations=len(run.evals),
                  runtime_s=round(time.time() - run.t0),
                  warm_start=warm.summary(run), first_screen_pass_eval=first_pass_eval(run.evals),
                  run_dir=rundir, design=best, pareto=pareto, pareto_objectives=names,
                  failing=[c for c in best["checks"] if not c["ok"]] if best else [],
                  other_verified=[dict(all_ok=v["all_ok"], worst_margin=v["worst_margin"], area_um2=v["area_um2"])
                                  for v in verified if v is not best],
                  top_verified=top_verified(verified, best, 3))
    json.dump(result, open(os.path.join(rundir, "result.json"), "w"), indent=1)
    run.progress("done", echo=False)
    print(f"\nSTATUS: {status}\n{msg}")
    if len(pareto) > 1:
        print(f"Trade-off front ({', '.join(names)}): {len(pareto)} verified designs")
        for d in pareto:
            print("  " + "  ".join(f"{k}={v}" for k, v in d["objectives"].items()) + f"  n_pass={d['n_pass']}")
    if best:
        print(f"n_pass={best['n_pass']} ({best['pass_unit_um'][0]}x{best['pass_unit_um'][1]} um), "
              f"R1={best['R1_ohm']/1e3:.2f}k, R2={best['R2_ohm']/1e3:.0f}k, area={best['area_um2']} um2, "
              f"min Cout={best['min_cout_uF']} uF")
        for c in best["checks"]:
            print(f"  {'OK  ' if c['ok'] else 'FAIL'} {c['name']}: {c['value']} (limit {c['op']} {c['limit']})")
    print("RESULT_JSON: " + os.path.join(rundir, "result.json"))


if __name__ == "__main__":
    main()

