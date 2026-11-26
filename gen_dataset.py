#!/usr/bin/env python3
"""Overnight dataset generator for the LDO (9T wide-swing cascode + PMOS follower buffer), GF180 or SKY130.

Every sample is one sizing (the same 8 variables BO searches) evaluated at all five
process corners: DC (3 conditions), loop-gain AC (phase margin over loads) and PSRR.
Results are appended one JSON line per sample, so the run can be stopped and resumed.

  python gen_dataset.py --vout 5.0 --timing 60     # short timing run, separate file
  python gen_dataset.py --vout 5.0                 # full run (resumes automatically)
  python gen_dataset.py --pdk sky130 --vout 3.3 --timing 60
  python gen_dataset.py --pdk sky130 --vout 3.3    # SKY130 dataset -> dataset/data_sky130_3.3V.jsonl

Must run with THIS folder's ldo_pdk.py (single pass device with m=n_pass, trimmed SKY130 library at
~/pdks/sky130_trim, made by make_sky130_trim.py). GF180 sampling and ids are unchanged from the
desktop version, so existing GF180 files resume correctly.
Since 2026-11-03 every row also stores <corner>_psrr10k_20mA: ldo_opt.py checks PSRR at tt/20 mA,
so the surrogate's PSRR target now matches the objective exactly (older rows lack it).
"""
import argparse, glob, json, math, multiprocessing as mp, os, re, subprocess, sys, time

# One thread per ngspice, and no OpenMP spin-waiting: without PASSIVE/SPINCOUNT a single SKY130 op sim
# took 9.9 s instead of 0.5 s (measured 2026-11-03); ldo_flow.py sets the same variables.
os.environ.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")

import numpy as np
from scipy.stats import qmc
import ldo_pdk as L

PDK = "gf180"                                        # set from --pdk in main() (workers inherit via fork)
NAMES = ["Win", "Lin", "Wtail", "n_pass", "Wfol", "Wbn2", "Wmirror", "Wvb"]
ALL_BOUNDS = {"gf180": [(4, 100), (0.7, 3), (4, 48), (20, 200), (4, 100), (1, 12), (4, 100), (1, 6)],
              "sky130": [(4, 100), (0.7, 3), (4, 48), (150, 2500), (4, 100), (1, 12), (4, 100), (1, 6)]}  # = ldo_flow.BROAD
BOUNDS = ALL_BOUNDS[PDK]
LOG_DIMS = {0, 2, 3, 4, 5, 6, 7}                     # sampled log-uniformly (widths, n_pass)
FIXED = dict(Lmirror=1.0, Lcas=1.0, Ltail=1.0, Lbias=1.0, Lcs=1.0, Lfol=0.5,
             Wcas=8, Wbias=8, Wbn=8, Wcs=8)
CORN = ["tt", "ss", "ff", "sf", "fs"]
AC_LOADS = [0.0, 1e-4, 2e-3, 5e-3, 2e-2, 5e-2, 0.1]
PSRR_LOADS = [0.1, 0.002, 0.02]                       # 20 mA added 2026-11-03 (ldo_opt checks tt @ 20 mA)
ALL_HISTORY = {"gf180": {5.0: ["opt_gf180_5v*.log", "flow_cold_5v*.log", "runs/*_gf180_5V/log.txt"],
                         3.3: ["opt_gf180_3v3*.log", "flow_cold_3v3*.log", "runs/*_gf180_3.3V/log.txt"]},
               "sky130": {3.3: ["sky_*.log", "flow_cold_sky130*.log", "runs/*_sky130_3.3V*/log.txt",
                                "runs_test/*_sky130_3.3V*/log.txt"]}}
ALL_KNOWN = {"gf180": {5.0: [[69.4, 1.48, 15.5, 109, 70.5, 4.83, 93.9, 1.20],     # frozen Design C
                             [76.1, 1.98, 15.3, 106, 64.4, 4.57, 98.6, 1.22]],    # frozen Design B
                       3.3: []},
             "sky130": {3.3: []}}                  # no 9T SKY130 design is frozen yet: anchors come from BO logs
HISTORY, KNOWN = ALL_HISTORY[PDK], ALL_KNOWN[PDK]
TAG = re.compile(r"Win=([\d.]+) Lin=([\d.]+) Wt=([\d.]+) n=(\d+) Wfol=([\d.]+) Wbn2=([\d.]+) "
                 r"Wm=([\d.]+) Wvb=([\d.]+).*?total=([\d.e+-]+)")


# ----------------------------------------------------------------------------- sampling
def unit_to_x(u):
    x = []
    for j, (lo, hi) in enumerate(BOUNDS):
        v = math.exp(math.log(lo) + u[j] * (math.log(hi) - math.log(lo))) if j in LOG_DIMS \
            else lo + u[j] * (hi - lo)
        x.append(int(round(v)) if j == 3 else round(float(v), 4))
    return x


def clip(x):
    out = []
    for j, (lo, hi) in enumerate(BOUNDS):
        v = min(max(x[j], lo), hi)
        out.append(int(round(v)) if j == 3 else round(float(v), 4))
    return out


def bo_history(vout):
    """All sizings BO has evaluated for this VOUT, parsed from the logs, with their scores."""
    pts = {}
    for pat in HISTORY.get(vout, []):
        for f in sorted(glob.glob(os.path.join(L.OPT, pat))):
            for line in open(f, errors="ignore"):
                m = TAG.search(line)
                if m:
                    g = [float(v) for v in m.groups()]
                    x = clip(g[:8])
                    key = tuple(x)
                    pts[key] = min(g[8], pts.get(key, 1e9))
    for f in glob.glob(os.path.join(L.OPT, f"best_{PDK}_{vout:g}V*.json")):
        try:
            d = json.load(open(f))
            s = d["sizes"]
            x = clip([s["Win"], s["Lin"], s["Wtail"], d["n_pass"], s["Wfol"], s["Wbn2"], s["Wmirror"], s["Wvb"]])
            pts[tuple(x)] = min(float(d.get("score", 0.0)), pts.get(tuple(x), 1e9))
        except Exception:
            pass
    for x in KNOWN.get(vout, []):
        pts[tuple(clip(x))] = 0.0
    return sorted(pts.items(), key=lambda kv: kv[1])


def build_tasks(vout, n_sobol, n_local, n_test, n_anchor):
    hist = bo_history(vout)
    anchors = [list(k) for k, s in hist if s < 5.0][:n_anchor] or [list(k) for k, _ in hist[:n_anchor]]
    test = [("test", i, unit_to_x(u)) for i, u in
            enumerate(np.random.default_rng(1234).random((n_test, len(NAMES))))]
    bo = [("bo", i, list(k)) for i, (k, _) in enumerate(hist)]
    m = max(1, int(math.ceil(math.log2(max(n_sobol, 2)))))
    sob = qmc.Sobol(d=len(NAMES), scramble=True, seed=7).random_base2(m)[:n_sobol]
    sobol = [("sobol", i, unit_to_x(u)) for i, u in enumerate(sob)]
    local = []
    if anchors:
        rng = np.random.default_rng(99)
        sig = [0.05, 0.15, 0.3]
        for i in range(n_local):
            a = anchors[i % len(anchors)]
            s = sig[(i // len(anchors)) % len(sig)]
            x = [a[j] * math.exp(rng.normal(0, s)) for j in range(len(NAMES))]
            x[3] = a[3] + rng.normal(0, s * a[3])
            local.append(("local", i, clip(x)))
    # interleave space-filling and local samples so a partial run stays balanced
    mixed = sorted([(i / max(len(sobol), 1), t) for i, t in enumerate(sobol)] +
                   [((i + 0.5) / max(len(local), 1), t) for i, t in enumerate(local)], key=lambda z: z[0])
    tasks = test + bo + [t for _, t in mixed]
    return [dict(id=f"{src}-{i:06d}", source=src, split="test" if src == "test" else "train", x=x)
            for src, i, x in tasks], len(hist), len(anchors)


# ----------------------------------------------------------------------------- evaluation
def evaluate(task, vout, cout_uF):
    t0 = time.time()
    task = dict(task, pdk=PDK)
    sp = L.make_spec(vout)
    cout, esr = cout_uF * 1e-6, sp["ESR"]
    d = dict(zip(NAMES, task["x"]))
    n = int(d.pop("n_pass"))
    sz = dict(FIXED); sz.update(d)
    p = L.base_params(sp, sz)
    row = dict(task, vout=vout, cout_uF=cout_uF, status="ok")
    ov_lim = L.PDKS[PDK]["vrating"] - 0.2
    try:
        R1 = sp["R1"]
        for _ in range(3):                                         # trim VOUT at tt, as BO does
            p["R1_val"] = f"{R1:.1f}"
            res, _c = L.op_run(PDK, p, n, sp)
            if res is None:
                row["status"] = "dc_fail"; return finish(row, t0)
            v = res["c0"]["v(vout)"]
            if v < 0.5 or abs(v - sp["VOUT"]) < 1e-3:
                break
            R1 = sp["R2"] * (sp["VOUT"] / (v / (1 + R1 / sp["R2"])) - 1)
            if not 0.3 * sp["R1"] < R1 < 3 * sp["R1"]:
                row["status"] = "trim_fail"; return finish(row, t0)
        p["R1_val"] = f"{R1:.1f}"
        row["R1"] = round(R1, 1)
        for c in CORN:
            res, _c = L.op_run(PDK, p, n, sp, c, 27)
            if res is None:
                row["status"] = f"dc_fail_{c}"; return finish(row, t0)
            if "-i(vin)" not in res["c0"]:
                row["status"] = f"meas_fail_iq_{c}"; return finish(row, t0)
            vn, vl, vh = (res[f"c{i}"]["v(vout)"] for i in range(3))
            devs = [r for i in range(3) for r in L.devices(PDK, res[f"c{i}"]) if not r["err"]]
            on = [r for r in devs if r["region"] != "subVt"]
            row.update({
                f"{c}_vout": vn, f"{c}_vout_light": vl, f"{c}_vout_vinmin": vh,
                f"{c}_load_mV": abs(vl - vn) * 1e3, f"{c}_line_mV": abs(vh - vn) * 1e3,
                f"{c}_err_mV": max(abs(q - sp["VOUT"]) for q in (vn, vl, vh)) * 1e3,
                f"{c}_iq_uA": (res["c0"]["-i(vin)"] - sp["IOUT"]) * 1e6,
                f"{c}_minsat_mV": min((r["margin"] for r in on), default=float("nan")) * 1e3,
                f"{c}_n_subvt": sum(r["region"] == "subVt" for r in devs),
                f"{c}_n_triode": sum(r["region"] == "TRIODE" for r in devs),
                f"{c}_maxv": max((r["maxv"] for r in devs), default=float("nan")),
            })
            for il, (pm, ugf) in zip(AC_LOADS, L.ac_run(PDK, p, n, AC_LOADS, cout, esr, c, 27)):
                row[f"{c}_pm_{il*1e3:g}mA"] = pm
                if il == 0.1:
                    row[f"{c}_ugf_100mA"] = ugf
            ps = L.psrr_run(PDK, p, n, PSRR_LOADS, cout, esr, c, 27)
            row[f"{c}_psrr1k_100mA"], row[f"{c}_psrr10k_100mA"], row[f"{c}_psrr100k_100mA"] = ps[0]
            row[f"{c}_psrr10k_2mA"] = ps[1][1]
            row[f"{c}_psrr10k_20mA"] = ps[2][1]
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
    except Exception as e:
        row["status"] = "error"; row["error"] = repr(e)[:300]
    return finish(row, t0)


def finish(row, t0):
    row["sim_s"] = round(time.time() - t0, 3)
    if row["status"] == "ok":
        sp = L.make_spec(row["vout"])
        g = lambda k: [row[f"{c}_{k}"] for c in CORN]
        pms = [row[k] if row[k] is not None else 0.0 for k in row if re.match(r"^\w\w_pm_", k)]
        ps = [v if v is not None else 0.0 for v in g("psrr10k_100mA") + g("psrr10k_2mA")]
        w = dict(line_mV=max(g("line_mV")), load_mV=max(g("load_mV")), iq_uA=max(g("iq_uA")),
                 err_mV=max(g("err_mV")), maxv=max(g("maxv")),
                 minsat_mV=float("nan") if any(math.isnan(v) for v in g("minsat_mV")) else min(g("minsat_mV")),
                 pm=min(pms), psrr10k=min(ps))
        lim = dict(line_mV=sp["VOUT"] * (sp["VIN_MAX"] - sp["VIN_MIN"]) * 10, load_mV=10 * sp["VOUT"],
                   iq_uA=100.0, err_mV=15 * sp["VOUT"], minsat_mV=50.0, maxv=L.PDKS[PDK]["vrating"] - 0.2,
                   pm=45.0, psrr10k=40.0)
        for k, v in w.items():
            row[f"worst_{k}"] = v
            ok = v >= lim[k] if k in ("minsat_mV", "pm", "psrr10k") else v <= lim[k]
            row[f"pass_{k}"] = bool(ok)
        row["pass_all"] = all(row[f"pass_{k}"] for k in w)
        P = L.PDKS[PDK]
        sz = dict(FIXED); sz.update(dict(zip(NAMES, row["x"])))
        row["area_um2"] = round(sum(sz[a] * k * sz[b] for *_, a, b, k in L.AMP) + sz["Wbias"] * sz["Lbias"]
                                + row["x"][3] * P["pass_w"] * P["pass_l"], 1)
    return row


# ----------------------------------------------------------------------------- runner
def init_worker(scratch, pdk):
    global PDK, BOUNDS
    PDK, BOUNDS = pdk, ALL_BOUNDS[pdk]      # Python 3.14 uses forkserver: workers re-import this module
    d = os.path.join(scratch, f"w{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    L.OPT = d
    orig = L.run
    L.run = lambda pdk, text, name="run", timeout=120: orig(pdk, text, name, timeout)


def work(args):
    task, vout, cout = args
    return evaluate(task, vout, cout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", choices=["gf180", "sky130"], default="gf180")
    ap.add_argument("--vout", type=float, required=True, choices=[3.3, 5.0])
    ap.add_argument("--cout", type=float, default=4.0, help="effective output cap, uF")
    ap.add_argument("--sobol", type=int, default=8000, help="space-filling samples")
    ap.add_argument("--local", type=int, default=4000, help="perturbations around good BO designs")
    ap.add_argument("--test", type=int, default=800, help="held-out uniform-random samples")
    ap.add_argument("--anchors", type=int, default=60, help="best distinct BO designs to perturb")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--timing", type=int, default=0, help="run N samples only, report speed, exit")
    ap.add_argument("--out", default=os.path.join(L.OPT, "dataset"))
    a = ap.parse_args()
    global PDK, BOUNDS, HISTORY, KNOWN
    PDK, BOUNDS, HISTORY = a.pdk, ALL_BOUNDS[a.pdk], ALL_HISTORY[a.pdk]
    KNOWN = ALL_KNOWN[a.pdk]
    if a.vout not in HISTORY:
        raise SystemExit(f"{PDK} supports VOUT {sorted(HISTORY)} only")

    os.makedirs(a.out, exist_ok=True)
    tag = f"{PDK}_{a.vout:g}V"
    tasks, n_hist, n_anch = build_tasks(a.vout, a.sobol, a.local, a.test, a.anchors)
    if a.timing:
        tasks = [t for t in tasks if t["source"] == "sobol"][:a.timing]
        path = os.path.join(a.out, f"timing_{tag}.jsonl")
        open(path, "w").close()
    else:
        path = os.path.join(a.out, f"data_{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [t for t in tasks if t["id"] not in done]
    print(f"{tag}: BO history {n_hist} points, {n_anch} anchors | tasks {len(tasks)}, "
          f"already done {len(done)}, to run {len(todo)} | {a.workers} workers | -> {path}", flush=True)
    if not todo:
        return

    t0 = time.time(); n = ok = good = 0; window = []
    with open(path, "a") as f, mp.Pool(a.workers, init_worker, (os.path.join(a.out, "scratch"), PDK)) as pool:
        for row in pool.imap_unordered(work, [(t, a.vout, a.cout) for t in todo], chunksize=1):
            f.write(json.dumps(row) + "\n"); f.flush()
            n += 1; ok += row["status"] == "ok"; good += bool(row.get("pass_all"))
            window.append(time.time()); window = window[-200:]
            if n % 50 == 0 or n == len(todo):
                rate = (len(window) - 1) / max(window[-1] - window[0], 1e-9) if len(window) > 1 else 0.0
                eta = (len(todo) - n) / rate / 3600 if rate else float("nan")
                msg = (f"[{time.strftime('%H:%M:%S')}] {n}/{len(todo)}  {rate:.2f}/s  ETA {eta:.1f} h  "
                       f"ok {100*ok/n:.0f}%  pass_all {good}")
                print(msg, flush=True)
                json.dump(dict(tag=tag, done=n, total=len(todo), rate_per_s=round(rate, 3), eta_h=round(eta, 2),
                               ok_frac=round(ok / n, 3), pass_all=good, elapsed_s=round(time.time() - t0)),
                          open(os.path.join(a.out, f"progress_{tag}.json"), "w"))
    el = time.time() - t0
    print(f"finished {n} samples in {el/60:.1f} min  ({n/el:.2f}/s, {3600*n/el:.0f}/h)", flush=True)
    if a.timing:
        rows = [json.loads(l) for l in open(path)]
        st = {}
        for r in rows:
            st[r["status"]] = st.get(r["status"], 0) + 1
        print("status counts:", st)
        print(f"median single-sample time {np.median([r['sim_s'] for r in rows]):.2f} s; "
              f"projected {3600*n/el:.0f} samples per hour with {a.workers} workers")


if __name__ == "__main__":
    main()
