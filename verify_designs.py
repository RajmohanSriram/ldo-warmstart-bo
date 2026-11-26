#!/usr/bin/env python3
"""Robustness checks for finished designs: temperature sweep and Monte Carlo mismatch yield.

Designs come from flow results (runs/*/result.json, status met_spec) or from best_*.json files.
Default (no --design): the best-margin passing design per PDK/VOUT found under runs/.
Uses the same limits as the flow's verify step. Each simulation runs in its own scratch folder.

  python verify_designs.py --list                              # which designs it would check
  python verify_designs.py --temp                              # 5 corners x (-40, 27, 125 C)
  python verify_designs.py --mc 200                            # 200 mismatch samples at tt, 27 C
  python verify_designs.py --design best_gf180_5V_4uF.json --cout 4 --temp --mc 200
Outputs go to verify_out/<design>/ (temperature.csv, mc.csv, summary.json).
"""
import argparse, concurrent.futures as cf, csv, glob, json, math, os, re, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ldo_pdk as L

CORN = ["tt", "ss", "ff", "sf", "fs"]
PM_LOADS = [0.0, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
# Mismatch-only switches. Global/process switches (SKY130 mc_pr_switch, GF180 sw_stat_global) are deliberately
# NOT listed: the MC here is a local-mismatch yield at tt, with process spread covered by the corners.
MC_SWITCH_NAMES = ["mc_mm_switch", "mismatch_switch", "mm_switch", "sw_stat_mismatch",
                   "sw_mismatch", "mismatchflag", "mc_mismatch"]


# ----------------------------------------------------------------------------- designs
def load_design(path, cout=None):
    j = json.load(open(path))
    if "design" in j:                                   # flow result.json
        d, s = j["design"], j["spec"]
        return dict(name=os.path.basename(os.path.dirname(path)), pdk=s["pdk"], vout=s["vout"],
                    cout=cout or s["cout_uF"], psrr=s.get("psrr_min_dB", 40.0), line_pct=s.get("line_pct_per_V", 1.0),
                    tol_pct=s.get("vout_tol_pct", 1.5), pm_floor=s.get("pm_floor_deg", 45.0),
                    dropout=round(s["vin_min"] - s["vout"], 3), vin_over=round(s["vin_max"] - s["vout"], 3),
                    sizes=d["sizes"], n_pass=d["n_pass"], R1=d["R1_ohm"], source=path)
    m = re.search(r"_([\d.]+)uF", os.path.basename(path))   # best_*.json from ldo_opt
    return dict(name=os.path.splitext(os.path.basename(path))[0], pdk=j["pdk"], vout=j["vout"],
                cout=cout or (float(m.group(1)) if m else 4.0), psrr=40.0, line_pct=1.0, tol_pct=1.5,
                pm_floor=45.0, dropout=0.3, vin_over=0.4, sizes=j["sizes"], n_pass=j["n_pass"], R1=j["R1"],
                source=path)


def best_from_runs(root):
    best = {}
    for f in glob.glob(os.path.join(root, "*", "result.json")):
        try:
            r = json.load(open(f))
        except ValueError:
            continue
        if r.get("status") != "met_spec" or not r.get("design"):
            continue
        k = (r["spec"]["pdk"], r["spec"]["vout"])
        if k not in best or r["design"]["worst_margin"] > best[k][0]:
            best[k] = (r["design"]["worst_margin"], f)
    return [load_design(f) for _, f in sorted(best.values(), key=lambda t: t[1])]


# ----------------------------------------------------------------------------- simulation jobs
_X = {}


def _init(scratch, extra_lines):
    os.environ.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")
    d = os.path.join(scratch, f"w{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    L.OPT = d
    orig = L.build

    def build(*args, **kw):                           # inject MC lines right after the model header
        txt = orig(*args, **kw)
        if not _X.get("lines"):
            return txt
        return txt.replace("\n.temp ", "\n" + "\n".join(_X["lines"]) + "\n.temp ", 1)   # after the model .lib
    L.build = build
    _X["base_lines"] = extra_lines


def spec_of(D):
    sp = L.make_spec(D["vout"], dropout=D["dropout"], vin_over=D["vin_over"])
    sp["COUT"] = D["cout"] * 1e-6
    return sp


def measure(job):
    D, corner, temp, mc_seed = job
    _X["lines"] = None
    if mc_seed is not None:
        _X["lines"] = [l.replace("{SEED}", str(mc_seed)) for l in _X["base_lines"]]
    try:
        sp = spec_of(D)
        p = L.base_params(sp, D["sizes"]); p["R1_val"] = f"{D['R1']:.1f}"
        n = D["n_pass"]
        r, _ = L.op_run(D["pdk"], p, n, sp, corner, temp)
        if r is None:
            return dict(corner=corner, temp=temp, seed=mc_seed, failed=True)
        v0, v1, v2 = (r[f"c{i}"]["v(vout)"] for i in range(3))
        dev = [d for i in range(3) for d in L.devices(D["pdk"], r[f"c{i}"]) if not d["err"]]
        loads = PM_LOADS if mc_seed is None else [2e-3, 2e-2, 0.1]
        ac = L.ac_run(D["pdk"], p, n, loads, sp["COUT"], sp["ESR"], corner, temp)
        pr = L.psrr_run(D["pdk"], p, n, [0.1, 0.002], sp["COUT"], sp["ESR"], corner, temp)
        # a failed measurement (None) is flagged and scored as 0 (= counts as a FAIL), never as a pass
        mfail = [f"pm@{il*1e3:g}mA" for il, q in zip(loads, ac) if q[0] is None]
        mfail += [f"psrr@{il*1e3:g}mA" for il, q in zip([0.1, 0.002], pr) if q[1] is None]
        if "-i(vin)" not in r["c0"]:
            mfail.append("iq")
        if not any(d["region"] != "subVt" for d in dev):
            mfail.append("min_sat(no devices)")
        pms = [(q[0] if q[0] is not None else 0.0) for q in ac]
        ps = [q[1] if q[1] is not None else 0.0 for q in pr]
        return dict(corner=corner, temp=temp, seed=mc_seed, failed=False, meas_failed=";".join(mfail), vout=v0, vout_err_mV=max(abs(v - D["vout"]) for v in (v0, v1, v2)) * 1e3,
                    load_mV=abs(v1 - v0) * 1e3, line_mV=abs(v2 - v0) * 1e3,
                    iq_uA=(r["c0"]["-i(vin)"] - sp["IOUT"]) * 1e6 if "-i(vin)" in r["c0"] else float("inf"),
                    min_sat_mV=min((d["margin"] for d in dev if d["region"] != "subVt"), default=0.0) * 1e3,
                    max_dev_V=max((d["maxv"] for d in dev), default=0.0), pm=min(pms), psrr=min(ps))
    except Exception as e:
        return dict(corner=corner, temp=temp, seed=mc_seed, failed=True, error=repr(e))


def checks(D, row):
    sp = spec_of(D)
    lim = dict(psrr=(">=", D["psrr"]), pm=(">=", D["pm_floor"]),
               line_mV=("<=", D["line_pct"] / 100 * D["vout"] * (sp["VIN_MAX"] - sp["VIN_MIN"]) * 1e3),
               load_mV=("<=", 0.01 * D["vout"] * 1e3), iq_uA=("<=", 100.0),
               vout_err_mV=("<=", D["tol_pct"] / 100 * D["vout"] * 1e3), min_sat_mV=(">=", 50.0),
               max_dev_V=("<=", L.PDKS[D["pdk"]]["vrating"] - 0.2))
    fails = [k for k, (op, v) in lim.items() if not (row[k] >= v if op == ">=" else row[k] <= v)]
    return fails, lim


# ----------------------------------------------------------------------------- MC set-up
def detect_mc(pdk):
    """Find the statistical/mismatch switch parameter(s) the PDK's ngspice models use."""
    found = {}
    for f in glob.glob(os.path.join(L.PDKS[pdk]["cwd"], "*")):
        if not os.path.isfile(f) or os.path.getsize(f) > 200e6:
            continue
        try:
            txt = open(f, errors="replace").read().lower()
        except OSError:
            continue
        for nm in MC_SWITCH_NAMES:
            if re.search(r"\b" + nm + r"\b", txt):
                found.setdefault(nm, os.path.basename(f))
    return found


def run_pool(jobs, workers, scratch, lines):
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(scratch, lines)) as pool:
        out = []
        for i, r in enumerate(pool.map(measure, jobs), 1):
            out.append(r)
            if i % max(1, len(jobs) // 10) == 0:
                print(f"   {i}/{len(jobs)}", flush=True)
        return out


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", nargs="*", default=None, help="result.json or best_*.json files")
    ap.add_argument("--cout", type=float, default=None, help="override Cout (uF)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--temp", action="store_true")
    ap.add_argument("--temps", default="-40,27,125")
    ap.add_argument("--mc", type=int, default=0, help="number of Monte Carlo mismatch samples")
    ap.add_argument("--mc-lines", default=None,
                    help="custom netlist lines for MC, ';'-separated, {SEED} is replaced (overrides auto-detect)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()

    designs = [load_design(p, a.cout) for p in a.design] if a.design else best_from_runs(os.path.join(HERE, a.runs))
    if not designs:
        raise SystemExit("no designs: pass --design FILE, or have met_spec results in runs/")
    for D in designs:
        print(f"{D['name']}: {D['pdk']} {D['vout']} V, Cout {D['cout']} uF, n_pass {D['n_pass']}, R1 {D['R1']/1e3:.1f}k "
              f"({os.path.relpath(D['source'], HERE)})")
    if a.list or not (a.temp or a.mc):
        return

    for D in designs:
        od = os.path.join(HERE, a.out, D["name"]); os.makedirs(od, exist_ok=True)
        scratch = os.path.join(od, "scratch")
        summ = dict(design=D["name"], source=D["source"], pdk=D["pdk"], vout=D["vout"], cout_uF=D["cout"])
        if a.temp:
            temps = [float(t) for t in a.temps.split(",")]
            print(f"\n== {D['name']}: temperature sweep, {len(CORN)} corners x {temps} C ==", flush=True)
            t0 = time.time()
            rows = run_pool([(D, c, t, None) for t in temps for c in CORN], a.workers, scratch, [])
            for r in rows:
                if not r["failed"]:
                    r["fails"] = ";".join(checks(D, r)[0])
            write_csv(os.path.join(od, "temperature.csv"), rows)
            print(f"{'corner':<7}{'T(C)':>6}{'VOUT':>9}{'line mV':>9}{'load mV':>9}{'Iq uA':>8}{'PSRR dB':>9}{'PM deg':>8}{'sat mV':>8}  fails")
            for r in rows:
                if r["failed"]:
                    print(f"{r['corner']:<7}{r['temp']:>6.0f}  SIMULATION FAILED {r.get('error', '')}"); continue
                print(f"{r['corner']:<7}{r['temp']:>6.0f}{r['vout']:>9.4f}{r['line_mV']:>9.2f}{r['load_mV']:>9.1f}{r['iq_uA']:>8.1f}"
                      f"{r['psrr']:>9.1f}{r['pm']:>8.1f}{r['min_sat_mV']:>8.0f}  {r['fails'] or 'ok'}"
                      + (f"  [MEASUREMENT FAILED: {r['meas_failed']}]" if r.get("meas_failed") else ""))
            per_t = {}
            for t in temps:
                rr = [r for r in rows if r["temp"] == t]
                per_t[str(t)] = dict(all_pass=all(not r["failed"] and not r["fails"] for r in rr),
                                     failing=sorted({f for r in rr if not r["failed"] for f in r["fails"].split(";") if f}),
                                     worst_psrr=min((r["psrr"] for r in rr if not r["failed"]), default=None),
                                     worst_pm=min((r["pm"] for r in rr if not r["failed"]), default=None))
            summ["temperature"] = per_t
            print(f"   ({time.time()-t0:.0f}s) " + "  ".join(f"{t} C: {'PASS' if v['all_pass'] else 'FAIL ' + ','.join(v['failing'])}"
                                                    for t, v in per_t.items()), flush=True)
        if a.mc:
            if a.mc_lines:
                lines = [l.strip() for l in a.mc_lines.split(";") if l.strip()]
            else:
                found = detect_mc(D["pdk"])
                print(f"\n== {D['name']}: Monte Carlo. Mismatch switches found in the PDK model files: "
                      f"{found or 'none'}", flush=True)
                lines = [f".param {nm}=1" for nm in found] + [".option seed={SEED}"]
            print("   MC netlist lines: " + " | ".join(lines), flush=True)
            # sanity check: do different seeds actually give different circuits?
            probe = run_pool([(D, "tt", 27, s) for s in (11, 22, 33, 44)], min(4, a.workers), scratch, lines)
            vs = [r["vout"] for r in probe if not r["failed"]]
            spread = (max(vs) - min(vs)) * 1e3 if len(vs) >= 2 else 0.0
            print(f"   sanity check, 4 seeds: VOUT spread {spread:.3f} mV", flush=True)
            if len(vs) < 2 or spread < 1e-3:
                summ["mc"] = dict(error="mismatch not active: different seeds gave identical results")
                print("   MISMATCH IS NOT ACTIVE with these lines, so no MC yield is reported.\n"
                      "   Send Claude the output of:\n"
                      f"   grep -il 'mismatch\\|agauss\\|mc_' {L.PDKS[D['pdk']]['cwd']}/* | head; "
                      f"grep -io '\\.param[^=]*\\(mm\\|mismatch\\|stat\\)[^=]*=' {L.PDKS[D['pdk']]['cwd']}/*.ngspice | sort -u | head -20")
            else:
                t0 = time.time()
                rows = run_pool([(D, "tt", 27, 1000 + i) for i in range(a.mc)], a.workers, scratch, lines)
                ok_rows = [r for r in rows if not r["failed"]]
                for r in ok_rows:
                    r["fails"] = ";".join(checks(D, r)[0])
                write_csv(os.path.join(od, "mc.csv"), rows)
                npass = sum(1 for r in ok_rows if not r["fails"])
                st = lambda k: (statistics.mean(r[k] for r in ok_rows), statistics.pstdev(r[k] for r in ok_rows))
                fail_counts = {}
                for r in ok_rows:
                    for f in filter(None, r["fails"].split(";")):
                        fail_counts[f] = fail_counts.get(f, 0) + 1
                summ["mc"] = dict(samples=len(rows), simulated=len(ok_rows), passing=npass,
                                  sim_failed=len(rows) - len(ok_rows),
                                  meas_failed=sum(1 for r in ok_rows if r.get("meas_failed")),
                                  mismatch_only=", ".join(l.split(None, 1)[1] for l in lines if l.lower().startswith(".param")) + ", tt 27 C",
                                  yield_pct=round(100 * npass / max(1, len(ok_rows)), 1), lines=lines,
                                  vout_mean=st("vout")[0], vout_sigma_mV=st("vout")[1] * 1e3,
                                  psrr_mean=st("psrr")[0], psrr_sigma=st("psrr")[1], fail_counts=fail_counts)
                print(f"   yield {npass}/{len(ok_rows)} = {summ['mc']['yield_pct']}%  VOUT sigma {summ['mc']['vout_sigma_mV']:.2f} mV  "
                      f"PSRR {summ['mc']['psrr_mean']:.1f} +/- {summ['mc']['psrr_sigma']:.2f} dB  fails {fail_counts}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(summ, open(os.path.join(od, "summary.json"), "w"), indent=1, default=str)
        print(f"   saved {os.path.relpath(od, HERE)}/", flush=True)


if __name__ == "__main__":
    main()
