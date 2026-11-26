import argparse, json, subprocess
import ldo_pdk as L

NAMES = ["Win", "Lin", "Wtail", "n_pass", "Wfol", "Wbn2", "Wmirror", "Wvb"]
FIXED = dict(Lmirror=1.0, Lcas=1.0, Ltail=1.0, Lbias=1.0, Lcs=1.0, Lfol=0.5,
             Wcas=8, Wbias=8, Wbn=8, Wcs=8)
CORNERS = [("tt", 27), ("ss", 27), ("ff", 27), ("sf", 27), ("fs", 27)]
AC_AT = {("tt", 27), ("ss", 27), ("ff", 27), ("sf", 27), ("fs", 27)}
AC_LOADS = [0.002, 0.005, 0.02, 0.05, 0.1]
PSRR_LOADS = {("tt", 27): [0.002, 0.02, 0.1], ("ss", 27): [0.1], ("ff", 27): [0.1], ("sf", 27): [0.1], ("fs", 27): [0.1]}
PM_TT, PM_CORNER = 60.0, 45.0
IQ_MAX, SAT_MIN, AREA_W = 100e-6, 0.05, 1e-4
SEEDS = [[95.1,1.74,15.4,111,72.8,5.85,80.9,1.18],[69.4,1.48,15.5,109,70.5,4.83,93.9,1.20],
         [50.0,2.86,27.7,105,44.4,7.23,90.6,1.59],[70.1,2.68,27.8,101,41.1,4.08,67.6,1.59],
         [44.1,2.93,28.1,96,58.6,7.34,74.2,1.64]]
over = lambda v, lim: ((v - lim) / lim) ** 2 if v > lim else 0.0


def make_objective(pdk, sp, line_pct, tol_pct):
    P = L.PDKS[pdk]
    line_lim = line_pct / 100 * sp["VOUT"] * (sp["VIN_MAX"] - sp["VIN_MIN"])
    load_lim = 0.01 * sp["VOUT"]
    tol = tol_pct / 100 * sp["VOUT"]
    ov_lim = P["vrating"] - 0.2
    state = {}

    def area(sz, n):
        a = sum(sz[w] * k * sz[l] for _, _, _, _, _, _, w, l, k in L.AMP)
        return a + sz["Wbias"] * sz["Lbias"] + n * P["pass_w"] * P["pass_l"]

    def obj(x):
        d = dict(zip(NAMES, [float(v) for v in x]))
        n = int(round(d.pop("n_pass")))
        sz = dict(FIXED); sz.update(d)
        tag = (f"Win={sz['Win']:.1f} Lin={sz['Lin']:.2f} Wt={sz['Wtail']:.1f} n={n} Wfol={sz['Wfol']:.1f} "
               f"Wbn2={sz['Wbn2']:.2f} Wm={sz['Wmirror']:.1f} Wvb={sz['Wvb']:.2f}")
        p = L.base_params(sp, sz)
        try:
            R1 = sp["R1"]
            for _ in range(3):                                   # trim at tt / 27C
                p["R1_val"] = f"{R1:.1f}"
                res, _c = L.op_run(pdk, p, n, sp)
                if res is None:
                    print(f"  [debug] {tag} | DC failed"); return 1e6
                v = res["c0"]["v(vout)"]
                if v < 0.5 or abs(v - sp["VOUT"]) < 1e-3:
                    break
                R1 = sp["R2"] * (sp["VOUT"] / (v / (1 + R1 / sp["R2"])) - 1)
                if not 0.3 * sp["R1"] < R1 < 3 * sp["R1"]:
                    print(f"  [debug] {tag} | trim out of range"); return 1e6
            p["R1_val"] = f"{R1:.1f}"
            pen = dict(line=0.0, load=0.0, iq=0.0, acc=0.0, pm=0.0, psrr=0.0, sat=0.0, ov=0.0)
            w = dict(line=0.0, load=0.0, iq=0.0, acc=0.0, pm=999.0, psrr=999.0)
            sat = []
            mfail = []              # measurements that failed: penalised via placeholder, logged as meas_failed=
            for corner, temp in CORNERS:
                res, _c = L.op_run(pdk, p, n, sp, corner, temp)
                if res is None:
                    print(f"  [debug] {tag} | DC failed at {corner}"); return 1e6
                vn, vl, vh = (res[f"c{i}"]["v(vout)"] for i in range(3))
                if "-i(vin)" not in res["c0"]:
                    print(f"  [debug] {tag} | Iq not measured at {corner}"); return 1e6
                iq = res["c0"]["-i(vin)"] - sp["IOUT"]
                ld, ln = abs(vl - vn), abs(vh - vn)
                acc = max(abs(vv - sp["VOUT"]) for vv in (vn, vl, vh))
                pen["line"] += over(ln, line_lim); pen["load"] += over(ld, load_lim)
                pen["iq"] += over(iq, IQ_MAX); pen["acc"] += over(acc, tol)
                w["line"], w["load"] = max(w["line"], ln), max(w["load"], ld)
                w["iq"], w["acc"] = max(w["iq"], iq), max(w["acc"], acc)
                for i in range(3):
                    for r in L.devices(pdk, res[f"c{i}"]):
                        if r["err"]:
                            continue
                        if r["region"] != "subVt":
                            sat.append((r["margin"], f"{r['name']}@{corner}"))
                        if r["maxv"] > ov_lim:
                            pen["ov"] += 10 * (r["maxv"] - ov_lim) ** 2
                if (corner, temp) in AC_AT:
                    lim = PM_TT if corner == "tt" else PM_CORNER
                    for il, (pm, _u) in zip(AC_LOADS, L.ac_run(pdk, p, n, AC_LOADS, sp["COUT"], sp["ESR"], corner, temp)):
                        if pm is None:      # no 0 dB crossing / meas failed: 0 deg = penalty placeholder
                            mfail.append(f"pm@{corner}/{il*1e3:g}mA"); pm = 0.0
                        w["pm"] = min(w["pm"], pm)
                        if pm < lim:
                            pen["pm"] += 10 * ((lim - pm) / lim) ** 2
                    ploads = PSRR_LOADS[(corner, temp)]
                    for il, ps in zip(ploads, L.psrr_run(pdk, p, n, ploads, sp["COUT"], sp["ESR"], corner, temp)):
                        val = ps[1]
                        if val is None:     # 0 dB = penalty placeholder
                            mfail.append(f"psrr@{corner}/{il*1e3:g}mA"); val = 0.0
                        w["psrr"] = min(w["psrr"], val)
                        if val < sp["PSRR_MIN"]:
                            pen["psrr"] += 10 * ((sp["PSRR_MIN"] - val) / sp["PSRR_MIN"]) ** 2
        except subprocess.TimeoutExpired:
            print(f"  [debug] {tag} | timeout"); return 1e6
        pen["sat"] = sum(((SAT_MIN - m) / SAT_MIN) ** 2 for m, _ in sat if m < SAT_MIN)
        wm, wd = min(sat) if sat else (float("nan"), "-")
        a = area(sz, n)
        total = sum(pen.values()) + AREA_W * a
        state.update(R1=R1, sizes=sz, n_pass=n)
        print(f"  [debug] {tag} R1={R1/1e3:.1f}k | line={w['line']*1e3:.1f}mV(<={line_lim*1e3:.1f}) "
              f"load={w['load']*1e3:.1f}mV iq={w['iq']*1e6:.0f}uA vout_err={w['acc']*1e3:.0f}mV "
              f"PM={w['pm']:.0f} PSRR={w['psrr']:.1f}dB minsat={wm*1e3:.0f}mV({wd}) area={a:.0f} | pen "
              + " ".join(f"{k}={v:.2f}" for k, v in pen.items()) + f" area={AREA_W*a:.2f} | total={total:.4f}"
              + (f" | meas_failed={','.join(mfail)}" if mfail else ""))
        return total
    return obj, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", default="gf180")
    ap.add_argument("--vout", type=float, default=3.3)
    ap.add_argument("--calls", type=int, default=150)
    ap.add_argument("--init", type=int, default=30)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--vin-over", type=float, default=0.4)
    ap.add_argument("--line-pct", type=float, default=1.0)
    ap.add_argument("--tol-pct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pm-tt", type=float, default=60.0)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--cout", type=float, default=0.8, help="effective output cap, uF")
    ap.add_argument("--psrr", type=float, default=40.0, help="min PSRR at 10 kHz, dB")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sp = L.make_spec(a.vout, dropout=a.dropout, vin_over=a.vin_over)
    sp["COUT"] = a.cout * 1e-6
    sp["PSRR_MIN"] = a.psrr
    globals()["PM_TT"] = a.pm_tt
    obj, state = make_objective(a.pdk, sp, a.line_pct, a.tol_pct)
    X0 = [28.5, 1.5, 10, 148 if a.vout < 4 else 110, 46, 8, 19.1, 2.28]
    print(f"{a.pdk}: VOUT {a.vout} V, VIN {sp['VIN_MIN']}-{sp['VIN_MAX']} V, {sp['IOUT']*1e3:g} mA, "
          f"line <= {a.line_pct}%/V, PSRR >= {sp['PSRR_MIN']:.0f} dB @ {sp['PSRR_F']/1e3:g} kHz")
    if a.check:
        print("score:", obj(X0)); return
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    space = [Real(4, 100), Real(0.7, 3), Real(4, 48), Integer(20, 200), Real(4, 100),
             Real(1, 12), Real(4, 100), Real(1, 6)]
    if a.refine:
        space = [Real(20,100), Real(1.2,3), Real(8,32), Integer(85,130), Real(20,100),
                 Real(3,9), Real(40,100), Real(1,2.5)]
        seeds = SEEDS
    else:
        seeds = [X0]
    r = gp_minimize(obj, space, x0=seeds, n_calls=a.calls, n_initial_points=a.init,
                    random_state=a.seed, verbose=True)
    print("\n=== Optimization complete ===")
    print("Best", NAMES, [float(v) for v in r.x], "score", r.fun)
    obj(r.x)
    out = f"best_{a.pdk}_{a.vout:g}V_{a.cout:g}uF.json"
    json.dump(dict(pdk=a.pdk, vout=a.vout, R1=state["R1"], n_pass=state["n_pass"],
                   sizes=state["sizes"], score=float(r.fun)), open(out, "w"), indent=1)
    print("saved", out)

if __name__ == "__main__":
    main()
