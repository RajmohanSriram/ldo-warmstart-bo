"""PDK-agnostic LDO netlist builder and checks. All sizes in um."""
import os, re, subprocess

OPT = os.path.expanduser("~/ldo_project/optimizer")
_GF = os.path.expanduser("~/pdks/gf180mcuD/libs.tech/ngspice")
_SK = os.path.expanduser("~/pdks/sky130A/libs.tech/ngspice")

PDKS = {
    "gf180": dict(cwd=_GF, sc=1e-6, nmos="nfet_06v0", pmos="pfet_06v0", vrating=6.0,
                  inner=lambda m: "m0",
                  header=lambda c: f".include {_GF}/design.ngspice\n.lib {_GF}/sm141064.ngspice {c}",
                  corners={"tt": "typical", "ss": "ss", "ff": "ff", "sf": "sf", "fs": "fs"},
                  pass_w=100, pass_l=0.5),
    "sky130": dict(cwd=_SK, sc=1.0, nmos="sky130_fd_pr__nfet_g5v0d10v5",
                   pmos="sky130_fd_pr__pfet_g5v0d10v5", vrating=5.0,
                   inner=lambda m: f"m{m}",
                   header=lambda c: f".lib {os.path.expanduser("~/pdks/sky130_trim/sky130_trim.lib.spice")} {c}",
                   corners={k: k for k in ("tt", "ss", "ff", "sf", "fs")},
                   pass_w=10, pass_l=0.5),
}

# (name, drain, gate, source, bulk, type, W-param, L-param, W multiplier)
AMP = [
    ("x1",  "d1",   "vfb",  "tail", "gnd", "n", "Win",     "Lin",     1),
    ("x2",  "aout", "vref", "tail", "gnd", "n", "Win",     "Lin",     1),
    ("x3",  "nx",   "d1",   "vdd",  "vdd", "p", "Wmirror", "Lmirror", 1),
    ("x4",  "ny",   "d1",   "vdd",  "vdd", "p", "Wmirror", "Lmirror", 1),
    ("x6",  "d1",   "vb",   "nx",   "vdd", "p", "Wcas",    "Lcas",    1),
    ("x7",  "aout", "vb",   "ny",   "vdd", "p", "Wcas",    "Lcas",    1),
    ("x5",  "tail", "ibias", "gnd", "gnd", "n", "Wtail",   "Ltail",   1),
    ("x8",  "vb",   "ibias", "gnd", "gnd", "n", "Wbn",     "Lbias",   1),
    ("x9",  "vb",   "vb",   "vdd",  "vdd", "p", "Wvb",     "Lcas",    1),
    ("x10", "vbp",  "ibias", "gnd", "gnd", "n", "Wbn2",    "Lbias",   1),
    ("x11", "vbp",  "vbp",  "vdd",  "vdd", "p", "Wcs",     "Lcs",     0.25),
    ("x12", "vout", "vbp",  "vdd",  "vdd", "p", "Wcs",     "Lcs",     1),
    ("x13", "gnd",  "aout", "vout", "vout", "p", "Wfol",   "Lfol",    1),
]
PORT = {"vfb": "fb", "vref": "vref", "vout": "ampout", "vdd": "vin", "gnd": "0", "ibias": "ibias"}

def net(n):
    return PORT.get(n, "0" if n == "0" else "xamp." + n)

NETS = sorted({net(x) for d in AMP for x in d[1:5]} - {"0"})


def make_spec(vout, iout=0.1, dropout=0.3, vin_over=0.4, vref=1.2, idiv=10e-6):
    r2 = vref / idiv
    return dict(VOUT=vout, IOUT=iout, VIN_MIN=round(vout + dropout, 3), VIN_MAX=round(vout + vin_over, 3),
                VREF=vref, R2=r2, R1=r2 * (vout / vref - 1), PSRR_MIN=40.0, PSRR_F=1e4,
                PM=60.0, COUT=0.8e-6, ESR=0.005)


def base_params(sp, sizes):
    p = dict(sizes)
    p.update(Vin_val=sp["VIN_MAX"], Vref_val=sp["VREF"], R1_val=f"{sp['R1']:.1f}",
             R2_val=f"{sp['R2']:.1f}", Iload_val=sp["IOUT"], Ibias_val="10u")
    return p


def build(pdk, p, n_pass, corner="tt", temp=27, control="", inject=False,
          cout=0.0, esr=0.0, vin_ac=False):
    P = PDKS[pdk]; sc = P["sc"]
    L = [f"* LDO {pdk}", P["header"](P["corners"][corner]), f".temp {temp}"]
    L += [f".param {k}={v}" for k, v in p.items()]
    if "Ccomp" not in p:
        L.append(".param Ccomp=0")
    L.append(".subckt error_amp vfb vref vout vdd gnd ibias")
    for name, d, g, s, b, t, wn, ln, k in AMP:
        mdl = P["nmos"] if t == "n" else P["pmos"]
        L.append(f"{name.upper()} {d} {g} {s} {b} {mdl} W={{{wn}*{k*sc:g}}} L={{{ln}*{sc:g}}}")
    L.append("CCOMP aout 0 {Ccomp+1e-18}")
    L.append(".ends error_amp")
    L.append(".subckt pass_fet d g s b")
    L.append(f"X1 d g s b {P['pmos']} W={P['pass_w']*sc:g} L={P['pass_l']*sc:g} m={n_pass}")
    L.append(".ends pass_fet")
    L += ["VIN vin 0 DC {Vin_val}" + (" AC 1" if vin_ac else ""),
          "VREF vref 0 DC {Vref_val}",
          "R1 vout fb {R1_val}", "R2 fb 0 {R2_val}",
          "IBIASGEN vin ibias DC {Ibias_val}",
          f"XBIAS ibias ibias 0 0 {P['nmos']} W={{Wbias*{sc:g}}} L={{Lbias*{sc:g}}}",
          f"XAMP {'fba' if inject else 'fb'} vref ampout vin 0 ibias error_amp",
          "XPASS vout ampout vin vin pass_fet"]
    if inject:
        L.append("VINJ fba fb DC 0 AC 1")
    if cout > 0:
        L += [f"COUT vout nesr {cout}", f"RESR nesr 0 {esr}"]
    L += ["ILOAD vout 0 DC {Iload_val}", control, ".end"]
    return "\n".join(L)


def run(pdk, text, name="run", timeout=600):
    path = os.path.join(OPT, f"{pdk}_{name}.spice")
    open(path, "w").write(text)
    r = subprocess.run(["ngspice", "-b", path], cwd=PDKS[pdk]["cwd"],
                       capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).lower()


KV = re.compile(r"^\s*(\S+)\s*=\s*([-+\d.e]+)\s*$")

def kv(text):
    out = {}
    for line in text.splitlines():
        m = KV.match(line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def _inst(pdk, d):
    P = PDKS[pdk]
    mdl = P["nmos"] if d[5] == "n" else P["pmos"]
    return f"m.xamp.{d[0]}.{P['inner'](mdl)}"


def op_run(pdk, p, n, sp, corner="tt", temp=27):
    conds = [("nominal", sp["VIN_MAX"], sp["IOUT"]), ("light", sp["VIN_MAX"], 1e-4),
             ("vinmin", sp["VIN_MIN"], sp["IOUT"])]
    ctl = [".control"]
    for i, (_, vin, il) in enumerate(conds):
        ctl += [f"alter VIN dc = {vin}", f"alter ILOAD dc = {il}", "op", f"echo @@c{i}",
                "print v(vout)", "print -i(VIN)"]
        ctl += [f"print v({x})" for x in NETS]
        for d in AMP:
            ctl += [f"print @{_inst(pdk, d)}[{q}]" for q in ("vth", "vdsat", "id")]
    ctl.append(".endc")
    out = run(pdk, build(pdk, p, n, corner, temp, "\n".join(ctl)), "op")
    parts = re.split(r"@@(c\d)", out)
    res = {parts[i]: kv(parts[i + 1]) for i in range(1, len(parts) - 1, 2)}
    if not all(k in res and "v(vout)" in res[k] for k in ("c0", "c1", "c2")):
        print("   OP simulation failed. ngspice tail:\n" + "\n".join(out.splitlines()[-20:]))
        return None, conds
    return res, conds


def devices(pdk, vals):
    V = lambda x: 0.0 if x == "0" else vals.get(f"v({x})")
    rows = []
    for d in AMP:
        name, dd, g, s, b, t = d[:6]
        nd, ng, ns, nb = (net(x) for x in (dd, g, s, b))
        vth, vdsat, idd = (vals.get(f"@{_inst(pdk, d)}[{q}]") for q in ("vth", "vdsat", "id"))
        if None in (V(nd), V(ng), V(ns), V(nb), vth, vdsat, idd):
            rows.append(dict(name=name, err=True)); continue
        pm = t == "p"
        vgs = (V(ns) - V(ng)) if pm else (V(ng) - V(ns))
        vds = (V(ns) - V(nd)) if pm else (V(nd) - V(ns))
        vth, vdsat = abs(vth), abs(vdsat)
        margin = vds - vdsat
        region = "subVt" if vgs < vth else ("sat" if margin >= 0 else "TRIODE")
        maxv = max(abs(a - c) for a, c in [(V(ng), V(ns)), (V(ng), V(nd)), (V(nd), V(ns)),
                                           (V(ng), V(nb)), (V(nd), V(nb)), (V(ns), V(nb))])
        rows.append(dict(name=name, t=t, vgs=vgs, vds=vds, vth=vth, vdsat=vdsat, margin=margin,
                         id=abs(idd), region=region, maxv=maxv, err=False))
    return rows


def ac_run(pdk, p, n, loads, cout, esr, corner="tt", temp=27):
    ctl = [".control"]
    for i, il in enumerate(loads):
        ctl += [f"alter ILOAD dc = {il}", "ac dec 20 1 1e9", "let T = -v(fb)/v(fba)",
                "let Tdb = db(T)", "let Tph = cph(T)*180/pi",
                f"meas ac phx{i} find Tph when Tdb=0 fall=1", f"meas ac ugf{i} when Tdb=0 fall=1"]
    ctl.append(".endc")
    v = kv(run(pdk, build(pdk, p, n, corner, temp, "\n".join(ctl), inject=True, cout=cout, esr=esr), "ac"))
    return [((180 + v[f"phx{i}"]) if f"phx{i}" in v else None, v.get(f"ugf{i}")) for i in range(len(loads))]


def psrr_run(pdk, p, n, loads, cout, esr, corner="tt", temp=27):
    ctl = [".control"]
    for i, il in enumerate(loads):
        ctl += [f"alter ILOAD dc = {il}", "ac dec 20 10 1e7", "let ps = -db(v(vout))",
                f"meas ac pa{i} find ps at=1e3", f"meas ac pb{i} find ps at=1e4",
                f"meas ac pc{i} find ps at=1e5"]
    ctl.append(".endc")
    v = kv(run(pdk, build(pdk, p, n, corner, temp, "\n".join(ctl), vin_ac=True, cout=cout, esr=esr), "psrr"))
    return [tuple(v.get(f"{c}{i}") for c in ("pa", "pb", "pc")) for i in range(len(loads))]


def ron(pdk, w, l, vs, vg=0.0, dv=0.05):
    P = PDKS[pdk]; sc = P["sc"]
    txt = "\n".join([f"* ron", P["header"](P["corners"]["tt"]),
                     f"VS s 0 {vs}", f"VG g 0 {vg}", f"VD d 0 {vs - dv}",
                     f"X1 d g s s {P['pmos']} W={w*sc:g} L={l*sc:g}",
                     ".control", "op", f"print @m.x1.{P['inner'](P['pmos'])}[id]", ".endc", ".end"])
    v = kv(run(pdk, txt, "ron"))
    k = f"@m.x1.{P['inner'](P['pmos'])}[id]"
    return dv / abs(v[k]) if k in v and v[k] != 0 else None
