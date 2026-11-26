#!/usr/bin/env python3
"""Build a trimmed SKY130 model library holding only the 5 V devices (g5v0d10v5) the LDO uses.
Drop-in replacement for sky130.lib.spice: same `.lib <corner>` sections (tt ss ff sf fs).
Checks against the full library (drain currents must agree) and reports the speed-up."""
import os, re, subprocess, sys, tempfile, time

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/pdks/sky130A/libs.tech/ngspice")
OUTDIR = os.path.expanduser("~/pdks/sky130_trim")
CORNERS = ["tt", "ss", "ff", "sf", "fs"]
INC = re.compile(r'^\s*\.include\s+"([^"]+)"', re.I)


def keep(path):
    n = os.path.basename(path)
    return "g5v0d10v5" in n and "esd" not in n


def corner_includes(c, with_all):
    base = os.path.join(ROOT, "corners")
    out = []
    for line in open(os.path.join(base, c + ".spice")):
        m = INC.match(line)
        if not m:
            continue
        p = os.path.normpath(os.path.join(base, m.group(1)))
        if keep(p) or (with_all and os.path.basename(p) == "all.spice"):
            out.append(p)
    return out


def write_lib(path, with_all):
    with open(path, "w") as f:
        f.write("* trimmed SKY130 library: 5 V devices only (generated)\n")
        for c in CORNERS:
            f.write(f".lib {c}\n.param mc_mm_switch=0\n.param mc_pr_switch=0\n")
            for p in corner_includes(c, with_all):
                f.write(f'.include "{p}"\n')
            f.write(f".endl {c}\n\n")


def test(lib, corner):
    net = f"""trim test
.lib "{lib}" {corner}
Vd d 0 1.5
Vg g 0 1.2
Vp pd 0 0
Vs ps 0 3.3
Vgp pg 0 2.2
XN d g 0 0 sky130_fd_pr__nfet_g5v0d10v5 w=10 l=0.5
XP pd pg ps ps sky130_fd_pr__pfet_g5v0d10v5 w=10 l=0.5
.control
op
print i(vd) i(vp)
quit
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(net)
    t = time.time()
    r = subprocess.run(["ngspice", "-b", f.name], capture_output=True, text=True)
    dt = time.time() - t
    os.unlink(f.name)
    vals = re.findall(r"i\((vd|vp)\)\s*=\s*([-\d.eE+]+)", r.stdout + r.stderr)
    return dict((k, float(v)) for k, v in vals), dt


os.makedirs(OUTDIR, exist_ok=True)
full = os.path.join(ROOT, "sky130.lib.spice")
ref = {c: test(full, c) for c in CORNERS}
print("full library:", {c: round(ref[c][1], 2) for c in CORNERS}, "seconds per launch")
if any(len(ref[c][0]) < 2 for c in CORNERS):
    sys.exit("could not read reference currents from the full library; ngspice output needed")

for with_all in (False, True):
    lib = os.path.join(OUTDIR, "sky130_trim.lib.spice")
    write_lib(lib, with_all)
    ok, times = True, []
    for c in CORNERS:
        got, dt = test(lib, c)
        times.append(dt)
        same = len(got) == 2 and all(abs(got[k] - ref[c][0][k]) <= 1e-9 * max(1.0, abs(ref[c][0][k])) +
                                      1e-12 for k in ref[c][0])
        print(f"  [{'with' if with_all else 'without'} all.spice] {c}: {'match' if same else 'MISMATCH'}  "
              f"{dt:.2f} s  {got}")
        ok &= same
    if ok:
        print(f"\nOK: trimmed library {lib}\nmean {sum(times)/len(times):.2f} s per launch "
              f"(full library {sum(ref[c][1] for c in CORNERS)/len(CORNERS):.2f} s)")
        sys.exit(0)
sys.exit("trimmed library did not reproduce the full library; paste this output")
