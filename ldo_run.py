#!/usr/bin/env python3
"""Retry wrapper: run ldo_flow.py up to N times with fresh seeds; stop at first met_spec.
Every attempt is a cold run in its own folder. Extra args are passed through to ldo_flow.py."""
import argparse, json, os, subprocess, sys, time

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--attempts", type=int, default=5)
a, rest = ap.parse_known_args()
here = os.path.dirname(os.path.abspath(__file__))
if "--seed" in rest:
    sys.exit("ldo_run.py picks a new seed per attempt; do not pass --seed")

results = []
for k in range(1, a.attempts + 1):
    seed = int(time.time() * 1000) % (2 ** 31)
    print(f"\n=== attempt {k}/{a.attempts} (seed {seed}) ===", flush=True)
    p = subprocess.Popen([sys.executable, "-u", os.path.join(here, "ldo_flow.py"), *rest, "--seed", str(seed)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    path = None
    for line in p.stdout:
        print(line, end="", flush=True)
        if line.startswith("RESULT_JSON: "):
            path = line.split("RESULT_JSON: ", 1)[1].strip()
    p.wait()
    if not path or not os.path.exists(path):
        print("attempt produced no result (error or refused); stopping")
        break
    r = json.load(open(path))
    results.append(r)
    if r["status"] == "met_spec":
        break
    if r["status"] == "refused":
        break

if not results:
    sys.exit(1)
ok = [r for r in results if r["status"] == "met_spec"]
if ok:
    final = ok[0]
else:
    final = max(results, key=lambda r: (r["design"] is not None and
                                        sum(c["ok"] for c in r["design"]["checks"]),
                                        r["design"]["worst_margin"] if r["design"] else -9))
final = dict(final, attempts=len(results), attempt_statuses=[r["status"] for r in results])
if final["status"] != "met_spec" and final.get("failing"):
    names = [c["name"] for c in final["failing"]]
    cout = final["spec"]["cout_uF"]
    if names and all(n.startswith("PSRR") for n in names):
        nxt = next((c for c in [1, 2.2, 3, 4, 6.8, 10, 22] if c > cout * 1.01), None)
        if nxt:
            final["suggestion"] = (f"PSRR is the only failing item, and it improves with a larger output capacitor. "
                                   f"Try Cout = {nxt} uF (not verified for this spec).")
            print("SUGGESTION: " + final["suggestion"])
out = os.path.join(final["run_dir"], "final.json")
json.dump(final, open(out, "w"), indent=1)
print(f"\nFINAL: {final['status']} after {len(results)} attempt(s)")
print("FINAL_JSON: " + out)
