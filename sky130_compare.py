"""SKY130 3.3 V / 4 uF end-to-end flow: cold vs warm (new MLP, + dataset) vs warm (old MLP, runs only).
Same seeds for every arm. Output: runs_sky130_compare/. Re-running skips finished (arm, seed) pairs.
NOTE: both MLPs were trained on SKY130 3.3 V / 4 uF data, so this measures warm start from prior data
at the SAME spec (not an unseen-spec test like the GF180 batch)."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "runs_sky130_compare")
ARMS = {"cold": None,
        "warm_new": os.path.join(HERE, "surrogate_ws", "sky130.pt"),
        "warm_old": os.path.join(HERE, "backup_pre_mlp_20261103", "models_sky130_pre_dataset", "sky130.pt")}
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
WORKERS = str(max(1, (os.cpu_count() or 4) - 2))
os.makedirs(ROOT, exist_ok=True)
summ = os.path.join(ROOT, "summary.json")
res = json.load(open(summ)) if os.path.exists(summ) else []
for seed in SEEDS:
    for arm, mdl in ARMS.items():
        if any(r["arm"] == arm and r["seed"] == seed for r in res):
            continue
        env = dict(os.environ, LDO_WARM="0" if mdl is None else "1", PYTHONUNBUFFERED="1")
        if mdl:
            env["LDO_SURROGATE"] = mdl
        cmd = [sys.executable, "-u", os.path.join(HERE, "ldo_flow.py"), "--pdk", "sky130", "--vout", "3.3", "--cout", "4",
               "--seed", str(7000 + seed), "--workers", WORKERS, "--runs-dir", os.path.join(ROOT, arm)]
        print(f"=== {arm} seed {seed}", flush=True)
        p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
        open(os.path.join(ROOT, f"{arm}_s{seed}.log"), "w").write(p.stdout + p.stderr)
        rj = [l.split("RESULT_JSON: ", 1)[1].strip() for l in p.stdout.splitlines() if l.startswith("RESULT_JSON: ")]
        if not rj or not os.path.exists(rj[-1]):
            print("FAILED:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-15:]), flush=True); continue
        r = json.load(open(rj[-1]))
        res.append(dict(arm=arm, seed=seed, status=r["status"], evaluations=r["evaluations"],
                        first_pass=r.get("first_screen_pass_eval"), runtime_s=r["runtime_s"], result=rj[-1]))
        json.dump(res, open(summ, "w"), indent=1)
        print(f"{arm} seed {seed}: {r['status']}, {r['evaluations']} evals, first pass @{r.get('first_screen_pass_eval')}, "
              f"{r['runtime_s']} s", flush=True)
print("ALL DONE", flush=True)
