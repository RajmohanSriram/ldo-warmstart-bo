#!/usr/bin/env python3
"""Surrogate-guided sampling: let the trained model choose which designs to simulate.

1. Generate a large pool of random candidate sizings (whole BO search box), plus a pool
   of perturbations around the best designs already in the dataset.
2. Score every candidate with the surrogate (predicted worst-case margin to spec).
3. Pick the top candidates, spread out so they are not near-copies of each other.
4. Also pick a small UNSCREENED random control from the same pool.
5. Simulate all of them in SPICE (same code as gen_dataset.py) and report the real pass rate
   of surrogate-picked vs random designs.

Results are appended to dataset/active_gf180_<V>V.jsonl (resumable, same format as the dataset).

  python active_sample.py --vout 3.3 --dry-run     # select and show predictions only (seconds)
  python active_sample.py --vout 3.3               # select + simulate (~30 min for 3300 designs)
  python active_sample.py --vout 3.3 --report      # hit-rate report from what has been simulated
"""
import argparse, glob, json, math, multiprocessing as mp, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import gen_dataset as G
import train_surrogate as T

HERE = os.path.dirname(os.path.abspath(__file__))
DSET = os.path.join(HERE, "dataset")


def load_model(path):
    import torch, torch.nn as nn
    ck = torch.load(path, weights_only=False)
    layers, d = [], len(ck["xm"])
    for _ in range(ck["depth"]):
        layers += [nn.Linear(d, ck["width"]), nn.SiLU()]; d = ck["width"]
    layers.append(nn.Linear(d, len(ck["ym"])))
    net = nn.Sequential(*layers); net.load_state_dict(ck["state_dict"]); net.eval()

    def predict(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 100000):
                z = torch.tensor((X[i:i + 100000] - ck["xm"]) / ck["xs"], dtype=torch.float32)
                out.append(net(z).numpy() * ck["ys"] + ck["ym"])
        return np.vstack(out)
    return predict


def to_unit(X):
    """Sizings -> [0,1]^8 using the same (log) scaling as the sampler, for distance checks."""
    U = np.empty_like(X, dtype=float)
    for j, (lo, hi) in enumerate(G.BOUNDS):
        if j in G.LOG_DIMS:
            U[:, j] = (np.log(X[:, j]) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            U[:, j] = (X[:, j] - lo) / (hi - lo)
    return U


def feats(X, vout):
    F = X.astype(float).copy()
    F[:, T.LOG_X] = np.log(F[:, T.LOG_X])
    return np.hstack([F, np.full((len(F), 1), vout)])


def global_pool(n, rng):
    U = rng.random((n, len(G.NAMES)))
    X = np.empty_like(U)
    for j, (lo, hi) in enumerate(G.BOUNDS):
        X[:, j] = np.exp(math.log(lo) + U[:, j] * (math.log(hi) - math.log(lo))) if j in G.LOG_DIMS \
            else lo + U[:, j] * (hi - lo)
    X[:, 3] = np.round(X[:, 3])
    return X


def local_pool(anchors, n, rng):
    A = anchors[rng.integers(0, len(anchors), n)]
    sig = rng.choice([0.05, 0.12, 0.25], size=(n, 1))
    X = A * np.exp(rng.normal(0, 1, A.shape) * sig)
    X[:, 3] = A[:, 3] + rng.normal(0, 1, n) * sig[:, 0] * A[:, 3]
    return np.array([G.clip(list(x)) for x in X], dtype=float)


def pick_diverse(X, score, k, d_min=0.04, top_mult=30):
    order = np.argsort(-score)[:top_mult * k]
    U = to_unit(X[order])
    chosen = []
    while True:
        chosen, acc = [], np.empty((0, U.shape[1]))
        for i in range(len(order)):
            if len(acc) and np.min(np.linalg.norm(acc - U[i], axis=1)) < d_min:
                continue
            chosen.append(order[i]); acc = np.vstack([acc, U[i]])
            if len(chosen) == k:
                return np.array(chosen)
        if d_min < 1e-3:
            return np.array(chosen)
        d_min /= 2                                     # not enough distinct candidates: relax


def dataset_anchors(vout, top=300):
    rows, _, _ = T.load(os.path.join(DSET, f"data_gf180_{vout:g}V.jsonl"))
    if not rows:
        return np.empty((0, 8)), 0
    m = T.margins(T.targets(rows), np.array([r["vout"] for r in rows])).min(1)
    o = np.argsort(-m)[:top]
    return np.array([rows[i]["x"] for i in o], dtype=float), int((m > 0).sum())


def build(a):
    rng = np.random.default_rng(a.seed)
    predict = load_model(os.path.join(HERE, "surrogate", "model.pt"))
    anchors, n_pass_known = dataset_anchors(a.vout)
    t0 = time.time()
    Xg = global_pool(a.pool, rng)
    pools = [("active_global", Xg, a.k_global)]
    if len(anchors):
        pools.append(("active_local", local_pool(anchors, a.pool // 5, rng), a.k_local))
    tasks, info = [], {}
    for name, X, k in pools:
        m = T.margins(predict(feats(X, a.vout)), np.full(len(X), a.vout))
        score = m.min(1)
        sel = pick_diverse(X, score, k)
        info[name] = dict(pool=len(X), pred_pass_frac=float((score > 0).mean()), picked=len(sel),
                          pred_pass_picked=float((score[sel] > 0).mean()),
                          median_pred_margin_picked=float(np.median(score[sel])))
        for i, s in enumerate(sel):
            tasks.append(dict(id=f"{name}-{i:06d}", source=name, split="train",
                              x=[int(v) if j == 3 else round(float(v), 4) for j, v in enumerate(X[s])],
                              pred_margin=float(score[s])))
    # unscreened control: random designs from the same global pool, never filtered by the model
    ctrl = rng.choice(len(Xg), a.control, replace=False)
    mg = T.margins(predict(feats(Xg[ctrl], a.vout)), np.full(len(ctrl), a.vout)).min(1)
    for i, c in enumerate(ctrl):
        tasks.append(dict(id=f"random_control-{i:06d}", source="random_control", split="train",
                          x=[int(v) if j == 3 else round(float(v), 4) for j, v in enumerate(Xg[c])],
                          pred_margin=float(mg[i])))
    info["random_control"] = dict(picked=len(ctrl), pred_pass_picked=float((mg > 0).mean()))
    print(f"{a.vout:g} V: {n_pass_known} passing designs already in the dataset, "
          f"{len(anchors)} anchors | scored {sum(len(p[1]) for p in pools)} candidates in "
          f"{time.time()-t0:.1f} s")
    for k, v in info.items():
        print(f"  {k:15s} " + "  ".join(f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}"
                                          for kk, vv in v.items()))
    return tasks


def report(path):
    rows = []
    for line in open(path, errors="ignore"):
        try:
            rows.append(json.loads(line.strip("\x00 \n")))
        except Exception:
            pass
    print(f"\nSPICE results for {len(rows)} simulated designs ({os.path.basename(path)}):")
    print(f"  {'group':16s}{'simulated':>10}{'pass_all':>10}{'hit rate':>10}{'pred pass':>11}{'agree':>8}")
    out = {}
    for src in ("active_global", "active_local", "random_control"):
        g = [r for r in rows if r["source"] == src]
        if not g:
            continue
        ok = [r for r in g if r["status"] == "ok"]
        hit = sum(bool(r.get("pass_all")) for r in ok)
        pp = sum(r["pred_margin"] > 0 for r in ok)
        agree = np.mean([(r["pred_margin"] > 0) == bool(r.get("pass_all")) for r in ok]) if ok else float("nan")
        out[src] = dict(n=len(g), ok=len(ok), pass_all=hit, hit_rate=hit / max(len(ok), 1))
        print(f"  {src:16s}{len(g):>10}{hit:>10}{100*hit/max(len(ok),1):>9.1f}%{pp:>11}{100*agree:>7.0f}%")
    if "active_global" in out and "random_control" in out and out["random_control"]["hit_rate"] > 0:
        print(f"\n  surrogate-picked vs random: {out['active_global']['hit_rate']/out['random_control']['hit_rate']:.0f}x "
              f"higher pass rate in real SPICE")
    elif "active_global" in out:
        print("\n  (no random-control design passed, so the improvement factor is a lower bound; "
              "compare with the 0.2% pass rate of the 31k-design random dataset)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vout", type=float, default=3.3, choices=[3.3, 5.0])
    ap.add_argument("--pool", type=int, default=1_000_000, help="random candidates scored by the model")
    ap.add_argument("--k-global", type=int, default=2000, help="designs picked from the random pool")
    ap.add_argument("--k-local", type=int, default=1000, help="designs picked near the best known designs")
    ap.add_argument("--control", type=int, default=300, help="unscreened random designs (baseline)")
    ap.add_argument("--cout", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    path = os.path.join(DSET, f"active_gf180_{a.vout:g}V.jsonl")
    if a.report:
        report(path); return
    tasks = build(a)
    if a.dry_run:
        return
    done = set()
    if os.path.exists(path):
        for line in open(path, errors="ignore"):
            try:
                done.add(json.loads(line.strip("\x00 \n"))["id"])
            except Exception:
                pass
    todo = [t for t in tasks if t["id"] not in done]
    print(f"to simulate: {len(todo)} (already done {len(done)}) with {a.workers} workers -> {path}", flush=True)
    t0, n, good = time.time(), 0, 0
    with open(path, "a") as f, mp.Pool(a.workers, G.init_worker, (os.path.join(DSET, "scratch"),)) as pool:
        for row in pool.imap_unordered(G.work, [(t, a.vout, a.cout) for t in todo], chunksize=1):
            f.write(json.dumps(row) + "\n"); f.flush()
            n += 1; good += bool(row.get("pass_all"))
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(todo)}  {n/el:.2f}/s  "
                      f"ETA {(len(todo)-n)/(n/el)/60:.0f} min  pass_all so far {good}", flush=True)
    report(path)


if __name__ == "__main__":
    main()
