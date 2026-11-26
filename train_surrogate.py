#!/usr/bin/env python3
"""Train and evaluate learned surrogates for the GF180 LDO dataset.

Input : dataset/data_gf180_*V.jsonl from gen_dataset.py (all VOUTs found are combined).
Output: surrogate/  - model.pt (MLP + scalers, for the warm-start demo)
                    - metrics.json, summary.txt
                    - parity_*.png, learning_curve.png, screening.png, importance.png

Models: a multi-output MLP (PyTorch) predicting the worst-case value of every spec across
the five corners, and a random-forest baseline. Pass/fail is judged from the predicted specs.

Evaluation sets (never used for training):
  test_global : uniform-random designs over the whole BO search box
  test_near   : 10% of the perturbations around good BO designs (the region that matters)

  python train_surrogate.py              # full run (~10-20 min on a 5600X)
  python train_surrogate.py --quick      # fewer epochs, no learning curve (~2 min)
"""
import argparse, glob, json, math, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Win", "Lin", "Wtail", "n_pass", "Wfol", "Wbn2", "Wmirror", "Wvb"]
LOG_X = [0, 2, 3, 4, 5, 6, 7]                         # widths and n_pass enter the model as log()
# target, unit, transform ("sym" = signed log, for heavy-tailed quantities), pass direction
TARGETS = [("worst_psrr10k", "dB", None, ">="), ("worst_pm", "deg", None, ">="),
           ("worst_iq_uA", "uA", "sym", "<="), ("worst_load_mV", "mV", "sym", "<="),
           ("worst_line_mV", "mV", "sym", "<="), ("worst_err_mV", "mV", "sym", "<="),
           ("worst_minsat_mV", "mV", "sym", ">="), ("worst_maxv", "V", None, "<=")]
NICE = {"worst_psrr10k": "PSRR @10 kHz", "worst_pm": "Phase margin", "worst_iq_uA": "Quiescent current",
        "worst_load_mV": "Load regulation", "worst_line_mV": "Line regulation",
        "worst_err_mV": "VOUT error", "worst_minsat_mV": "Min saturation margin",
        "worst_maxv": "Max device voltage"}


def limits(vout):
    """Spec limits exactly as gen_dataset.finish() applies them (VIN span 0.1 V)."""
    return {"worst_line_mV": vout * 0.1 * 10, "worst_load_mV": 10 * vout, "worst_iq_uA": 100.0,
            "worst_err_mV": 15 * vout, "worst_minsat_mV": 50.0, "worst_maxv": 5.8,
            "worst_pm": 45.0, "worst_psrr10k": 40.0}


def fwd(y, t):
    return np.sign(y) * np.log1p(np.abs(y)) if t == "sym" else y


def inv(z, t):
    return np.sign(z) * np.expm1(np.abs(z)) if t == "sym" else z


# ----------------------------------------------------------------------------- data
def load(path_glob):
    rows, bad = [], 0
    globs = [path_glob] if isinstance(path_glob, str) else path_glob
    for f in sorted({f for g in globs for f in glob.glob(g)}):
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line.strip("\x00 \n"))
            except Exception:
                bad += 1; continue
            if r.get("status") == "ok":
                rows.append(r)
    keep = [r for r in rows if all(r.get(t) is not None and np.isfinite(r[t]) for t, *_ in TARGETS)]
    return keep, bad, len(rows) - len(keep)


def features(rows):
    X = np.array([r["x"] for r in rows], dtype=float)
    X[:, LOG_X] = np.log(X[:, LOG_X])
    v = np.array([[r["vout"]] for r in rows], dtype=float)
    return np.hstack([X, v])


def targets(rows):
    return np.column_stack([fwd(np.array([r[t] for r in rows], float), tr) for t, _, tr, _ in TARGETS])


def splits(rows, seed=0):
    rng = np.random.default_rng(seed)
    idx = {"train": [], "val": [], "test_global": [], "test_near": []}
    for i, r in enumerate(rows):
        if r["split"] == "test":
            idx["test_global"].append(i)
        elif r["source"] == "local" and rng.random() < 0.10:
            idx["test_near"].append(i)
        elif rng.random() < 0.08:
            idx["val"].append(i)
        else:
            idx["train"].append(i)
    return {k: np.array(v, dtype=int) for k, v in idx.items()}


# ----------------------------------------------------------------------------- models
def train_mlp(Xtr, Ytr, Xva, Yva, epochs=300, width=256, depth=4, seed=0, quiet=False):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-9
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-9
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    xt, yt = T((Xtr - xm) / xs), T((Ytr - ym) / ys)
    xv, yv = T((Xva - xm) / xs), T((Yva - ym) / ys)
    layers, d = [], Xtr.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]; d = width
    layers.append(nn.Linear(d, Ytr.shape[1]))
    net = nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = min(512, max(32, len(xt) // 8))
    best, best_state, wait, patience = 1e9, None, 0, 40
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            loss = nn.functional.smooth_l1_loss(net(xt[b]), yt[b], beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = nn.functional.smooth_l1_loss(net(xv), yv, beta=0.5).item()
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait > patience:
                break
        if not quiet and ep % 25 == 0:
            print(f"    epoch {ep:4d}  train {loss.item():.4f}  val {vl:.4f}", flush=True)
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            return net(T((X - xm) / xs)).numpy() * ys + ym
    return predict, dict(net=net, xm=xm, xs=xs, ym=ym, ys=ys, width=width, depth=depth)


def train_rf(Xtr, Ytr, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, Ytr)
    return rf.predict, rf


# ----------------------------------------------------------------------------- evaluation
def regression_metrics(Yz_true, Yz_pred):
    out = {}
    for j, (t, unit, tr, _) in enumerate(TARGETS):
        yt, yp = inv(Yz_true[:, j], tr), inv(Yz_pred[:, j], tr)
        ss = np.sum((Yz_true[:, j] - Yz_true[:, j].mean()) ** 2)
        r2 = 1 - np.sum((Yz_true[:, j] - Yz_pred[:, j]) ** 2) / ss if ss > 0 else float("nan")
        out[t] = dict(r2=float(r2), mae=float(np.mean(np.abs(yt - yp))),
                      medae=float(np.median(np.abs(yt - yp))), unit=unit)
    return out


def margins(Yz, vout):
    """Normalised margin to each spec (>0 passes); the design's score is its worst margin."""
    m = np.zeros((len(Yz), len(TARGETS)))
    for j, (t, _, tr, op) in enumerate(TARGETS):
        y = inv(Yz[:, j], tr)
        lim = np.array([limits(v)[t] for v in vout])
        m[:, j] = (y - lim) / np.abs(lim) if op == ">=" else (lim - y) / np.abs(lim)
    return m


def pass_metrics(Yz_true, Yz_pred, vout):
    mt, mp = margins(Yz_true, vout).min(1), margins(Yz_pred, vout).min(1)
    t, p = mt > 0, mp > 0
    tp, fp, fn = int(np.sum(t & p)), int(np.sum(~t & p)), int(np.sum(t & ~p))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    acc = float(np.mean(t == p))
    # screening: send the top-k ranked designs to SPICE; how many real passers do we get?
    order = np.argsort(-mp)
    k = max(1, int(t.sum()))
    hit_k = float(t[order[:k]].mean())
    return dict(n=len(t), true_pass=int(t.sum()), pred_pass=int(p.sum()), tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, accuracy=acc, base_rate=float(t.mean()),
                precision_at_k=hit_k, k=k), mt, mp


# ----------------------------------------------------------------------------- plots
def parity_plot(path, Yz_true, preds, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for j, (t, unit, tr, _) in enumerate(TARGETS):
        ax = axes.flat[j]
        yt = inv(Yz_true[:, j], tr)
        for name, (Yp, col) in preds.items():
            ax.scatter(yt, inv(Yp[:, j], tr), s=4, alpha=0.35, color=col, label=name, rasterized=True)
        lo, hi = np.percentile(yt, [0.5, 99.5])
        pad = 0.05 * (hi - lo + 1e-9); lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#444", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if tr == "sym" and lo > 0:
            ax.set_xscale("log"); ax.set_yscale("log")
        r2 = 1 - np.sum((Yz_true[:, j] - list(preds.values())[0][0][:, j]) ** 2) / \
            np.sum((Yz_true[:, j] - Yz_true[:, j].mean()) ** 2)
        ax.set_title(f"{NICE[t]} ({unit})   R² = {r2:.3f}", fontsize=10)
        ax.set_xlabel("SPICE", fontsize=9); ax.set_ylabel("predicted", fontsize=9)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(markerscale=3, fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def learning_plot(path, curve):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    n = [c["n_train"] for c in curve]
    for t, *_ in TARGETS:
        axes[0].plot(n, [1 - c["near"][t]["r2"] for c in curve], marker="o", ms=4, label=NICE[t])
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("training samples"); axes[0].set_ylabel("1 − R² (near-optimum test set)")
    axes[0].set_title("Prediction error vs dataset size"); axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot(n, [c["pass_near"]["precision_at_k"] for c in curve], marker="o", label="near-optimum test")
    axes[1].plot(n, [c["pass_global"]["precision_at_k"] for c in curve], marker="s", label="global test")
    axes[1].set_xscale("log"); axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("training samples"); axes[1].set_ylabel("precision@k (fraction of top-k that pass)")
    axes[1].set_title("Screening quality vs dataset size"); axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def screening_plot(path, sets):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(sets), figsize=(6.5 * len(sets), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, (mt, mp, pm)) in zip(axes, sets.items()):
        t = mt > 0
        ax.scatter(mp[~t], mt[~t], s=4, alpha=0.3, color="#c0392b", label="fails in SPICE", rasterized=True)
        ax.scatter(mp[t], mt[t], s=6, alpha=0.6, color="#1f77b4", label="passes in SPICE", rasterized=True)
        ax.axvline(0, color="#444", lw=0.8); ax.axhline(0, color="#444", lw=0.8)
        ax.set_xlim(-1.5, 0.6); ax.set_ylim(-1.5, 0.6)
        ax.set_xlabel("predicted worst margin (> 0 = predicted pass)")
        ax.set_ylabel("SPICE worst margin")
        ax.set_title(f"{name}: precision {pm['precision']:.2f}, recall {pm['recall']:.2f}, "
                     f"P@{pm['k']} {pm['precision_at_k']:.2f}", fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def importance_plot(path, rf):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = NAMES + ["VOUT"]
    imp = rf.feature_importances_
    o = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([names[i] for i in o], imp[o], color="#1f4e79")
    ax.set_xlabel("random-forest importance (all specs)"); ax.set_title("Which sizes drive the specs")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[os.path.join(HERE, "dataset", "data_gf180_*V.jsonl"),
                                                  os.path.join(HERE, "dataset", "active_gf180_*V.jsonl")])
    ap.add_argument("--out", default=os.path.join(HERE, "surrogate"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-curve", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.no_curve = 60, True
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    rows, bad, nan = load(a.data)
    if not rows:
        raise SystemExit(f"no usable rows found for {a.data}")
    X, Y = features(rows), targets(rows)
    vout = np.array([r["vout"] for r in rows])
    S = splits(rows)
    vouts = sorted(set(vout.tolist()))
    print(f"{len(rows)} usable designs ({bad} unreadable lines, {nan} with missing values), VOUT {vouts}")
    print("  " + ", ".join(f"{k} {len(v)}" for k, v in S.items()), flush=True)

    tr, va = S["train"], S["val"]
    print("\nTraining MLP on all training data ...", flush=True)
    t1 = time.time()
    mlp, state = train_mlp(X[tr], Y[tr], X[va], Y[va], epochs=a.epochs)
    t_mlp = time.time() - t1
    print(f"  done in {t_mlp:.0f} s\nTraining random forest ...", flush=True)
    t1 = time.time()
    rf_pred, rf = train_rf(X[tr], Y[tr])
    t_rf = time.time() - t1
    print(f"  done in {t_rf:.0f} s", flush=True)

    res = dict(n_rows=len(rows), splits={k: int(len(v)) for k, v in S.items()}, vouts=vouts,
               train_seconds=dict(mlp=round(t_mlp, 1), rf=round(t_rf, 1)), models={})
    screen_sets = {}
    for mname, f in (("MLP", mlp), ("RandomForest", rf_pred)):
        res["models"][mname] = {}
        for sname in ("test_global", "test_near"):
            idx = S[sname]
            if len(idx) == 0:
                continue
            P = f(X[idx])
            pm, mt, mp = pass_metrics(Y[idx], P, vout[idx])
            res["models"][mname][sname] = dict(regression=regression_metrics(Y[idx], P), passfail=pm)
            if mname == "MLP":
                screen_sets[{"test_global": "Global test", "test_near": "Near-optimum test"}[sname]] = (mt, mp, pm)
        for v in vouts:                                   # per-VOUT screening on the near-optimum set
            idx = S["test_near"][vout[S["test_near"]] == v]
            if len(idx):
                pm, _, _ = pass_metrics(Y[idx], f(X[idx]), vout[idx])
                res["models"][mname][f"test_near_{v:g}V"] = dict(passfail=pm)

    # timing: surrogate vs SPICE
    tt = time.time(); _ = mlp(np.repeat(X[:1], 100000, axis=0)); per = (time.time() - tt) / 100000
    sim_s = np.median([r["sim_s"] for r in rows])
    res["speed"] = dict(mlp_seconds_per_design=per, spice_seconds_per_design=float(sim_s),
                        speedup=float(sim_s / per))

    for sname, title in (("test_near", "near-optimum test set"), ("test_global", "global test set")):
        if len(S[sname]):
            idx = S[sname]
            parity_plot(os.path.join(a.out, f"parity_{sname}.png"), Y[idx],
                        {"MLP": (mlp(X[idx]), "#1f77b4"), "Random forest": (rf_pred(X[idx]), "#ff7f0e")},
                        f"Surrogate vs SPICE, worst case over 5 corners: {title} ({len(idx)} designs)")
    screening_plot(os.path.join(a.out, "screening.png"), screen_sets)
    importance_plot(os.path.join(a.out, "importance.png"), rf)

    if not a.no_curve:
        print("\nLearning curve ...", flush=True)
        curve = []
        rng = np.random.default_rng(1)
        sizes = [n for n in (250, 500, 1000, 2000, 4000, 8000, 16000, 32000) if n < len(tr)] + [len(tr)]
        for n in sizes:
            sub = rng.choice(tr, n, replace=False)
            f, _ = train_mlp(X[sub], Y[sub], X[va], Y[va], epochs=a.epochs, quiet=True)
            near = regression_metrics(Y[S["test_near"]], f(X[S["test_near"]]))
            pn, _, _ = pass_metrics(Y[S["test_near"]], f(X[S["test_near"]]), vout[S["test_near"]])
            pg, _, _ = pass_metrics(Y[S["test_global"]], f(X[S["test_global"]]), vout[S["test_global"]])
            curve.append(dict(n_train=int(n), near=near, pass_near=pn, pass_global=pg))
            print(f"  n={n:6d}  mean R2 near {np.mean([near[t]['r2'] for t, *_ in TARGETS]):.3f}  "
                  f"P@k near {pn['precision_at_k']:.2f}", flush=True)
        res["learning_curve"] = curve
        learning_plot(os.path.join(a.out, "learning_curve.png"), curve)

    import torch
    torch.save(dict(state_dict=state["net"].state_dict(), xm=state["xm"], xs=state["xs"], ym=state["ym"],
                    ys=state["ys"], width=state["width"], depth=state["depth"], names=NAMES,
                    log_x=LOG_X, targets=TARGETS), os.path.join(a.out, "model.pt"))
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1, default=float)

    L = [f"Surrogate results  ({len(rows)} designs, VOUT {vouts}, total {time.time()-t0:.0f} s)", ""]
    for sname in ("test_near", "test_global"):
        if sname not in res["models"]["MLP"]:
            continue
        L.append(f"== {sname} ({res['splits'][sname]} designs) ==")
        L.append(f"{'spec':<24}{'MLP R2':>8}{'MLP MAE':>12}{'RF R2':>8}{'RF MAE':>12}")
        for t, unit, *_ in TARGETS:
            m, r = res["models"]["MLP"][sname]["regression"][t], res["models"]["RandomForest"][sname]["regression"][t]
            L.append(f"{NICE[t]:<24}{m['r2']:>8.3f}{m['mae']:>9.2f} {unit:<2}{r['r2']:>8.3f}{r['mae']:>9.2f} {unit}")
        for mn in ("MLP", "RandomForest"):
            p = res["models"][mn][sname]["passfail"]
            L.append(f"{mn:<13} pass/fail: accuracy {p['accuracy']:.3f}, precision {p['precision']:.2f}, "
                     f"recall {p['recall']:.2f} | true passers {p['true_pass']}/{p['n']} "
                     f"(base rate {p['base_rate']:.3f}), precision@{p['k']} {p['precision_at_k']:.2f}")
        L.append("")
    for v in vouts:
        k = f"test_near_{v:g}V"
        if k in res["models"]["MLP"]:
            p = res["models"]["MLP"][k]["passfail"]
            L.append(f"MLP near-optimum {v:g} V: true passers {p['true_pass']}/{p['n']}, precision {p['precision']:.2f}, "
                     f"recall {p['recall']:.2f}, precision@{p['k']} {p['precision_at_k']:.2f}")
    sp = res["speed"]
    L.append(f"\nSpeed: SPICE {sp['spice_seconds_per_design']:.2f} s/design (all corners), MLP "
             f"{sp['mlp_seconds_per_design']*1e6:.1f} us/design  ->  {sp['speedup']:.0f}x faster")
    txt = "\n".join(L)
    open(os.path.join(a.out, "summary.txt"), "w").write(txt)
    print("\n" + txt + f"\n\nSaved to {a.out}/")


if __name__ == "__main__":
    main()
