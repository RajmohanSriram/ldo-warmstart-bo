#!/usr/bin/env python3
"""MLP surrogate for warm-starting the LDO Bayesian optimisation (and online retraining).

Training data = the flow's own evaluation logs: every runs/*/evals.jsonl row holds the 8 sizing
variables and the simulated worst-case metrics (line, load, Iq, VOUT error, PM, PSRR, min sat, ov).
Inputs are the sizing (log-scaled) plus the spec that changes the physics: VOUT and Cout.
The model never stores or returns designs; it only ranks candidate sizings, and every point it
proposes is simulated in SPICE before anything is reported.

  python surrogate_ws.py train --pdk gf180              # train from runs/ -> surrogate_ws/gf180.pt
  python surrogate_ws.py info  --pdk gf180              # data coverage + accuracy of the saved model
  python surrogate_ws.py propose --pdk gf180 --vout 3.3 --cout 4   # show what it would seed BO with
"""
import argparse, glob, json, math, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Win", "Lin", "Wtail", "n_pass", "Wfol", "Wbn2", "Wmirror", "Wvb"]
TARGETS = ["line", "load", "iq", "err", "pm", "psrr", "minsat", "ov"]
LOGT = {"line", "load", "err", "ov"}          # positive, heavy-tailed -> log1p
BROAD = {"gf180": [(4, 100), (0.7, 3), (4, 48), (20, 200), (4, 100), (1, 12), (4, 100), (1, 6)],
         "sky130": [(4, 100), (0.7, 3), (4, 48), (150, 2500), (4, 100), (1, 12), (4, 100), (1, 6)]}


def default_path(pdk):
    return os.environ.get("LDO_SURROGATE") or os.path.join(HERE, "surrogate_ws", f"{pdk}.pt")


# ----------------------------------------------------------------------------- data
def read_spec(d):
    """Spec of one run folder: spec.json (written at start) or result.json (written at end)."""
    for fn in ("spec.json", "result.json"):
        f = os.path.join(d, fn)
        if os.path.exists(f):
            try:
                j = json.load(open(f))
            except ValueError:
                continue
            s = j.get("spec", j)
            if "vout" in s and ("cout_uF" in s or "cout" in s):
                return dict(pdk=s.get("pdk", "gf180"), vout=float(s["vout"]),
                            cout=float(s.get("cout_uF", s.get("cout"))))
    return None


def targets_from_eval(m):
    """Metric dict from evals.jsonl -> target row. ldo_opt.py writes PM = 0.0 when the loop gain has no
    0 dB crossing (meas failed) and PSRR = 0.0 when the PSRR measurement failed; those are penalty
    placeholders, not measurements, so they become NaN (masked out of the loss), never a label."""
    y = [float(m[t]) for t in TARGETS]
    flagged = {f.split("@")[0] for f in m.get("meas_failed") or []}   # logs from 2026-11-04 on
    for t in ("pm", "psrr"):
        if m[t] == 0.0 or t in flagged:
            y[TARGETS.index(t)] = float("nan")
    return y


def load_runs(roots, pdk, exclude=(), stats=None):
    """Rows (x, vout, cout, metrics) from every run folder under the given roots.
    exclude: iterable of (vout, cout) pairs to hold out (for fair unseen-spec tests).
    Returns X, Y, S (spec per row), nrun, G (group = run folder, used for leakage-free validation)."""
    X, Y, S, G, seen, nrun = [], [], [], [], set(), 0
    st = stats if stats is not None else {}
    ex = {(round(v, 3), round(c, 3)) for v, c in exclude}
    for root in roots:
        for ej in sorted(glob.glob(os.path.join(root, "**", "evals.jsonl"), recursive=True)):
            d = os.path.dirname(ej)
            sp = read_spec(d)
            if not sp:
                st["runs_without_spec"] = st.get("runs_without_spec", 0) + 1
                continue
            if sp["pdk"] != pdk or (round(sp["vout"], 3), round(sp["cout"], 3)) in ex:
                continue
            nrun += 1
            for line in open(ej, errors="replace"):
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                m = e.get("m")
                if not m or e.get("total", 1e6) >= 1e5 or len(e.get("x", [])) != 8:
                    st["failed_sims_excluded"] = st.get("failed_sims_excluded", 0) + 1
                    continue
                key = (tuple(round(v, 3) for v in e["x"]), sp["vout"], sp["cout"])
                if key in seen:
                    continue
                seen.add(key)
                y = targets_from_eval(m)
                for t in ("pm", "psrr"):
                    if math.isnan(y[TARGETS.index(t)]):
                        st[f"{t}_masked"] = st.get(f"{t}_masked", 0) + 1
                X.append(list(e["x"]) + [sp["vout"], sp["cout"]])
                Y.append(y)
                S.append((sp["vout"], sp["cout"]))
                G.append("run:" + os.path.relpath(d, HERE))
    return (np.array(X, float).reshape(-1, 10), np.array(Y, float).reshape(-1, len(TARGETS)), S, nrun, G)


# ----------------------------------------------------------------------------- desktop dataset (gen_dataset.py)
CORN = ["tt", "ss", "ff", "sf", "fs"]
_PM_LOADS = ("2", "5", "20", "50", "100")          # = ldo_opt.AC_LOADS (dataset also has 0 / 0.1 mA: unused)


def default_dataset_dir():
    d = os.environ.get("LDO_DATASET")
    if d is not None:
        return d or None                              # LDO_DATASET="" disables
    d = os.path.join(os.path.dirname(HERE), "optimizer_pc", "dataset")
    return d if os.path.isdir(d) else None


def dataset_files(ddir, pdk):
    """data_<pdk>_*.jsonl + active_<pdk>_*.jsonl; *.backup.* files are older copies of the same rows."""
    if not ddir:
        return []
    fs = glob.glob(os.path.join(ddir, f"data_{pdk}_*.jsonl")) + glob.glob(os.path.join(ddir, f"active_{pdk}_*.jsonl"))
    return sorted(f for f in fs if ".backup" not in os.path.basename(f))


def targets_from_dataset(r, ov_lim):
    """gen_dataset.py row -> the same worst-case targets ldo_opt.py logs (checked by re-simulating
    32 test-split rows with ldo_opt: DC metrics identical; PM null <-> ldo_opt's 0.0 placeholder).
    Differences, documented: PSRR here is the min over tt@2mA/100mA + other corners @100mA; ldo_opt
    also checks tt@20mA, so this can be slightly optimistic (1 of 32 checked rows, by 0.24 dB).
    ov: ldo_opt logs a penalty summed over devices; only the per-corner max voltage is stored, so ov is
    exact (0) when max V <= limit and masked (NaN) otherwise. Missing measurements -> NaN (masked)."""
    nan = float("nan")
    bad = lambda q: q is None or (isinstance(q, float) and math.isnan(q))
    mx = lambda v: nan if any(bad(q) for q in v) else float(max(v))
    mn = lambda v: nan if any(bad(q) for q in v) else float(min(v))
    g = lambda k: [r.get(f"{c}_{k}") for c in CORN]
    pm = mn([r.get(f"{c}_pm_{l}mA") for c in CORN for l in _PM_LOADS])
    tt = [r.get("tt_psrr10k_2mA"), r.get("tt_psrr10k_100mA")]
    if "tt_psrr10k_20mA" in r:              # rows from gen_dataset.py since 2026-11-03: exact match to ldo_opt
        tt.append(r["tt_psrr10k_20mA"])
    ps = mn(tt + [r.get(f"{c}_psrr10k_100mA") for c in CORN[1:]])
    maxv = mx(g("maxv"))
    ov = nan if math.isnan(maxv) or maxv > ov_lim else 0.0
    y = dict(line=mx(g("line_mV")), load=mx(g("load_mV")), iq=mx(g("iq_uA")), err=mx(g("err_mV")),
             pm=pm, psrr=ps, minsat=mn(g("minsat_mV")), ov=ov)
    return [y[t] for t in TARGETS]


def load_dataset(ddir, pdk, exclude=(), stats=None):
    """Rows from the desktop dataset. Returns (train part, test part), each (X, Y, S, G).
    Only status == 'ok' rows are used (dc_fail / trim_fail / timeout / error rows are failed sims).
    The dataset's own split is respected: split == 'test' rows are NEVER used for training."""
    import ldo_pdk as L
    ov_lim = L.PDKS[pdk]["vrating"] - 0.2
    st = stats if stats is not None else {}
    ex = {(round(v, 3), round(c, 3)) for v, c in exclude}
    parts = {"train": ([], [], [], []), "test": ([], [], [], [])}
    for f in dataset_files(ddir, pdk):
        base = os.path.basename(f)
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("pdk", pdk) != pdk:          # rows record their PDK since 2026-11-03
                raise SystemExit(f"{f}: row {r.get('id')} is {r.get('pdk')}, expected {pdk}")
            if r.get("status") != "ok" or len(r.get("x", [])) != 8:
                st["dataset_failed_excluded"] = st.get("dataset_failed_excluded", 0) + 1
                continue
            v, c = float(r["vout"]), float(r["cout_uF"])
            if (round(v, 3), round(c, 3)) in ex:
                continue
            y = targets_from_dataset(r, ov_lim)
            if any(math.isnan(y[TARGETS.index(t)]) for t in ("line", "load", "iq", "err")):
                st["dataset_incomplete_dc"] = st.get("dataset_incomplete_dc", 0) + 1
                continue
            for t in ("pm", "psrr", "minsat", "ov"):
                if math.isnan(y[TARGETS.index(t)]):
                    st[f"dataset_{t}_masked"] = st.get(f"dataset_{t}_masked", 0) + 1
            X, Y, S, G = parts["test" if r.get("split") == "test" else "train"]
            X.append([float(q) for q in r["x"]] + [v, c]); Y.append(y); S.append((v, c))
            # sobol rows are i.i.d. (row-level validation is fine); bo/local/active rows sit near
            # earlier designs, so they are grouped by source and kept in training only.
            G.append(f"ds:{base}:{r.get('source')}" + (f":{r['id']}" if r.get("source") == "sobol" else ""))
    out = {}
    for k, (X, Y, S, G) in parts.items():
        out[k] = (np.array(X, float).reshape(-1, 10), np.array(Y, float).reshape(-1, len(TARGETS)), S, G)
    return out["train"], out["test"]


def unit_rows(X):
    """Rows in the unit-scaled log space used for near-duplicate checks (spec columns kept separate)."""
    lo, hi = np.log([b[0] for b in BROAD["gf180"]]), np.log([b[1] for b in BROAD["gf180"]])
    return (np.log(np.maximum(X[:, :8], 1e-9)) - lo) / (hi - lo)


def near_mask(Xa, Xb, eps=0.01):
    """True for rows of Xa within eps (RMS unit distance) of any row of Xb at the same (vout, cout)."""
    from scipy.spatial import cKDTree
    out = np.zeros(len(Xa), bool)
    if not len(Xa) or not len(Xb):
        return out
    ua, ub = unit_rows(Xa), unit_rows(Xb)
    ka = [tuple(np.round(r, 3)) for r in Xa[:, 8:]]
    kb = [tuple(np.round(r, 3)) for r in Xb[:, 8:]]
    for s in set(ka):
        ia = np.array([i for i, k in enumerate(ka) if k == s])
        ib = np.array([i for i, k in enumerate(kb) if k == s])
        if not len(ib):
            continue
        d, _ = cKDTree(ub[ib]).query(ua[ia], k=1)
        out[ia] = d / math.sqrt(8) < eps
    return out


def feat(X):
    X = np.asarray(X, float)
    return np.column_stack([np.log(np.maximum(X[:, :8], 1e-6)), X[:, 8], np.log(np.maximum(X[:, 9], 1e-3))])


def ty(Y):
    Y = np.array(Y, float)
    for j, t in enumerate(TARGETS):
        if t in LOGT:
            Y[:, j] = np.log1p(np.maximum(Y[:, j], 0.0))
    return Y


def inv_ty(Y):
    Y = np.array(Y, float)
    for j, t in enumerate(TARGETS):
        if t in LOGT:
            Y[:, j] = np.expm1(Y[:, j])
    return Y


def r2_scores(Y, P):
    """R^2 per target on the transformed scale, ignoring masked (NaN) labels. None = constant/too few."""
    r2 = {}
    A, Q = ty(Y), ty(P)
    for j, t in enumerate(TARGETS):
        ok = ~np.isnan(A[:, j])
        a, p = A[ok, j], Q[ok, j]
        ss = ((a - a.mean()) ** 2).sum() if len(a) > 1 else 0.0
        r2[t] = round(float(1 - ((a - p) ** 2).sum() / ss), 4) if ss > 0 else None
    return r2


def pm_ok_scores(Y, P):
    """Accuracy / ROC-AUC of the PM-measurable head (label: PM not masked)."""
    if P.shape[1] <= len(TARGETS):
        return {}
    y = ~np.isnan(np.asarray(Y, float)[:, TARGETS.index("pm")])
    q = P[:, len(TARGETS)]
    out = dict(pm_ok_acc=round(float(((q > 0.5) == y).mean()), 4), pm_ok_frac_measurable=round(float(y.mean()), 4))
    if 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score
        out["pm_ok_auc"] = round(float(roc_auc_score(y, q)), 4)
    return out


def fmt_r2(r2):
    return "  ".join(f"{t}={r2[t]:.3f}" if r2[t] is not None else f"{t}=const" for t in TARGETS)


# ----------------------------------------------------------------------------- model
class Surrogate:
    def __init__(self, width=128, depth=3, seed=0, pm_head=True):
        """pm_head: extra logit output 'is PM measurable' (loop gain has a 0 dB crossing). Trained on
        every row, so designs whose PM could not be measured (masked in the PM regression) still teach
        the model that they fail. predict() then returns len(TARGETS)+1 columns, the last = P(pm_ok)."""
        import torch, torch.nn as nn
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        torch.manual_seed(seed)
        self.torch, self.width, self.depth, self.pm_head = torch, width, depth, pm_head
        layers, d = [], 10
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]; d = width
        layers.append(nn.Linear(d, len(TARGETS) + (1 if pm_head else 0)))
        self.net = nn.Sequential(*layers)
        self.xm = self.xs = self.ym = self.ys = None
        self.meta = {}

    def _norm_fit(self, F, T):
        self.xm, self.xs = F.mean(0), F.std(0) + 1e-6
        self.ym, self.ys = np.nanmean(T, 0), np.nanstd(T, 0) + 1e-6

    @staticmethod
    def split_groups(G, n, val_frac, seed):
        """Validation = whole groups (run folders / dataset sources), so near-identical points from one
        BO run never sit on both sides of the split. Returns (val idx, train idx)."""
        rng = np.random.default_rng(seed)
        if G is None:
            idx = rng.permutation(n); nv = int(n * val_frac)
            return idx[:nv], idx[nv:]
        G = np.asarray(G)
        # bo/local/active dataset rows are near earlier designs: training only
        eligible = np.array([not (g.startswith("ds:") and g.count(":") < 3) for g in G])
        ug = np.unique(G[eligible])
        rng.shuffle(ug)
        target, take, cnt = val_frac * n, set(), 0
        sizes = {g: c for g, c in zip(*np.unique(G, return_counts=True))}
        for g in ug:
            if cnt >= target:
                break
            take.add(g); cnt += sizes[g]
        isv = np.isin(G, list(take))
        return np.where(isv)[0], np.where(~isv)[0]

    def fit(self, X, Y, epochs=400, patience=30, lr=2e-3, val_frac=0.1, renorm=True, seed=0, quiet=False,
            max_seconds=600, groups=None):
        torch = self.torch
        F, T = feat(X), ty(Y)
        if renorm or self.xm is None:
            self._norm_fit(F, T)
        Fn = torch.tensor((F - self.xm) / self.xs, dtype=torch.float32)
        Tn_np = (T - self.ym) / self.ys
        Mk = torch.tensor(~np.isnan(Tn_np), dtype=torch.float32)          # 0 = masked (failed measurement)
        Tn = torch.tensor(np.nan_to_num(Tn_np, nan=0.0), dtype=torch.float32)
        n = len(Fn)
        if n >= 50 and val_frac > 0:
            vi, ti = self.split_groups(groups, n, val_frac, seed)
            if len(vi) and len(ti) and groups is not None:                # drop val rows that duplicate train rows
                dup = near_mask(np.asarray(X)[vi], np.asarray(X)[ti])
                self.meta["val_near_dup_dropped"] = int(dup.sum())
                vi = vi[~dup]
        else:
            vi, ti = np.array([], int), np.arange(n)
        rng = np.random.default_rng(seed)
        nv = len(vi)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        _sl1 = torch.nn.SmoothL1Loss(beta=0.1, reduction="none")
        _bce = torch.nn.BCEWithLogitsLoss()
        jpm, nt = TARGETS.index("pm"), len(TARGETS)

        def lossf(P, T, M):
            l = (_sl1(P[:, :nt], T) * M).sum() / M.sum().clamp(min=1.0)
            if self.pm_head:                      # label: PM was measurable (= not masked)
                l = l + 0.5 * _bce(P[:, nt], M[:, jpm])
            return l
        best, best_state, bad, t0 = 1e18, None, 0, time.time()
        bs = 512 if n > 2000 else 128
        for ep in range(epochs):
            self.net.train()
            perm = ti[rng.permutation(len(ti))]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(self.net(Fn[b]), Tn[b], Mk[b])
                loss.backward(); opt.step()
            if nv:
                self.net.eval()
                with torch.no_grad():
                    vl = float(lossf(self.net(Fn[vi]), Tn[vi], Mk[vi]))
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                if bad >= patience:
                    break
            if time.time() - t0 > max_seconds:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        if nv:
            P = self.predict(X[vi])
            r2 = r2_scores(Y[vi], P)
            self.meta.update(val_r2=r2, n_val=int(nv), val_split="grouped by run folder / dataset source")
            self.meta.update({f"val_{k}": v for k, v in pm_ok_scores(Y[vi], P).items()})
            self._val = (Y[vi], P)
            if not quiet:
                print("  validation R^2 (transformed scale): " + fmt_r2(r2))
                if self.pm_head:
                    print("  validation PM-measurable classifier: " + str(pm_ok_scores(Y[vi], P)))
        self.meta.update(n_train=int(len(ti)), epochs=ep + 1, trained_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        return self

    def predict(self, X):
        torch = self.torch
        F = (feat(X) - self.xm) / self.xs
        out = []
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(F), 100000):
                out.append(self.net(torch.tensor(F[i:i + 100000], dtype=torch.float32)).numpy())
        O = np.vstack(out)
        nt = len(TARGETS)
        R = inv_ty(O[:, :nt] * self.ys + self.ym)
        if self.pm_head:
            R = np.column_stack([R, 1.0 / (1.0 + np.exp(-O[:, nt]))])
        return R

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.torch.save(dict(state_dict=self.net.state_dict(), width=self.width, depth=self.depth, pm_head=self.pm_head,
                             xm=self.xm, xs=self.xs, ym=self.ym, ys=self.ys, meta=self.meta,
                             targets=TARGETS, names=NAMES), path)
        m = {k: v for k, v in self.meta.items() if k != "val_idx_sample"}
        json.dump(m, open(path + ".json", "w"), indent=1)

    @classmethod
    def load(cls, path):
        import torch
        ck = torch.load(path, weights_only=False)
        s = cls(ck["width"], ck["depth"], pm_head=ck.get("pm_head", False))   # older models: no head
        s.net.load_state_dict(ck["state_dict"])
        s.xm, s.xs, s.ym, s.ys, s.meta = ck["xm"], ck["xs"], ck["ym"], ck["ys"], ck.get("meta", {})
        return s


def load_or_none(path):
    try:
        return Surrogate.load(path) if path and os.path.exists(path) else None
    except Exception as e:
        print(f"surrogate: could not load {path}: {e!r}")
        return None


# ----------------------------------------------------------------------------- spec screening
def limits(spec):
    """spec: vout, psrr (search target), line_pct, tol_pct, pm_floor, dropout, vin_over."""
    v = spec["vout"]
    return dict(line=spec["line_pct"] / 100 * v * (spec["vin_over"] - spec["dropout"]) * 1e3,
                load=0.01 * v * 1e3, iq=100.0, err=spec["tol_pct"] / 100 * v * 1e3,
                pm=spec["pm_floor"], psrr=spec["psrr"], minsat=50.0)


def margin(P, spec):
    """Worst normalised margin to spec for predicted metrics P (n x len(TARGETS)). >0 = predicted pass."""
    lim = limits(spec)
    col = {t: P[:, j] for j, t in enumerate(TARGETS)}
    ms = [(lim[t] - col[t]) / lim[t] for t in ("line", "load", "iq", "err")]
    ms += [(col[t] - lim[t]) / abs(lim[t]) for t in ("pm", "psrr", "minsat")]
    # ov is a penalty that is 0 for every legal design, so -ov/0.05 was <= 0 by construction and,
    # being the min, ranked passing candidates by prediction noise around 0 (seen on real runs).
    # Pass/fail with a small tolerance instead: +1 at ov = 0, crosses 0 at OV_TOL (~20 mV over limit).
    ms.append((OV_TOL - col["ov"]) / OV_TOL)
    if P.shape[1] > len(TARGETS):               # P(PM measurable): binds when the model expects no 0 dB crossing
        ms.append((P[:, len(TARGETS)] - 0.5) / 0.5)
    return np.min(np.column_stack(ms), axis=1)


PRED_COLS = TARGETS + ["pm_ok"]


def pred_row(p):
    """Predicted-metric dict (from propose) -> one row for margin(); includes pm_ok when present."""
    return np.array([[p[t] for t in PRED_COLS if t in p]])


OV_TOL = 0.005


def sample_box(rng, dims, n):
    """Log-uniform samples inside the box (n_pass rounded to integer)."""
    lo = np.log([d[0] for d in dims]); hi = np.log([d[1] for d in dims])
    U = np.exp(lo + rng.random((n, len(dims))) * (hi - lo))
    U[:, 3] = np.round(U[:, 3])
    return U


def unit(X, dims):
    lo = np.log([d[0] for d in dims]); hi = np.log([d[1] for d in dims])
    return (np.log(np.maximum(np.asarray(X, float), 1e-9)) - lo) / np.maximum(hi - lo, 1e-9)


def propose(model, spec, dims, k, n_pool=200000, seed=0, avoid=(), dmin=0.08):
    """Top-k diverse candidates by predicted spec margin. Returns (points, predicted-metric dicts, margins)."""
    rng = np.random.default_rng(seed)
    C = sample_box(rng, dims, n_pool)
    feats = np.column_stack([C, np.full(len(C), spec["vout"]), np.full(len(C), spec["cout"])])
    P = model.predict(feats)
    mg = margin(P, spec)
    order = np.argsort(-mg)
    taken = [unit(np.array(a, float)[None, :], dims)[0] for a in avoid]
    UC = unit(C, dims)
    pts, preds, mgs = [], [], []
    for i in order:
        u = UC[i]
        if taken and min(np.linalg.norm(u - t) / math.sqrt(len(u)) for t in taken) < dmin:
            continue
        taken.append(u)
        x = [float(v) for v in C[i]]; x[3] = int(round(x[3]))
        pts.append(x); preds.append({t: float(P[i, j]) for j, t in enumerate(PRED_COLS[:P.shape[1]])}); mgs.append(float(mg[i]))
        if len(pts) == k:
            break
    return pts, preds, mgs


class Online:
    """Retrains a copy of the base model on this run's own simulations, then proposes new points.
    This is the BO -> MLP -> BO loop: SPICE results from the search flow back into the model."""
    def __init__(self, model, spec, base_X=None, base_Y=None, seed=0):
        import copy
        self.model = Surrogate(model.width, model.depth, seed, pm_head=model.pm_head)   # private copy
        self.model.net.load_state_dict(copy.deepcopy(model.net.state_dict()))
        self.model.xm, self.model.xs, self.model.ym, self.model.ys = model.xm, model.xs, model.ym, model.ys
        self.model.meta = dict(model.meta)
        self.spec, self.seed = spec, seed
        self.bX, self.bY = base_X, base_Y
        self.n_retrains = 0

    def update(self, evals):
        rows = [e for e in evals if e.get("m") and e.get("total", 1e6) < 1e5]
        if len(rows) < 8:
            return False
        X = np.array([list(e["x"]) + [self.spec["vout"], self.spec["cout"]] for e in rows], float)
        Y = np.array([targets_from_eval(e["m"]) for e in rows], float)   # PM/PSRR placeholders masked
        reps = max(1, min(20, 2000 // len(rows)))      # weight this run's own data heavily
        X, Y = np.repeat(X, reps, 0), np.repeat(Y, reps, 0)
        if self.bX is not None and len(self.bX):
            rng = np.random.default_rng(self.seed + self.n_retrains)
            take = rng.choice(len(self.bX), size=min(len(self.bX), 4000), replace=False)
            X, Y = np.vstack([X, self.bX[take]]), np.vstack([Y, self.bY[take]])
        self.model.fit(X, Y, epochs=40, patience=1000, lr=5e-4, val_frac=0.0, renorm=False, quiet=True,
                       max_seconds=60)
        self.n_retrains += 1
        return True

    def propose(self, dims, k, avoid, seed):
        return propose(self.model, self.spec, dims, k, n_pool=50000, seed=seed, avoid=avoid)


# ----------------------------------------------------------------------------- CLI
def train(pdk, roots, out, exclude=(), seed=0, quiet=False, dataset="auto"):
    """dataset: 'auto' (LDO_DATASET or ../optimizer_pc/dataset if present), a directory, or None.
    exclude applies to both sources. Dataset test-split rows form a fixed TEST set: never trained on,
    and any training row within a near-duplicate distance of a test row is dropped."""
    st = {}
    X, Y, S, nrun, G = load_runs(roots, pdk, exclude, st)
    n_runs_rows = len(X)
    ddir = default_dataset_dir() if dataset == "auto" else dataset
    files = dataset_files(ddir, pdk)
    Xt = Yt = None
    if files:
        (Xd, Yd, Sd, Gd), (Xt, Yt, St, _) = load_dataset(ddir, pdk, exclude, st)
        X, Y, S, G = np.vstack([X, Xd]), np.vstack([Y, Yd]), S + Sd, G + Gd
        if len(Xt):
            near = near_mask(X, Xt)
            st["train_rows_near_test_dropped"] = int(near.sum())
            X, Y, S, G = X[~near], Y[~near], [s for s, k in zip(S, near) if not k], [g for g, k in zip(G, near) if not k]
    if not quiet:
        print(f"training data: {n_runs_rows} unique evaluations from {nrun} run folders"
              + (f" + {len(X) - n_runs_rows + st.get('train_rows_near_test_dropped', 0)} dataset rows "
                 f"({', '.join(os.path.basename(f) for f in files)})" if files else "") + f" ({pdk})")
        for (v, c), cnt in sorted({s: S.count(s) for s in set(S)}.items()):
            print(f"   VOUT {v:g} V, Cout {c:g} uF: {cnt}")
        print("   data hygiene: " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))
    if len(X) < 200:
        raise SystemExit(f"only {len(X)} usable evaluations; need at least 200. Run the flow a few times first.")
    m = Surrogate(seed=seed).fit(X, Y, seed=seed, quiet=quiet, groups=G)
    m.meta.update(pdk=pdk, roots=[os.path.relpath(r, HERE) for r in roots], n_runs=nrun,
                  dataset_files=[os.path.abspath(f) for f in files], data_hygiene=st,
                  coverage={f"{v:g}V_{c:g}uF": S.count((v, c)) for v, c in set(S)},
                  excluded=[list(e) for e in exclude])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if Xt is not None and len(Xt):
        Pt = m.predict(Xt)
        m.meta.update({f"test_{k}": v for k, v in pm_ok_scores(Yt, Pt).items()})
        m.meta.update(test_r2=r2_scores(Yt, Pt), n_test=int(len(Xt)),
                      test_set="dataset split=='test' rows (never used for training or early stopping)")
        np.savez(out + ".test.npz", y=Yt, p=Pt, x=Xt, targets=np.array(TARGETS))
        if not quiet:
            print(f"  TEST R^2 on {len(Xt)} held-out dataset rows:  " + fmt_r2(m.meta["test_r2"]))
            print("  TEST PM-measurable classifier: " + str(pm_ok_scores(Yt, Pt)))
    m.save(out)
    if getattr(m, "_val", None) is not None:
        np.savez(out + ".val.npz", y=m._val[0], p=m._val[1], targets=np.array(TARGETS))
    if not quiet:
        print(f"saved {out}")
    return m, X, Y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "info", "propose"])
    ap.add_argument("--pdk", default="gf180")
    ap.add_argument("--roots", default="runs", help="comma-separated run roots, e.g. runs,runs_batch")
    ap.add_argument("--out", default=None)
    ap.add_argument("--vout", type=float, default=3.3)
    ap.add_argument("--cout", type=float, default=4.0)
    ap.add_argument("--psrr", type=float, default=41.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dataset", default="auto",
                    help="desktop dataset dir (gen_dataset.py jsonl); 'auto' = $LDO_DATASET or ../optimizer_pc/dataset; 'none' = off")
    a = ap.parse_args()
    path = a.out or default_path(a.pdk)
    roots = [os.path.join(HERE, r.strip()) for r in a.roots.split(",") if r.strip()]
    ds = None if a.dataset.lower() == "none" else a.dataset
    if a.cmd == "train":
        train(a.pdk, roots, path, dataset=ds)
    elif a.cmd == "info":
        X, Y, S, nrun, _ = load_runs(roots, a.pdk)
        print(f"{len(X)} usable evaluations in {nrun} run folders")
        for (v, c) in sorted(set(S)):
            print(f"   VOUT {v:g} V, Cout {c:g} uF: {S.count((v, c))}")
        if os.path.exists(path + ".json"):
            print(json.dumps(json.load(open(path + ".json")), indent=1))
        else:
            print(f"no model at {path}")
    else:
        m = load_or_none(path)
        if m is None:
            raise SystemExit(f"no model at {path}; run: python surrogate_ws.py train")
        spec = dict(vout=a.vout, cout=a.cout, psrr=a.psrr, line_pct=1.0, tol_pct=1.5, pm_floor=45.0,
                    dropout=0.3, vin_over=0.4)
        t0 = time.time()
        pts, preds, mg = propose(m, spec, BROAD[a.pdk], a.k)
        print(f"{len(pts)} proposals in {time.time()-t0:.1f}s (predicted worst margin, then metrics)")
        for x, p, g in zip(pts, preds, mg):
            print(f"  margin={g:+.3f}  x=[{', '.join(f'{v:.3g}' for v in x)}]  "
                  + " ".join(f"{t}={p[t]:.1f}" for t in TARGETS))


if __name__ == "__main__":
    main()
