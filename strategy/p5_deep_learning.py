#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 5 - Deep learning (PyTorch): ANN/MLP, LSTM, GRU, Transformer encoder, CNN-LSTM hybrid,
plus a causal stacking ensemble with the best Phase-4 gradient-boosting model.

Daily (CNC) problem: for every (date t, stock i) the network sees the last SEQ_LEN days of
that stock's cross-sectionally ranked features (+ market regime features) and predicts the
gauss-ranked 2-day excess return. Intraday (MIS) problem: tabular MLP on decision-point features.

Validation mirrors Phase 4 exactly (expanding walk-forward, embargo) so ICs are comparable.
Hyper-parameters: small Optuna search on fold 1 (if installed), then frozen.
Adoption rule: the ensemble is used downstream ONLY if its OOS daily IC beats Phase 4's best
with a paired t-test on the daily IC series (p < 0.10). Otherwise Phase 6 keeps the Phase-4 model.

Uses a GPU automatically if available. --quick: 2 models, fewer folds/epochs.
Output folder: phase5_deep_learning/
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy import stats

import p1_data_costs as L
import p4_ml_models as P4

plt = L.plt
PHASE = "phase5_deep_learning"

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

SEQ_LEN = 20


# =============================================================================
# Models
# =============================================================================
if torch is not None:
    class MLP(nn.Module):
        def __init__(self, d, h=64, drop=0.1, seq=False):
            super().__init__()
            self.seq = seq
            self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(h, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, 1))

        def forward(self, x):
            return self.net(x[:, -1, :] if self.seq else x).squeeze(-1)

    class RNN(nn.Module):
        def __init__(self, d, h=48, cell="lstm", drop=0.1):
            super().__init__()
            R = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = R(d, h, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(h), nn.Dropout(drop), nn.Linear(h, 1))

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1, :]).squeeze(-1)

    class CNNLSTM(nn.Module):
        def __init__(self, d, h=48, drop=0.1):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(d, h, kernel_size=3, padding=2, dilation=2), nn.GELU())
            self.rnn = nn.LSTM(h, h, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(h), nn.Dropout(drop), nn.Linear(h, 1))

        def forward(self, x):
            z = self.conv(x.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)   # causal crop
            o, _ = self.rnn(z)
            return self.head(o[:, -1, :]).squeeze(-1)

    class TransformerEnc(nn.Module):
        def __init__(self, d, h=48, heads=4, layers=2, drop=0.1, seq_len=SEQ_LEN):
            super().__init__()
            self.inp = nn.Linear(d, h)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, h))
            layer = nn.TransformerEncoderLayer(h, heads, dim_feedforward=2 * h, dropout=drop,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers)
            self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)   # causal attention
            self.register_buffer("mask", mask)

        def forward(self, x):
            z = self.inp(x) + self.pos[:, : x.shape[1]]
            z = self.enc(z, mask=self.mask[: x.shape[1], : x.shape[1]])
            return self.head(z[:, -1, :]).squeeze(-1)


def make_model(kind, d, hp):
    if kind == "MLP":
        return MLP(d, hp.get("h", 64), hp.get("drop", 0.1), seq=True)
    if kind == "LSTM":
        return RNN(d, hp.get("h", 48), "lstm", hp.get("drop", 0.1))
    if kind == "GRU":
        return RNN(d, hp.get("h", 48), "gru", hp.get("drop", 0.1))
    if kind == "CNN-LSTM":
        return CNNLSTM(d, hp.get("h", 48), hp.get("drop", 0.1))
    if kind == "Transformer":
        return TransformerEnc(d, hp.get("h", 48), 4, hp.get("layers", 2), hp.get("drop", 0.1))
    raise ValueError(kind)


# =============================================================================
# Data: panel tensor + index sampling (memory-efficient sequences)
# =============================================================================
def build_panel(D: pd.DataFrame, feats, target):
    dates = pd.DatetimeIndex(D["date"].drop_duplicates().sort_values())
    names = sorted(D["name"].unique())
    di = pd.Series(np.arange(len(dates)), index=dates)
    ni = pd.Series(np.arange(len(names)), index=names)
    X = np.zeros((len(dates), len(names), len(feats)), dtype=np.float32)
    Y = np.full((len(dates), len(names)), np.nan, dtype=np.float32)
    M = np.zeros((len(dates), len(names)), dtype=bool)
    t, i = di[D["date"]].to_numpy(), ni[D["name"]].to_numpy()
    X[t, i] = np.nan_to_num(D[feats].to_numpy(dtype=np.float32))
    yr = P4.gauss_rank(D[target], D["date"]).to_numpy()
    Y[t, i] = yr
    M[t, i] = True
    return dates, names, X, Y, M


def gather(X, idx_t, idx_i, L_):
    offs = np.arange(-L_ + 1, 1)
    tt = np.clip(idx_t[:, None] + offs[None, :], 0, None)
    return X[tt, idx_i[:, None], :]


def train_one(kind, X, Y, tr_idx, va_idx, hp, dev, epochs, seq=True, bs=1024, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = X.shape[-1]
    model = make_model(kind, d, hp).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.get("lr", 1e-3), weight_decay=hp.get("wd", 1e-4))
    steps = max(1, epochs * int(np.ceil(len(tr_idx[0]) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp.get("lr", 1e-3), total_steps=steps, pct_start=0.2)
    lossf = nn.HuberLoss(delta=1.0)
    best, best_state, bad = -np.inf, None, 0

    def batches(idx, shuffle):
        n = len(idx[0])
        order = np.random.permutation(n) if shuffle else np.arange(n)
        for k in range(0, n, bs):
            b = order[k:k + bs]
            xb = gather(X, idx[0][b], idx[1][b], SEQ_LEN) if seq else X[idx[0][b], idx[1][b]]
            yield torch.from_numpy(xb).to(dev), torch.from_numpy(Y[idx[0][b], idx[1][b]]).to(dev)

    for ep in range(epochs):
        model.train()
        for xb, yb in batches(tr_idx, True):
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        # validation IC (early stopping)
        p = predict(model, X, va_idx, dev, seq)
        df = pd.DataFrame({"t": va_idx[0], "p": p, "y": Y[va_idx[0], va_idx[1]]})
        ic = df.groupby("t").apply(lambda g: g["p"].corr(g["y"], method="spearman"), include_groups=False).mean()
        if ic > best:
            best, best_state, bad = ic, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def predict(model, X, idx, dev, seq=True, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(idx[0]), bs):
            xb = gather(X, idx[0][k:k + bs], idx[1][k:k + bs], SEQ_LEN) if seq else X[idx[0][k:k + bs], idx[1][k:k + bs]]
            out.append(model(torch.from_numpy(xb).to(dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def tune_hp(kind, X, Y, tr_idx, va_idx, dev, n_trials, epochs):
    base = {"h": 48, "drop": 0.1, "lr": 1e-3, "wd": 1e-4, "layers": 2}
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return base
    if n_trials <= 0:
        return base

    def obj(tr):
        hp = {"h": tr.suggest_categorical("h", [32, 48, 64]), "drop": tr.suggest_float("drop", 0.0, 0.3),
              "lr": tr.suggest_float("lr", 3e-4, 3e-3, log=True), "wd": tr.suggest_float("wd", 1e-5, 1e-2, log=True),
              "layers": tr.suggest_int("layers", 1, 3)}
        sub = (tr_idx[0][::3], tr_idx[1][::3])
        _, ic = train_one(kind, X, Y, sub, va_idx, hp, dev, max(2, epochs // 2), seq=True)
        return ic

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    base.update(st.best_params)
    return base


# =============================================================================
def daily_walk_forward(D, feats, kinds, dev, quick, trials, epochs):
    target = "fwd2_ex"
    D = D.dropna(subset=["ret_21"])
    dates, names, X, Y, M = build_panel(D, feats, target)
    valid = M & np.isfinite(Y)
    tt, ii = np.nonzero(valid)
    keep = tt >= SEQ_LEN
    tt, ii = tt[keep], ii[keep]
    years = sorted({d.year for d in dates if d >= pd.Timestamp("2019-01-01")})
    if quick:
        years = years[-3:]
    preds = []
    hps = {}
    for fi, y in enumerate(years):
        ts = dates.searchsorted(pd.Timestamp(f"{y}-01-01"))
        te_end = dates.searchsorted(pd.Timestamp(f"{y+1}-01-01"))
        emb = 3
        tr_m = tt < ts - emb
        va_cut = int(ts - emb - 0.15 * (ts - emb))
        itr = (tt[tr_m & (tt < va_cut - emb)], ii[tr_m & (tt < va_cut - emb)])
        iva = (tt[tr_m & (tt >= va_cut)], ii[tr_m & (tt >= va_cut)])
        te_m = (tt >= ts) & (tt < te_end)
        ite = (tt[te_m], ii[te_m])
        # prediction rows for the test year must include rows with unknown targets too (latest days)
        tm = M.copy(); tm[:ts] = False; tm[te_end:] = False
        pt, pi = np.nonzero(tm)
        if len(ite[0]) == 0 and len(pt) == 0:
            continue
        out = pd.DataFrame({"date": dates[pt], "name": np.array(names)[pi]})
        for kind in kinds:
            if kind not in hps:
                hps[kind] = tune_hp(kind, X, Y, itr, iva, dev, trials if kind in ("LSTM", "Transformer") else 0, epochs)
            model, vic = train_one(kind, X, Y, itr, iva, hps[kind], dev, epochs, seq=True, seed=fi)
            out[f"pred_{kind}"] = predict(model, X, (pt, pi), dev, seq=True)
            print(f"   [daily {kind}] fold {y}: val IC {vic:.4f}")
        preds.append(out)
    P = pd.concat(preds, ignore_index=True)
    P = P.merge(D[["date", "name", "fwd2", "fwd2_ex", "fwd5_ex", "vol_20_raw"]], on=["date", "name"], how="left")
    return P, hps


def intraday_mlp(I, feats, dev, quick, epochs):
    I = I.sort_values("date").reset_index(drop=True)
    I["y"] = P4.gauss_rank(I["ret_to_exit"], I["date"].astype(str) + I["slot"])
    dates = pd.DatetimeIndex(I["date"].drop_duplicates().sort_values())
    first = dates[min(250, len(dates) // 2)]
    step = 84 if quick else 42
    starts = dates[dates >= first][::step]
    X = np.nan_to_num(I[feats].to_numpy(dtype=np.float32))
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
    preds = []
    for k, ts in enumerate(starts):
        te_end = starts[k + 1] if k + 1 < len(starts) else dates[-1] + pd.Timedelta(days=1)
        tr = I.index[(I["date"] < ts - pd.Timedelta(days=1)) & I["y"].notna()].to_numpy()
        te = I.index[(I["date"] >= ts) & (I["date"] < te_end)].to_numpy()
        if len(tr) < 5000 or len(te) == 0:
            continue
        # scale with train statistics only
        m_, s_ = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = ((X - m_) / s_).astype(np.float32)
        Ys = I["y"].to_numpy(dtype=np.float32)
        va = tr[int(len(tr) * 0.85):]
        trn = tr[: int(len(tr) * 0.85)]
        torch.manual_seed(k)
        model = MLP(Xs.shape[1], 64, 0.2).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        lossf = nn.HuberLoss()
        best, best_state = -np.inf, None
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(trn)
            for b in range(0, len(perm), 2048):
                idx = perm[b:b + 2048]
                xb = torch.from_numpy(Xs[idx]).to(dev); yb = torch.from_numpy(Ys[idx]).to(dev)
                opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(torch.from_numpy(Xs[va]).to(dev)).cpu().numpy()
            ic = stats.spearmanr(pv, Ys[va]).statistic
            if ic > best:
                best, best_state = ic, {kk: vv.clone() for kk, vv in model.state_dict().items()}
        model.load_state_dict(best_state)
        with torch.no_grad():
            p = model(torch.from_numpy(Xs[te]).to(dev)).cpu().numpy()
        o = I.loc[te, ["date", "name", "slot"]].copy()
        o["pred_MLP"] = p
        preds.append(o)
        print(f"   [intraday MLP] fold {k+1}/{len(starts)} val IC {best:.4f}")
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def causal_blend(df, cols, target, by="date", grid=(0, .25, .5, .75, 1.0), refit_days=63, zby=None):
    """Blend z-scored predictions a*c1 + (1-a)*c2 with `a` chosen on PAST OOS rows only.
    zby: key for the cross-sectional z-score (fix: intraday uses date+slot so a 09:30 prediction is
    never standardised with later slots of the same day)."""
    z = df[cols].groupby(df[zby or by]).transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    dates = pd.DatetimeIndex(df[by].drop_duplicates().sort_values())
    out = pd.Series(np.nan, index=df.index)
    chosen = []
    for k in range(0, len(dates), refit_days):
        d0 = dates[k]
        d1 = dates[k + refit_days] if k + refit_days < len(dates) else dates[-1] + pd.Timedelta(days=1)
        hist = df[by] < d0 - pd.Timedelta(days=8)          # > 2-day target horizon + weekends
        a_best = 0.5
        if hist.sum() > 20000:
            best = -np.inf
            for a in grid:
                s = a * z.loc[hist, cols[0]] + (1 - a) * z.loc[hist, cols[1]]
                ic = pd.DataFrame({"s": s, "y": df.loc[hist, target], "d": df.loc[hist, by]}).dropna() \
                    .groupby("d").apply(lambda g: g["s"].corr(g["y"], method="spearman"), include_groups=False).mean()
                if ic > best:
                    best, a_best = ic, a
        m = (df[by] >= d0) & (df[by] < d1)
        out[m] = a_best * z.loc[m, cols[0]] + (1 - a_best) * z.loc[m, cols[1]]
        chosen.append({"from": str(d0.date()), "weight_on_" + cols[0]: a_best})
    return out, chosen


def main():
    ap = L.base_args("Phase 5: deep learning + causal ensemble")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()
    out = L.out_dir(PHASE)
    t0 = time.time()
    if torch is None:
        msg = "PyTorch not installed (pip install torch). Phase 5 skipped; Phase 6 will use Phase-4 models."
        print("[P5] " + msg)
        (out / "report.md").write_text("# Phase 5\n\n" + msg)
        (out / "adoption.json").write_text(json.dumps({"adopt_daily": False, "adopt_intraday": False, "reason": msg}))
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))
    print(f"[P5] device={dev}")
    p3, p4 = L.HERE / "phase3_features", L.HERE / "phase4_ml_models"
    fl = json.loads((p3 / "feature_lists.json").read_text())
    D = pd.read_csv(p3 / "daily_features.csv", parse_dates=["date"])
    feats = fl["daily_features"] + fl["daily_market_features"]
    for c in fl["daily_market_features"]:
        # (fix) expanding z-score using only PAST dates (was a full-sample mean/std = look-ahead)
        mk_ = D.groupby("date")[c].first().sort_index()
        mu_, sd_ = mk_.expanding(60).mean().shift(1), mk_.expanding(60).std().shift(1)
        D[c] = ((D[c] - D["date"].map(mu_)) / (D["date"].map(sd_) + 1e-9)).fillna(0.0)
    kinds = ["MLP", "LSTM"] if args.quick else ["MLP", "LSTM", "GRU", "CNN-LSTM", "Transformer"]
    epochs = 2 if args.quick else args.epochs
    trials = 0 if args.quick else args.trials
    print(f"[P5] daily sequence models {kinds} (seq_len={SEQ_LEN}, epochs={epochs})")
    PD, hps = daily_walk_forward(D, feats, kinds, dev, args.quick, trials, epochs)
    (out / "dl_hyperparams.json").write_text(json.dumps(hps, indent=2))

    # ---------- compare with Phase-4 best ------------------------------------
    p4e = pd.read_csv(p4 / "model_evaluation.csv")
    b4 = p4e[(p4e.target == "fwd2_ex") & (p4e.model != "prob_up")].sort_values("ic_t", ascending=False).iloc[0]["model"]
    P4d = pd.read_csv(p4 / "oos_pred_daily_fwd2_ex.csv", parse_dates=["date"])[["date", "name", b4]]
    M = PD.merge(P4d, on=["date", "name"], how="inner")
    ev = P4.evaluate(M.dropna(subset=["fwd2_ex"]), "fwd2_ex", "date", [f"pred_{k}" for k in kinds] + [b4])
    best_dl = ev[ev.model != b4].sort_values("ic_t", ascending=False).iloc[0]["model"]
    M["pred_ensemble"], blend_w = causal_blend(M, [b4, best_dl], "fwd2_ex")
    ev = pd.concat([ev, P4.evaluate(M.dropna(subset=["fwd2_ex", "pred_ensemble"]), "fwd2_ex", "date", ["pred_ensemble"])])
    ev.to_csv(out / "daily_model_comparison.csv", index=False)
    ic_e = P4.daily_ic(M.dropna(subset=["fwd2_ex", "pred_ensemble"]), "pred_ensemble", "fwd2_ex")
    ic_b = P4.daily_ic(M.dropna(subset=["fwd2_ex"]), b4, "fwd2_ex").reindex(ic_e.index)
    tt = stats.ttest_rel(ic_e, ic_b, nan_policy="omit")
    adopt_d = bool(tt.statistic > 0 and tt.pvalue / 2 < 0.10)
    M.to_csv(out / "oos_pred_daily_ensemble.csv", index=False, float_format="%.6g")

    # ---------- intraday MLP + blend with Phase-4 ridge/lgbm -------------------
    I = pd.read_csv(p3 / "intraday_features.csv", parse_dates=["date"])
    I["slot"] = I["slot"].astype(str).str.zfill(4)
    for s in ("0930", "1100", "1300"):
        I[f"slot_{s}"] = (I["slot"] == s).astype(int)
    ifeats = [c for c in fl["intraday_features"] + ["mkt_ret_since_open", "dow", "slot_0930", "slot_1100", "slot_1300"] if c in I]
    print("[P5] intraday MLP ...")
    PI = intraday_mlp(I, ifeats, dev, args.quick, 3 if args.quick else 8)
    pi4 = pd.read_csv(p4 / "oos_pred_intraday_ret_to_exit.csv", parse_dates=["date"])
    pi4["slot"] = pi4["slot"].astype(str).str.zfill(4)
    ie = p4e[(p4e.target == "ret_to_exit") & (p4e.model != "prob_up")].sort_values("ic_t", ascending=False).iloc[0]["model"]
    MI = pi4.merge(PI, on=["date", "name", "slot"], how="inner")
    MI["dk"] = MI["date"].dt.strftime("%Y-%m-%d") + MI["slot"]
    MI["dkd"] = pd.to_datetime(MI["date"])
    MI["pred_ensemble"], iblend = causal_blend(MI, [ie, "pred_MLP"], "ret_to_exit", by="dkd", refit_days=42, zby="dk")
    evi = P4.evaluate(MI.dropna(subset=["ret_to_exit"]), "ret_to_exit", "dk", [ie, "pred_MLP", "pred_ensemble"])
    evi.to_csv(out / "intraday_model_comparison.csv", index=False)
    ic_ie = P4.daily_ic(MI.dropna(subset=["pred_ensemble"]), "pred_ensemble", "ret_to_exit", "dk")
    ic_ib = P4.daily_ic(MI, ie, "ret_to_exit", "dk").reindex(ic_ie.index)
    tti = stats.ttest_rel(ic_ie, ic_ib, nan_policy="omit")
    adopt_i = bool(tti.statistic > 0 and tti.pvalue / 2 < 0.10)
    MI.drop(columns=["dk", "dkd"]).to_csv(out / "oos_pred_intraday_ensemble.csv", index=False, float_format="%.6g")
    adoption = {"adopt_daily": adopt_d, "daily_paired_t": round(float(tt.statistic), 3), "daily_p_one_sided": round(float(tt.pvalue / 2), 4),
                "phase4_daily_model": b4, "best_dl_daily": best_dl,
                "adopt_intraday": adopt_i, "intraday_paired_t": round(float(tti.statistic), 3),
                "intraday_p_one_sided": round(float(tti.pvalue / 2), 4), "phase4_intraday_model": ie,
                "blend_weights_daily": blend_w, "blend_weights_intraday": iblend}
    (out / "adoption.json").write_text(json.dumps(adoption, indent=2, default=str))

    # ---------- candidate strategies -----------------------------------------
    data = L.data_dir(args.data)
    Pn = L.load_daily_panel(data)
    hs = L.half_spread_bps(Pn["adj_high"], Pn["adj_low"], Pn["turnover"])
    liq = L.liquid_returns(data, Pn["adj_close"].index)
    sims = {}
    col_d = "pred_ensemble" if adopt_d else best_dl
    W = P4.cnc_from_pred(M.rename(columns={col_d: "s"}), "s", k=4, hold=2, buffer=6)
    sims[f"CNC DL top-4 ({col_d})"] = L.simulate_weights(W, Pn["adj_open"], Pn["adj_close"], liq, pledge=True,
                                                        half_spread=hs)["ret"].loc[W.index.min():]
    MI["exp_ret"] = P4.causal_calibrate(MI, "pred_ensemble" if adopt_i else ie, "ret_to_exit")
    mis = P4.mis_from_pred(MI.dropna(subset=["exp_ret"]), "exp_ret", mode="directional", gross=2.0,
                           hurdle=L.mis_rate_roundtrip() + 2 * 2.2e-4 + 3e-4)
    sims["Liquid-pledge + MIS DL/ensemble directional (cost hurdle)"] = liq.reindex(mis.index).fillna(0) + mis
    cards = [L.scorecard(s, k) for k, s in sims.items()]
    best = L.save_scorecards(out, cards, PHASE, series=sims, notes=f"DL adoption: {adoption['adopt_daily']} (daily), {adoption['adopt_intraday']} (intraday)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    e = ev.sort_values("ic_t")
    ax.barh(e["model"], e["ic_t"], color=["tab:orange" if m == "pred_ensemble" else "tab:blue" for m in e["model"]])
    ax.set_title("Daily OOS rank-IC t-stat: DL vs Phase-4 vs ensemble")
    fig.tight_layout(); fig.savefig(out / "daily_model_comparison.png", dpi=120); plt.close(fig)

    md = f"""# Phase 5 - Deep learning + causal ensemble

Device: {dev} | sequence length {SEQ_LEN} | models: {kinds} | epochs {epochs} | Optuna trials {trials}

## Daily (2-day excess return), OOS walk-forward, yearly refits
{L.md_table(ev)}

Paired t-test (ensemble IC - Phase-4 {b4} IC): t = {tt.statistic:.2f}, one-sided p = {tt.pvalue/2:.3f}
-> **ensemble adopted for CNC: {adopt_d}**

## Intraday (return to 15:15)
{L.md_table(evi)}

Paired t-test: t = {tti.statistic:.2f}, one-sided p = {tti.pvalue/2:.3f} -> **adopted for MIS: {adopt_i}**

## Candidate scorecards
{L.md_table(pd.DataFrame(cards)[L.SC_COLS])}

**Best overall (incl. carried):** {best.get('strategy')} | P(profitable 12m) {best.get('P_profitable_12m')}
| predicted monthly {best.get('pred_monthly_mean')} | goal score {best.get('goal_score')}
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    L.print_scorecard(best, PHASE)
    print(f"[P5] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
