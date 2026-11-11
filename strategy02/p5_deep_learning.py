#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 5 - Small deep-learning challengers on the 21-day target + adoption rule.

What changed versus strategy/p5
-------------------------------
  * Same 21-day target and the same walk-forward / purge rules as phase 4 (no 2-day labels).
  * Two deliberately small models (the volumes: "deeper is not better" with ~100 names):
      MLP  - 2 x 64 SiLU, LayerNorm, dropout 0.1, AdamW (wd 1e-3), 10% linear warm-up + cosine decay
      GRU  - 1 layer x 16 on the last 21 days of [vol-scaled return, relative volume, intraday range]
             concatenated with the cross-sectional features
    Each trained with 3 seeds (deep ensemble) on every 3rd date (labels overlap 21 days anyway),
    refit every 6 months on an expanding window.
  * Adoption rule (Level-4 must earn its place): the DL / LightGBM rank ensemble replaces the
    phase-4 score only if the paired t-stat of monthly rank-IC differences is > 2.0.
Output folder: phase5_deep_learning/   (oos_pred_final.csv is what phase 6 reads)
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

import p1_data_costs as L
import p3_features as F3
import p4_ml_models as M4

warnings.filterwarnings("ignore")
plt = L.plt
PHASE = "phase5_deep_learning"
SEQ = 21
try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
except ImportError:                 # pragma: no cover
    torch = None


def sequences(P, stocks):
    """(date, name) -> last SEQ days of [ret / vol63, log(vol / vol63avg), (h-l)/c]."""
    c, h, l, v = (P[k][stocks] for k in ("adj_close", "adj_high", "adj_low", "adj_volume"))
    r = c.pct_change()
    sd = r.rolling(63, min_periods=40).std()
    x1 = (r / sd).clip(-6, 6)
    x2 = np.log((v / v.rolling(63, min_periods=40).mean()).replace(0, np.nan)).clip(-3, 3)
    x3 = ((h - l) / c / sd).clip(0, 10)
    arr = np.stack([x1.to_numpy(), x2.to_numpy(), x3.to_numpy()], -1)       # T x N x 3
    return np.nan_to_num(arr), {d: i for i, d in enumerate(c.index)}, {n: j for j, n in enumerate(stocks)}


class MLP(nn.Module if torch else object):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x, s=None):
        return self.net(x).squeeze(-1)


class GRUNet(nn.Module if torch else object):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(3, 16, batch_first=True)
        self.head = nn.Sequential(nn.Linear(16 + d, 32), nn.SiLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x, s):
        _, hN = self.gru(s)
        return self.head(torch.cat([hN[-1], x], -1)).squeeze(-1)


def train_net(kind, X, S, y, w, seed, epochs=6, bs=512, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = MLP(X.shape[1]) if kind == "mlp" else GRUNet(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    n = len(X); steps = epochs * int(np.ceil(n / bs)); warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + np.cos(np.pi * (k - warm) / max(1, steps - warm))))
    Xt, St, yt, wt = (torch.tensor(a, dtype=torch.float32) for a in (X, S, y, w / w.mean()))
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = (lossf(net(Xt[b], St[b]), yt[b]) * wt[b]).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
    net.eval()
    return net


def predict(net, X, S):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(X, dtype=torch.float32), torch.tensor(S, dtype=torch.float32))).numpy()


def main():
    ap = L.base_args("Strategy02 Phase 5: small DL challengers on the 21-day target")
    args = ap.parse_args()
    if torch is None:
        raise SystemExit("pip install torch")
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    df, meta = M4.load_features()
    feats = meta["model_features"]
    lg = json.loads((L.HERE / "phase4_ml_models" / "lgbm_params.json").read_text())
    target = lg["gate"].get("target", "y_bin")
    df["y_xs"] = np.where(df["y_rank"].notna(), (df["y_rank"] > 0).astype(float), np.nan)
    df = df.dropna(subset=feats[:4]).reset_index(drop=True)
    X_raw = df[feats].to_numpy(np.float32)                 # (fix) fill + standardise per refit with TRAIN rows only
    P = L.load_daily_panel(data)
    stocks = [c for c in P["adj_close"].columns if c != L.INDEX]
    arr, di, ni = sequences(P, stocks)
    ti = df["date"].map(di).to_numpy(); nj = df["name"].map(ni).to_numpy()
    S_all = np.stack([arr[np.clip(ti - SEQ + 1 + k, 0, None), nj] for k in range(SEQ)], 1).astype(np.float32)

    dates = np.sort(df["date"].unique())
    refits = pd.date_range(M4.OOS_START, pd.Timestamp(dates[-1]) + pd.Timedelta(days=1), freq="6MS")
    seeds = [0] if args.quick else [0, 1, 2]
    epochs = 3 if args.quick else 6
    pred = {"mlp": np.full(len(df), np.nan), "gru": np.full(len(df), np.nan)}
    udates = pd.Series(dates)
    sub_dates = set(udates.iloc[::3])
    for a, b in zip(refits, list(refits[1:]) + [pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)]):
        tr = ((df["date"] <= a - pd.Timedelta(days=M4.PURGE_DAYS)) & df[target].notna() & df["date"].isin(sub_dates)).to_numpy()
        te = ((df["date"] >= a) & (df["date"] < b)).to_numpy()
        if tr.sum() < 5000 or not te.any():
            continue
        y = df.loc[tr, target].to_numpy(np.float32); w = df.loc[tr, "uniq_w"].fillna(0.05).to_numpy(np.float32)
        med = np.nanmedian(X_raw[tr], axis=0)
        Xtr = np.where(np.isnan(X_raw[tr]), med, X_raw[tr]); Xte = np.where(np.isnan(X_raw[te]), med, X_raw[te])
        mu, sdv = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = ((Xtr - mu) / sdv).astype(np.float32); Xte = ((Xte - mu) / sdv).astype(np.float32)
        for kind in pred:
            ps = [predict(train_net(kind, Xtr, S_all[tr], y, w, sd_, epochs=epochs), Xte, S_all[te]) for sd_ in seeds]
            pred[kind][te] = np.mean(ps, 0)
        print(f"   refit {a.date()}  train {tr.sum():>6}  predict {te.sum():>5}")

    lgbm = pd.read_csv(L.HERE / "phase4_ml_models" / "oos_pred_lgbm.csv", parse_dates=["date"])
    d = df[["date", "name"]].copy(); d["mlp"] = pred["mlp"]; d["gru"] = pred["gru"]
    d = lgbm.merge(d, on=["date", "name"], how="left")
    for c_ in ("prob_raw", "mlp", "gru"):
        d[c_ + "_pct"] = d.groupby("date")[c_].rank(pct=True)
    d["ensemble"] = d[["prob_raw_pct", "mlp_pct", "gru_pct"]].mean(1)

    me_dates = set(pd.Series(d["date"].unique()).groupby(pd.Series(d["date"].unique()).dt.to_period("M")).max())
    ev = d[d["date"].isin(me_dates)]
    ics = {k: M4.daily_ic(ev, c_) for k, c_ in [("LightGBM", "prob_raw"), ("MLP", "mlp"), ("GRU", "gru"), ("Ensemble", "ensemble")]}
    rows = []
    for k, ic in ics.items():
        diff = (ic - ics["LightGBM"]).dropna()
        rows.append({"model": k, "monthly_mean_ic": ic.mean(), "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                     "paired_t_vs_lgbm": diff.mean() / diff.std() * np.sqrt(len(diff)) if k != "LightGBM" and diff.std() > 0 else np.nan,
                     "n_months": len(ic)})
    cmp_ = pd.DataFrame(rows)
    cmp_.round(4).to_csv(out / "model_comparison.csv", index=False)
    t_ens = float(cmp_.set_index("model").loc["Ensemble", "paired_t_vs_lgbm"])
    adopt = bool(np.isfinite(t_ens) and t_ens > 2.0)
    d["score"] = d["ensemble"] if adopt else d["prob_raw_pct"]
    d.to_csv(out / "oos_pred_final.csv", index=False, float_format="%.6g")
    (out / "adoption.json").write_text(json.dumps({"rule": "adopt ensemble if paired t(monthly IC diff vs LightGBM) > 2",
                                                    "paired_t": round(t_ens, 3), "adopted": adopt,
                                                    "score_used_downstream": "ensemble" if adopt else "LightGBM prob_raw",
                                                    "dl_trials": 2}, indent=2))

    fig, ax = plt.subplots(figsize=(11, 5))
    for k, ic in ics.items():
        ax.plot(ic.rolling(12).mean(), label=f"{k} (mean {ic.mean():+.3f})")
    ax.axhline(0, color="k", lw=0.6); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("12-month rolling monthly rank-IC vs 21-day residual return (walk-forward OOS)")
    fig.tight_layout(); fig.savefig(out / "dl_vs_lgbm_ic.png", dpi=120); plt.close(fig)

    close, opn = P["adj_close"], P["adj_open"]
    liq = L.liquid_returns(data, close.index)
    hs = L.half_spread_bps(P["adj_high"][stocks], P["adj_low"][stocks], P["turnover"][stocks])
    me = L.month_end_mask(close.index)
    start = d["date"].min()
    sims = {}
    for k, c_ in [("Ensemble top10 monthly (CNC)", "ensemble"), ("MLP top10 monthly (CNC)", "mlp"), ("GRU top10 monthly (CNC)", "gru")]:
        sc = d.pivot(index="date", columns="name", values=c_).reindex(close.index)
        sims[k] = L.simulate_weights(F3.top_k_buffer(sc, 10, 20, me), opn, close, liq, pledge=False, half_spread=hs,
                                     rebal_mask=me)["ret"].loc[start:]
    for nm in ("LightGBM top10 monthly (CNC)", "Composite rank top10 (phase 3)"):
        s_ = L.find_series(PHASE, nm)
        if s_ is not None:
            sims[nm] = s_.loc[start:]
    cards = [L.scorecard(v, k) for k, v in sims.items()]
    best = L.save_scorecards(out, cards, PHASE, notes=f"DL adoption: {adopt} (t={t_ens:.2f})", series=sims)
    L.multi_panel(sims, out / "dl_books_all_in_one.png", "Phase 5 - DL challengers vs LightGBM vs Level-1 (CNC, OOS)",
                  highlight="Ensemble top10 monthly (CNC)")
    tab = pd.DataFrame(cards)[L.SC_COLS]
    (out / "report.md").write_text(f"""# Strategy02 - Phase 5: deep-learning challengers

Target `{target}` (from phase 4), refit every 6 months, {len(seeds)} seeds per model.

{L.md_table(cmp_)}

Adoption (paired t > 2 vs LightGBM): **{adopt}** (t = {t_ens:.2f}). Score passed to phase 6:
**{'ensemble' if adopt else 'LightGBM prob_raw'}**.

{L.md_table(tab)}

![ic](dl_vs_lgbm_ic.png)
![books](dl_books_all_in_one.png)
""")
    L.print_scorecard(best, PHASE)
    print(f"[P5] adopted ensemble: {adopt} (t={t_ens:.2f}); done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
