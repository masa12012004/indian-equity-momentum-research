#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 9 - DAY-BY-DAY risk overlay and trade-quality model with ML + DL.

Why
---
Phase 8 showed flat stretches in the test window (Jan-Apr 2025, Mar-Oct 2026): the monthly
SMA200 trend exit had moved the whole book into cash / the liquid ETF. The exit itself saved
money, but it was checked only at month-ends, so the book re-entered late after the April 2025
rebound and sat idle. This phase decides the exposure EVERY DAY.

What it does (all walk-forward, nothing fitted on the future)
  1. Market-state features every day: NIFTY distance to SMA50/SMA200, 5/21/63-day returns,
     21-day vol and vol ratio, drawdown, breadth (above SMA50/SMA200, at 52-week highs),
     dispersion, HMM bull/hostile probabilities, Kalman trend t-stat, absorption ratio.
  2. Target: NIFTY beats the liquid ETF over the next 21 trading days (1/0).
  3. Daily P(risk-on) from
       rule   : daily SMA200 with a 2% exit band (Faber, checked daily, 3-day confirmation)
       lgbm   : LightGBM (small, monotone-free), refit every 21 days, 30-day purge
       gru    : GRU(16) on the last 63 days of the features, 3 seeds, refit every 126 days
       ens    : average of lgbm and gru
     Probability -> exposure 1 / 0.5 / 0 with fixed thresholds 0.55 / 0.40 and a 3-day
     confirmation (fewer whipsaw trades). Thresholds are NOT tuned.
  4. Trade quality: stock score mom_vn_ml = z(vol-normalised momentum) + 0.5 x z(ML probability
     from phases 4/5), so names the ML model dislikes are skipped at entry.
  5. Grid (exposure x score x k) selected on TRAIN (first 70%) by the goal score, frozen, TEST
     reported. Sharpe is shown for every line.
Outputs: phase9_daily_ml_overlay/ (train_70pct.png, test_30pct.png, exposure_timeline.png,
         timing_model_eval.csv, grid_results.csv, metrics.csv, report.md)
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

import p1_data_costs as L
import p6_portfolio_backtest as E
import p8_train_test_split as S8

warnings.filterwarnings("ignore")
plt = L.plt
PHASE = "phase9_daily_ml_overlay"
H = 21
PURGE = 30
SEQ = 63
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
except ImportError:  # pragma: no cover
    torch = None


# ----------------------------------------------------------------------------
def market_features(mk: E.Market) -> pd.DataFrame:
    n = mk.nifty.reindex(mk.dates)
    C = pd.DataFrame(mk.C, index=mk.dates, columns=mk.stocks)
    lr = np.log(n).diff()
    reg = pd.read_csv(L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv", index_col=0, parse_dates=True).reindex(mk.dates)
    f = pd.DataFrame({
        "dist_sma50": n / n.rolling(50).mean() - 1,
        "dist_sma200": n / n.rolling(200).mean() - 1,
        "ret_5": n.pct_change(5), "ret_21": n.pct_change(21), "ret_63": n.pct_change(63),
        "vol_21": lr.rolling(21).std() * np.sqrt(252),
        "vol_ratio": lr.rolling(5).std() / lr.rolling(63).std(),
        "drawdown_252": n / n.rolling(252).max() - 1,
        "breadth_50": (C > C.rolling(50).mean()).astype(float).where(C.notna()).mean(1),
        "breadth_200": reg["breadth_above_200"],
        "new_high_share": (C >= C.rolling(252, min_periods=200).max() * 0.98).astype(float).where(C.notna()).mean(1),
        "dispersion_21": reg["dispersion_21"],
        "p_bull": reg["p_bull"], "p_hostile": reg["p_hostile"],
        "trend_t": reg["trend_t"], "absorption": reg["absorption_ratio"].ffill(),
    }, index=mk.dates)
    return f


def label(mk: E.Market) -> pd.Series:
    n = mk.nifty.reindex(mk.dates)
    liq = pd.Series(mk.liq, index=mk.dates)
    fwd = n.shift(-H) / n - 1
    liq_fwd = (1 + liq).rolling(H).apply(np.prod, raw=True).shift(-H) - 1
    y = (fwd > liq_fwd).astype(float)
    return y.where(fwd.notna())


def walk_lgbm(F: pd.DataFrame, y: pd.Series, start: pd.Timestamp, step: int = 21) -> pd.Series:
    out = pd.Series(np.nan, index=F.index)
    idx = F.index
    ok = F.notna().all(1)
    i0 = int(np.searchsorted(idx, start))
    for a in range(i0, len(idx), step):
        cut = idx[a] - pd.Timedelta(days=PURGE + 14)
        tr = ok & y.notna() & (idx <= cut)
        if tr.sum() < 250:
            continue
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=2)
        m.fit(F[tr], y[tr].astype(int))
        b = min(a + step, len(idx))
        te = F.iloc[a:b]
        good = te.notna().all(1)
        if good.any():
            out.iloc[a:b] = np.where(good, m.predict_proba(te.fillna(0))[:, 1], np.nan)
    return out


class GRUNet(nn.Module if torch else object):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, 16, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(16, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def walk_gru(F: pd.DataFrame, y: pd.Series, start: pd.Timestamp, step: int = 126, seeds=(0, 1, 2), epochs=40) -> pd.Series:
    out = pd.Series(np.nan, index=F.index)
    X = F.ffill().to_numpy(np.float32)
    idx = F.index
    valid = np.isfinite(X).all(1)
    i0 = int(np.searchsorted(idx, start))
    for a in range(i0, len(idx), step):
        cut = idx[a] - pd.Timedelta(days=PURGE + 14)
        tr_end = int(np.searchsorted(idx, cut, "right"))
        tr_idx = [t for t in range(SEQ, tr_end) if valid[t - SEQ + 1:t + 1].all() and np.isfinite(y.iloc[t])]
        if len(tr_idx) < 250:
            continue
        mu = X[:tr_end][valid[:tr_end]].mean(0); sd = X[:tr_end][valid[:tr_end]].std(0) + 1e-6
        Z = np.nan_to_num((X - mu) / sd).clip(-6, 6)
        seq = lambda ts: np.stack([Z[t - SEQ + 1:t + 1] for t in ts])
        Xt = torch.tensor(seq(tr_idx)); yt = torch.tensor(y.iloc[tr_idx].to_numpy(np.float32))
        b = min(a + step, len(idx))
        te_idx = [t for t in range(a, b) if t >= SEQ - 1 and valid[t - SEQ + 1:t + 1].all()]
        if not te_idx:
            continue
        Xe = torch.tensor(seq(te_idx))
        preds = []
        for sd_ in seeds:
            torch.manual_seed(sd_)
            net = GRUNet(Z.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
            lossf = nn.BCEWithLogitsLoss()
            net.train()
            for _ in range(epochs):
                perm = torch.randperm(len(Xt))
                for k in range(0, len(Xt), 128):
                    bb = perm[k:k + 128]
                    loss = lossf(net(Xt[bb]), yt[bb])
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(net(Xe)).numpy())
        out.iloc[te_idx] = np.mean(preds, 0)
    return out


def to_exposure(p: pd.Series, hi=0.55, lo=0.40, confirm=3) -> pd.Series:
    """Probability -> 1 / 0.5 / 0 with a 3-day confirmation (debounce)."""
    raw = np.where(p >= hi, 1.0, np.where(p <= lo, 0.0, 0.5))
    raw = np.where(np.isfinite(p.to_numpy()), raw, np.nan)
    out = np.full(len(raw), np.nan); state = 1.0; cand, cnt = None, 0
    for t, r in enumerate(raw):
        if np.isnan(r):
            out[t] = np.nan if t == 0 or np.isnan(out[t - 1]) else state
            continue
        if r != state:
            cnt = cnt + 1 if r == cand else 1
            cand = r
            if cnt >= confirm:
                state, cnt = r, 0
        else:
            cnt = 0
        out[t] = state
    return pd.Series(out, index=p.index)


def rule_daily(mk: E.Market, band=0.02, confirm=3) -> pd.Series:
    n = mk.nifty.reindex(mk.dates)
    sma = n.rolling(200, min_periods=150).mean()
    out = np.full(len(n), np.nan); state = 1.0; cnt = 0
    for t in range(len(n)):
        if not np.isfinite(sma.iloc[t]):
            continue
        want = 0.0 if n.iloc[t] < sma.iloc[t] * (1 - band) else (1.0 if n.iloc[t] > sma.iloc[t] else state)
        cnt = cnt + 1 if want != state else 0
        if cnt >= confirm:
            state, cnt = want, 0
        out[t] = state
    return pd.Series(out, index=mk.dates)


def reentry_daily(mk: E.Market, confirm=3) -> pd.Series:
    """Exit only at a month-end close below SMA200 (as phase 8); RE-ENTER on any day once NIFTY has
    closed above SMA200 for 3 days - fixes the late re-entry after the April 2025 rebound."""
    n = mk.nifty.reindex(mk.dates)
    sma = n.rolling(200, min_periods=150).mean()
    out = np.full(len(n), np.nan); state = 1.0; cnt = 0
    for t in range(len(n)):
        if not np.isfinite(sma.iloc[t]):
            continue
        above = n.iloc[t] > sma.iloc[t]
        if state == 1.0 and mk.me[t] and not above:
            state = 0.0
        elif state == 0.0:
            cnt = cnt + 1 if above else 0
            if cnt >= confirm:
                state, cnt = 1.0, 0
        out[t] = state
    return pd.Series(out, index=mk.dates)


def vol_managed(mk: E.Market, target=0.15, confirm=3) -> pd.Series:
    """Moreira & Muir (2017) volatility-managed exposure: target / HAR forecast of NIFTY vol,
    in 0.25 steps between 0.25 and 1, 3-day confirmation."""
    n = mk.nifty.reindex(mk.dates)
    lr = np.log(n).diff()
    har = np.sqrt(252) * (0.5 * lr.rolling(5).std() + 0.3 * lr.rolling(21).std() + 0.2 * lr.rolling(63).std())
    e = ((target / har).clip(0.25, 1.0) * 4).round() / 4
    out = np.full(len(e), np.nan); state = 1.0; cand = None; cnt = 0
    for t, r in enumerate(e.to_numpy()):
        if np.isnan(r):
            out[t] = state
            continue
        if r != state:
            cnt = cnt + 1 if r == cand else 1; cand = r
            if cnt >= confirm:
                state, cnt = r, 0
        else:
            cnt = 0
        out[t] = state
    return pd.Series(out, index=mk.dates)


def trade_quality(tr: pd.DataFrame) -> dict:
    if tr is None or tr.empty:
        return {}
    sells = tr[tr["side"] == "SELL"].dropna(subset=["pnl_pct"])
    return {"orders": int(len(tr)), "closed_positions": int(len(sells)),
            "hit_rate": round(float((sells["pnl_pct"] > 0).mean()), 3),
            "avg_pnl_pct_per_exit": round(float(sells["pnl_pct"].mean()), 4),
            "avg_win_pct": round(float(sells.loc[sells["pnl_pct"] > 0, "pnl_pct"].mean()), 4),
            "avg_loss_pct": round(float(sells.loc[sells["pnl_pct"] <= 0, "pnl_pct"].mean()), 4),
            "costs_rs": round(float(tr["cost"].sum()))}


def main():
    ap = L.base_args("Strategy02 Phase 9: daily ML/DL risk overlay + trade-quality score")
    args = ap.parse_args()
    if lgb is None or torch is None:
        raise SystemExit("pip install lightgbm torch")
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    mk = E.Market(data)
    scores, meta = E.load_scores(mk)
    regime = pd.read_csv(L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv", index_col=0, parse_dates=True)
    split = json.loads((L.HERE / "phase8_train_test_split" / "split.json").read_text())
    tr_w = (pd.Timestamp(split["train"][0]), pd.Timestamp(split["train"][1]))
    te_w = (pd.Timestamp(split["test"][0]), pd.Timestamp(split["test"][1]))
    start = tr_w[0]

    # ---------------- 1-3: daily timing models
    print("[P9] market features + labels ...")
    F = market_features(mk); y = label(mk)
    pred_start = pd.Timestamp("2018-07-01")       # predictions before the backtest start warm up the overlay
    print("[P9] LightGBM walk-forward (refit every 21 days) ...")
    p_lgbm = walk_lgbm(F, y, pred_start)
    print("[P9] GRU walk-forward (refit every 126 days, 3 seeds) ...")
    p_gru = walk_gru(F, y, pred_start, seeds=(0,) if args.quick else (0, 1, 2), epochs=15 if args.quick else 40)
    p_ens = pd.concat([p_lgbm, p_gru], axis=1).mean(1, skipna=False)
    probs = pd.DataFrame({"lgbm": p_lgbm, "gru": p_gru, "ens": p_ens, "label": y, **{c: F[c] for c in ["dist_sma200", "breadth_200"]}})
    probs.to_csv(out / "daily_timing_probabilities.csv", float_format="%.5g")

    # model evaluation (AUC / Brier / accuracy) on train and test windows
    from sklearn.metrics import roc_auc_score
    ev = []
    for nm in ("lgbm", "gru", "ens"):
        for wn, w in (("train", tr_w), ("test", te_w)):
            d = probs.loc[w[0]:w[1], [nm, "label"]].dropna()
            if len(d) > 30 and d["label"].nunique() == 2:
                ev.append({"model": nm, "window": wn, "n_days": len(d), "AUC": roc_auc_score(d["label"], d[nm]),
                           "brier": float(((d[nm] - d["label"]) ** 2).mean()),
                           "accuracy": float(((d[nm] > 0.5) == d["label"]).mean()),
                           "base_rate": float(d["label"].mean())})
    ev = pd.DataFrame(ev)
    ev.round(4).to_csv(out / "timing_model_eval.csv", index=False)
    print(ev.round(3).to_string(index=False))

    exposure = {"rule_daily": rule_daily(mk), "reentry_daily": reentry_daily(mk), "volman": vol_managed(mk),
                "lgbm": to_exposure(p_lgbm), "gru": to_exposure(p_gru), "ens": to_exposure(p_ens)}
    for k in list(exposure):
        exposure[k] = exposure[k].fillna(exposure["rule_daily"]).fillna(1.0)
    mk.exposure = {k: v.to_numpy() for k, v in exposure.items()}
    pd.DataFrame(exposure).to_csv(out / "daily_exposure.csv")

    # ---------------- 4: trade-quality score (momentum + ML probability)
    ml = scores["final"].reindex(mk.dates)
    scores["mom_vn_ml"] = E._xs_z(scores["mom_vn"]) + 0.5 * E._xs_z(ml).reindex_like(scores["mom_vn"]).fillna(0)

    # ---------------- 5: grid, selected on TRAIN
    prev = json.loads((L.HERE / "phase8_train_test_split" / "split.json").read_text())["params"]
    prev_p = E.Params(**prev)
    grid = []
    for ex in ["monthly_sma200", "reentry_daily", "volman+sma200", "rule_daily", "lgbm", "gru", "ens"]:
        for sc in ["mom_vn", "mom_fip", "mom_vn_ml", "final"]:
            for k in ([8] if args.quick else [8, 12]):
                p = E.Params(name=f"{sc}|EW|k{k}|stop_off|{ex}", score=sc, opt="EW", k=k, stops=False,
                             trend_exit="sma200" if ex in ("monthly_sma200", "volman+sma200") else "none",
                             daily_exposure={"monthly_sma200": "none", "volman+sma200": "volman"}.get(ex, ex))
                grid.append(p)
    rows, G = [], {}
    for p in grid:
        d, _ = E.run(mk, p, scores[p.score], meta if p.score == "final" else None, regime, start)
        G[p.name] = d["ret"]
        a = L.scorecard(d["ret"].loc[tr_w[0]:tr_w[1]], p.name, n_boot=300)
        b = E.metrics(d["ret"].loc[te_w[0]:te_w[1]])
        rows.append({"variant": p.name, "train_goal_score": a["goal_score"], "train_sharpe": a["sharpe"], "train_CAGR": a["CAGR"],
                     "train_max_dd": a["max_drawdown"], "test_sharpe": b["sharpe"], "test_CAGR": b["CAGR"],
                     "test_max_dd": b["max_drawdown"], "trades_rs_cost": round(float(d["cost"].sum()))})
        print(f"   {p.name:<36} train Sharpe {a['sharpe']:>6}  test Sharpe {b['sharpe']:>6}  test CAGR {b['CAGR']:+.3f}")
    gt = pd.DataFrame(rows).sort_values(["train_goal_score", "train_sharpe"], ascending=False)
    gt.to_csv(out / "grid_results.csv", index=False)
    pd.DataFrame(G).to_csv(out / "grid_daily_returns.csv")
    chosen = gt.iloc[0]["variant"]
    bp = next(p for p in grid if p.name == chosen)
    print(f"[P9] chosen on TRAIN: {chosen}")

    # The highlighted CNC-only line now parks idle money in LIQUIDBEES - itself a CNC delivery BUY -
    # so it no longer goes flat while the trend exit has the book out of stocks.
    CNC = "CNC-only (stocks + LIQUIDBEES when out; all delivery buys) - chosen on train"
    lines, logs, tq = {}, {}, {}
    for lab, p in {CNC: bp, "same book, idle cash in bank at 0% (old flat line)": replace(bp, sweep=False),
                   "Phase 8 book (monthly SMA200 exit)": prev_p}.items():
        d, tr = E.run(mk, p, scores[p.score], meta if p.score == "final" else None, regime, start, log_trades=True)
        lines[lab] = d["ret"]; logs[lab] = d
        for wn, w in (("train", tr_w), ("test", te_w)):
            t_ = tr[(tr["date"] >= w[0]) & (tr["date"] <= w[1])] if len(tr) else tr
            tq[f"{lab} | {wn}"] = trade_quality(t_)
        if lab == CNC:
            tr.to_csv(out / "trades_cnc_only.csv", index=False); d.to_csv(out / "daily_cnc_only.csv")
    (out / "trade_quality.json").write_text(json.dumps(tq, indent=2))
    for ex in ["reentry_daily", "volman+sma200", "rule_daily", "lgbm", "gru", "ens"]:
        v = gt[gt["variant"].str.endswith("|" + ex)].iloc[0]["variant"]
        if v != chosen:
            lines[f"best {ex} overlay (train): {v}"] = G[v]
    l6 = pd.read_csv(L.HERE / "phase6_portfolio_backtest" / "lines_daily_returns.csv", index_col=0, parse_dates=True)
    for b in ["NIFTY50 buy&hold", "EW NIFTY100 monthly", "Liquid ETF only"] + [c for c in l6 if c.startswith("Random")]:
        lines[b] = l6[b]
    dates = G[chosen].index
    lines = {k: v.reindex(dates) for k, v in lines.items()}
    pd.DataFrame(lines).to_csv(out / "lines_daily_returns.csv")
    nullD = pd.read_csv(L.HERE / "phase6_portfolio_backtest" / "null_daily_returns.csv", index_col=0, parse_dates=True)

    met = []
    for wn, w in (("TRAIN 70%", tr_w), ("TEST 30%", te_w)):
        win = {k: v.loc[w[0]:w[1]] for k, v in lines.items()}
        nd = nullD.loc[w[0]:w[1]]; cg = (1 + nd).prod().sort_values()
        band = (nd[cg.index[int(0.05 * (len(cg) - 1))]], nd[cg.index[int(0.95 * (len(cg) - 1))]])
        S8.split_panel(win, out / ("train_70pct.png" if wn.startswith("TRAIN") else "test_30pct.png"),
                       f"Strategy02 + daily ML/DL overlay - {wn}: {w[0].date()} .. {w[1].date()} (chosen on TRAIN: {chosen}); "
                       f"shaded = book out of stocks", highlight=CNC, band=band,
                       shade=(logs[CNC]["stock_exposure"] < 0.05).reindex(dates).fillna(False).loc[w[0]:w[1]])
        for k, s in win.items():
            met.append(S8.metrics_row(s, k, wn))
    met = pd.DataFrame(met)
    met.to_csv(out / "metrics.csv", index=False)

    # exposure timeline: explains the flat stretches
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    n = mk.nifty.loc[start:]
    ax[0].plot(n, color="k", lw=1, label="NIFTY50"); ax[0].plot(mk.nifty.rolling(200).mean().loc[start:], color="orange", lw=1, label="SMA200")
    expo = logs[CNC]["stock_exposure"]
    ax[0].fill_between(expo.index, n.min(), n.max(), where=(expo < 0.05).reindex(expo.index).to_numpy(), color="grey", alpha=0.2, label="book in cash")
    ax[0].axvline(te_w[0], color="red"); ax[0].legend(fontsize=8); ax[0].set_title("NIFTY50, SMA200 and the periods the chosen book held no stocks")
    for nm in ("lgbm", "gru", "ens"):
        ax[1].plot(probs[nm].loc[start:], lw=0.9, label=f"P(risk-on) {nm}")
    ax[1].axhline(0.55, color="g", ls="--", lw=0.8); ax[1].axhline(0.40, color="r", ls="--", lw=0.8); ax[1].legend(fontsize=8)
    ax[1].set_title("Daily ML / DL probability that NIFTY beats the liquid ETF over the next 21 days")
    ax[2].plot(expo, color="k", lw=1.2, label="stock exposure (chosen book)")
    ax[2].plot(pd.Series(mk.exposure.get(bp.daily_exposure, np.full(len(mk.dates), np.nan)), index=mk.dates).loc[start:],
               color="tab:blue", lw=0.8, alpha=0.7, label=f"overlay level ({bp.daily_exposure})")
    ax[2].legend(fontsize=8); ax[2].set_title("Exposure")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "exposure_timeline.png", dpi=120); plt.close(fig)

    keys = [CNC, "same book, idle cash in bank at 0% (old flat line)", "Phase 8 book (monthly SMA200 exit)",
            "NIFTY50 buy&hold", "Liquid ETF only"]
    cols = ["window", "line", "final_value_rs", "CAGR", "sharpe", "max_drawdown", "worst_month", "months_ge_-0.5%",
            "P_profitable_12m", "pred_monthly_mean"]
    key = met[met["line"].isin(keys)]
    (out / "report.md").write_text(f"""# Strategy02 - Phase 9: day-by-day ML/DL overlay

The flat stretches in phase 8 were the monthly SMA200 exit holding the book in cash (or the liquid
ETF). Phase 9 checks the exposure every day with a rule, LightGBM, a GRU and their ensemble, and
adds an ML trade-quality score. Chosen on TRAIN only: **{chosen}**.

## Timing models (walk-forward)
{L.md_table(ev, 3)}

## Trade quality (closed positions)
```
{json.dumps(tq, indent=2)}
```

## Key lines (rebased to Rs 1,00,000 per window)
{L.md_table(key[cols], 4)}

## Grid (selected by TRAIN goal score)
{L.md_table(gt, 3)}

![train](train_70pct.png)
![test](test_30pct.png)
![exposure](exposure_timeline.png)
""")
    print("\n" + "=" * 92)
    for _, r in key.iterrows():
        print(f"  {r['window']:<10} {r['line'][:52]:<52} Rs {r['final_value_rs']:>9,}  CAGR {r['CAGR']:+.3f}  Sharpe {r['sharpe']:>6}"
              f"  maxDD {r['max_drawdown']:+.3f}  P(profit 12m) {r['P_profitable_12m']}")
    print("=" * 92)
    print(f"[P9] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
