#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 3 - Features and LABELS for a 21-day swing horizon.

What changed versus strategy/p3 (Vols 2-4 of the strategy volumes)
-----------------------------------------------------------------
  * Horizon 21 trading days (monthly book) instead of 2-5 days: the round-trip cost (~0.36%)
    is now small next to the 21-day dispersion of returns.
  * Training target  y_bin  = triple-barrier label: barriers at +/- 1.0 x sigma_63 x sqrt(21),
    vertical barrier 21 days, entry at the NEXT OPEN, intraday high/low decide the first touch
    (both touched on the same day -> counted as the stop, conservative).
  * Economic target  y_rank = 21-day forward return (next open -> close t+21) minus the
    cross-sectional mean = the residual return a long-only ranking book actually earns.
  * Uniqueness weights (Lopez de Prado ch.4): overlapping labels get weight mean(1/concurrency).
  * Feature set from the volumes: 12-1 momentum, 6/12-month returns, distance to 52-week high,
    distance to the 200-day average, residual momentum, volatility, idiosyncratic vol, MAX,
    skew/kurtosis, Amihud, turnover, ATR; all ranked cross-sectionally each day (0..1).
    Market features (breadth, market vol/trend, dispersion, HMM probabilities) from phase 2.
  * Statistics only on the TRAIN window (<= 2021-08-02 minus the 21-day label horizon):
    daily rank-IC with Newey-West (21 lags), Benjamini-Hochberg FDR at 10%.
  * Pruning: one feature per correlation cluster (|rho| > 0.7), the one with the strongest IC.
  * Leakage audits: shuffled-label IC (max must be < 0.02), truncation test (features computed
    on data cut at date T must equal the full-data values at T), leak ceiling |IC| > 0.15.
Output folder: phase3_features/
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm

import p1_data_costs as L

plt = L.plt
PHASE = "phase3_features"
H = 21                         # label horizon (trading days)
BARRIER_K = 1.0                # barrier width in units of sigma * sqrt(H)
TRAIN_END = pd.Timestamp("2021-08-02")
TEST_END = pd.Timestamp("2025-08-01")
# Size / liquidity features are EXCLUDED from the model by default: the universe is today's NIFTY100,
# so names that were small in 2018 are in it only because they grew -> "small/illiquid wins" is
# survivorship bias, not alpha (Vol 6, survivorship section). --allow-size-features to override.
SIZE_FEATURES = ["turnover_21", "amihud_21"]
MARKET_COLS = ["breadth_above_200", "mkt_vol_21", "mkt_trend_63", "dispersion_21", "p_bull", "p_hostile"]


# ----------------------------------------------------------------------------
def stock_features(P: dict[str, pd.DataFrame], stocks: list[str]) -> dict[str, pd.DataFrame]:
    c, o, h, l = (P[k][stocks] for k in ("adj_close", "adj_open", "adj_high", "adj_low"))
    to = P["turnover"][stocks]
    n = P["adj_close"][L.INDEX]
    r = c.pct_change()
    rm = n.pct_change()
    lr = np.log(c).diff()
    var_m = rm.rolling(252, min_periods=180).var()
    beta = r.rolling(252, min_periods=180).cov(rm).div(var_m, axis=0)
    resid = r - beta.mul(rm, axis=0)
    tr = pd.concat({"a": h - l, "b": (h - c.shift()).abs(), "c": (l - c.shift()).abs()}).groupby(level=1).max()
    tr = tr.reindex(c.index)
    F = {
        "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "ret_252": c / c.shift(252) - 1,
        "ret_126": c / c.shift(126) - 1,
        "ret_63": c / c.shift(63) - 1,
        "ret_21": c / c.shift(21) - 1,
        "dist_52w_high": c / c.rolling(252, min_periods=200).max() - 1,
        "dist_sma200": c / c.rolling(200, min_periods=150).mean() - 1,
        "dist_sma50": c / c.rolling(50, min_periods=40).mean() - 1,
        "resid_mom_126": resid.shift(21).rolling(105, min_periods=80).sum() / resid.rolling(252, min_periods=180).std(),
        "vol_63": lr.rolling(63, min_periods=50).std() * np.sqrt(L.TRADING_DAYS),
        "idio_vol_63": resid.rolling(63, min_periods=50).std() * np.sqrt(L.TRADING_DAYS),
        "beta_252": beta,
        "max_ret_21": r.rolling(21).max(),
        "skew_63": r.rolling(63, min_periods=50).skew(),
        "kurt_63": r.rolling(63, min_periods=50).kurt(),
        "amihud_21": (r.abs() / (to / 1e7)).replace([np.inf, -np.inf], np.nan).rolling(21, min_periods=15).mean(),
        "turnover_21": np.log(to.rolling(21, min_periods=15).mean()),
        "turnover_trend": np.log(to.rolling(21, min_periods=15).mean() / to.rolling(126, min_periods=90).mean()),
        "atr14_pct": tr.rolling(14).mean() / c,
    }
    return F


def labels(P: dict[str, pd.DataFrame], stocks: list[str], vol63: pd.DataFrame):
    """Triple-barrier labels + economic target. Decision at close t, entry at open t+1."""
    c, o, h, l = (P[k][stocks].to_numpy() for k in ("adj_close", "adj_open", "adj_high", "adj_low"))
    T, N = c.shape
    entry = np.full((T, N), np.nan); entry[:-1] = o[1:]
    exit_ = np.full((T, N), np.nan); exit_[:-H] = c[H:]
    fwd = exit_ / entry - 1
    sig = (vol63.to_numpy() / np.sqrt(L.TRADING_DAYS)) * np.sqrt(H) * BARRIER_K
    sig = np.clip(sig, 0.03, 0.40)
    y_bin = np.full((T, N), np.nan)
    t1 = np.full((T, N), np.nan)
    done = np.zeros((T, N), dtype=bool)
    for k in range(1, H + 1):                       # day t+k of the holding period
        hi = np.full((T, N), np.nan); lo = np.full((T, N), np.nan)
        hi[:-k] = h[k:]; lo[:-k] = l[k:]
        up = (hi / entry - 1) >= sig
        dn = (lo / entry - 1) <= -sig
        new_dn = dn & ~done
        new_up = up & ~dn & ~done
        y_bin[new_dn] = 0; t1[new_dn] = k
        y_bin[new_up] = 1; t1[new_up] = k
        done |= new_dn | new_up
    vert = ~done & np.isfinite(fwd)
    y_bin[vert] = (fwd[vert] > 0).astype(float); t1[vert] = H
    y_bin[~np.isfinite(fwd)] = np.nan
    y_rank = fwd - np.nanmean(fwd, axis=1, keepdims=True)
    # uniqueness: label (t, i) is "alive" on days t+1 .. t+t1
    w = np.full((T, N), np.nan)
    for j in range(N):
        span = t1[:, j]
        conc = np.zeros(T + H + 2)
        idx = np.where(np.isfinite(span))[0]
        for t in idx:
            conc[t + 1: t + 1 + int(span[t])] += 1
        inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
        cs = np.r_[0, np.cumsum(inv)]
        for t in idx:
            a, b = t + 1, t + 1 + int(span[t])
            w[t, j] = (cs[b] - cs[a]) / (b - a)
    idx = P["adj_close"].index
    mk = lambda a: pd.DataFrame(a, index=idx, columns=stocks)
    return {"fwd_ret": mk(fwd), "y_rank": mk(y_rank), "y_bin": mk(y_bin), "t1": mk(t1), "uniq_w": mk(w)}


def xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def rank_ic(feat: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    a = feat.rank(axis=1); b = target.where(feat.notna()).rank(axis=1)
    a = a.where(b.notna())
    a = a.sub(a.mean(1), axis=0); b = b.sub(b.mean(1), axis=0)
    ic = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return ic[(feat.notna() & target.notna()).sum(1) >= 20]


def nw_t(x: pd.Series, lags: int = H) -> float:
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n))


def bh_fdr(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    n = len(p); order = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    passed = p[order] <= thr
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool); out[order[:k]] = True
    return out


def top_k_buffer(score: pd.DataFrame, k: int, buffer: int, mask: pd.Series) -> pd.DataFrame:
    """Equal-weight top-k book on rebalance days; a held name stays while its rank <= buffer
    (cuts turnover - Vol 6 'costs on the delta')."""
    W = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    held: list[str] = []
    for d in score.index[mask.reindex(score.index).fillna(False).to_numpy()]:
        s = score.loc[d].dropna().sort_values(ascending=False)
        if len(s) < k:
            continue
        rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
        keep = [n for n in held if n in rank and rank[n] <= buffer]
        new = [n for n in s.index if n not in keep][: max(0, k - len(keep))]
        held = keep + new
        W.loc[d, held] = 1.0 / len(held)
    return W.where(mask.reindex(score.index).fillna(False), np.nan).ffill().fillna(0.0)


def main():
    ap = L.base_args("Strategy02 Phase 3: 21-day features, triple-barrier labels, audits")
    ap.add_argument("--allow-size-features", action="store_true")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    P = L.load_daily_panel(data)
    close, opn, high, low = P["adj_close"], P["adj_open"], P["adj_high"], P["adj_low"]
    stocks = [c for c in close.columns if c != L.INDEX]
    liq = L.liquid_returns(data, close.index)
    reg = pd.read_csv(L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv", index_col=0, parse_dates=True)

    print("[P3] features ...")
    F = stock_features(P, stocks)
    print("[P3] triple-barrier labels + uniqueness weights ...")
    Y = labels(P, stocks, F["vol_63"])
    R = {k: xs_rank(v) for k, v in F.items()}

    # ---------------- IC on the TRAIN window only (labels must end before TRAIN_END)
    train_dates = close.index[(close.index <= TRAIN_END - pd.Timedelta(days=35)) & (close.index >= "2017-10-01")]
    test_dates = close.index[(close.index > TRAIN_END) & (close.index <= TEST_END)]
    rows = []
    for k, v in R.items():
        ic_tr = rank_ic(v.loc[train_dates], Y["y_rank"].loc[train_dates])
        ic_bin = rank_ic(v.loc[train_dates], Y["y_bin"].loc[train_dates])
        ic_te = rank_ic(v.loc[test_dates], Y["y_rank"].loc[test_dates])
        t = nw_t(ic_tr)
        rows.append({"feature": k, "ic_train_yrank": ic_tr.mean(), "nw_t_train": t,
                     "p_value": 2 * (1 - norm.cdf(abs(t))) if np.isfinite(t) else 1.0,
                     "ic_train_ybin": ic_bin.mean(), "ic_test_yrank": ic_te.mean(),
                     "sign_stable": bool(np.sign(ic_tr.mean()) == np.sign(ic_te.mean()))})
    ic = pd.DataFrame(rows)
    ic["fdr_10pct"] = bh_fdr(ic["p_value"].to_numpy())
    ic["leak_flag_gt_0.15"] = ic["ic_train_yrank"].abs() > 0.15

    # ---------------- pruning: one feature per |rho| > 0.7 cluster (train window, month-ends)
    me_tr = [d for d in train_dates if L.month_end_mask(close.index).loc[d]]
    names_f = list(R)
    C = np.zeros((len(names_f), len(names_f)))
    for d in me_tr:
        X = pd.DataFrame({k: R[k].loc[d] for k in names_f}).dropna()
        if len(X) > 30:
            C += X.corr(method="spearman").fillna(0).to_numpy()
    C /= max(len(me_tr), 1)
    np.fill_diagonal(C, 1.0)
    pd.DataFrame(C, index=names_f, columns=names_f).round(3).to_csv(out / "feature_correlation_train.csv")
    Z = linkage(squareform(1 - np.abs(C), checks=False), method="average")
    cl = fcluster(Z, t=0.3, criterion="distance")
    ic["cluster"] = cl
    keep = []
    eligible = ic if args.allow_size_features else ic[~ic["feature"].isin(SIZE_FEATURES)]
    for g, sub in eligible.groupby("cluster"):
        keep.append(sub.loc[sub["nw_t_train"].abs().idxmax(), "feature"])
    ic["kept_after_pruning"] = ic["feature"].isin(keep)
    pruned = ic[ic["kept_after_pruning"]].sort_values("nw_t_train", key=abs, ascending=False)["feature"].tolist()
    selected = ic[ic["kept_after_pruning"] & ic["fdr_10pct"]]["feature"].tolist()
    if len(selected) < 6:          # FDR is strict with ~4 years of training data: keep the 6 strongest pruned
        selected = pruned[:6]
    ic.round(5).to_csv(out / "ic_table.csv", index=False)

    # ---------------- leakage audits
    rng = np.random.default_rng(0)
    shuf = Y["y_rank"].loc[train_dates].apply(lambda r: pd.Series(rng.permutation(r.to_numpy()), index=r.index), axis=1)
    shuffled = {k: float(rank_ic(R[k].loc[train_dates], shuf).mean()) for k in names_f}
    trunc_ok = True
    cut_dates = [close.index[800], close.index[1500], close.index[2200]]
    for cd in cut_dates:
        Pc = {k: v.loc[:cd] for k, v in P.items()}
        Fc = stock_features(Pc, stocks)
        for k in names_f:
            a, b = Fc[k].loc[cd], F[k].loc[cd]
            if not np.allclose(a.fillna(-9).to_numpy(), b.fillna(-9).to_numpy(), rtol=1e-8, atol=1e-10):
                trunc_ok = False
    audits = {"shuffled_label_max_abs_ic": round(max(abs(v) for v in shuffled.values()), 4),
              "shuffled_pass_lt_0.02": bool(max(abs(v) for v in shuffled.values()) < 0.02),
              "truncation_test_pass": bool(trunc_ok), "truncation_dates": [str(d.date()) for d in cut_dates],
              "leak_ceiling_flags": ic.loc[ic["leak_flag_gt_0.15"], "feature"].tolist(),
              "mean_uniqueness_weight": round(float(np.nanmean(Y["uniq_w"].to_numpy())), 4),
              "label_balance_y_bin": round(float(np.nanmean(Y["y_bin"].to_numpy())), 3),
              "median_label_length_days": float(np.nanmedian(Y["t1"].to_numpy()))}
    (out / "leakage_audit.json").write_text(json.dumps(audits, indent=2))

    # ---------------- write long feature table
    print("[P3] writing daily_features.csv ...")
    long = pd.concat({**{k: R[k] for k in names_f}, "vol_63_raw": F["vol_63"], "atr14_raw": F["atr14_pct"],
                      "amihud_raw": F["amihud_21"], **Y}, axis=1).stack(future_stack=True)
    long.index.names = ["date", "name"]
    long = long.reset_index()
    long = long[long[names_f].notna().sum(1) >= len(names_f) - 3]
    long = long[long["date"] >= "2017-10-01"]
    long = long.merge(reg[MARKET_COLS], left_on="date", right_index=True, how="left")
    long.to_csv(out / "daily_features.csv", index=False, float_format="%.6g")
    (out / "feature_lists.json").write_text(json.dumps({
        "all_stock_features": names_f, "selected_stock_features": selected, "pruned_stock_features": pruned,
        "excluded_size_features": [] if args.allow_size_features else SIZE_FEATURES,
        "market_features": MARKET_COLS, "model_features": pruned + MARKET_COLS, "horizon": H, "barrier_k": BARRIER_K,
        "train_end": str(TRAIN_END.date()), "test_end": str(TEST_END.date())}, indent=2))

    # ---------------- simple Level-1 candidates (monthly, top 10, buffer 20)
    me = L.month_end_mask(close.index)
    hs = L.half_spread_bps(high[stocks], low[stocks], P["turnover"][stocks])
    sgn = ic.set_index("feature")["ic_train_yrank"].apply(np.sign)
    comp = sum(R[k] * sgn[k] for k in selected) / len(selected)
    books = {
        "Mom 12-1 top10 monthly (CNC)": top_k_buffer(R["mom_12_1"], 10, 20, me),
        "Composite rank (selected features) top10 monthly (CNC)": top_k_buffer(comp, 10, 20, me),
    }
    start = pd.Timestamp("2017-11-01")
    sims = {k: L.simulate_weights(w, opn, close, liq, pledge=False, half_spread=hs, rebal_mask=me)["ret"].loc[start:]
            for k, w in books.items()}
    sims["EW NIFTY100 monthly"] = L.simulate_weights(close[stocks].notna().astype(float).div(close[stocks].notna().sum(1), axis=0),
                                                     opn, close, liq, pledge=False, half_spread=hs, rebal_mask=me)["ret"].loc[start:]
    sims["NIFTY50 buy&hold"] = close[L.INDEX].pct_change().fillna(0).loc[start:]
    cards = [L.scorecard(v, k) for k, v in sims.items()]
    best = L.save_scorecards(out, cards, PHASE, notes="Level-1 ranking books on the 21-day horizon.", series=sims)
    L.multi_panel(sims, out / "level1_books_all_in_one.png", "Phase 3 - Level-1 monthly ranking books (CNC only)",
                  highlight="Composite rank (selected features) top10 monthly (CNC)")

    fig, ax = plt.subplots(figsize=(11, 6))
    d = ic.sort_values("nw_t_train")
    ax.barh(d["feature"], d["nw_t_train"], color=np.where(d["feature"].isin(selected), "#1f77b4", "#bbbbbb"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Rank-IC Newey-West t (train window), blue = selected after FDR + pruning")
    fig.tight_layout(); fig.savefig(out / "ic_tstats.png", dpi=120); plt.close(fig)

    tab = pd.DataFrame(cards)[L.SC_COLS]
    (out / "report.md").write_text(f"""# Strategy02 - Phase 3: 21-day features and labels

Label: triple barrier (+/- {BARRIER_K} sigma sqrt(21), 21-day vertical), entry next open.
Economic target y_rank = 21-day forward return minus the cross-sectional mean.
Train window for every statistic: up to {TRAIN_END.date()} (minus the label horizon).

## Rank IC (train) and out-of-sample sign check (test {TRAIN_END.date()} .. {TEST_END.date()})
{L.md_table(ic[['feature','ic_train_yrank','nw_t_train','p_value','fdr_10pct','cluster','kept_after_pruning','ic_test_yrank','sign_stable']], 4)}

Selected stock features: **{', '.join(selected)}**  + market features {', '.join(MARKET_COLS)}

## Leakage audits
```
{json.dumps(audits, indent=2)}
```

## Level-1 books (monthly, top 10, keep while rank <= 20)
{L.md_table(tab)}

![ic](ic_tstats.png)
![books](level1_books_all_in_one.png)
""")
    L.print_scorecard(best, PHASE)
    print(f"[P3] selected: {selected}")
    print(f"[P3] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
