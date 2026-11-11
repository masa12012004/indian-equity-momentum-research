#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3 - Feature engineering, leakage tests, information-coefficient analysis.

Two prediction problems (matching the two sleeves of the mandate):

  A. CNC sleeve (daily, cross-sectional): decide at close t, buy at open t+1, hold 2 days.
     target  fwd2_ex = open[t+3]/open[t+1]-1  minus the cross-sectional mean
  B. MIS sleeve (intraday, 5-min bars): decide at 09:30 / 11:00 / 13:00, exit at the
     15:15 bar close (before Flattrade's MIS auto square-off).
     targets ret_to_exit (+ stop-loss path outcomes for 0.5/1.0/1.5% stops, long & short)

Leakage protection
------------------
  1. Every daily feature is a function of rows <= t (verified by a truncation test).
  2. Every intraday feature only reads bars that have CLOSED before the decision time
     (verified by a future-perturbation test: scrambling later bars must not change features).
  3. Targets start strictly after the decision point (open t+1 / next bar).
  4. Feature selection & signs are learned on the first 60% of dates only (train);
     the candidate strategies below are scored on the last 40% (test).

Statistics: daily Spearman rank IC, Newey-West t-stats (overlapping targets),
Benjamini-Hochberg false-discovery control across all features.
Output folder: phase3_features/
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

import p1_data_costs as L

plt = L.plt
PHASE = "phase3_features"
SLOTS = {"0930": 3, "1100": 21, "1300": 45}      # number of 5-min bars closed at decision
EXIT_BAR = 71                                     # bar opening 15:10 -> closes 15:15
NBARS = 75
STOPS = [0.005, 0.010, 0.015]
TRAIN_FRAC = 0.60


# =============================================================================
# Daily cross-sectional features
# =============================================================================
def fracdiff_weights(d=0.4, n=100):
    w = [1.0]
    for k in range(1, n):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def daily_features(P: dict, regimes: pd.DataFrame | None, industry: dict) -> pd.DataFrame:
    C, O, H, Lo, V, TO = (P[k] for k in ("adj_close", "adj_open", "adj_high", "adj_low", "adj_volume", "turnover"))
    stocks = [c for c in C.columns if c != L.INDEX]
    C, O, H, Lo, V, TO = (x[stocks] for x in (C, O, H, Lo, V, TO))
    m = P["adj_close"][L.INDEX].pct_change()
    r = C.pct_change()
    F = {}
    for k in (1, 5, 10, 21, 63):
        F[f"ret_{k}"] = C.pct_change(k)
    F["mom_12_1"] = C.shift(21) / C.shift(252) - 1
    F["vol_20"] = r.rolling(20).std()
    F["park_vol_20"] = np.sqrt((np.log(H / Lo) ** 2 / (4 * np.log(2))).rolling(20).mean())
    F["down_vol_20"] = r.clip(upper=0).pow(2).rolling(20).mean().pow(0.5)
    cov = r.rolling(63).cov(m)
    beta = cov.div(m.rolling(63).var(), axis=0)
    F["beta_63"] = beta
    resid = r - beta.mul(m, axis=0)
    F["resid_ret_5"] = resid.rolling(5).sum()
    F["resid_ret_21"] = resid.rolling(21).sum()
    F["idio_vol_63"] = resid.rolling(63).std()
    F["max_ret_21"] = r.rolling(21).max()
    F["skew_63"] = r.rolling(63).skew()
    lv = np.log(V.replace(0, np.nan))
    F["volume_z_20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    F["amihud_21"] = (r.abs() / TO.replace(0, np.nan)).rolling(21).mean() * 1e9
    F["log_turnover_63"] = np.log(TO.rolling(63).median().replace(0, np.nan))
    F["dist_52w_high"] = C / C.rolling(252, min_periods=120).max() - 1
    F["clv"] = ((C - Lo) - (H - C)) / (H - Lo).replace(0, np.nan)
    F["gap"] = O / C.shift(1) - 1
    up = r.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    F["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    w = fracdiff_weights()
    lc = np.log(C)
    fd = lc.apply(lambda s: pd.Series(np.convolve(s.ffill().to_numpy(), w, mode="full")[: len(s)], index=s.index)
                  .where(s.notna().rolling(len(w)).sum() == len(w)))
    F["fracdiff_04"] = (fd - fd.rolling(252, min_periods=60).mean()) / fd.rolling(252, min_periods=60).std()
    pu = (r > 0).rolling(21).mean()
    F["sign_entropy_21"] = -(pu * np.log(pu.clip(1e-9)) + (1 - pu) * np.log((1 - pu).clip(1e-9)))
    ind = pd.Series({s: industry.get(s, "NA") for s in stocks})
    r21 = F["ret_21"]
    ind_mean = r21.T.groupby(ind).transform("mean").T
    F["ind_rel_ret_21"] = r21 - ind_mean

    # cross-sectional rank-normalisation to [-0.5, 0.5]
    rows = []
    for k, df in F.items():
        rk = df.rank(axis=1, pct=True) - 0.5
        rows.append(rk.stack(future_stack=True).rename(k))
    X = pd.concat(rows, axis=1)
    X.index.names = ["date", "name"]

    # targets
    fwd2 = O.shift(-3) / O.shift(-1) - 1
    fwd5 = O.shift(-6) / O.shift(-1) - 1
    T = pd.concat([fwd2.stack(future_stack=True).rename("fwd2"), fwd5.stack(future_stack=True).rename("fwd5"),
                   F["vol_20"].stack(future_stack=True).rename("vol_20_raw"),
                   C.stack(future_stack=True).rename("adj_close")], axis=1)
    T.index.names = ["date", "name"]
    X = X.join(T)
    X["fwd2_ex"] = X["fwd2"] - X.groupby(level="date")["fwd2"].transform("mean")
    X["fwd5_ex"] = X["fwd5"] - X.groupby(level="date")["fwd5"].transform("mean")
    if regimes is not None:
        mk = regimes[["hmm_p_bull", "trend_t", "garch_vol_fcst", "absorption_ratio", "risk_score"]].copy()
        mk.columns = ["mkt_" + c for c in mk.columns]
        X = X.join(mk, on="date")
    return X.dropna(subset=["ret_21", "vol_20"])


# =============================================================================
# Intraday decision-point features
# =============================================================================
def to_grid(df: pd.DataFrame):
    """5-min regular bars -> arrays (days x 75) for o,h,l,c,v (adjusted prices)."""
    t = df["datetime"]
    k = ((t.dt.hour * 60 + t.dt.minute) - 555) // 5
    d = t.dt.normalize()
    days = pd.DatetimeIndex(d.unique())
    di = pd.Series(np.arange(len(days)), index=days)[d].to_numpy()
    shape = (len(days), NBARS)
    arr = {}
    for col, key in [("adj_open", "o"), ("adj_high", "h"), ("adj_low", "l"), ("adj_close", "c"), ("adj_volume", "v")]:
        a = np.full(shape, np.nan)
        a[di, k.to_numpy()] = df[col].to_numpy()
        arr[key] = a
    c = pd.DataFrame(arr["c"]).ffill(axis=1).to_numpy()
    arr["c"] = c
    for key in ("o", "h", "l"):
        arr[key] = np.where(np.isnan(arr[key]), c, arr[key])
    arr["v"] = np.nan_to_num(arr["v"])
    return days, arr


def intraday_features_one(days, a, prev_close: np.ndarray, mkt: dict | None):
    """Features at each slot using bars [0, k) only. mkt = dict slot -> market ret_since_open."""
    o, h, l, c, v = a["o"], a["h"], a["l"], a["c"], a["v"]
    day_open = o[:, 0]
    tp = (h + l + c) / 3
    cum_v = np.cumsum(v, axis=1)
    cum_pv = np.cumsum(tp * v, axis=1)
    lr = np.diff(np.log(c), axis=1, prepend=np.log(o[:, :1]))
    out = []
    for slot, k in SLOTS.items():
        px = c[:, k - 1]
        feat = {
            "slot": slot,
            "gap_on": day_open / prev_close - 1,
            "ret_since_open": px / day_open - 1,
            "ret_last_15m": px / c[:, max(k - 4, 0)] - 1 if k > 3 else px / day_open - 1,
            "ret_last_30m": px / c[:, max(k - 7, 0)] - 1 if k > 6 else px / day_open - 1,
            "orb_pos": (px - l[:, :3].min(1)) / np.maximum(h[:, :3].max(1) - l[:, :3].min(1), 1e-9),
            "range_sofar_pct": (h[:, :k].max(1) - l[:, :k].min(1)) / day_open,
            "vwap_dist": px / np.where(cum_v[:, k - 1] > 0, cum_pv[:, k - 1] / np.maximum(cum_v[:, k - 1], 1), px) - 1,
            "rv_sofar": np.sqrt((lr[:, :k] ** 2).sum(1)),
            "cum_vol": cum_v[:, k - 1],
            "entry_px": px,
        }
        if mkt is not None:
            feat["mkt_ret_since_open"] = mkt[slot]
            feat["rel_strength"] = feat["ret_since_open"] - mkt[slot]
        # targets (strictly after the decision bar)
        ex = c[:, EXIT_BAR]
        feat["ret_to_exit"] = ex / px - 1
        lows = l[:, k:EXIT_BAR + 1].min(1) / px - 1
        highs = h[:, k:EXIT_BAR + 1].max(1) / px - 1
        feat["mae_long"], feat["mfe_long"] = lows, highs
        for s in STOPS:
            tag = f"{int(s*1000):03d}"
            feat[f"long_sl{tag}"] = np.where(lows <= -s, -s, ex / px - 1)
            feat[f"short_sl{tag}"] = np.where(highs >= s, -s, -(ex / px - 1))
        df = pd.DataFrame(feat)
        df["date"] = days
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    return res


def build_intraday(data, names_, daily_X: pd.DataFrame, quick=False):
    # market (NIFTY50) since-open returns per slot
    nf = L.load_intraday(data, L.INDEX, "5min")
    ndays, na = to_grid(nf)
    mkt = {s: pd.Series(na["c"][:, k - 1] / na["o"][:, 0] - 1, index=ndays) for s, k in SLOTS.items()}
    frames = []
    for i, n in enumerate(names_):
        df = L.load_intraday(data, n, "5min")
        if df.empty:
            continue
        days, a = to_grid(df)
        dcl = L._read(L.find(data, "1D", n))
        dcl = dcl.set_index(dcl["datetime"].dt.normalize())["adj_close"]
        prev_close = dcl.shift(1).reindex(days).to_numpy()
        mk = {s: mkt[s].reindex(days).to_numpy() for s in SLOTS}
        f = intraday_features_one(days, a, prev_close, mk)
        # same-time-of-day volume ratio vs trailing 20 days (past days only)
        f = f.sort_values(["slot", "date"])
        f["vol_ratio_tod"] = f["cum_vol"] / f.groupby("slot")["cum_vol"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean())
        f["rv_ratio"] = f["rv_sofar"] / f.groupby("slot")["rv_sofar"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean())
        f["name"] = n
        frames.append(f)
        if (i + 1) % 20 == 0:
            print(f"   intraday features {i+1}/{len(names_)}")
    I = pd.concat(frames, ignore_index=True)
    # previous-day daily features (known before the open)
    prev = daily_X[["ret_1", "ret_5", "ret_21", "vol_20", "clv", "volume_z_20", "resid_ret_5"]].copy()
    prev.columns = ["prevday_" + c for c in prev.columns]
    prev = prev.reset_index()
    prev["date"] = prev.groupby("name")["date"].shift(-1)          # features of t-1 attached to day t
    I = I.merge(prev.dropna(subset=["date"]), on=["date", "name"], how="left")
    I["dow"] = I["date"].dt.dayofweek
    I["ret_to_exit_ex"] = I["ret_to_exit"] - I.groupby(["date", "slot"])["ret_to_exit"].transform("mean")
    return I


# =============================================================================
# Leakage tests
# =============================================================================
def leakage_tests(data, P, regimes, industry, sample="RELIANCE") -> list[dict]:
    res = []
    # (1) daily truncation test: features on date t must not change when data after t is removed
    cut = P["adj_close"].index[-300]
    full = daily_features(P, regimes, industry)
    trunc_P = {k: v.loc[:cut] for k, v in P.items()}
    tr = daily_features(trunc_P, regimes.loc[:cut] if regimes is not None else None, industry)
    feats = [c for c in tr.columns if not c.startswith(("fwd", "mkt_")) and c not in ("vol_20_raw", "adj_close")]
    a = full.loc[cut][feats]
    b = tr.loc[cut][feats]
    diff = float((a - b.reindex(a.index)).abs().max().max())
    res.append({"test": "daily truncation (features at t unchanged when future removed)",
                "max_abs_diff": diff, "pass": diff < 1e-9})
    # (2) intraday future-perturbation test
    df = L.load_intraday(data, sample, "5min")
    if len(df):
        days, a = to_grid(df)
        prev = np.full(len(days), np.nan)
        base = intraday_features_one(days, a, prev, None)
        rng = np.random.default_rng(0)
        ok = True
        worst = 0.0
        for slot, k in SLOTS.items():
            b2 = {key: arr.copy() for key, arr in a.items()}
            for key in ("o", "h", "l", "c"):
                b2[key][:, k:] *= rng.uniform(0.9, 1.1, size=b2[key][:, k:].shape)
            b2["v"][:, k:] *= rng.uniform(0.5, 2.0, size=b2["v"][:, k:].shape)
            pert = intraday_features_one(days, b2, prev, None)
            cols = ["gap_on", "ret_since_open", "ret_last_15m", "ret_last_30m", "orb_pos",
                    "range_sofar_pct", "vwap_dist", "rv_sofar", "cum_vol", "entry_px"]
            m1 = base[base.slot == slot][cols].to_numpy()
            m2 = pert[pert.slot == slot][cols].to_numpy()
            d = np.nanmax(np.abs(m1 - m2))
            worst = max(worst, float(d))
            ok &= d < 1e-12
        res.append({"test": f"intraday future-perturbation ({sample}, all slots)", "max_abs_diff": worst, "pass": bool(ok)})
    # (3) target alignment: target must begin after decision -> corr(feature_t, target_t) should not be ~1
    return res


# =============================================================================
# IC analysis
# =============================================================================
def ic_table(X: pd.DataFrame, feats, target, group_cols, nw_lag=2):
    rows = []
    g = X.groupby(group_cols)
    for f in feats:
        ic = g.apply(lambda d: d[f].corr(d[target], method="spearman") if d[f].notna().sum() > 10 else np.nan,
                     include_groups=False).dropna()
        if len(ic) < 30:
            continue
        x = ic.to_numpy() - ic.mean()
        T = len(x)
        gamma0 = (x @ x) / T
        s = gamma0 + 2 * sum((1 - j / (nw_lag + 1)) * (x[j:] @ x[:-j]) / T for j in range(1, nw_lag + 1))
        t = ic.mean() / np.sqrt(max(s, 1e-12) / T)
        rows.append({"feature": f, "mean_ic": ic.mean(), "ic_std": ic.std(), "ic_ir": ic.mean() / ic.std(),
                     "t_nw": t, "p_value": 2 * (1 - stats.norm.cdf(abs(t))), "hit_rate": (ic > 0).mean(), "n": T})
    df = pd.DataFrame(rows)
    if len(df):
        # Benjamini-Hochberg
        p = df["p_value"].to_numpy()
        order = np.argsort(p)
        q = np.empty_like(p)
        m = len(p)
        prev = 1.0
        for rank, i in enumerate(order[::-1]):
            k = m - rank
            prev = min(prev, p[i] * m / k)
            q[i] = prev
        df["q_bh"] = q
        df["significant_fdr10"] = df["q_bh"] < 0.10
    return df.sort_values("t_nw", key=np.abs, ascending=False)


# =============================================================================
# Simple candidate strategies (composite of FDR-significant features)
# =============================================================================
def composite(X, ic, feats):
    sel = ic[ic["significant_fdr10"]] if "significant_fdr10" in ic and ic["significant_fdr10"].any() else ic.head(5)
    w = np.sign(sel.set_index("feature")["mean_ic"])
    cols = [c for c in w.index if c in feats]
    return (X[cols].fillna(0) * w[cols]).sum(axis=1) / max(len(cols), 1), list(cols)


def cnc_weights_from_scores(score: pd.Series, k=4, hold=2, buffer=4) -> pd.DataFrame:
    s = score.unstack("name")
    dates = s.index
    held: list[str] = []
    age = {}
    W = []
    for d in dates:
        row = s.loc[d].dropna()
        if row.empty:
            W.append(pd.Series(dtype=float, name=d)); continue
        rk = row.rank(ascending=False)
        keep = [n for n in held if (age.get(n, 0) < hold) or (n in rk.index and rk[n] <= k + buffer)]
        for n in rk.sort_values().index:
            if len(keep) >= k:
                break
            if n not in keep:
                keep.append(n)
        for n in keep:
            age[n] = age.get(n, 0) + 1 if n in held else 1
        held = keep
        W.append(pd.Series(1.0 / k, index=keep, name=d))
    return pd.DataFrame(W).reindex(columns=s.columns).fillna(0.0)


def mis_daily_pnl(I: pd.DataFrame, score: pd.Series, gross_mult=2.0, n_side=2, stop="010", slot="0930"):
    """Market-neutral MIS book at one slot: long top-n, short bottom-n; daily return on capital."""
    d = I.assign(score=score)
    d = d[d["slot"] == slot].dropna(subset=["score", f"long_sl{stop}"])
    rt_cost = L.mis_rate_roundtrip() + 2 * 2.2e-4
    per = gross_mult / (2 * n_side)
    out = {}
    for day, g in d.groupby("date"):
        if len(g) < 2 * n_side + 2:
            continue
        g = g.sort_values("score")
        longs, shorts = g.tail(n_side), g.head(n_side)
        pnl = per * ((longs[f"long_sl{stop}"] - rt_cost).sum() + (shorts[f"short_sl{stop}"] - rt_cost).sum())
        out[day] = pnl
    return pd.Series(out).sort_index()


# =============================================================================
def main():
    ap = L.base_args("Phase 3: features, leakage tests, IC analysis")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    P = L.load_daily_panel(data)
    reg_path = L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv"
    regimes = pd.read_csv(reg_path, index_col=0, parse_dates=True) if reg_path.exists() else None
    inst = data / "meta" / "instruments.csv"
    industry = {}
    if inst.exists():
        it = pd.read_csv(inst)
        industry = {str(n).replace("&", "_"): i for n, i in zip(it["name"], it.get("industry", pd.Series(["NA"] * len(it))))}

    print("[P3] daily features ...")
    X = daily_features(P, regimes, industry)
    print("[P3] leakage tests ...")
    lk = leakage_tests(data, P, regimes, industry)
    pd.DataFrame(lk).to_csv(out / "leakage_tests.csv", index=False)
    for r in lk:
        print(f"   {'PASS' if r['pass'] else 'FAIL'}  {r['test']}  (max diff {r['max_abs_diff']:.2e})")

    stock_feats = [c for c in X.columns if not c.startswith(("fwd", "mkt_")) and c not in ("vol_20_raw", "adj_close")]
    dates = X.index.get_level_values("date").unique().sort_values()
    split = dates[int(len(dates) * TRAIN_FRAC)]
    Xtr = X[X.index.get_level_values("date") < split]
    print(f"[P3] daily IC (train < {split.date()}) ...")
    ic_d = ic_table(Xtr.dropna(subset=["fwd2_ex"]).reset_index(), stock_feats, "fwd2_ex", "date")
    ic_d.to_csv(out / "ic_daily_train.csv", index=False)
    ic_d_test = ic_table(X[X.index.get_level_values("date") >= split].dropna(subset=["fwd2_ex"]).reset_index(),
                         stock_feats, "fwd2_ex", "date")
    ic_d_test.to_csv(out / "ic_daily_test.csv", index=False)

    names_ = L.names(data)
    if args.quick:
        names_ = names_[:30]
    print(f"[P3] intraday decision-point features for {len(names_)} stocks ...")
    I = build_intraday(data, names_, X, quick=args.quick)
    intra_feats = ["gap_on", "ret_since_open", "ret_last_15m", "ret_last_30m", "orb_pos", "range_sofar_pct",
                   "vwap_dist", "rv_sofar", "rel_strength", "vol_ratio_tod", "rv_ratio",
                   "prevday_ret_1", "prevday_ret_5", "prevday_ret_21", "prevday_vol_20", "prevday_clv",
                   "prevday_volume_z_20", "prevday_resid_ret_5"]
    idates = I["date"].drop_duplicates().sort_values()
    isplit = idates.iloc[int(len(idates) * TRAIN_FRAC)]
    Itr = I[I["date"] < isplit]
    print(f"[P3] intraday IC (train < {isplit.date()}) ...")
    ic_i = []
    for slot in SLOTS:
        t = ic_table(Itr[Itr.slot == slot], intra_feats, "ret_to_exit_ex", "date", nw_lag=0)
        t["slot"] = slot
        ic_i.append(t)
    ic_i = pd.concat(ic_i)
    ic_i.to_csv(out / "ic_intraday_train.csv", index=False)

    # ---------------- save feature tables (csv) -----------------------------
    print("[P3] writing feature tables ...")
    X.reset_index().to_csv(out / "daily_features.csv", index=False, float_format="%.6g")
    I.to_csv(out / "intraday_features.csv", index=False, float_format="%.6g")
    (out / "feature_lists.json").write_text(pd.Series({
        "daily_features": stock_feats, "daily_market_features": [c for c in X.columns if c.startswith("mkt_")],
        "intraday_features": intra_feats, "daily_split_date": str(split.date()),
        "intraday_split_date": str(isplit.date())}).to_json(indent=2))

    # ---------------- candidate strategies on TEST period ------------------
    score_d, used_d = composite(X, ic_d, stock_feats)
    W = cnc_weights_from_scores(score_d, k=4, hold=2)
    W = W.loc[split:]
    hs = L.half_spread_bps(P["adj_high"], P["adj_low"], P["turnover"])
    liq = L.liquid_returns(data, P["adj_close"].index)
    cnc = L.simulate_weights(W, P["adj_open"], P["adj_close"], liq, pledge=True, half_spread=hs)["ret"].loc[split:]
    if regimes is not None:
        tv = (0.10 / regimes["garch_vol_fcst"]).clip(upper=1).reindex(W.index).ffill().fillna(0)
        cnc_vt = L.simulate_weights(W.mul(tv, axis=0), P["adj_open"], P["adj_close"], liq, pledge=True, half_spread=hs)["ret"].loc[split:]
    best_slot = ic_i.groupby("slot")["t_nw"].apply(lambda s: s.abs().max()).idxmax()
    icb = ic_i[ic_i.slot == best_slot]
    score_i, used_i = composite(I.set_index(I.index), icb, intra_feats)
    mis = mis_daily_pnl(I[I.date >= isplit], score_i.loc[I.date >= isplit], gross_mult=2.0, slot=best_slot)
    liq_i = liq.reindex(mis.index).fillna(0)
    sims = {"CNC top-4 composite, 2-day hold (Goal 3 sleeve)": cnc,
            "Liquid-pledge + MIS composite (2x gross, neutral)": liq_i + mis}
    if regimes is not None:
        sims["CNC top-4 composite, vol-targeted 10%"] = cnc_vt
    cards = [L.scorecard(s, k) for k, s in sims.items()]
    best = L.save_scorecards(out, cards, PHASE, series=sims, notes=f"Test period only (daily from {split.date()}, intraday from {isplit.date()}).")
    L.plot_equity(sims, out / "candidates_equity.png", "Phase 3 candidates (test period)", log=False)

    # IC figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    top = ic_d.head(15).iloc[::-1]
    ax[0].barh(top["feature"], top["t_nw"], color=["g" if s else "grey" for s in top.get("significant_fdr10", [False] * len(top))])
    ax[0].set_title("Daily IC t-stat (train, green = FDR 10%)")
    ti = icb.head(15).iloc[::-1]
    ax[1].barh(ti["feature"], ti["t_nw"], color=["g" if s else "grey" for s in ti.get("significant_fdr10", [False] * len(ti))])
    ax[1].set_title(f"Intraday IC t-stat slot {best_slot} (train)")
    fig.tight_layout(); fig.savefig(out / "ic_tstats.png", dpi=120); plt.close(fig)

    md = f"""# Phase 3 - Features, leakage tests, IC analysis

## Leakage tests
{L.md_table(pd.DataFrame(lk))}

## Daily cross-sectional features (target: 2-day open-to-open excess return)
Train < {split.date()} | features significant at FDR 10%: {int(ic_d.get('significant_fdr10', pd.Series(dtype=bool)).sum())}/{len(ic_d)}

{L.md_table(ic_d.head(12)[['feature','mean_ic','ic_ir','t_nw','q_bh']])}

Out-of-sample check (test >= {split.date()}), same features:

{L.md_table(ic_d_test.head(12)[['feature','mean_ic','ic_ir','t_nw']])}

## Intraday features (target: return from decision to 15:15, cross-sectionally demeaned)
Best slot: {best_slot}. Composite uses: {used_i}

{L.md_table(icb.head(12)[['feature','mean_ic','ic_ir','t_nw','q_bh']])}

## Candidate scorecards (TEST period only)
{L.md_table(pd.DataFrame(cards)[L.SC_COLS])}

**Best:** {best.get('strategy')} | P(profitable 12m) {best.get('P_profitable_12m')} | predicted monthly {best.get('pred_monthly_mean')}
| P(all months >= -0.5%) {best.get('P_all_12_months_ge_floor')} | goal score {best.get('goal_score')}

Rows: daily_features.csv {len(X):,} | intraday_features.csv {len(I):,}
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    L.print_scorecard(best, PHASE)
    print(f"[P3] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
