#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 2 - Market regime model for a MONTHLY swing book.

What changed versus strategy/p2 (Vols 2, 3 and 5 of the strategy volumes)
-----------------------------------------------------------------------
  * Market-state features that the volumes found useful for a 21-day horizon:
      breadth_above_200  share of NIFTY100 names above their 200-day average
      mkt_vol_21         21-day realised NIFTY volatility (annualised)
      mkt_trend_63       63-day NIFTY log return
      dispersion_21      cross-sectional std of 21-day stock returns
  * 3-state Gaussian HMM (bull / neutral / hostile) on [5d return, log vol, breadth],
    refit every 21 trading days on an expanding window, inputs standardised with the
    TRAINING window's mean/std only, probabilities from the forward filter (never smoothed).
  * Regime cash floor (Vol 5): hostile -> keep at least 30% in cash / liquid ETF.
  * Vol-target scale for a 14% annual book vol, clipped so that cash never exceeds 30%.
  * Kalman local-linear trend and RMT absorption ratio kept from strategy/ (diagnostics).

Everything is causal: a value on date t uses data up to the close of t only.
Output folder: phase2_regimes_stochastic/
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import p1_data_costs as L

plt = L.plt
PHASE = "phase2_regimes_stochastic"
REG = {"refit_every": 21, "min_obs": 500, "K": 3, "hostile_floor": 0.30, "neutral_floor": 0.0,
       "vol_target": 0.14, "cash_cap": 0.30}


# ----------------------------------------------------------------------------
# Gaussian HMM (full covariance) - Baum-Welch with warm start + forward filter
# ----------------------------------------------------------------------------
def _logpdf(X, mu, cov):
    d = X.shape[1]
    inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    z = X - mu
    return -0.5 * (np.einsum("ij,jk,ik->i", z, inv, z) + logdet + d * np.log(2 * np.pi))


def hmm_fit(X, K=3, iters=60, init=None, seed=0):
    n, d = X.shape
    if init is None:
        q = np.quantile(X[:, 1], np.linspace(0, 1, K + 1))          # seed states by volatility tercile
        lab = np.clip(np.searchsorted(q, X[:, 1]) - 1, 0, K - 1)
        mu = np.array([X[lab == k].mean(0) for k in range(K)])
        cov = np.array([np.cov(X[lab == k].T) + 1e-3 * np.eye(d) for k in range(K)])
        A = np.full((K, K), 0.04 / (K - 1)); np.fill_diagonal(A, 0.96)
        pi = np.full(K, 1 / K)
    else:
        mu, cov, A, pi = (init[k].copy() for k in ("mu", "cov", "A", "pi"))
    prev = -np.inf
    for _ in range(iters):
        lB = np.column_stack([_logpdf(X, mu[k], cov[k]) for k in range(K)])
        mx = lB.max(1, keepdims=True)
        B = np.exp(lB - mx) + 1e-300
        al = np.zeros((n, K)); c = np.zeros(n)
        al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
        for t in range(1, n):
            al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
        be = np.ones((n, K))
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        gam = al * be; gam /= gam.sum(1, keepdims=True)
        xi = (al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :]) / c[1:, None, None]
        xi = xi / xi.sum((1, 2), keepdims=True)
        xi = xi.sum(0)
        pi = gam[0]; A = xi / xi.sum(1, keepdims=True)
        for k in range(K):
            w = gam[:, k]
            mu[k] = (w[:, None] * X).sum(0) / w.sum()
            z = X - mu[k]
            cov[k] = (w[:, None, None] * np.einsum("ij,ik->ijk", z, z)).sum(0) / w.sum() + 1e-3 * np.eye(d)
        ll = float(np.log(c).sum() + mx.sum())
        if abs(ll - prev) < 1e-5:
            break
        prev = ll
    return {"pi": pi, "A": A, "mu": mu, "cov": cov, "loglik": ll}


def hmm_filter(X, m):
    K = len(m["pi"])
    lB = np.column_stack([_logpdf(X, m["mu"][k], m["cov"][k]) for k in range(K)])
    B = np.exp(lB - lB.max(1, keepdims=True)) + 1e-300
    a = m["pi"] * B[0]; a /= a.sum()
    out = np.empty((len(X), K)); out[0] = a
    for t in range(1, len(X)):
        a = (a @ m["A"]) * B[t]; a /= a.sum(); out[t] = a
    return out


def hmm_walk_forward(feat: pd.DataFrame, quick=False):
    """Expanding-window refits every REG['refit_every'] days. States are re-labelled after each
    refit by their mean standardised return (bull = highest, hostile = lowest)."""
    X = feat.astype(float).dropna()
    vals = X.to_numpy(dtype=float)
    P = pd.DataFrame(np.nan, index=X.index, columns=["p_bull", "p_neutral", "p_hostile"])
    fits, prev = [], None
    step = REG["refit_every"] * (3 if quick else 1)
    for start in range(REG["min_obs"], len(X), step):
        tr = vals[:start]
        mu_, sd_ = tr.mean(0), tr.std(0)                     # TRAINING-window standardisation
        m = hmm_fit((tr - mu_) / sd_, K=REG["K"], iters=60 if prev is None else 25, init=prev)
        prev = m
        order = np.argsort(-(m["mu"][:, 0] - 0.5 * m["mu"][:, 1]))   # high return, low vol first
        end = min(start + step, len(X))
        pr = hmm_filter((vals[:end] - mu_) / sd_, m)[start:end]
        P.iloc[start:end] = pr[:, order]
        fits.append({"fit_date": str(X.index[start - 1].date()), "loglik": round(m["loglik"], 2),
                     **{f"mu_ret_{n}": round(float(m["mu"][o, 0]), 3) for n, o in zip(("bull", "neu", "hostile"), order)},
                     **{f"mu_vol_{n}": round(float(m["mu"][o, 1]), 3) for n, o in zip(("bull", "neu", "hostile"), order)},
                     **{f"stay_{n}": round(float(m["A"][o, o]), 3) for n, o in zip(("bull", "neu", "hostile"), order)}})
    return P, pd.DataFrame(fits)


def kalman_trend(y: pd.Series, q_level=1e-5, q_slope=1e-7, r_obs=1e-4):
    v = y.to_numpy()
    F = np.array([[1, 1], [0, 1]]); H = np.array([[1, 0]])
    Q = np.diag([q_level, q_slope]); R = np.array([[r_obs]])
    x = np.array([v[0], 0.0]); P = np.eye(2)
    slope, sstd = np.empty(len(v)), np.empty(len(v))
    for t in range(len(v)):
        x = F @ x; P = F @ P @ F.T + Q
        if np.isfinite(v[t]):
            S = H @ P @ H.T + R
            K = P @ H.T / S
            x = x + (K * (v[t] - H @ x)).ravel()
            P = (np.eye(2) - K @ H) @ P
        slope[t], sstd[t] = x[1], np.sqrt(P[1, 1])
    return pd.DataFrame({"trend_slope": slope * L.TRADING_DAYS, "trend_t": slope / sstd}, index=y.index)


def absorption_ratio(rets: pd.DataFrame, win=252, step=21) -> pd.Series:
    out = {}
    for end in range(win, len(rets), step):
        R = rets.iloc[end - win:end].dropna(axis=1, thresh=int(win * 0.9)).fillna(0)
        if R.shape[1] < 20:
            continue
        ev = np.sort(np.linalg.eigvalsh(np.corrcoef(R.T.to_numpy())))[::-1]
        out[rets.index[end - 1]] = ev[: max(1, R.shape[1] // 5)].sum() / ev.sum()
    return pd.Series(out, name="absorption_ratio")


def market_features(close: pd.DataFrame, stocks: list[str]) -> pd.DataFrame:
    n = close[L.INDEX]
    lr = np.log(n).diff()
    S = close[stocks]
    sma200 = S.rolling(200, min_periods=150).mean()
    above = (S > sma200).astype(float).where(sma200.notna())
    r21 = S.pct_change(21)
    return pd.DataFrame({
        "nifty_ret5": lr.rolling(5).sum(),
        "mkt_vol_21": lr.rolling(21).std() * np.sqrt(L.TRADING_DAYS),
        "mkt_trend_63": lr.rolling(63).sum(),
        "breadth_above_200": above.mean(1),
        "dispersion_21": r21.std(1),
        "mkt_drawdown": n / n.cummax() - 1,
    })


def main():
    ap = L.base_args("Strategy02 Phase 2: market regimes for a monthly book")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    P = L.load_daily_panel(data)
    close, opn, high, low = P["adj_close"], P["adj_open"], P["adj_high"], P["adj_low"]
    stocks = [c for c in close.columns if c != L.INDEX]
    liq = L.liquid_returns(data, close.index)

    print("[P2] market features ...")
    mf = market_features(close, stocks)
    feat = pd.DataFrame({"ret5": mf["nifty_ret5"], "logvol": np.log(mf["mkt_vol_21"]),
                         "breadth": mf["breadth_above_200"]})
    print("[P2] 3-state HMM, walk-forward refit every 21 days ...")
    probs, fits = hmm_walk_forward(feat, quick=args.quick)
    fits.to_csv(out / "hmm_refits.csv", index=False)
    trend = kalman_trend(np.log(close[L.INDEX]))
    ar = absorption_ratio(close[stocks].pct_change()).reindex(close.index).ffill()

    reg = pd.concat([mf, probs, trend, ar], axis=1)
    lab = probs[["p_bull", "p_neutral", "p_hostile"]].to_numpy()
    ok = np.isfinite(lab).all(1)
    regime = np.full(len(probs), "unknown", dtype=object)
    regime[ok] = np.array(["bull", "neutral", "hostile"])[np.nanargmax(np.where(ok[:, None], lab, 0), 1)[ok]]
    reg["regime"] = pd.Series(regime, index=probs.index).reindex(reg.index).fillna("unknown")
    floor = reg["regime"].map({"hostile": REG["hostile_floor"], "neutral": REG["neutral_floor"]}).fillna(0.0)
    vt = (REG["vol_target"] / reg["mkt_vol_21"].clip(lower=0.05)).clip(upper=1.0)
    reg["cash_floor"] = floor
    reg["vol_target_scale"] = vt
    reg["stock_frac"] = np.minimum(1 - floor, vt).clip(lower=1 - REG["cash_cap"])
    reg.to_csv(out / "regimes_daily.csv", index_label="date")

    # forward 21-day NIFTY return by regime (descriptive only)
    fwd = close[L.INDEX].shift(-21) / close[L.INDEX] - 1
    cond = pd.DataFrame({"regime": reg["regime"], "fwd21": fwd}).dropna()
    rc = cond.groupby("regime")["fwd21"].agg(["count", "mean", "std", lambda x: (x > 0).mean()])
    rc.columns = ["n_days", "mean_fwd21", "sd_fwd21", "frac_positive"]
    rc.to_csv(out / "regime_conditional_returns.csv")

    # ---------------- candidates (monthly hold, decision at month-end close)
    hs = L.half_spread_bps(high[stocks], low[stocks], P["turnover"][stocks])
    me = L.month_end_mask(close.index)
    avail = close[stocks].notna().astype(float)
    ew = avail.div(avail.sum(1), axis=0)
    frac = reg["stock_frac"].reindex(close.index).fillna(1.0)
    switch = (reg["regime"] == "hostile").astype(int).diff().fillna(0) != 0
    mask = me | switch.reindex(close.index).fillna(False)
    sims = {
        "EW NIFTY100 monthly": L.simulate_weights(ew, opn, close, liq, pledge=False, half_spread=hs, rebal_mask=me)["ret"],
        "EW NIFTY100 + regime floor + vol target": L.simulate_weights(ew.mul(frac, axis=0), opn, close, liq, pledge=False,
                                                                      half_spread=hs, rebal_mask=mask)["ret"],
        "NIFTY50 buy&hold": close[L.INDEX].pct_change().fillna(0),
        "Liquid ETF only": liq,
    }
    start = reg["p_bull"].first_valid_index()
    sims = {k: v.loc[start:] for k, v in sims.items()}
    cards = [L.scorecard(v, k) for k, v in sims.items()]
    best = L.save_scorecards(out, cards, PHASE, notes="Regime layer for a monthly book.", series=sims)
    L.multi_panel(sims, out / "regime_candidates_all_in_one.png", "Phase 2 - regime overlay vs plain books",
                  highlight="EW NIFTY100 + regime floor + vol target")

    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(close[L.INDEX].loc[start:], color="k", lw=1); ax[0].set_yscale("log"); ax[0].set_title("NIFTY50")
    colors = {"bull": "#c8f0c8", "neutral": "#f5f0c0", "hostile": "#f5c8c8"}
    rr = reg["regime"].loc[start:]
    for r, ccol in colors.items():
        m_ = (rr == r).to_numpy()
        ax[0].fill_between(rr.index, 0, 1, where=m_, color=ccol, transform=ax[0].get_xaxis_transform(), alpha=0.8)
    ax[1].stackplot(probs.loc[start:].index, probs.loc[start:].T.fillna(0).to_numpy(),
                    labels=["bull", "neutral", "hostile"], colors=["#2ca02c", "#bcbd22", "#d62728"], alpha=0.7)
    ax[1].legend(loc="upper left", fontsize=8); ax[1].set_title("HMM filtered probabilities (walk-forward)")
    ax[2].plot(reg["breadth_above_200"].loc[start:], label="breadth > SMA200")
    ax[2].plot(reg["stock_frac"].loc[start:], label="stock fraction (floor + vol target)")
    ax[2].legend(fontsize=8); ax[2].set_title("Breadth and allowed stock fraction")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "regimes.png", dpi=120); plt.close(fig)

    tab = pd.DataFrame(cards)[[c for c in L.SC_COLS if c in pd.DataFrame(cards)]]
    (out / "report.md").write_text(f"""# Strategy02 - Phase 2: regimes

3-state HMM on [NIFTY 5d return, log 21d vol, breadth above 200d], refit every 21 days
(walk-forward, training-window standardisation, forward filter). Cash floor 30% in the hostile
state, vol target 14% with cash capped at 30%.

## Forward 21-day NIFTY return by regime (descriptive, not a backtest)
{L.md_table(rc.reset_index())}

## Candidates
{L.md_table(tab)}

Best (goal score): **{best.get('strategy')}**

![regimes](regimes.png)
![all](regime_candidates_all_in_one.png)
""")
    (out / "regime_config.json").write_text(json.dumps(REG, indent=2))
    L.print_scorecard(best, PHASE)
    print(f"[P2] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
