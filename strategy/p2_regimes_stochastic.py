#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 2 - Stochastic volatility, regime detection and random-matrix structure.

Everything here is CAUSAL: every value on date t uses data up to and including t only
(expanding-window refits, forward-filtered HMM probabilities - never smoothed).

Models
------
  * Range-based volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
  * GJR-GARCH(1,1) with Student-t innovations, MLE (scipy), expanding refits -> sigma_{t+1|t}
  * Realized variance from 5-min NIFTY bars, HAR-RV forecast (Corsi 2009),
    rough-volatility Hurst exponent of log-RV (Gatheral-Jaisson-Rosenbaum 2018)
  * 2-state Gaussian HMM on [return, log vol] (Baum-Welch EM, forward filter)
  * Local-linear-trend Kalman filter on log NIFTY -> filtered slope and t-stat
  * Random matrix theory on the NIFTY100 correlation matrix: Marchenko-Pastur edge,
    number of informative eigenvalues, absorption ratio (Kritzman et al. 2011)
  * Ornstein-Uhlenbeck half-life of each stock's market-residual (mean reversion speed)

Candidate strategies scored: regime-gated equal-weight and volatility-targeted books.
Output folder: phase2_regimes_stochastic/
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

import p1_data_costs as L

plt = L.plt
PHASE = "phase2_regimes_stochastic"


# ----------------------------------------------------------------------------
# Range volatility estimators (daily variance), rolling mean -> annualised vol
# ----------------------------------------------------------------------------
def range_vols(o, h, l, c, win=20):
    lo, lh, ll, lc = np.log(o), np.log(h), np.log(l), np.log(c)
    pc = lc.shift(1)
    park = (lh - ll) ** 2 / (4 * np.log(2))
    gk = 0.5 * (lh - ll) ** 2 - (2 * np.log(2) - 1) * (lc - lo) ** 2
    rs = (lh - lc) * (lh - lo) + (ll - lc) * (ll - lo)
    on = lo - pc
    oc = lc - lo
    k = 0.34 / (1.34 + (win + 1) / (win - 1))
    yz = on.rolling(win).var() + k * oc.rolling(win).var() + (1 - k) * rs.rolling(win).mean()
    ann = np.sqrt(L.TRADING_DAYS)
    return pd.DataFrame({
        "vol_cc": np.log(c).diff().rolling(win).std() * ann,
        "vol_parkinson": np.sqrt(park.rolling(win).mean()) * ann,
        "vol_gk": np.sqrt(gk.rolling(win).mean().clip(lower=0)) * ann,
        "vol_rs": np.sqrt(rs.rolling(win).mean().clip(lower=0)) * ann,
        "vol_yz": np.sqrt(yz.clip(lower=0)) * ann,
    })


# ----------------------------------------------------------------------------
# GJR-GARCH(1,1)-t
# ----------------------------------------------------------------------------
def _gjr_filter(r, mu, omega, alpha, gamma, beta):
    e = r - mu
    s2 = np.empty(len(r) + 1)
    s2[0] = np.var(r)
    for t in range(len(r)):
        s2[t + 1] = omega + (alpha + gamma * (e[t] < 0)) * e[t] ** 2 + beta * s2[t]
    return s2  # s2[t] = variance for r[t] given info to t-1; s2[-1] = forecast


def _gjr_nll(p, r):
    mu, lw, a, g, b, lnu = p
    omega, nu = np.exp(lw), 2.05 + np.exp(lnu)
    if a < 0 or g < 0 or b < 0 or a + g / 2 + b >= 0.999:
        return 1e10
    s2 = _gjr_filter(r, mu, omega, a, g, b)[:-1]
    z2 = (r - mu) ** 2 / s2
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi * s2)
          - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
    return -ll.sum()


def fit_gjr(r, x0=None):
    x0 = x0 if x0 is not None else np.array([r.mean(), np.log(np.var(r) * 0.05), 0.05, 0.08, 0.85, np.log(6)])
    res = minimize(_gjr_nll, x0, args=(r,), method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-6})
    return res.x


def garch_forecasts(ret: pd.Series, refit_every=21, min_obs=500):
    r = ret.dropna().to_numpy() * 100
    idx = ret.dropna().index
    fc = np.full(len(r), np.nan)
    params, p = [], None
    for start in range(min_obs, len(r), refit_every):
        p = fit_gjr(r[:start], p)
        params.append((idx[start - 1], *p))
        end = min(start + refit_every, len(r))
        s2 = _gjr_filter(r[:end], p[0], np.exp(p[1]), p[2], p[3], p[4])
        # s2[t+1] is the forecast for day t+1 made at t  -> store on date t
        fc[start - 1:end] = np.sqrt(s2[start:end + 1])[: end - start + 1][: end - start + 1]
    s = pd.Series(fc / 100 * np.sqrt(L.TRADING_DAYS), index=idx, name="garch_vol_fcst")
    pr = pd.DataFrame(params, columns=["fit_date", "mu", "log_omega", "alpha", "gamma", "beta", "log_nu_m2"])
    return s, pr


# ----------------------------------------------------------------------------
# Realized variance, HAR-RV, rough-vol Hurst
# ----------------------------------------------------------------------------
def realized_var(intra: pd.DataFrame) -> pd.Series:
    lr = np.log(intra["adj_close"]).diff()
    same = intra["datetime"].dt.date == intra["datetime"].dt.date.shift(1)
    lr = lr.where(same)
    return (lr ** 2).groupby(intra["datetime"].dt.normalize()).sum(min_count=10)


def har_forecast(rv: pd.Series, min_obs=120) -> pd.Series:
    """Corsi HAR: RV_{t+1} = b0 + b1 RV_t + b2 mean5 + b3 mean22 (on log RV), expanding OLS."""
    x = np.log(rv.replace(0, np.nan)).dropna()
    X = pd.DataFrame({"d": x, "w": x.rolling(5).mean(), "m": x.rolling(22).mean()})
    y = x.shift(-1)
    out = pd.Series(np.nan, index=x.index)
    for t in range(min_obs, len(x)):
        Xt = X.iloc[:t - 1].dropna()
        yt = y.loc[Xt.index]
        A = np.column_stack([np.ones(len(Xt)), Xt.to_numpy()])
        beta = np.linalg.lstsq(A, yt.to_numpy(), rcond=None)[0]
        xv = X.iloc[t]
        if xv.notna().all():
            out.iloc[t] = beta[0] + beta[1:] @ xv.to_numpy()
    return np.sqrt(np.exp(out) * L.TRADING_DAYS).rename("har_vol_fcst")


def hurst_logvol(rv: pd.Series) -> dict:
    lv = 0.5 * np.log(rv.replace(0, np.nan).dropna())
    lags = np.arange(1, 31)
    res = {}
    for q in (1.0, 2.0):
        m = [np.mean(np.abs(lv.shift(-d) - lv).dropna() ** q) for d in lags]
        slope = np.polyfit(np.log(lags), np.log(m), 1)[0]
        res[f"H_q{int(q)}"] = round(float(slope / q), 4)
    res["interpretation"] = ("H << 0.5 => rough volatility (anti-persistent increments of log-vol); "
                             "short-horizon vol forecasts should weight recent RV heavily (HAR/rough-vol)")
    return res


# ----------------------------------------------------------------------------
# Gaussian HMM (2 states, full covariance) - Baum-Welch + forward filter
# ----------------------------------------------------------------------------
def _gauss_pdf(X, mu, cov):
    d = X.shape[1]
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    z = X - mu
    return np.exp(-0.5 * np.einsum("ij,jk,ik->i", z, inv, z)) / np.sqrt((2 * np.pi) ** d * det)


def hmm_fit(X, K=2, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    q = np.quantile(X[:, 1], np.linspace(0, 1, K + 1))
    lab = np.clip(np.searchsorted(q, X[:, 1]) - 1, 0, K - 1)
    mu = np.array([X[lab == k].mean(0) if (lab == k).sum() > d else X[rng.integers(0, n)] for k in range(K)])
    cov = np.array([np.cov(X.T) + 1e-6 * np.eye(d) for _ in range(K)])
    A = np.full((K, K), 0.05 / (K - 1))
    np.fill_diagonal(A, 0.95)
    pi = np.full(K, 1 / K)
    prev = -np.inf
    for _ in range(iters):
        B = np.column_stack([_gauss_pdf(X, mu[k], cov[k]) for k in range(K)]) + 1e-300
        al = np.zeros((n, K)); c = np.zeros(n)
        al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
        for t in range(1, n):
            al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
        be = np.ones((n, K))
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        gam = al * be
        gam /= gam.sum(1, keepdims=True)
        xi = np.zeros((K, K))
        for t in range(n - 1):
            m = np.outer(al[t], B[t + 1] * be[t + 1]) * A / c[t + 1]
            xi += m / m.sum()
        pi = gam[0]
        A = xi / xi.sum(1, keepdims=True)
        for k in range(K):
            w = gam[:, k]
            mu[k] = (w[:, None] * X).sum(0) / w.sum()
            z = X - mu[k]
            cov[k] = (w[:, None, None] * np.einsum("ij,ik->ijk", z, z)).sum(0) / w.sum() + 1e-6 * np.eye(d)
        ll = np.log(c).sum()
        if abs(ll - prev) < 1e-6:
            break
        prev = ll
    return {"pi": pi, "A": A, "mu": mu, "cov": cov, "loglik": ll}


def hmm_filter(X, m, p0=None):
    K = len(m["pi"])
    B = np.column_stack([_gauss_pdf(X, m["mu"][k], m["cov"][k]) for k in range(K)]) + 1e-300
    a = (m["pi"] if p0 is None else p0) * B[0]
    a /= a.sum()
    out = [a]
    for t in range(1, len(X)):
        a = (a @ m["A"]) * B[t]
        a /= a.sum()
        out.append(a)
    return np.array(out)


def hmm_causal(feat: pd.DataFrame, refit_every=63, min_obs=500, quick=False):
    X = feat.dropna()
    vals = X.to_numpy()
    mu_, sd_ = None, None
    pbull = pd.Series(np.nan, index=X.index)
    last = None
    for start in range(min_obs, len(X), refit_every):
        tr = vals[:start]
        mu_, sd_ = tr.mean(0), tr.std(0)
        m = hmm_fit((tr - mu_) / sd_, iters=40 if quick else 100)
        bull = int(np.argmax(m["mu"][:, 0] - m["mu"][:, 1]))   # higher return, lower vol
        end = min(start + refit_every, len(X))
        # forward filter from the beginning of the training window (cheap) -> probabilities to `end`
        P = hmm_filter((vals[:end] - mu_) / sd_, m)
        pbull.iloc[start:end] = P[start:end, bull]
        last = m, bull, mu_, sd_
    return pbull.rename("hmm_p_bull"), last


# ----------------------------------------------------------------------------
# Kalman local linear trend
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Random matrix theory
# ----------------------------------------------------------------------------
def rmt_rolling(rets: pd.DataFrame, win=252, step=21):
    rows = []
    for end in range(win, len(rets), step):
        R = rets.iloc[end - win:end].dropna(axis=1, thresh=int(win * 0.9)).fillna(0)
        T, N = R.shape
        if N < 20:
            continue
        Z = (R - R.mean()) / R.std().replace(0, np.nan)
        C = np.corrcoef(Z.fillna(0).T.to_numpy())
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        q = N / T
        lam_plus = (1 + np.sqrt(q)) ** 2
        n_top = max(1, N // 5)
        rows.append({"date": rets.index[end - 1], "lambda_max": ev[0], "lambda_mp_plus": lam_plus,
                     "n_signal_eigs": int((ev > lam_plus).sum()), "market_mode_share": ev[0] / N,
                     "absorption_ratio": ev[:n_top].sum() / ev.sum()})
    return pd.DataFrame(rows).set_index("date")


def ou_half_life(rets: pd.DataFrame, mkt: pd.Series, win=252):
    rows = []
    R = rets.iloc[-win:]
    m = mkt.reindex(R.index)
    for c in R.columns:
        y = R[c].dropna()
        if len(y) < 120:
            continue
        x = m.loc[y.index]
        b = np.cov(y, x)[0, 1] / x.var()
        resid = (y - b * x).cumsum()
        dx = resid.diff().dropna()
        lag = resid.shift(1).loc[dx.index]
        k = np.polyfit(lag, dx, 1)[0]
        hl = -np.log(2) / k if k < 0 else np.inf
        rows.append({"name": c, "beta_252d": round(b, 3), "ou_half_life_days": round(hl, 1)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main():
    ap = L.base_args("Phase 2: stochastic vol, regimes, RMT")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    P = L.load_daily_panel(data)
    close, opn, high, low = P["adj_close"], P["adj_open"], P["adj_high"], P["adj_low"]
    stocks = [c for c in close.columns if c != L.INDEX]
    nC, nO, nH, nL = (x[L.INDEX] for x in (close, opn, high, low))
    nret = np.log(nC).diff()
    liq = L.liquid_returns(data, close.index)

    print("[P2] range vols, GJR-GARCH-t ...")
    rv_daily = range_vols(nO, nH, nL, nC)
    garch, gparams = garch_forecasts(nret, refit_every=63 if args.quick else 21)
    gparams.to_csv(out / "garch_params.csv", index=False)

    print("[P2] realized variance (5-min), HAR, rough-vol Hurst ...")
    intra = L.load_intraday(data, L.INDEX, "5min")
    rv = realized_var(intra) if len(intra) else pd.Series(dtype=float)
    har = har_forecast(rv) if len(rv) > 200 else pd.Series(dtype=float, name="har_vol_fcst")
    rough = hurst_logvol(rv) if len(rv) > 100 else {}
    (out / "rough_vol.json").write_text(json.dumps(rough, indent=2))

    print("[P2] HMM regimes (causal) ...")
    feat = pd.DataFrame({"ret5": nret.rolling(5).sum(), "logvol": np.log(rv_daily["vol_yz"])})
    pbull, last = hmm_causal(feat, refit_every=126 if args.quick else 63, quick=args.quick)
    if last:
        m, bull, mu_, sd_ = last
        (out / "hmm_params.json").write_text(json.dumps({
            "bull_state": bull, "transition_matrix": m["A"].round(4).tolist(),
            "state_means_std_units": m["mu"].round(3).tolist(), "feature_mean": mu_.tolist(),
            "feature_std": sd_.tolist(), "features": ["5d log return", "log Yang-Zhang vol"]}, indent=2))

    print("[P2] Kalman trend, RMT, OU ...")
    trend = kalman_trend(np.log(nC))
    srets = close[stocks].pct_change()
    rmt = rmt_rolling(srets)
    ou = ou_half_life(srets, nret)
    ou.to_csv(out / "stock_ou_halflife.csv", index=False)

    reg = pd.concat([nret.rename("nifty_logret"), rv_daily, garch, har, pbull, trend,
                     rmt.reindex(close.index).ffill()], axis=1)
    reg["rv_5min_vol"] = np.sqrt(rv * L.TRADING_DAYS).reindex(reg.index)
    gv = reg["garch_vol_fcst"]
    reg["garch_vol_pctile"] = gv.expanding(250).rank(pct=True)
    reg["risk_on"] = ((reg["hmm_p_bull"] > 0.5) & (reg["trend_t"] > 0) & (reg["garch_vol_pctile"] < 0.85)).astype(int)
    reg["risk_score"] = (0.5 * reg["hmm_p_bull"].fillna(0.5) + 0.3 * (reg["trend_t"].clip(-3, 3) / 6 + 0.5)
                         + 0.2 * (1 - reg["garch_vol_pctile"].fillna(0.5)))
    reg.index.name = "date"
    reg.to_csv(out / "regimes_daily.csv")

    # ---------------- candidate strategies ----------------------------------
    hs = L.half_spread_bps(high[stocks], low[stocks], P["turnover"][stocks])
    avail = close[stocks].notna().astype(float)
    ew = avail.div(avail.sum(1), axis=0)
    ro = reg["risk_on"].reindex(close.index).fillna(0)
    me = ew.index.to_series().groupby(ew.index.to_period("M")).transform("max") == ew.index
    switch = ro.diff().fillna(0) != 0
    ew_m = ew.where(me | switch).ffill().fillna(0)
    gated = ew_m.mul(ro, axis=0)
    tv = (0.10 / reg["garch_vol_fcst"].reindex(close.index)).clip(upper=1.0).fillna(0)
    tv_w = tv.where(me | switch | (tv.diff().abs() > 0.15)).ffill().fillna(0)
    voltgt = ew_m.mul(tv_w * (0.5 + 0.5 * ro), axis=0)

    start = "2018-07-01"
    m1 = (me | switch)
    m2 = (me | switch | (tv.diff().abs() > 0.15))
    sims = {
        "EW NIFTY100 monthly (P1 best)": L.simulate_weights(ew_m, opn, close, liq, pledge=False, half_spread=hs, rebal_mask=me)["ret"],
        "Regime-gated EW (HMM+Kalman+GARCH)": L.simulate_weights(gated, opn, close, liq, pledge=False, half_spread=hs, rebal_mask=m1)["ret"],
        "Vol-targeted 10% + regime tilt": L.simulate_weights(voltgt, opn, close, liq, pledge=False, half_spread=hs, rebal_mask=m2)["ret"],
        "NIFTY50 buy&hold": nret.apply(np.expm1),
    }
    sims = {k: v.loc[start:] for k, v in sims.items()}       # (fix) same window for new and carried candidates
    cards = [L.scorecard(s, k) for k, s in sims.items()]
    best = L.save_scorecards(out, cards, PHASE, series=sims, notes=f"Evaluated from {start} (after 500-day model warm-up). "
                             "Survivorship bias: universe = today's NIFTY100.")

    # regime-conditional stats
    nd = nret.loc[start:]
    cond = pd.DataFrame({"ret": nd, "risk_on": reg["risk_on"].shift(1).loc[start:]}).dropna()
    rc = cond.groupby("risk_on")["ret"].agg(["mean", "std", "count"])
    rc["ann_ret"] = rc["mean"] * 252
    rc["ann_vol"] = rc["std"] * np.sqrt(252)
    rc.to_csv(out / "regime_conditional_returns.csv")

    # ---------------- figures -----------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(nC.loc[start:], color="k", lw=1)
    on = reg["risk_on"].loc[start:]
    ax[0].fill_between(on.index, nC.loc[start:].min(), nC.loc[start:].max(), where=on == 0, color="red", alpha=0.12, label="risk-off")
    ax[0].set_title("NIFTY50 with causal risk-off regimes"); ax[0].legend()
    ax[1].plot(reg["garch_vol_fcst"].loc[start:], label="GJR-GARCH-t forecast")
    ax[1].plot(reg["vol_yz"].loc[start:], label="Yang-Zhang 20d", alpha=0.7)
    if reg["har_vol_fcst"].notna().any():
        ax[1].plot(reg["har_vol_fcst"].loc[start:], label="HAR-RV forecast", alpha=0.8)
    ax[1].legend(); ax[1].set_title("Volatility (annualised)")
    ax[2].plot(reg["hmm_p_bull"].loc[start:], label="HMM P(bull)")
    ax[2].plot(reg["absorption_ratio"].loc[start:], label="RMT absorption ratio")
    ax[2].legend()
    fig.tight_layout(); fig.savefig(out / "regimes.png", dpi=120); plt.close(fig)
    L.plot_equity({k: v.loc[start:] for k, v in sims.items()}, out / "candidates_equity.png", "Phase 2 candidates")
    L.plot_monthly_hist({k: v.loc[start:] for k, v in sims.items()}, out / "candidates_monthly_hist.png",
                        "Phase 2 monthly returns")

    md = f"""# Phase 2 - Stochastic volatility, regimes, random-matrix structure

## Volatility
* GJR-GARCH(1,1)-t refit every {63 if args.quick else 21} days on an expanding window (latest params:
  alpha={gparams.iloc[-1]['alpha']:.3f}, gamma(leverage)={gparams.iloc[-1]['gamma']:.3f}, beta={gparams.iloc[-1]['beta']:.3f}).
* Rough-volatility Hurst exponent of log realized vol (5-min NIFTY): {rough}
* HAR-RV forecast available from {har.first_valid_index() if len(har) and har.notna().any() else 'n/a'}.

## Regimes
* 2-state Gaussian HMM on [5d return, log vol], forward-filtered (no look-ahead).
* Kalman local-linear trend t-stat; risk_on = P(bull)>0.5 AND trend_t>0 AND GARCH vol < 85th pct.
* NIFTY next-day return by regime (from {start}):

{L.md_table(rc.reset_index())}

## Random matrix theory (NIFTY100, 252d windows)
Latest: lambda_max={rmt['lambda_max'].iloc[-1]:.1f}, MP edge={rmt['lambda_mp_plus'].iloc[-1]:.2f},
informative eigenvalues={int(rmt['n_signal_eigs'].iloc[-1])}, absorption ratio={rmt['absorption_ratio'].iloc[-1]:.3f}.
Only a handful of eigenvalues exceed the Marchenko-Pastur edge -> most of the sample correlation matrix is noise;
phase 6 uses the RMT-cleaned covariance.

## Candidate scorecards
{L.md_table(pd.DataFrame(cards)[L.SC_COLS])}

**Best:** {best.get('strategy')} | P(profitable 12m) {best.get('P_profitable_12m')} | predicted monthly {best.get('pred_monthly_mean')}
| P(all months >= -0.5%) {best.get('P_all_12_months_ge_floor')} | goal score {best.get('goal_score')}
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    L.print_scorecard(best, PHASE)
    print(f"[P2] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
