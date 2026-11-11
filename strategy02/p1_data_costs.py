#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 1 - Data, Flattrade cost model (Vol 6 itemisation), margin/pledge model,
baselines, random-selection null, scorecard + shared library.

Adjusted from strategy/ following the "How a quantitative swing-trading system is built" volumes:
  * DP charge per sell day now charged (Vol 6: Rs 15.34) - the charge that hurts small accounts
  * slippage model: 10 bp floor + Amihud illiquidity (<=40 bp) + high-vol add-on (<=20 bp),
    25 bp on triggered stops (Vol 6 Ch.3)
  * Lo (2002) standard error of the Sharpe ratio, PSI drift statistic, random-pick null (Vol 6/8/9)
  * multi-panel comparison chart: many strategies as separate lines in ONE png

This file is also the shared library for phases 2-7 (they `import p1_data_costs as L`),
so that the pipeline stays at exactly one Python file per phase.

Mandate (from Indian_Equity_Market_Ultimate_Strategy_File.pdf + user answers)
-----------------------------------------------------------------------------
  Broker            Flattrade, zero brokerage, pledge Rs 20 + GST per pledge order
  Capital           Rs 1,00,000
  Goal 1            trailing max drawdown never worse than -20%  AND
                    trailing 12-month return > +20% at every point
  Goal 2            monthly return consistently inside [-0.5%, +3%] (higher is better)
  Goal 3            CNC basket held ~2 days, pledged, MIS (5x) traded on the collateral,
                    rebalanced whenever needed
  Goal 4            80% CNC / 20% MIS variant
  Goal 5            other distributions (pledging / liquid-ETF collateral, ...)
  Outputs           csv, json, md, png only - one folder per phase

Run:  python p1_data_costs.py            (from the strategy folder)
Data: ../data_clean  (output of clean_nifty100.py). Override: --data PATH or env QUANT_DATA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
IST = "Asia/Kolkata"
TRADING_DAYS = 252
INDEX = "NIFTY50"

# =============================================================================
# CONFIG  (every assumption in one place - edit here)
# =============================================================================
CFG = {
    "capital": 100_000.0,
    # ---- Flattrade charges (from the user's charges table) --------------------
    "brokerage": 0.0,
    "stt_delivery": 0.001,          # 0.1% buy & sell
    "stt_intraday_sell": 0.00025,   # 0.025% sell side only
    "exch_txn": 0.0000307,          # NSE 0.00307%
    "sebi_per_rs": 10 / 1e7,        # Rs 10 per crore
    "ipft_per_rs": 10 / 1e7,        # Rs 10 per crore
    "gst": 0.18,                    # on brokerage + transaction + SEBI
    "stamp_delivery_buy": 0.00015,  # 0.015% buy only
    "stamp_intraday_buy": 0.00003,  # 0.003% buy only
    "pledge_fee": 20.0 * 1.18,      # Rs 20 + GST per pledge order (per ISIN)
    "unpledge_fee": 0.0,            # VERIFY with Flattrade (set >0 if charged)
    "dp_fee_per_sell": 15.34,       # DP charge per ISIN per selling day (Vol 6 figure; VERIFY Flattrade's)
    # ---- execution frictions -----------------------------------------------
    "min_half_spread_bps": 1.0,     # floor on estimated half-spread
    "max_half_spread_bps": 10.0,
    "impact_coef": 0.7,             # square-root law  impact = c * sigma_d * sqrt(Q/ADV)
    # ---- margin / collateral (SEBI + Flattrade) ------------------------------
    "mis_leverage": 5.0,            # MIS exposure = 5 x usable margin (20% margin)
    "haircut_floor": 0.11,          # VaR(>=7.5%) + ELM(3.5%) style floor for NIFTY100 stocks
    "haircut_sigma_mult": 3.5,      # haircut = max(floor, 3.5*sigma_daily + 3.5%)
    "liquid_haircut": 0.10,         # cash-equivalent collateral (LIQUIDBEES-type)
    "liquid_yield": 0.05,           # annual yield of the liquid ETF proxy: overnight rate (~5.25-5.5%) minus
                                    # LIQUIDBEES TER (~0.7%). Download LIQUIDBEES 1D data to use the real series.
    "liquid_cost_side": 0.00005,    # ~0.5 bp per side for liquid ETF trades
    "settlement_lag_days": 1,       # bought shares can be pledged only after T+1
    # ---- Vol 6 execution realism -----------------------------------------------
    "slip_floor_bps": 10.0,         # minimum slippage per side
    "slip_amihud_cap_bps": 40.0,    # illiquidity add-on cap
    "slip_vol_cap_bps": 20.0,       # add-on for annual vol above 25%
    "stop_slip_bps": 25.0,          # a triggered stop is a market order in a falling market
    "circuit_limit": 0.195,         # open > 19.5% away from decision price -> no fill
    "volume_cap": 0.01,             # fills capped at 1% of the day's volume
    "min_position_rs": 4000.0,      # Vol 5: smaller positions make the DP charge too large
    # ---- goals --------------------------------------------------------------
    "goal_month_floor": -0.005,
    "goal_month_cap": 0.03,
    "goal_max_dd": -0.20,
    "goal_trailing_12m": 0.20,
    # ---- bootstrap ----------------------------------------------------------
    "boot_paths": 3000,
    "boot_block": 10,               # mean block length (days) - stationary bootstrap
    "seed": 7,
}


# =============================================================================
# Paths / IO
# =============================================================================
def data_dir(cli: str | None = None) -> Path:
    p = Path(cli or os.getenv("QUANT_DATA") or (HERE.parent / "data_clean"))
    if not (p / "1D").exists():
        sys.exit(f"[!] data folder not found: {p}  (run clean_nifty100.py first, or pass --data)")
    return p


def out_dir(name: str) -> Path:
    p = HERE / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def base_args(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default=None, help="data_clean folder")
    ap.add_argument("--quick", action="store_true", help="smaller/faster run (for testing)")
    return ap


def _read(path: Path, cols=None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols)
        if "session_date" in df:
            df["session_date"] = pd.to_datetime(df["session_date"])
    dt = df["datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, utc=True, format="ISO8601").dt.tz_convert(IST)
    elif dt.dt.tz is None:
        dt = dt.dt.tz_localize(IST)
    df["datetime"] = dt.dt.tz_convert(IST).dt.tz_localize(None)   # naive IST everywhere
    return df


def find(data: Path, tf: str, name: str) -> Path | None:
    for ext in (".parquet", ".csv"):
        p = data / tf / f"{name}{ext}"
        if p.exists():
            return p
    return None


def names(data: Path, include_index: bool = False) -> list[str]:
    ns = sorted({p.stem for p in (data / "1D").iterdir() if p.suffix in (".parquet", ".csv")})
    ns = [n for n in ns if n != "LIQUIDBEES"]
    return ns if include_index else [n for n in ns if n != INDEX]


def load_daily_panel(data: Path, fields=("adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
                                          "raw_open", "raw_close", "turnover")) -> dict[str, pd.DataFrame]:
    """Wide panels: dict field -> DataFrame(date x name). Includes the NIFTY50 index."""
    frames = {f: {} for f in fields}
    for n in names(data, include_index=True):
        df = _read(find(data, "1D", n))
        df["date"] = df["datetime"].dt.normalize()
        df = df.set_index("date")
        for f in fields:
            if f in df:
                frames[f][n] = df[f]
    return {f: pd.DataFrame(v).sort_index() for f, v in frames.items()}


def load_intraday(data: Path, name: str, tf: str = "5min", regular_only: bool = True) -> pd.DataFrame:
    p = find(data, tf, name)
    if p is None:
        return pd.DataFrame()
    df = _read(p)
    if regular_only and "session" in df:
        df = df[df["session"] == "regular"]
    return df.reset_index(drop=True)


def liquid_returns(data: Path, dates: pd.DatetimeIndex) -> pd.Series:
    """Daily return of the collateral parking asset. Uses LIQUIDBEES if you downloaded it
    (python iifl_fetch_nifty100.py --symbols LIQUIDBEES --timeframes 1D), else a constant
    yield proxy."""
    p = find(data, "1D", "LIQUIDBEES")
    if p is not None:
        df = _read(p)
        s = df.set_index(df["datetime"].dt.normalize())["adj_close"].pct_change()
        return s.reindex(dates).fillna((1 + CFG["liquid_yield"]) ** (1 / TRADING_DAYS) - 1)
    return pd.Series((1 + CFG["liquid_yield"]) ** (1 / TRADING_DAYS) - 1, index=dates)


# =============================================================================
# Cost model (Flattrade table) - all functions return RUPEES
# =============================================================================
def _statutory(value: float, side: str, product: str) -> float:
    c = CFG
    exch = value * c["exch_txn"]
    sebi = value * c["sebi_per_rs"]
    ipft = value * c["ipft_per_rs"]
    gst = c["gst"] * (c["brokerage"] + exch + sebi)
    if product == "CNC":
        stt = value * c["stt_delivery"]
        stamp = value * c["stamp_delivery_buy"] if side == "buy" else 0.0
    else:
        stt = value * c["stt_intraday_sell"] if side == "sell" else 0.0
        stamp = value * c["stamp_intraday_buy"] if side == "buy" else 0.0
    return stt + exch + sebi + ipft + gst + stamp + c["brokerage"]


def cost_cnc(value: float, side: str, half_spread_bps: float = 2.0, pledge: bool = True) -> float:
    fees = _statutory(abs(value), side, "CNC") + abs(value) * half_spread_bps / 1e4
    if side == "buy" and pledge:
        fees += CFG["pledge_fee"]
    if side == "sell":
        fees += CFG["dp_fee_per_sell"] + (CFG["unpledge_fee"] if pledge else 0.0)
    return fees


def cost_mis(value: float, side: str, half_spread_bps: float = 2.0) -> float:
    return _statutory(abs(value), side, "MIS") + abs(value) * half_spread_bps / 1e4


def roundtrip_pct(value: float, product: str, half_spread_bps: float = 2.0, pledge: bool = True) -> float:
    if product == "CNC":
        return (cost_cnc(value, "buy", half_spread_bps, pledge) + cost_cnc(value, "sell", half_spread_bps, pledge)) / value
    return (cost_mis(value, "buy", half_spread_bps) + cost_mis(value, "sell", half_spread_bps)) / value


# vectorised (fraction of notional, excluding fixed fees) - used by fast backtests
def cnc_rate(side: str) -> float:
    return _statutory(1e7, side, "CNC") / 1e7


def mis_rate_roundtrip() -> float:
    return (_statutory(1e7, "buy", "MIS") + _statutory(1e7, "sell", "MIS")) / 1e7


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Corwin & Schultz (2012) high-low spread estimator (daily), returns full spread."""
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    h2 = np.maximum(high, high.shift(1))
    l2 = np.minimum(low, low.shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


def half_spread_bps(high: pd.DataFrame, low: pd.DataFrame, turnover: pd.DataFrame | None = None,
                    window: int = 60, causal: bool = True):
    """Half-spread estimate in bps. Daily-bar spread estimators (Corwin-Schultz) are biased
    upward by intraday volatility, so for the cost model we use a liquidity-scaled rule calibrated
    to typical NSE large-cap quoted spreads: ~1.5 bp at Rs 500 cr/day turnover, rising with
    1/sqrt(turnover); clipped to [min, max]. CS is still reported as a diagnostic."""
    if turnover is not None and causal:
        # TIME-VARYING and causal (fix): spread on day t uses the median turnover of the 60 days
        # BEFORE t. (Earlier versions used the last 60 days of the whole file for every date, which
        # made 2017-2020 costs look ~45% too cheap because turnover has grown ~5x.)
        to = turnover.rolling(window, min_periods=20).median().shift(1).replace(0, np.nan) / 1e7
        hs = 1.5 * np.sqrt(500.0 / to)
        return hs.clip(CFG["min_half_spread_bps"], CFG["max_half_spread_bps"]).bfill().fillna(3.0)
    if turnover is not None:
        to = turnover.iloc[-window:].median().replace(0, np.nan) / 1e7          # Rs crore (latest snapshot)
        hs = 1.5 * np.sqrt(500.0 / to)
    else:
        cs = corwin_schultz_spread(high, low).rolling(window, min_periods=20).mean()
        hs = cs.iloc[-1] * 1e4 / 2
    return hs.clip(CFG["min_half_spread_bps"], CFG["max_half_spread_bps"]).fillna(3.0)


def haircut(sigma_daily: pd.Series | float):
    return np.maximum(CFG["haircut_floor"], CFG["haircut_sigma_mult"] * sigma_daily + 0.035)


def usable_margin(free_cash: float, liquid_val: float, stock_collateral_val: float) -> float:
    """Flattrade/SEBI 50:50 rule: non-cash collateral (pledged stocks, after haircut) is usable
    only up to the amount of cash + cash-equivalent collateral (liquid ETF after haircut)."""
    cash_side = max(free_cash, 0.0) + liquid_val * (1 - CFG["liquid_haircut"])
    return cash_side + min(stock_collateral_val, cash_side)


def mis_capacity(free_cash, liquid_val, stock_collateral_val) -> float:
    return CFG["mis_leverage"] * usable_margin(free_cash, liquid_val, stock_collateral_val)


# =============================================================================
# Statistics / scorecard (used by every phase)
# =============================================================================
def to_monthly(daily: pd.Series) -> pd.Series:
    d = daily.dropna()
    return (1 + d).groupby(d.index.to_period("M")).prod() - 1


def max_drawdown(daily: pd.Series) -> float:
    eq = (1 + daily.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def rolling_12m(daily: pd.Series) -> pd.Series:
    eq = (1 + daily.fillna(0)).cumprod()
    return eq / eq.shift(TRADING_DAYS) - 1


def stationary_bootstrap(x: np.ndarray, n_paths: int, horizon: int, block: float, rng) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap. Returns (n_paths, horizon) array."""
    n = len(x)
    p = 1.0 / block
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_paths)
    jump = rng.random((n_paths, horizon)) < p
    fresh = rng.integers(0, n, (n_paths, horizon))
    for t in range(1, horizon):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return x[idx]


def goal_score(sc: dict) -> float:
    """One number (0..1) to rank strategies against the mandate.
      30% consistency  : share of months >= -0.5%
      25% Goal 1 (both): P(maxDD better than -20% AND 12m return > 20%) over a bootstrapped year
      15% drawdown     : P(maxDD better than -20%)
      30% return       : mean monthly return relative to the +3% upper target
    Hard-constraint penalty: x0.6 if the historical max drawdown breached -20%."""
    s = (0.30 * sc["frac_months_ge_floor"] + 0.25 * sc["P_goal1_both"] + 0.15 * sc["P_maxdd_ok_12m"]
         + 0.30 * float(np.clip(sc["mean_monthly"] / CFG["goal_month_cap"], 0, 1)))
    if not sc.get("hist_goal1_dd_ok", True):
        s *= 0.6
    return round(s, 4)


def scorecard(daily: pd.Series, label: str, n_boot: int | None = None) -> dict:
    d = daily.dropna().astype(float)
    if len(d) < 40:
        return {"strategy": label, "n_days": len(d), "note": "too short"}
    m = to_monthly(d)
    ann = (1 + d).prod() ** (TRADING_DAYS / len(d)) - 1
    vol = d.std() * np.sqrt(TRADING_DAYS)
    r12 = rolling_12m(d).dropna()
    sc = {
        "strategy": label, "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
        "n_days": int(len(d)), "n_months": int(len(m)),
        "CAGR": round(float(ann), 4), "ann_vol": round(float(vol), 4),
        "sharpe": round(float(d.mean() / d.std() * np.sqrt(TRADING_DAYS)) if d.std() > 1e-8 else 0.0, 3),
        "max_drawdown": round(max_drawdown(d), 4),
        "mean_monthly": round(float(m.mean()), 5), "median_monthly": round(float(m.median()), 5),
        "worst_month": round(float(m.min()), 4), "best_month": round(float(m.max()), 4),
        "frac_months_positive": round(float((m > 0).mean()), 3),
        "frac_months_ge_floor": round(float((m >= CFG["goal_month_floor"]).mean()), 3),
        "frac_months_in_band": round(float(((m >= CFG["goal_month_floor"]) & (m <= CFG["goal_month_cap"])).mean()), 3),
        "min_trailing_12m": round(float(r12.min()), 4) if len(r12) else None,
        "hist_goal1_dd_ok": bool(max_drawdown(d) >= CFG["goal_max_dd"]),
        "hist_goal1_12m_ok": bool(len(r12) > 0 and r12.min() > CFG["goal_trailing_12m"]) if len(r12) else None,
        "hist_goal2_ok": bool((m >= CFG["goal_month_floor"]).all()),
    }
    # ---- forward-looking probabilities via stationary bootstrap (12 months ahead)
    rng = np.random.default_rng(CFG["seed"])
    nb = n_boot or CFG["boot_paths"]
    paths = stationary_bootstrap(d.to_numpy(), nb, 12 * 21, CFG["boot_block"], rng)
    months = (1 + paths.reshape(nb, 12, 21)).prod(axis=2) - 1
    tot = (1 + paths).prod(axis=1) - 1
    eq = np.cumprod(1 + paths, axis=1)
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    sc.update({
        "P_profitable_12m": round(float((tot > 0).mean()), 3),          # probability of profitability
        "P_12m_gt_20pct": round(float((tot > CFG["goal_trailing_12m"]).mean()), 3),
        "P_maxdd_ok_12m": round(float((dd > CFG["goal_max_dd"]).mean()), 3),
        "P_goal1_both": round(float(((dd > CFG["goal_max_dd"]) & (tot > CFG["goal_trailing_12m"])).mean()), 3),
        "P_month_ge_floor": round(float((months >= CFG["goal_month_floor"]).mean()), 3),
        "P_all_12_months_ge_floor": round(float((months >= CFG["goal_month_floor"]).all(axis=1).mean()), 4),
        "pred_monthly_mean": round(float(months.mean()), 5),                # predicted profitability
        "pred_monthly_p05": round(float(np.percentile(months, 5)), 4),
        "pred_monthly_p95": round(float(np.percentile(months, 95)), 4),
        "pred_12m_median": round(float(np.median(tot)), 4),
    })
    sc["goal_score"] = goal_score(sc)
    return sc


def best_of(cards: list[dict]) -> dict:
    ok = [c for c in cards if "goal_score" in c]
    return max(ok, key=lambda c: c["goal_score"]) if ok else {}


def save_scorecards(folder: Path, cards: list[dict], phase: str, notes: str = "",
                    series: dict[str, pd.Series] | None = None) -> dict:
    """Writes scorecards.csv / scorecard.json. The previous phase's best is carried forward as a
    candidate, so the 'best' reported by a phase is never worse than before - if a new idea does
    not beat it, the report says so explicitly."""
    prev = previous_best(phase)
    allc = list(cards)
    if prev and "goal_score" in prev:
        base_name = prev["strategy"].replace("[carried] ", "")
        carried = dict(prev)
        s_prev = find_series(phase, base_name)
        if series and s_prev is not None:
            # re-score the previous best on THIS phase's evaluation window -> apples-to-apples
            lo = min(pd.Series(v).dropna().index.min() for v in series.values())
            hi = max(pd.Series(v).dropna().index.max() for v in series.values())
            win = s_prev.loc[lo:hi].dropna()
            if len(win) >= 40:
                carried = scorecard(win, base_name)
        carried["strategy"] = f"[carried] {base_name}"
        allc.append(carried)
    best = best_of(allc)
    pd.DataFrame(allc).to_csv(folder / "scorecards.csv", index=False)
    new_best = best_of(cards)
    payload = {"phase": phase, "best": best, "best_new_this_phase": new_best, "all": allc,
               "previous_phase_best": prev,
               "improvement_goal_score": round(best.get("goal_score", 0) - prev.get("goal_score", 0), 4) if prev else None,
               "new_idea_beat_previous": bool(new_best and prev and new_best.get("goal_score", 0) > prev.get("goal_score", 0)),
               "notes": notes}
    (folder / "scorecard.json").write_text(json.dumps(payload, indent=2, default=str))
    if series:
        pd.DataFrame(series).to_csv(folder / "candidate_daily_returns.csv")
    return best


def find_series(phase: str, name: str) -> pd.Series | None:
    """Daily returns of a named candidate saved by an earlier phase (candidate_daily_returns.csv)."""
    i = PHASES.index(phase)
    for p in reversed(PHASES[:i]):
        f = HERE / p / "candidate_daily_returns.csv"
        if f.exists():
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if name in df.columns:
                return df[name]
    return None


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_trials_var: float | None = None) -> dict:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio on per-period returns."""
    from scipy.stats import norm, skew, kurtosis
    r = returns.dropna().to_numpy()
    T = len(r)
    sr = r.mean() / r.std(ddof=1) if r.std() > 0 else 0.0
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    v = sr_trials_var if sr_trials_var is not None else 1.0 / T
    emc = 0.5772156649
    n = max(n_trials, 2)
    sr0 = np.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return {"sr_per_period": round(float(sr), 4), "sr_annual": round(float(sr * np.sqrt(TRADING_DAYS)), 3),
            "sr0_threshold": round(float(sr0), 4), "n_trials": int(n_trials), "DSR": round(float(norm.cdf(z)), 4)}


def slippage_bps(amihud: float, ann_vol: float) -> float:
    """Vol 6 slippage model, per side, in bps."""
    c = CFG
    # amihud = mean(|daily return| / turnover in Rs crore); NIFTY100 median ~7e-5 -> ~2 bp, 95th pct ~30 bp
    add_illiq = min(c["slip_amihud_cap_bps"], max(0.0, amihud) * 1e4 * 3) if np.isfinite(amihud) else c["slip_amihud_cap_bps"] / 2
    # high-volatility add-on: +0.5 bp per vol point above 30% annualised, capped
    add_vol = min(c["slip_vol_cap_bps"], max(0.0, (ann_vol - 0.30) * 50)) if np.isfinite(ann_vol) else 0.0
    return c["slip_floor_bps"] + add_illiq + add_vol


def lo_sharpe_se(monthly: pd.Series) -> dict:
    """Standard error of the annualised Sharpe from monthly returns.
    Per-period SE: Mertens (2002) non-normal iid formula. Annualisation: Lo (2002) eq. (13),
    eta(q) = q / sqrt(q + 2 sum_{k<q} (q-k) rho_k) with q = 12, i.e. autocorrelation-aware."""
    from scipy.stats import skew, kurtosis
    m = monthly.dropna().to_numpy()
    n = len(m)
    if n < 6 or m.std() == 0:
        return {"sharpe": None, "se": None, "n_months": n}
    sr = m.mean() / m.std(ddof=1)
    g3, g4 = skew(m), kurtosis(m, fisher=False)
    se = np.sqrt((1 + 0.5 * sr ** 2 - g3 * sr + (g4 - 3) / 4 * sr ** 2) / n)
    q = 12
    x = m - m.mean()
    rho = [float((x[k:] @ x[:-k]) / (x @ x)) if n > k + 2 else 0.0 for k in range(1, q)]
    eta = q / np.sqrt(max(q + 2 * sum((q - k) * r for k, r in zip(range(1, q), rho)), 1e-6))
    return {"sharpe": round(float(sr * eta), 3), "se": round(float(se * eta), 3), "n_months": n,
            "sharpe_iid_sqrt12": round(float(sr * np.sqrt(12)), 3), "lo_eta": round(float(eta), 3)}


def psi(train: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
    """Population stability index of `recent` against `train` (decile bins of train)."""
    tr, rc = train[np.isfinite(train)], recent[np.isfinite(recent)]
    if len(tr) < 50 or len(rc) < 20:
        return np.nan
    edges = np.unique(np.quantile(tr, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(tr, edges)[0] / len(tr) + 1e-6
    q = np.histogram(rc, edges)[0] / len(rc) + 1e-6
    return float(((q - p) * np.log(q / p)).sum())


def multi_panel(series: dict[str, pd.Series], path: Path, title: str, highlight: str | None = None,
                band: tuple[pd.Series, pd.Series] | None = None) -> None:
    """ONE png, four panels, every strategy as its own line: equity (log, Rs), drawdown,
    rolling 12-month return, and monthly-return distribution (box plot)."""
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    cap = CFG["capital"]
    cmap = plt.get_cmap("tab20")
    bench = ("NIFTY50", "EW NIFTY100", "Liquid", "Random")
    for n, (k, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        lw, z = (2.8, 5) if k == highlight else (1.2, 2)
        col = "black" if k == highlight else cmap((2 * n) % 20 if n < 10 else (2 * n + 1) % 20)
        ls = "--" if any(b in k for b in bench) else "-"
        eq = cap * (1 + s).cumprod()
        ax[0, 0].plot(eq, label=k, lw=lw, zorder=z, color=col, ls=ls)
        ax[0, 1].plot((eq / eq.cummax() - 1) * 100, lw=lw * 0.7, zorder=z, color=col, ls=ls)
        ax[1, 0].plot(rolling_12m(s) * 100, lw=lw * 0.7, zorder=z, color=col, ls=ls)
    if band is not None:
        lo_, hi_ = band
        ax[0, 0].fill_between(lo_.index, cap * (1 + lo_).cumprod(), cap * (1 + hi_).cumprod(),
                              color="grey", alpha=0.15, label="random-pick null 5-95%")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_title("Equity (Rs, log)")
    ax[0, 1].axhline(CFG["goal_max_dd"] * 100, color="red", ls="--", lw=1); ax[0, 1].set_title("Drawdown % (Goal 1: -20%)")
    ax[1, 0].axhline(CFG["goal_trailing_12m"] * 100, color="green", ls="--", lw=1); ax[1, 0].axhline(0, color="k", lw=0.6)
    ax[1, 0].set_title("Rolling 12-month return % (Goal 1: > +20%)")
    mons = {k: to_monthly(v.dropna()) * 100 for k, v in series.items() if v.dropna().size}
    ax[1, 1].boxplot(list(mons.values()), vert=False, showfliers=True)
    ax[1, 1].set_yticks(range(1, len(mons) + 1)); ax[1, 1].set_yticklabels([k[:38] for k in mons], fontsize=7)
    ax[1, 1].axvline(CFG["goal_month_floor"] * 100, color="red", ls="--"); ax[1, 1].axvline(CFG["goal_month_cap"] * 100, color="green", ls="--")
    ax[1, 1].set_title("Monthly returns % (band -0.5% .. +3%)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02 + 0.022 * ((len(lab) + 2) // 3), 1, 1)); fig.savefig(path, dpi=120); plt.close(fig)


PHASES = ["phase1_data_costs", "phase2_regimes_stochastic", "phase3_features", "phase4_ml_models",
          "phase5_deep_learning", "phase6_portfolio_backtest", "phase7_robustness_report"]


def previous_best(phase: str) -> dict:
    i = PHASES.index(phase)
    for p in reversed(PHASES[:i]):
        f = HERE / p / "scorecard.json"
        if f.exists():
            return json.loads(f.read_text()).get("best", {})
    return {}


def print_scorecard(best: dict, phase: str) -> None:
    prev = previous_best(phase)
    print("\n" + "=" * 78)
    print(f"{phase.upper()}  -  best candidate: {best.get('strategy')}")
    print("=" * 78)
    keys = [("P_profitable_12m", "Probability of profitability (12m)"),
            ("pred_monthly_mean", "Predicted mean monthly return"),
            ("P_month_ge_floor", "P(month >= -0.5%)"),
            ("P_all_12_months_ge_floor", "P(all 12 months >= -0.5%)"),
            ("P_12m_gt_20pct", "P(12m return > 20%)"),
            ("P_maxdd_ok_12m", "P(max DD better than -20%)"),
            ("CAGR", "Historical CAGR"), ("max_drawdown", "Historical max DD"),
            ("frac_months_ge_floor", "Hist. months >= -0.5%"), ("goal_score", "Goal score (0-1)")]
    for k, lab in keys:
        v, pv = best.get(k), prev.get(k) if prev else None
        delta = f"   (prev {pv})" if pv is not None else ""
        print(f"  {lab:<36} {v}{delta}")
    print("=" * 78 + "\n")


def md_table(df: pd.DataFrame, floatfmt: int = 4) -> str:
    df = df.copy()
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append(f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


SC_COLS = ["strategy", "CAGR", "ann_vol", "sharpe", "max_drawdown", "mean_monthly", "worst_month",
           "frac_months_ge_floor", "P_profitable_12m", "P_month_ge_floor", "P_12m_gt_20pct",
           "P_maxdd_ok_12m", "goal_score"]


def plot_equity(series: dict[str, pd.Series], path: Path, title: str, log: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, s in series.items():
        s = s.dropna()
        if len(s):
            ax.plot((1 + s).cumprod(), label=k, lw=1.2)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_monthly_hist(series: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, s in series.items():
        m = to_monthly(s.dropna())
        if len(m):
            ax.hist(m * 100, bins=40, alpha=0.45, label=k)
    ax.axvline(CFG["goal_month_floor"] * 100, color="red", ls="--", label="-0.5% floor")
    ax.axvline(CFG["goal_month_cap"] * 100, color="green", ls="--", label="+3% target")
    ax.set_xlabel("monthly return %")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =============================================================================
# Simple daily-rebalanced portfolio simulator (used by phases 1-3 for candidates)
# =============================================================================
def simulate_weights(weights: pd.DataFrame, open_px: pd.DataFrame, close_px: pd.DataFrame,
                     liquid: pd.Series, capital: float = None, pledge: bool = True,
                     half_spread: pd.Series | pd.DataFrame | None = None, rebal_mask: pd.Series | None = None) -> pd.DataFrame:
    """Target weights decided at the CLOSE of day t are executed at the OPEN of t+1.
    Capital not in stocks sits in the liquid asset. Costs: Flattrade CNC statutory rates
    + half-spread + fixed pledge fee (Rs) per newly bought name (+ DP fee per sold name).
    rebal_mask (strategy02): if given, trades happen ONLY on the open after a True day; on all other
    days positions drift with prices (true monthly buy-and-hold, costs on the delta only).
    Returns DataFrame[ret, cost_frac, turnover] indexed by date."""
    capital = capital or CFG["capital"]
    cols = weights.columns
    w = weights.reindex(close_px.index).fillna(0.0)
    op = open_px[cols].reindex(close_px.index)
    cl = close_px[cols]
    # per-side cost on top of statutory charges = half-spread + the same 10 bp slippage floor the
    # random null and the phase-6 engine pay (fix: the books used to pay only the half-spread)
    slip = CFG["slip_floor_bps"] / 1e4
    if isinstance(half_spread, pd.DataFrame):
        HS = (half_spread.reindex(index=close_px.index, columns=cols).ffill().fillna(3.0) / 1e4).to_numpy() + slip
    else:
        hs_ = ((half_spread.reindex(cols).fillna(3.0) / 1e4) if half_spread is not None
               else pd.Series(2e-4, index=cols)).to_numpy() + slip
        HS = np.tile(hs_, (len(close_px), 1))
    buy_r, sell_r = cnc_rate("buy"), cnc_rate("sell")
    O, C, W = op.to_numpy(), cl.to_numpy(), w.to_numpy()
    liq = liquid.reindex(close_px.index).fillna(0.0).to_numpy()
    RB = (rebal_mask.reindex(close_px.index).fillna(False).to_numpy() if rebal_mask is not None
          else np.ones(len(close_px), dtype=bool))
    pos = np.zeros(len(cols))            # rupees per name at previous close
    cash = capital                       # liquid balance
    eq_prev = capital
    out = []
    for i in range(1, len(cl)):
        o, c, cp = O[i], C[i], C[i - 1]
        g = np.where(np.isfinite(o) & np.isfinite(cp) & (cp > 0), o / cp, 1.0)
        pos_open = pos * g
        eq_open = pos_open.sum() + cash
        tgt_w = np.where(np.isfinite(o), W[i - 1], 0.0)          # (fix) no peek at day i's close
        tgt_w = np.where(pos_open > 0, np.where(np.isfinite(o), tgt_w, pos_open / max(eq_open, 1e-9)), tgt_w)
        target = tgt_w * eq_open if RB[i - 1] else pos_open.copy()
        trade = target - pos_open
        buys, sells = np.clip(trade, 0, None), np.clip(-trade, 0, None)
        new_names = int(((target > 0) & (pos_open <= 0)).sum())
        closed = int(((target <= 0) & (pos_open > 0)).sum())
        sold_names = int((sells > 1.0).sum())                     # DP fee: per scrip per SELL day (partial sells too)
        hs = HS[i - 1]
        cost = float((buys * (buy_r + hs)).sum() + (sells * (sell_r + hs)).sum()
                     + (new_names * CFG["pledge_fee"] if pledge else 0.0)
                     + sold_names * CFG["dp_fee_per_sell"] + closed * (CFG["unpledge_fee"] if pledge else 0.0))
        cash = eq_open - target.sum() - cost
        r_ic = np.where(np.isfinite(c) & np.isfinite(o) & (o > 0), c / o, 1.0)
        pos = target * r_ic
        cash *= (1 + liq[i])
        eq = pos.sum() + cash
        out.append((cl.index[i], eq / eq_prev - 1, cost / eq_prev, float(np.abs(trade).sum() / max(eq_open, 1e-9))))
        eq_prev = eq
    return pd.DataFrame(out, columns=["date", "ret", "cost_frac", "turnover"]).set_index("date")


# =============================================================================
# PHASE 1 main
# =============================================================================
def month_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of each month (decision at that close, fill next open)."""
    s = index.to_series()
    return s.groupby(index.to_period("M")).transform("max") == s


def random_null(close: pd.DataFrame, opn: pd.DataFrame, k: int = 8, n_draws: int = 200, seed: int = 11,
                start: str = "2017-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vol 6 Ch.5 null: every month pick k names AT RANDOM from the same (survivorship-biased)
    universe, equal weight, decide at the month-end close, enter next open, pay CNC costs +
    10 bp slippage + DP fee on every exit. Returns (daily returns of each draw, summary per draw)."""
    rng = np.random.default_rng(seed)
    stocks = [c for c in close.columns if c != INDEX]
    C, O = close[stocks].loc[start:], opn[stocks].loc[start:]
    R = C.pct_change().fillna(0.0).to_numpy()
    ON = (O / C.shift(1) - 1).fillna(0.0).to_numpy()            # overnight leg of the entry day
    dates = C.index
    # (fix) decide at the month-END close like the strategy books, fill at the next open
    month_last = np.r_[dates.month[1:] != dates.month[:-1], False]
    rebal = np.where(month_last)[0]
    rt_cost = cnc_rate("buy") + cnc_rate("sell") + 2 * CFG["slip_floor_bps"] / 1e4
    dp_rs = CFG["dp_fee_per_sell"] * k          # Rs per full turnover; converted with the CURRENT equity (fix)
    avail = C.notna().to_numpy()
    G = np.ones((len(dates), n_draws))
    T = len(dates)
    buy_only = cnc_rate("buy") + CFG["slip_floor_bps"] / 1e4
    for j in range(n_draws):
        held = None
        w_end = None
        for a, b in zip(rebal, list(rebal[1:]) + [T]):
            # decided at the close of day a; old basket is sold / new basket bought at the open of a+1
            if a + 1 >= T:
                break
            pool = np.where(avail[a])[0]
            if len(pool) < k:
                continue
            pick = rng.choice(pool, k, replace=False)
            turn = 1.0 if held is None else 1 - len(set(pick) & set(held)) / k
            if held is not None:                       # old basket's overnight leg (drifted weights) into the open of a+1
                ww = w_end if w_end is not None else np.full(len(held), 1 / len(held))
                G[a + 1, j] *= 1 + float(ww @ ON[a + 1, held])
            first = held is None
            held = pick
            oc = (1 + R[a + 1, pick]) / (1 + ON[a + 1, pick])
            eq_now = CFG["capital"] * float(np.prod(G[:a + 1, j]))
            fee = buy_only if first else turn * rt_cost + turn * dp_rs / max(eq_now, 1.0)
            G[a + 1, j] *= oc.mean() * (1 - fee)
            w_end = oc / oc.sum()
            if a + 2 < T:
                # equal weight at entry, buy-and-hold inside the month (weights drift)
                rel = np.cumprod(1 + R[a + 2:b + 1][:, pick], axis=0) * oc
                val = np.vstack([oc[None, :], rel])
                G[a + 2:b + 1, j] *= val[1:].mean(1) / val[:-1].mean(1)
                w_end = val[-1] / val[-1].sum()
            if b < T:                                  # day b's close is held; the overnight into b+1 is
                pass                                   # charged in the next block
    out = G - 1
    D = pd.DataFrame(out, index=dates)
    m = (1 + D).groupby(D.index.to_period("M")).prod() - 1
    summ = pd.DataFrame({"sharpe": m.mean() / m.std() * np.sqrt(12), "cagr": (1 + D).prod() ** (252 / len(D)) - 1,
                         "max_dd": [(((1 + D[c]).cumprod()) / ((1 + D[c]).cumprod().cummax()) - 1).min() for c in D]})
    return D, summ


def main():
    ap = base_args("Phase 1: data, costs, margin model, baselines")
    args = ap.parse_args()
    data = data_dir(args.data)
    out = out_dir("phase1_data_costs")
    t0 = time.time()
    print(f"[P1] loading daily panel from {data}")
    P = load_daily_panel(data)
    close, opn, high, low = P["adj_close"], P["adj_open"], P["adj_high"], P["adj_low"]
    stocks = [c for c in close.columns if c != INDEX]
    dates = close.index
    liq = liquid_returns(data, dates)

    # ---------------- universe & liquidity statistics ------------------------
    rets = close[stocks].pct_change()
    sig = rets.iloc[-60:].std()
    hs = half_spread_bps(high[stocks], low[stocks], P["turnover"][stocks])
    cs_diag = (corwin_schultz_spread(high[stocks], low[stocks]).rolling(60, min_periods=20).mean().iloc[-1] * 1e4 / 2)
    to = P["turnover"][stocks].iloc[-60:].median()
    raw_last = P["raw_close"][stocks].ffill().iloc[-1]
    uni = pd.DataFrame({
        "last_raw_price": raw_last.round(2),
        "median_turnover_cr": (to / 1e7).round(1),
        "daily_vol_60d": sig.round(4),
        "half_spread_bps_model": hs.iloc[-1].round(2),
        "half_spread_bps_CS_diag": cs_diag.round(2),
        "pledge_haircut_est": pd.Series(haircut(sig.to_numpy()), index=stocks).round(3),
        "first_date": close[stocks].apply(lambda s: s.first_valid_index()).dt.date,
    })
    # affordability with Rs 1 lakh: per-position budget for k = 4 names (25%)
    budget = CFG["capital"] / 4
    uni["shares_at_25k"] = np.floor(budget / uni["last_raw_price"])
    uni["rounding_error_pct"] = ((budget - uni["shares_at_25k"] * uni["last_raw_price"]) / budget * 100).round(2)
    uni["affordable_k4"] = uni["shares_at_25k"] >= 1
    uni["mis_impact_bps_at_1L"] = (CFG["impact_coef"] * sig * np.sqrt(1e5 / to.replace(0, np.nan)) * 1e4).round(3)
    uni.sort_values("median_turnover_cr", ascending=False).to_csv(out / "universe_stats.csv")

    # ---------------- cost tables -------------------------------------------
    rows = []
    for ticket in [5_000, 10_000, 25_000, 50_000, 100_000, 500_000]:
        for prod in ["CNC", "MIS"]:
            rt = roundtrip_pct(ticket, prod, half_spread_bps=float(hs.iloc[-1].median()))
            rows.append({"ticket_rs": ticket, "product": prod,
                         "roundtrip_cost_pct": round(rt * 100, 4),
                         "breakeven_edge_bps": round(rt * 1e4, 2),
                         "of_which_pledge_pct": round((CFG["pledge_fee"] / ticket * 100) if prod == "CNC" else 0, 4)})
    costs = pd.DataFrame(rows)
    costs.to_csv(out / "cost_table.csv", index=False)
    # margin capacity under 50:50 for several distributions
    mrows = []
    h_med = float(np.median(uni["pledge_haircut_est"]))
    for lab, stock_w, liq_w, cash_w in [("100% cash", 0, 0, 1), ("80% CNC / 20% cash (Goal 4)", .8, 0, .2),
                                        ("100% CNC pledged (Goal 3)", 1, 0, 0), ("50% CNC / 50% liquid ETF", .5, .5, 0),
                                        ("100% liquid ETF pledged", 0, 1, 0), ("60% CNC / 40% liquid", .6, .4, 0)]:
        cap = CFG["capital"]
        m = usable_margin(cash_w * cap, liq_w * cap, stock_w * cap * (1 - h_med))
        mrows.append({"distribution": lab, "usable_margin_rs": round(m), "mis_buying_power_rs": round(5 * m),
                      "overnight_equity_beta": stock_w})
    margins = pd.DataFrame(mrows)
    margins.to_csv(out / "margin_capacity.csv", index=False)

    # ---------------- baselines ---------------------------------------------
    nifty = close[INDEX].pct_change()
    ew_w = close[stocks].notna().astype(float)
    ew_w = ew_w.div(ew_w.sum(axis=1), axis=0)
    month_end = ew_w.index.to_series().groupby(ew_w.index.to_period("M")).transform("max") == ew_w.index
    ew_w = ew_w.where(month_end).ffill().fillna(0)
    # top-4 short-term momentum rotated every 2 days (naive Goal 3) - shows cost drag
    mom5 = close[stocks].pct_change(5)
    rank = mom5.rank(axis=1, ascending=False)
    rot = (rank <= 4).astype(float) / 4
    every2 = pd.Series(np.arange(len(rot)) % 2 == 0, index=rot.index)
    rot = rot.where(every2, np.nan).ffill().fillna(0)

    print("[P1] simulating baselines ...")
    sims = {
        "NIFTY50 buy&hold": nifty,
        "Liquid ETF only (proxy)": liq,
        "EW NIFTY100 monthly (CNC)": simulate_weights(ew_w, opn, close, liq, pledge=False, half_spread=hs,
                                                      rebal_mask=month_end_mask(close.index))["ret"],
        "Naive 2-day rotation top4 mom (CNC+pledge)": simulate_weights(rot, opn, close, liq, pledge=True, half_spread=hs)["ret"],
    }
    print("[P1] random-pick null (200 draws of 8 names / month) ...")
    nullD, nsum = random_null(close, opn, k=8, n_draws=60 if args.quick else 200)
    nsum.to_csv(out / "null_random_pick_draws.csv", index_label="draw")
    order = nsum["cagr"].sort_values()
    pick = {q: order.index[int(q * (len(order) - 1))] for q in (0.05, 0.5, 0.95)}
    pd.DataFrame({f"null_p{int(q*100):02d}": nullD[c] for q, c in pick.items()}).to_csv(out / "null_paths.csv")
    sims["Random 8 names monthly (null median)"] = nullD[pick[0.5]]
    (out / "null_summary.json").write_text(json.dumps({
        "n_draws": len(nsum), "median_sharpe": round(float(nsum["sharpe"].median()), 3),
        "sd_sharpe": round(float(nsum["sharpe"].std()), 3), "p05_sharpe": round(float(nsum["sharpe"].quantile(.05)), 3),
        "p95_sharpe": round(float(nsum["sharpe"].quantile(.95)), 3), "median_cagr": round(float(nsum["cagr"].median()), 4),
        "note": "Random picking from today's NIFTY100 list. Theory says ~0 skill; anything above the market's own "
                "Sharpe is survivorship bias. Strategies are graded on their EXCESS over this median (Vol 6)."}, indent=2))
    cards = [scorecard(s.loc["2017-01-01":], k) for k, s in sims.items()]
    best = save_scorecards(out, cards, "phase1_data_costs", series={k: v.loc["2017-01-01":] for k, v in sims.items()},
                           notes="Baselines only. Probabilities from 12-month stationary bootstrap of daily returns.")
    pd.DataFrame({k: to_monthly(v.loc["2017-01-01":]) for k, v in sims.items()}).to_csv(out / "baseline_monthly_returns.csv")
    plot_equity({k: v.loc["2017-01-01":] for k, v in sims.items()}, out / "baseline_equity.png", "Phase 1 baselines (log)")
    multi_panel({k: v.loc["2017-01-01":] for k, v in sims.items()}, out / "baselines_all_in_one.png",
                "Phase 1 baselines - all lines in one chart",
                band=(nullD[pick[0.05]], nullD[pick[0.95]]))
    plot_monthly_hist({k: v.loc["2017-01-01":] for k, v in sims.items() if "Liquid" not in k},
                      out / "baseline_monthly_hist.png", "Monthly returns vs mandate band")
    fig, ax = plt.subplots(figsize=(9, 4))
    sub = costs[costs["ticket_rs"] <= 100_000].pivot(index="ticket_rs", columns="product", values="breakeven_edge_bps")
    sub.plot(kind="bar", ax=ax)
    ax.set_ylabel("round-trip break-even (bps)")
    ax.set_title("Flattrade round-trip cost incl. pledge fee & spread")
    fig.tight_layout()
    fig.savefig(out / "cost_breakeven.png", dpi=120)
    plt.close(fig)

    # ---------------- report -------------------------------------------------
    sc_df = pd.DataFrame(cards)[SC_COLS]
    rc_ = simulate_weights(rot, opn, close, liq, pledge=True, half_spread=hs)["cost_frac"].loc["2017":]
    rot_cost = rc_.sum() / (len(rc_) / TRADING_DAYS)
    md = f"""# Phase 1 - Data, costs, margin model, baselines

Data: `{data}` | instruments: {len(stocks)} stocks + NIFTY50 | daily history {dates[0].date()} -> {dates[-1].date()}
Capital: Rs {CFG['capital']:,.0f} | Broker: Flattrade (zero brokerage, pledge Rs 20 + GST per order)

## Key facts that shape the strategy
* **CNC round trip costs ~{costs.query('ticket_rs==25000 and product=="CNC"').roundtrip_cost_pct.iloc[0]:.3f}% on a Rs 25k ticket**
  (STT 0.1% each side dominates; pledge fee alone = {CFG['pledge_fee']/25000*100:.3f}%). A 2-day CNC hold must earn more than this.
* **MIS round trip ~{costs.query('ticket_rs==100000 and product=="MIS"').roundtrip_cost_pct.iloc[0]:.3f}%** -> intraday edge must beat ~{costs.query('ticket_rs==100000 and product=="MIS"').breakeven_edge_bps.iloc[0]:.1f} bps per trade.
* The naive "rotate top-4 every 2 days" (Goal 3 taken literally) pays ~{rot_cost*100:.1f}% of capital per year in costs.
* **50:50 rule** (Flattrade): pledged stocks are usable as margin only up to the amount of cash + cash-equivalent
  collateral. Pledging a liquid ETF (cash-equivalent) unlocks the stock collateral. Margin capacity by distribution:

{md_table(margins, 0)}

* With Rs 1 lakh, {int((~uni['affordable_k4']).sum())} stocks cannot be bought at a 25% position size; median rounding error {uni['rounding_error_pct'].median():.1f}%.

## Baseline scorecards (2017 -> today)
{md_table(sc_df)}

**Best baseline:** {best.get('strategy')} - probability of profitability (12m) {best.get('P_profitable_12m')},
predicted mean monthly return {best.get('pred_monthly_mean')}, P(all 12 months >= -0.5%) {best.get('P_all_12_months_ge_floor')}.

Interpretation: a long-only stock book carries ~4-6% monthly volatility, so the -0.5% monthly floor
cannot be met by stock beta alone. Later phases must (a) gate overnight stock exposure by regime,
(b) earn most of the return intraday (flat overnight) and (c) park idle capital in pledged liquid collateral.

Files: universe_stats.csv, cost_table.csv, margin_capacity.csv, baseline_monthly_returns.csv,
scorecards.csv, scorecard.json, baseline_equity.png, baseline_monthly_hist.png, cost_breakeven.png
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    print_scorecard(best, "phase1_data_costs")
    print(f"[P1] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
