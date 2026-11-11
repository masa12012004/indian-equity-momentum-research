#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 6 - Monthly CNC portfolio engine (Vols 5-6) and the comparison of
many books as separate lines in ONE chart.

The book (CNC delivery, long only, buy side only, no intraday)
---------------------------------------------------------------
  decision    at the close of the last trading day of each month, orders fill at the NEXT OPEN
  ranking     score from phase 5 (LightGBM, or the DL ensemble if it was adopted) or the
              Level-1 composite from phase 3 (both are grid options)
  selection   top-k with a hold buffer (a held name stays while its rank <= 2k), names whose
              60-day correlation with an already chosen name is > 0.85 are skipped
  weights     MV  : max mu'w - (8/2) w'Sigma w, 5% <= w <= 15%, iterative cardinality 8-12,
                    mu = (pct rank - 0.5) x meta confidence x 21-day vol x 0.25 shrinkage,
                    Sigma = 252-day covariance cleaned with random-matrix theory (MP clipping)
              HRP : hierarchical risk parity (Lopez de Prado 2016) on the cleaned covariance
              EW  : equal weight (the optimiser must beat this or EW is used)
  sizing      stock fraction = min(14% / predicted book vol, 1 - regime floor) clipped to
              [70%, 100%] (cash never above 30%); whole shares (raw price), min position
              Rs 4,000, greedy one-share top-up of the most under-weight name
  idle cash   swept into a liquid ETF (LIQUIDBEES proxy) - or left idle in the "pure CNC" line
  stops       tighter of entry - 2 x ATR14 and entry x 0.88; a gap through the stop fills at the
              open, an intraday touch fills at stop - 25 bp; lower-circuit lock (high == low)
              walks forward to the first tradable day; upper circuit at the open blocks a buy
  costs       Flattrade CNC statutory charges, slippage 10 bp floor + Amihud + vol add-on,
              DP fee Rs 15.34 per scrip per sell day, costs only on the traded delta,
              no-trade band Rs 2,000 for names that stay in the book
Validation windows: selection 2019-07 .. 2021-08-02 (hyper-parameters chosen here ONLY),
Windows: first 70% of the backtest days = TRAIN (all selection), last 30% = TEST.
Improve round: research scores (vol-normalised momentum, Frog-in-the-Pan, low-vol blend) and
an absolute-momentum / trend exit (Faber SMA200, Antonacci dual momentum) added to the grid.
Output folder: phase6_portfolio_backtest/
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

import p1_data_costs as L
import p3_features as F3

plt = L.plt
PHASE = "phase6_portfolio_backtest"
# Windows are set in main(): first 70% of the backtest days = TRAIN/selection, last 30% = TEST.
TRAIN_FRAC = 0.70
SEL = (pd.Timestamp("2019-07-01"), pd.Timestamp("2024-07-29"))
TEST = (pd.Timestamp("2024-07-30"), pd.Timestamp("2100-01-01"))
VALID = (pd.Timestamp("2100-01-01"), pd.Timestamp("2100-01-02"))      # (kept for compatibility; empty)
ERAS = {"2019-20 pre-COVID": ("2019-07-01", "2020-01-31"), "2020 COVID crash": ("2020-02-01", "2020-05-31"),
        "2020-21 melt-up": ("2020-06-01", "2021-12-31"), "2022 rates shock": ("2022-01-01", "2022-12-31"),
        "2023-24 bull": ("2023-01-01", "2024-09-30"), "2024-25 correction": ("2024-10-01", "2025-03-31"),
        "2025-26": ("2025-04-01", "2026-12-31")}


SECTOR = {
    **dict.fromkeys(["AXISBANK", "BANKBARODA", "CANBK", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "PNB", "SBIN", "UNIONBANK"], "Banks"),
    **dict.fromkeys(["BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", "CHOLAFIN", "HDFCAMC", "HDFCLIFE", "SBILIFE", "SHRIRAMFIN",
                     "MUTHOOTFIN", "JIOFIN", "PFC", "IRFC", "BSE", "TATACAP"], "Financials"),
    **dict.fromkeys(["HCLTECH", "INFY", "TCS", "TECHM", "WIPRO", "LTM"], "IT"),
    **dict.fromkeys(["BAJAJ-AUTO", "EICHERMOT", "MARUTI", "M_M", "TVSMOTOR", "HYUNDAI", "TMPV", "TMCV", "MOTHERSON", "BOSCHLTD"], "Auto"),
    **dict.fromkeys(["CIPLA", "DIVISLAB", "DRREDDY", "SUNPHARMA", "TORNTPHARM", "ZYDUSLIFE", "APOLLOHOSP", "MAXHEALTH"], "Health"),
    **dict.fromkeys(["BRITANNIA", "GODREJCP", "HINDUNILVR", "ITC", "NESTLEIND", "TATACONSUM", "VBL"], "FMCG"),
    **dict.fromkeys(["ASIANPAINT", "PIDILITIND", "TITAN", "TRENT", "DMART", "ETERNAL", "INDIGO"], "ConsumerDisc"),
    **dict.fromkeys(["BPCL", "IOC", "ONGC", "GAIL", "RELIANCE", "COALINDIA"], "Energy"),
    **dict.fromkeys(["NTPC", "POWERGRID", "TATAPOWER", "ADANIPOWER", "ADANIGREEN", "ADANIENSOL"], "Power"),
    **dict.fromkeys(["HINDALCO", "HINDZINC", "JINDALSTEL", "JSWSTEEL", "TATASTEEL", "VEDL", "VAML"], "Metals"),
    **dict.fromkeys(["ABB", "BEL", "CGPOWER", "CUMMINSIND", "HAL", "LT", "MAZDOCK", "POLYCAB", "POWERINDIA", "SIEMENS",
                     "SOLARINDS", "ENRIN"], "CapGoods"),
    **dict.fromkeys(["AMBUJACEM", "GRASIM", "ULTRACEMCO"], "Cement"),
    **dict.fromkeys(["BHARTIARTL", "IDEA"], "Telecom"),
    **dict.fromkeys(["DLF", "ADANIPORTS", "ADANIENT"], "InfraRealty"),
}


@dataclass
class Params:
    name: str = "base"
    score: str = "final"          # final (phase 5) | composite (phase 3)
    opt: str = "MV"               # MV | HRP | EW
    k: int = 10
    stops: bool = True
    sweep: bool = True            # idle cash in liquid ETF
    regime: bool = True           # regime floor + vol target
    pledge: bool = False          # pledge new names (only needed for the MIS overlay)
    risk_aversion: float = 8.0
    w_min: float = 0.05
    w_max: float = 0.15
    shrink: float = 0.25
    corr_cap: float = 0.85
    vol_target: float = 0.14
    cash_cap: float = 0.30
    atr_mult: float = 2.0
    stop_pct: float = 0.12
    stop_slip_bps: float = 25.0
    min_pos: float = 4000.0
    band_rs: float = 2000.0
    cost_mult: float = 1.0
    slip_floor_bps: float = 10.0
    liquid_yield_override: float | None = None
    trend_exit: str = "none"      # none | sma200 | dual : if the market state is risk-off at the
                                  # month-end decision, hold NO stocks (everything in the liquid ETF)
    sector_cap: int = 0           # phase 11: max names per sector (0 = no cap)
    abs_mom_filter: bool = False  # phase 11: only buy names whose 12-month return beats the liquid ETF
    skip_hot: bool = False        # phase 11: do not OPEN a position in a name whose 1-month return is in the
                                  # top 10% of the universe (short-term reversal, Jegadeesh 1990)
    riskoff_asset: str = "liquid" # phase 11: "liquid" or "gold_dual" (hold GOLDBEES while out of stocks if its
                                  # 12-month return beats the liquid ETF - Antonacci dual momentum)
    rebal_months: int = 1         # full rebalance every N month-ends (1 monthly, 3 quarterly, 6 semi-annual);
                                  # the trend exit is still checked every month-end
    daily_exposure: str = "none"  # phase 9: name of a DAILY exposure series in mk.exposure (values 0/0.5/1).
                                  # Checked every day; a change rebuilds the book (re-ranked) at the next open.


# ----------------------------------------------------------------------------
# covariance cleaning and weights
# ----------------------------------------------------------------------------
def rmt_cov(R: np.ndarray) -> np.ndarray:
    T, N = R.shape
    sd = R.std(0, ddof=1); sd[sd == 0] = 1e-4
    C = np.corrcoef(R.T)
    C = np.nan_to_num(C); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C)
    lam_plus = (1 + np.sqrt(N / T)) ** 2
    noise = ev < lam_plus
    if noise.any():
        ev[noise] = ev[noise].mean()
    Cc = V @ np.diag(ev) @ V.T
    d = np.sqrt(np.diag(Cc)); Cc = Cc / np.outer(d, d)
    return Cc * np.outer(sd, sd)


def hrp(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None)); np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), "single"))
    w = np.ones(len(cov)); clusters = [list(order)]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for a, b in zip(clusters[::2], clusters[1::2]):
            va = _ivp_var(cov, a); vb = _ivp_var(cov, b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha; w[b] *= 1 - alpha
    return w / w.sum()


def _ivp_var(cov, idx):
    c = cov[np.ix_(idx, idx)]
    ivp = 1 / np.diag(c); ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def clip_weights(w: np.ndarray, lo: float, hi: float, iters: int = 50) -> np.ndarray:
    w = w / w.sum()
    for _ in range(iters):
        w = np.clip(w, lo, hi); w = w / w.sum()
        if (w <= hi + 1e-9).all() and (w >= lo - 1e-9).all():
            break
    return w


def mv_weights(mu: np.ndarray, cov: np.ndarray, lam: float, lo: float, hi: float) -> np.ndarray:
    n = len(mu)
    f = lambda w: -(mu @ w - 0.5 * lam * w @ cov @ w)
    g = lambda w: -(mu - lam * cov @ w)
    res = minimize(f, np.full(n, 1 / n), jac=g, method="SLSQP", bounds=[(lo, hi)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 200, "ftol": 1e-10})
    return res.x if res.success else np.full(n, 1 / n)


# ----------------------------------------------------------------------------
# market data container
# ----------------------------------------------------------------------------
class Market:
    def __init__(self, data):
        P = L.load_daily_panel(data, fields=("adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
                                             "raw_open", "raw_close", "turnover"))
        self.stocks = [c for c in P["adj_close"].columns if c != L.INDEX]
        self.dates = P["adj_close"].index
        g = lambda k: P[k][self.stocks].to_numpy(float)
        self.O, self.H, self.Lo, self.C = g("adj_open"), g("adj_high"), g("adj_low"), g("adj_close")
        self.rawO, self.rawC, self.V = g("raw_open"), g("raw_close"), g("adj_volume")
        # raw (unadjusted) prices exist only where the cleaner had intraday data (2023-10 onwards);
        # before that whole-share sizing uses the adjusted price (approximation, see ADJUSTMENTS.md)
        self.rawO = np.where(np.isfinite(self.rawO), self.rawO, self.O)
        self.rawC = np.where(np.isfinite(self.rawC), self.rawC, self.C)
        self.nifty = P["adj_close"][L.INDEX]
        self.ret = np.vstack([np.full(len(self.stocks), np.nan), self.C[1:] / self.C[:-1] - 1])
        tr = np.nanmax(np.stack([self.H - self.Lo, np.abs(self.H - np.roll(self.C, 1, 0)),
                                 np.abs(self.Lo - np.roll(self.C, 1, 0))]), 0)
        self.atr = pd.DataFrame(tr).rolling(14).mean().to_numpy()
        self.liq = L.liquid_returns(data, self.dates).to_numpy()
        lr = np.log(self.C)
        self.vol63 = pd.DataFrame(np.diff(lr, axis=0, prepend=np.nan)).rolling(63, min_periods=40).std().to_numpy() * np.sqrt(252)
        to = g("turnover")
        am = pd.DataFrame(np.abs(self.ret) / (to / 1e7)).replace([np.inf, -np.inf], np.nan)
        self.amihud = am.rolling(21, min_periods=10).mean().to_numpy()
        self.me = L.month_end_mask(self.dates).to_numpy()
        Cdf = pd.DataFrame(self.C, index=self.dates)
        self.ret21 = (Cdf / Cdf.shift(21) - 1).to_numpy()
        self.ret252 = (Cdf / Cdf.shift(252) - 1).to_numpy()
        liq_s = pd.Series(self.liq, index=self.dates)
        self.liq252 = ((1 + liq_s).rolling(252, min_periods=200).apply(np.prod, raw=True) - 1).to_numpy()
        self.sector = {n: SECTOR.get(n, "Other") for n in self.stocks}
        gp = L.find(data, "1D", "GOLDBEES")
        if gp is not None:
            g = L._read(gp)
            g = g.set_index(g["datetime"].dt.normalize())["adj_close"].reindex(self.dates).ffill()
            self.gold = g.pct_change().fillna(0.0).to_numpy()
            self.gold252 = (g / g.shift(252) - 1).to_numpy()
        else:
            self.gold = None
        self.col = {n: j for j, n in enumerate(self.stocks)}
        self.P = P

    def slip(self, i, j, extra_floor=None):
        k = max(i - 1, 0)                     # (fix) only data known before today's open
        bps = L.slippage_bps(self.amihud[k, j], self.vol63[k, j])
        if extra_floor is not None:
            bps += extra_floor - L.CFG["slip_floor_bps"]
        return bps / 1e4


# ----------------------------------------------------------------------------
# the engine
# ----------------------------------------------------------------------------
def build_targets(mk: Market, i: int, score_row: pd.Series, meta_row: pd.Series | None, held: list[str],
                  stock_frac_regime: float, p: Params):
    s = score_row.dropna()
    s = s[[n for n in s.index if n in mk.col and np.isfinite(mk.C[i, mk.col[n]])]]
    if len(s) < p.k:
        return {}, 0.0, {}
    if p.abs_mom_filter:                   # stock-level absolute momentum (time-series momentum)
        hurdle = mk.liq252[i] if np.isfinite(mk.liq252[i]) else 0.05
        s = s[[n for n in s.index if np.isfinite(mk.ret252[i, mk.col[n]]) and mk.ret252[i, mk.col[n]] > hurdle]]
        if len(s) < max(4, p.k // 2):
            return {}, 0.0, {}
    s = s.sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    pct = s.rank(pct=True)
    keep = [n for n in held if n in rank and rank[n] <= 2 * p.k]
    ordered = keep + [n for n in s.index if n not in keep]
    lo_i = max(0, i - 251)
    R = mk.ret[lo_i:i + 1]
    chosen = []
    hot_cut = np.nanquantile(mk.ret21[i], 0.9) if p.skip_hot else np.inf
    sec_count: dict[str, int] = {}
    for n in ordered:
        j = mk.col[n]
        if p.skip_hot and n not in held and np.isfinite(mk.ret21[i, j]) and mk.ret21[i, j] >= hot_cut:
            continue
        if p.sector_cap and sec_count.get(mk.sector[n], 0) >= p.sector_cap:
            continue
        if chosen:
            r60 = R[-60:, [mk.col[c] for c in chosen] + [j]]
            ok = np.isfinite(r60).all(1)
            if ok.sum() > 40:
                cc = np.corrcoef(r60[ok].T)[-1, :-1]
                if np.nanmax(cc) > p.corr_cap:
                    continue
        chosen.append(n)
        sec_count[mk.sector[n]] = sec_count.get(mk.sector[n], 0) + 1
        if len(chosen) >= (12 if p.opt == "MV" else p.k) + (4 if p.opt == "MV" else 0):
            break
    cols = [mk.col[n] for n in chosen]
    Rs = R[:, cols]
    good = np.isfinite(Rs).mean(0) > 0.8
    chosen = [n for n, g in zip(chosen, good) if g]
    cols = [mk.col[n] for n in chosen]
    Rs = np.nan_to_num(R[:, cols])
    if len(chosen) < (4 if (p.abs_mom_filter or p.sector_cap) else 5):
        return {}, 0.0, {}
    cov_d = rmt_cov(Rs)
    if p.opt == "EW":
        names = chosen[:p.k]
        w = np.full(len(names), 1 / len(names))
    elif p.opt == "HRP":
        names = chosen[:p.k]
        idx = [chosen.index(n) for n in names]
        w = clip_weights(hrp(cov_d[np.ix_(idx, idx)]), p.w_min, p.w_max)
    else:
        cand = chosen[:]
        meta = (meta_row.reindex(cand).fillna(0.5).to_numpy() if meta_row is not None else np.full(len(cand), 0.5))
        vol_h = np.sqrt(np.diag(cov_d) * 21)
        mu_all = (pct.reindex(cand).to_numpy() - 0.5) * (2 * meta) * vol_h * p.shrink
        names = cand[:12]
        for _ in range(6):                                   # iterative cardinality 8..12
            idx = [cand.index(n) for n in names]
            w = mv_weights(mu_all[idx], cov_d[np.ix_(idx, idx)] * 21, p.risk_aversion, 0.0, p.w_max)
            names2 = [n for n, x in zip(names, w) if x >= p.w_min - 1e-6]
            if len(names2) < 8:
                extra = [n for n in cand if n not in names2][: 8 - len(names2)]
                names2 = names2 + extra
            if names2 == names:
                break
            names = names2
        idx = [cand.index(n) for n in names]
        w = mv_weights(mu_all[idx], cov_d[np.ix_(idx, idx)] * 21, p.risk_aversion, p.w_min, p.w_max)
    idx = [chosen.index(n) for n in names]
    port_vol = float(np.sqrt(w @ cov_d[np.ix_(idx, idx)] @ w * 252))
    frac = 1.0
    if p.regime:
        frac = min(1.0, p.vol_target / max(port_vol, 1e-4), stock_frac_regime)
        frac = float(np.clip(frac, 1 - p.cash_cap, 1.0))
    diag = {"port_vol_pred": port_vol, "n_names": len(names), "stock_frac": frac}
    return dict(zip(names, w)), frac, diag


def run(mk: Market, p: Params, score: pd.DataFrame, meta: pd.DataFrame | None, regime: pd.DataFrame,
        start: pd.Timestamp, end: pd.Timestamp | None = None, log_trades: bool = False):
    cap = L.CFG["capital"]
    buy_r, sell_r = L.cnc_rate("buy") * p.cost_mult, L.cnc_rate("sell") * p.cost_mult
    dp = L.CFG["dp_fee_per_sell"]; pledge_fee = L.CFG["pledge_fee"]
    dates = mk.dates
    i0 = int(np.searchsorted(dates, start)); i1 = len(dates) if end is None else int(np.searchsorted(dates, end, "right"))
    liq = mk.liq.copy()
    if p.liquid_yield_override is not None:
        liq[:] = (1 + p.liquid_yield_override) ** (1 / 252) - 1
    score = score.reindex(dates)
    meta = meta.reindex(dates) if meta is not None else None
    sfrac = regime["stock_frac"].reindex(dates).fillna(1.0).to_numpy()

    E_arr = (np.asarray(mk.exposure[p.daily_exposure], dtype=float)
             if p.daily_exposure != "none" else None)
    cur_e = None
    trend_off = False
    me_count = 0
    gold_on = False
    cash = cap                       # liquid-ETF (or idle) balance
    pos: dict[str, dict] = {}        # name -> {units (adj), shares (raw), entry_adj, stop_adj, entry_date}
    pending = None
    eq_prev = cap
    rows, trades = [], []
    for i in range(i0, i1):
        if mk.me[i]:
            me_count += 1
        if mk.me[i - 1] if i > 0 else False:
            # decided at yesterday's month-end close: idle cash sits in GOLDBEES while the book is out of
            # stocks and gold's 12-month return beats the liquid ETF; switching costs ~10 bp
            want = (p.riskoff_asset == "gold_dual" and mk.gold is not None and not pos
                    and np.isfinite(mk.gold252[i - 1]) and mk.gold252[i - 1] > (mk.liq252[i - 1] if np.isfinite(mk.liq252[i - 1]) else 0.05))
            if want != gold_on:
                cash *= 1 - 0.001
            gold_on = want
        day_cost = 0.0
        sold_today: set[str] = set()
        # ------------------------------------------------ open: execute pending rebalance
        if pending is not None:
            tgt_w, frac = pending
            pending = None
            val_open = {n: d["units"] * mk.O[i, mk.col[n]] if np.isfinite(mk.O[i, mk.col[n]]) else d["units"] * mk.C[i - 1, mk.col[n]]
                         for n, d in pos.items()}
            eq_open = cash + sum(val_open.values())
            budget = frac * eq_open
            tgt_val = {n: w * budget for n, w in tgt_w.items()}
            tgt_val = {n: v for n, v in tgt_val.items() if v >= p.min_pos}
            # sells first (full exits and down-sizing)
            for n in list(pos):
                j = mk.col[n]; o = mk.O[i, j]
                if not np.isfinite(o):
                    continue
                locked = mk.H[i, j] == mk.Lo[i, j] and o < mk.C[i - 1, j] * 0.9
                if locked:
                    continue
                cur = val_open[n]; tv = tgt_val.get(n, 0.0)
                if tv == 0.0 or (cur - tv) > p.band_rs:
                    raw_px = mk.rawO[i, j] if np.isfinite(mk.rawO[i, j]) else mk.O[i, j]
                    # (fix) shares held are derived from the adjusted units at TODAY's raw price, so
                    # splits / the 2023-10 switch to raw prices cannot desynchronise the share count
                    sh_held = max(int(round(pos[n]["units"] * o / raw_px)), 1)
                    sh_target = int(tv // raw_px) if tv > 0 else 0
                    sh_sell = sh_held - sh_target
                    if sh_sell <= 0:
                        continue
                    frac_sold = min(sh_sell / sh_held, 1.0)
                    value = cur * frac_sold
                    c = value * (sell_r + mk.slip(i, j, p.slip_floor_bps) * p.cost_mult) + (dp if n not in sold_today else 0.0)
                    sold_today.add(n)
                    cash += value - c; day_cost += c
                    if log_trades:
                        trades.append({"date": dates[i], "name": n, "side": "SELL", "shares": sh_sell, "price_raw": raw_px,
                                       "value": value, "cost": c, "reason": "rebalance",
                                       "pnl_pct": mk.O[i, j] / pos[n]["entry_adj"] - 1})
                    pos[n]["units"] *= (1 - frac_sold)
                    if frac_sold >= 1.0:
                        del pos[n]
            # buys
            buys = []
            for n, tv in sorted(tgt_val.items(), key=lambda x: -x[1]):
                j = mk.col[n]; o = mk.O[i, j]
                if not np.isfinite(o) or o >= mk.C[i - 1, j] * 1.195:        # upper circuit at the open
                    continue
                cur = pos[n]["units"] * o if n in pos else 0.0
                if tv - cur > (p.band_rs if n in pos else 0.0):
                    buys.append((n, j, tv - cur))
            for n, j, need in buys:
                raw_px = mk.rawO[i, j]; slip = mk.slip(i, j, p.slip_floor_bps) * p.cost_mult
                px = raw_px * (1 + slip)
                if not (np.isfinite(px) and px > 0 and np.isfinite(cash)):
                    continue
                sh = int(max(0.0, min(need, cash - 100)) // (px * (1 + buy_r)))
                if np.isfinite(mk.V[i - 1, j]):                   # volume cap: 1% of YESTERDAY's (raw) share volume
                    sh = min(sh, int(0.01 * mk.V[i - 1, j] * mk.O[i, j] / raw_px))
                if sh <= 0:
                    continue
                value = sh * raw_px
                c = value * (buy_r + slip) + (pledge_fee if (p.pledge and n not in pos) else 0.0)
                cash -= value + c; day_cost += c
                units = value / mk.O[i, j]
                if n in pos:
                    pos[n]["units"] += units
                else:
                    atr = mk.atr[i - 1, j]
                    stop = max(mk.O[i, j] * (1 - p.stop_pct), mk.O[i, j] - p.atr_mult * atr) if np.isfinite(atr) else mk.O[i, j] * (1 - p.stop_pct)
                    pos[n] = {"units": units, "entry_adj": mk.O[i, j], "stop_adj": stop, "entry_i": i}
                if log_trades:
                    trades.append({"date": dates[i], "name": n, "side": "BUY", "shares": sh, "price_raw": raw_px,
                                   "value": value, "cost": c, "reason": "rebalance", "pnl_pct": np.nan})
            # greedy one-share top-up towards the most under-weight name
            for _ in range(30):
                gaps = [(tgt_val[n] - pos[n]["units"] * mk.O[i, mk.col[n]], n) for n in tgt_val if n in pos]
                gaps = [g for g in gaps if g[0] > 0]
                if not gaps:
                    break
                g, n = max(gaps); j = mk.col[n]; raw_px = mk.rawO[i, j]
                px = raw_px * (1 + mk.slip(i, j, p.slip_floor_bps) * p.cost_mult) * (1 + buy_r)
                spare = cash - (1 - frac) * eq_open
                if px > spare or px > g * 1.5 or mk.O[i, j] >= mk.C[i - 1, j] * 1.195:
                    break
                cash -= px; day_cost += px - raw_px
                pos[n]["units"] += raw_px / mk.O[i, j]
        # ------------------------------------------------ intraday: protective stops
        if p.stops:
            for n in list(pos):
                d = pos[n]; j = mk.col[n]
                if d["entry_i"] > i or not np.isfinite(mk.O[i, j]):
                    continue
                if mk.H[i, j] == mk.Lo[i, j] and mk.C[i, j] < mk.C[i - 1, j]:   # lower-circuit lock: cannot sell
                    continue
                if d["entry_i"] < i and mk.O[i, j] <= d["stop_adj"]:      # gap through the stop (not on entry day)
                    px_adj, why = mk.O[i, j], "stop_gap"
                elif mk.Lo[i, j] <= d["stop_adj"]:
                    px_adj, why = d["stop_adj"] * (1 - p.stop_slip_bps / 1e4), "stop"
                else:
                    continue
                value = d["units"] * px_adj
                c = value * (sell_r + mk.slip(i, j, p.slip_floor_bps) * p.cost_mult) + (dp if n not in sold_today else 0.0)
                sold_today.add(n)
                cash += value - c; day_cost += c
                if log_trades:
                    trades.append({"date": dates[i], "name": n, "side": "SELL",
                                   "shares": int(round(d["units"] * mk.O[i, j] / mk.rawO[i, j])),
                                   "price_raw": px_adj * mk.rawO[i, j] / mk.O[i, j], "value": value, "cost": c,
                                   "reason": why, "pnl_pct": px_adj / d["entry_adj"] - 1})
                del pos[n]
        # ------------------------------------------------ close: mark to market, carry on cash
        if p.sweep:
            if gold_on and mk.gold is not None:
                cash *= 1 + mk.gold[i]
            else:
                cash *= 1 + liq[i]
        stock_val = 0.0
        for n, d in pos.items():
            c_ = mk.C[i, mk.col[n]]
            if not np.isfinite(c_):
                c_ = mk.C[i - 1, mk.col[n]] if np.isfinite(mk.C[i - 1, mk.col[n]]) else d["entry_adj"]
            stock_val += d["units"] * c_
        eq = cash + stock_val
        rows.append({"date": dates[i], "equity": eq, "ret": eq / eq_prev - 1, "stock_exposure": stock_val / eq,
                     "n_stocks": len(pos), "cost": day_cost, "cash": cash})
        eq_prev = eq
        # ------------------------------------------------ daily exposure overlay (phase 9)
        if E_arr is not None and i + 1 < i1:
            if mk.me[i] and p.trend_exit != "none":          # monthly trend exit can be combined with the overlay
                trend_off = bool(mk.trend_states[p.trend_exit][i])
            e = E_arr[i] if np.isfinite(E_arr[i]) else 1.0
            if p.trend_exit != "none" and trend_off:
                e = 0.0
            if cur_e is None and not mk.me[i]:
                e = cur_e                          # (fix) first book is built at the first month-end, like the others
            elif mk.me[i] or e != cur_e:
                if e <= 0:
                    pending = ({}, 0.0) if (pos or cur_e is None or cur_e > 0) else None
                else:
                    tw, frac, diag = build_targets(mk, i, score.iloc[i], meta.iloc[i] if meta is not None else None,
                                                   list(pos), sfrac[i] if p.regime else 1.0, p)
                    if tw:
                        pending = (tw, frac * e)
                cur_e = e
        # ------------------------------------------------ month-end decision
        elif mk.me[i] and i + 1 < i1 and p.trend_exit != "none" and mk.trend_states[p.trend_exit][i]:
            pending = ({}, 0.0)                       # absolute-momentum exit: sell everything next open
        elif mk.me[i] and i + 1 < i1 and ((me_count - 1) % max(p.rebal_months, 1) == 0 or not pos):
            row = score.iloc[i]
            mrow = meta.iloc[i] if meta is not None else None
            tw, frac, diag = build_targets(mk, i, row, mrow, list(pos), sfrac[i] if p.regime else 1.0, p)
            if tw:
                pending = (tw, frac)
    daily = pd.DataFrame(rows).set_index("date")
    return daily, pd.DataFrame(trades)


def load_scores(mk: Market):
    f5 = L.HERE / "phase5_deep_learning" / "oos_pred_final.csv"
    d = pd.read_csv(f5, parse_dates=["date"])
    final = d.pivot(index="date", columns="name", values="score")
    meta = d.pivot(index="date", columns="name", values="meta_conf")
    feats = pd.read_csv(L.HERE / "phase3_features" / "daily_features.csv", parse_dates=["date"])
    sel = json.loads((L.HERE / "phase3_features" / "feature_lists.json").read_text())["selected_stock_features"]
    sgn = pd.read_csv(L.HERE / "phase3_features" / "ic_table.csv").set_index("feature")["ic_train_yrank"].apply(np.sign)
    feats["composite"] = sum(feats[c] * sgn[c] for c in sel)
    comp = feats.pivot(index="date", columns="name", values="composite")
    out = {"final": final, "composite": comp}
    out.update(research_scores(mk))
    return out, meta


def _xs_z(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(1), axis=0).div(df.std(1).replace(0, np.nan), axis=0)


def research_scores(mk: Market) -> dict[str, pd.DataFrame]:
    """Strategy02 'improve' round - scores taken from published research, parameters NOT tuned.

      mom_vn   NSE Nifty200 Momentum 30 methodology: average cross-sectional z-score of the
               6-month and 12-month price return, each divided by the 1-year daily volatility.
      mom_fip  mom_vn minus 0.5 x z(information discreteness) - 'Frog in the Pan'
               (Da, Gurun & Warachka 2014): ID = sign(PRET) x (%neg days - %pos days) over the
               12-1 month window; low ID = continuous, more persistent momentum.
      lowvol   NSE Nifty100 Low Volatility 30: minus the 1-year std of daily returns.
      research 50/50 factor blend (momentum + low vol, top names of either sleeve via the max of
               the two percentile ranks). Crash protection (Daniel & Moskowitz 2016; Faber 2007):
               when NIFTY is below its 200-day average OR its 2-year return is negative, the
               momentum sleeve is switched off and the book holds the low-vol sleeve only.
    """
    C = pd.DataFrame(mk.C, index=mk.dates, columns=mk.stocks)
    r = C.pct_change()
    sig = r.rolling(252, min_periods=200).std() * np.sqrt(252)
    m6 = (C / C.shift(126) - 1) / sig
    m12 = (C / C.shift(252) - 1) / sig
    mom_vn = 0.5 * _xs_z(m6) + 0.5 * _xs_z(m12)
    rr = r.shift(21)
    pos = (rr > 0).astype(float).where(rr.notna()).rolling(231, min_periods=180).mean()
    neg = (rr < 0).astype(float).where(rr.notna()).rolling(231, min_periods=180).mean()
    pret = C.shift(21) / C.shift(252) - 1
    ID = np.sign(pret) * (neg - pos)
    mom_fip = mom_vn - 0.5 * _xs_z(ID)
    # residual (idiosyncratic) momentum, Blitz, Huij & Martens (2011): 12-1 month return net of the
    # NIFTY beta exposure, scaled by its own volatility - less exposed to momentum crashes
    rm = mk.nifty.reindex(mk.dates).pct_change()
    beta = r.rolling(252, min_periods=180).cov(rm).div(rm.rolling(252, min_periods=180).var(), axis=0)
    res = r - beta.mul(rm, axis=0)
    res_l = res.shift(21)
    resid_mom = res_l.rolling(231, min_periods=180).sum() / res_l.rolling(231, min_periods=180).std()
    consensus = (mom_vn.rank(axis=1, pct=True) + mom_fip.rank(axis=1, pct=True) + resid_mom.rank(axis=1, pct=True)) / 3
    # industry momentum (Moskowitz & Grinblatt 1999): each stock inherits its sector's average
    # 6-month return rank; blended 50/50 with the stock-level consensus score
    r126 = C / C.shift(126) - 1
    sec = pd.Series({c_: SECTOR.get(c_, "Other") for c_ in C.columns})
    sec_mom = r126.T.groupby(sec).transform("mean").T
    consensus_sector = 0.5 * consensus + 0.5 * sec_mom.rank(axis=1, pct=True)
    lowvol = -sig
    n = mk.nifty.reindex(mk.dates)
    risk_off = (n < n.rolling(200, min_periods=150).mean()) | (n / n.shift(504) - 1 < 0)
    p_m = mom_fip.rank(axis=1, pct=True)
    p_l = lowvol.rank(axis=1, pct=True)
    blend = np.maximum(p_m, p_l) + 1e-3 * (p_m + p_l)
    research = blend.mask(risk_off, p_l, axis=0)
    mk.risk_off = risk_off
    # trend / absolute-momentum states used by Params.trend_exit (all causal, decided at the close)
    liq_12m = pd.Series(mk.liq, index=mk.dates).add(1).rolling(252).apply(np.prod, raw=True) - 1
    mk.trend_states = {
        "sma200": (n < n.rolling(200, min_periods=150).mean()).to_numpy(),                       # Faber (2007)
        "dual": ((n < n.rolling(200, min_periods=150).mean()) |
                 (n / n.shift(252) - 1 < liq_12m)).to_numpy(),                                  # + Antonacci absolute momentum
        # exit only when the index is below its SMA200 AND fewer than half the stocks are above theirs
        # (two-signal confirmation -> fewer whipsaws)
        "sma200_breadth": ((n < n.rolling(200, min_periods=150).mean()) &
                           ((C > C.rolling(200, min_periods=150).mean()).astype(float).where(C.notna()).mean(1) < 0.5)).to_numpy(),
    }
    return {"mom_vn": mom_vn, "mom_fip": mom_fip, "lowvol": lowvol, "blend_no_filter": blend, "research": research,
            "resid_mom": resid_mom, "consensus": consensus, "consensus_sector": consensus_sector}


def window(s: pd.Series, w) -> pd.Series:
    return s.loc[w[0]:w[1]]


def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 20:
        return {}
    m = L.to_monthly(s)
    return {"final_value_rs": round(L.CFG["capital"] * float((1 + s).prod()), 0),
            "total_return": round(float((1 + s).prod() - 1), 4),
            "CAGR": round(float((1 + s).prod() ** (252 / len(s)) - 1), 4),
            "sharpe": round(float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 1e-8 else 0.0, 3),
            "max_drawdown": round(L.max_drawdown(s), 4), "worst_month": round(float(m.min()), 4),
            "months_positive": round(float((m > 0).mean()), 3),
            "months_ge_-0.5%": round(float((m >= -0.005).mean()), 3)}


def main():
    ap = L.base_args("Strategy02 Phase 6: monthly CNC engine, variant grid, all-lines chart")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    for f in out.glob("trades_*.csv"):
        f.unlink()
    t0 = time.time()
    mk = Market(data)
    scores, meta = load_scores(mk)
    regime = pd.read_csv(L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv", index_col=0, parse_dates=True)
    global SEL, TEST
    start = max(pd.Timestamp("2019-07-01"), scores["final"].index.min())
    bt = mk.dates[mk.dates >= start]
    cut = int(len(bt) * TRAIN_FRAC)
    SEL, TEST = (bt[0], bt[cut - 1]), (bt[cut], bt[-1])
    print(f"[P6] TRAIN (selection) {SEL[0].date()} .. {SEL[1].date()}  |  TEST {TEST[0].date()} .. {TEST[1].date()}")

    # ---------------- variant grid (36 variants -> enough for PBO / DSR trial counting)
    STOPS = {"off": dict(stops=False), "2atr12": dict(stops=True, atr_mult=2.0, stop_pct=0.12),
             "3atr15": dict(stops=True, atr_mult=3.0, stop_pct=0.15)}
    grid = []
    SCORES = ["final", "composite", "mom_vn", "mom_fip", "research"]
    TRENDS = ["none", "sma200", "dual"]
    for sc, opt, k, st, te in itertools.product(SCORES, ["MV", "HRP", "EW"], [8, 12], list(STOPS), TRENDS):
        if args.quick and (k == 8 or opt == "MV" or st == "2atr12"):
            continue
        grid.append(Params(name=f"{sc}|{opt}|k{k}|stop_{st}|trend_{te}", score=sc, opt=opt, k=k, trend_exit=te, **STOPS[st]))
    res, G = [], {}
    for p in grid:
        d, _ = run(mk, p, scores[p.score], meta if p.score == "final" else None, regime, start)
        G[p.name] = d["ret"]
        r = {"variant": p.name, **{f"sel_{k}": v for k, v in metrics(window(d["ret"], SEL)).items()},
             **{f"test_{k}": v for k, v in metrics(window(d["ret"], TEST)).items()},
             **{f"full_{k}": v for k, v in metrics(d["ret"]).items()}}
        sc_sel = L.scorecard(window(d["ret"], SEL), p.name, n_boot=300)
        r["sel_goal_score"] = sc_sel["goal_score"]
        r["avg_names"] = round(float(d["n_stocks"].mean()), 1)
        r["cost_rs_total"] = round(float(d["cost"].sum()), 0)
        res.append(r)
        print(f"   {p.name:<28} sel Sharpe {r.get('sel_sharpe')}  test CAGR {r.get('test_CAGR')}  full CAGR {r.get('full_CAGR')}")
    grid_tab = pd.DataFrame(res)
    grid_tab.to_csv(out / "grid_results.csv", index=False)
    pd.DataFrame(G).to_csv(out / "grid_daily_returns.csv")

    # selection on the SELECTION window only (mandate goal score, tie-break Sharpe)
    best_row = grid_tab.sort_values(["sel_goal_score", "sel_sharpe"]).iloc[-1]
    bp = next(p for p in grid if p.name == best_row["variant"])
    (out / "best_variant.json").write_text(json.dumps({"selected_on": f"{SEL[0].date()}..{SEL[1].date()}",
                                                       "train": [str(SEL[0].date()), str(SEL[1].date())],
                                                       "test": [str(TEST[0].date()), str(TEST[1].date())],
                                                       "params": asdict(bp), "row": best_row.to_dict()}, indent=2, default=str))
    print(f"[P6] selected on the selection window: {bp.name}")

    # ---------------- the lines for the all-in-one chart
    lines, daily_logs = {}, {}
    cfgs = {
        "CNC-only long (buy side, delivery, idle cash 0%)": replace(bp, sweep=False),
        "CNC + liquid-ETF sweep (selected)": bp,
        "CNC + sweep, no regime/vol-target": replace(bp, regime=False),
        "CNC + sweep, no stops": replace(bp, stops=False),
        "CNC + sweep, no trend filter": replace(bp, trend_exit="none"),
        "CNC + sweep, EW weights": replace(bp, opt="EW"),
        "CNC + sweep, HRP weights": replace(bp, opt="HRP"),
        "CNC + sweep, MV weights": replace(bp, opt="MV"),
    }
    seen = set()
    for lab, p in cfgs.items():
        key = json.dumps(asdict(replace(p, name="")), sort_keys=True)
        if key in seen and lab not in ("CNC-only long (buy side, delivery, idle cash 0%)", "CNC + liquid-ETF sweep (selected)"):
            continue
        seen.add(key)
        d, tr = run(mk, p, scores[p.score], meta if p.score == "final" else None, regime, start, log_trades=True)
        lines[lab] = d["ret"]; daily_logs[lab] = d
        if lab.startswith("CNC-only") or lab.startswith("CNC + liquid"):
            tag = "cnc_only" if lab.startswith("CNC-only") else "cnc_sweep"
            d.to_csv(out / f"daily_{tag}.csv")
            tr.to_csv(out / f"trades_{tag}.csv", index=False)

    # optional MIS overlay: strategy/ v1 intraday sleeve P&L added on top of the pledged CNC book
    v1 = L.HERE.parent / "strategy" / "phase6_portfolio_backtest" / "daily_V3.csv"
    if v1.exists():
        mis = pd.read_csv(v1, index_col=0, parse_dates=True)
        mis_r = (mis["mis_pnl"] - mis.get("mis_cost", 0) * 0) / mis["equity"].shift(1).fillna(L.CFG["capital"])
        d, _ = run(mk, replace(bp, pledge=True), scores[bp.score], meta if bp.score == "final" else None, regime, start)
        base = d["ret"]
        lines["CNC pledged + MIS overlay (v1 intraday sleeve)"] = base.add(mis_r.reindex(base.index).fillna(0.0), fill_value=0.0)

    P = mk.P
    liq = pd.Series(mk.liq, index=mk.dates)
    hs = L.half_spread_bps(P["adj_high"][mk.stocks], P["adj_low"][mk.stocks], P["turnover"][mk.stocks])
    ew = P["adj_close"][mk.stocks].notna().astype(float)
    ew = ew.div(ew.sum(1), axis=0)
    lines["EW NIFTY100 monthly"] = L.simulate_weights(ew, P["adj_open"], P["adj_close"], liq, pledge=False, half_spread=hs,
                                                      rebal_mask=L.month_end_mask(mk.dates))["ret"].loc[start:]
    lines["NIFTY50 buy&hold"] = mk.nifty.pct_change().fillna(0).loc[start:]
    lines["Liquid ETF only"] = liq.loc[start:]
    nullD, nsum = L.random_null(P["adj_close"], P["adj_open"], k=bp.k, n_draws=60 if args.quick else 200, start=str(start.date()))
    order = nsum["cagr"].sort_values()
    pick = {q: order.index[int(q * (len(order) - 1))] for q in (0.05, 0.5, 0.95)}
    lines[f"Random {bp.k} names monthly (null median)"] = nullD[pick[0.5]]
    nsum.to_csv(out / "null_draws.csv", index_label="draw")
    nullD.loc[start:].to_csv(out / "null_daily_returns.csv", float_format="%.6g")
    lines = {k: v.loc[start:] for k, v in lines.items()}
    pd.DataFrame(lines).to_csv(out / "lines_daily_returns.csv")

    L.multi_panel(lines, out / "strategy02_all_lines.png",
                  f"Strategy02 - every book from Rs 1,00,000 ({start.date()} .. {mk.dates[-1].date()}); selected = {bp.name}",
                  highlight="CNC-only long (buy side, delivery, idle cash 0%)",
                  band=(nullD[pick[0.05]].loc[start:], nullD[pick[0.95]].loc[start:]))
    cnc_set = {k: v for k, v in lines.items() if k.startswith("CNC") or k.startswith("NIFTY50") or k.startswith("Random")}
    L.multi_panel(cnc_set, out / "cnc_only_vs_benchmarks.png", "CNC-only (buy-side, delivery) books vs NIFTY50 and the random null",
                  highlight="CNC-only long (buy side, delivery, idle cash 0%)")

    # ---------------- long history (composite score exists from 2017-11): CNC only
    lh_start = pd.Timestamp("2017-11-01")
    lh = {}
    lh_score = bp.score if bp.score != "final" else "composite"
    for lab, p in {f"{lh_score} CNC-only (idle 0%)": replace(bp, score=lh_score, sweep=False),
                   f"{lh_score} CNC + sweep": replace(bp, score=lh_score)}.items():
        d, _ = run(mk, p, scores[lh_score], None, regime, lh_start)
        lh[lab] = d["ret"]
    lh["NIFTY50 buy&hold"] = mk.nifty.pct_change().fillna(0).loc[lh_start:]
    lh["EW NIFTY100 monthly"] = L.simulate_weights(ew, P["adj_open"], P["adj_close"], liq, pledge=False, half_spread=hs,
                                                   rebal_mask=L.month_end_mask(mk.dates))["ret"].loc[lh_start:]
    L.multi_panel(lh, out / "long_history_cnc_only.png", f"Long history 2017-11 .. now ({lh_score} score, selected settings)",
                  highlight=f"{lh_score} CNC-only (idle 0%)")
    pd.DataFrame(lh).to_csv(out / "long_history_daily_returns.csv")

    # ---------------- profit tables
    prof = []
    for lab, s in lines.items():
        for wn, w in (("full", (start, mk.dates[-1])), ("train", SEL), ("test", TEST)):
            m = metrics(window(s, w))
            if m:
                prof.append({"line": lab, "window": wn, **m})
    prof = pd.DataFrame(prof)
    prof.to_csv(out / "profit_table.csv", index=False)
    era = []
    for lab, s in lines.items():
        for en, (a, b) in ERAS.items():
            x = s.loc[a:b]
            if len(x) > 20:
                era.append({"line": lab, "era": en, "return": round(float((1 + x).prod() - 1), 4),
                            "max_dd": round(L.max_drawdown(x), 4)})
    era = pd.DataFrame(era)
    era.pivot(index="line", columns="era", values="return").to_csv(out / "era_table.csv")

    cnc = daily_logs["CNC-only long (buy side, delivery, idle cash 0%)"]
    tr = pd.read_csv(out / "trades_cnc_only.csv") if (out / "trades_cnc_only.csv").exists() else pd.DataFrame()
    trade_stats = {}
    if len(tr):
        sells = tr[tr["side"] == "SELL"]
        trade_stats = {"orders": int(len(tr)), "buys": int((tr["side"] == "BUY").sum()), "sells": int(len(sells)),
                       "stop_exits": int(sells["reason"].str.startswith("stop").sum()),
                       "total_costs_rs": round(float(tr["cost"].sum()), 0),
                       "avg_names_held": round(float(cnc["n_stocks"].mean()), 1),
                       "avg_stock_exposure": round(float(cnc["stock_exposure"].mean()), 3),
                       "years": round(len(cnc) / 252, 2)}
        trade_stats["costs_per_year_rs"] = round(trade_stats["total_costs_rs"] / trade_stats["years"], 0)
    (out / "cnc_only_trade_stats.json").write_text(json.dumps(trade_stats, indent=2))

    cards = [L.scorecard(v, k) for k, v in lines.items()]
    best = L.save_scorecards(out, cards, PHASE, notes=f"Selected variant {bp.name}", series=lines)
    full = prof[prof["window"] == "full"].drop(columns="window")
    tabw = prof[prof["line"].isin(["CNC-only long (buy side, delivery, idle cash 0%)", "CNC + liquid-ETF sweep (selected)",
                                   "NIFTY50 buy&hold", f"Random {bp.k} names monthly (null median)"])]
    gcols = ["variant", "sel_goal_score", "sel_sharpe", "sel_CAGR", "test_CAGR", "test_sharpe", "test_max_drawdown",
             "full_CAGR", "full_max_drawdown", "avg_names"]
    (out / "report.md").write_text(f"""# Strategy02 - Phase 6: monthly CNC engine

Selected on {SEL[0].date()} .. {SEL[1].date()} only: **{bp.name}**
(score = {bp.score}, weights = {bp.opt}, k = {bp.k}, stops = {bp.stops}).

## What Rs 1,00,000 became ({start.date()} .. {mk.dates[-1].date()})
{L.md_table(full, 4)}

## CNC-only line vs benchmarks by window (train 70% / test 30%)
{L.md_table(tabw, 4)}

## CNC-only trade statistics
```
{json.dumps(trade_stats, indent=2)}
```

## Variant grid ({len(grid)} variants)
{L.md_table(grid_tab[[c for c in gcols if c in grid_tab]], 4)}

## Eras (total return)
{L.md_table(era.pivot(index='line', columns='era', values='return').reset_index(), 3)}

![all](strategy02_all_lines.png)
![cnc](cnc_only_vs_benchmarks.png)
![long](long_history_cnc_only.png)
""")
    L.print_scorecard(best, PHASE)
    print(f"[P6] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
