#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 6 - Integrated CNC + pledge + MIS portfolio backtest (event-driven) with a causal
selection / holdout protocol.

Mandate variant families (Goals 3, 4, 5)
  V1  Goal-3 pledge-rotate   : 100% CNC stocks held ~2-5 days, pledged, MIS on the collateral
  V2  Goal-4 80/20           : 80% CNC stocks, 20% cash reserved for MIS margin
  V3  Pledge-optimised 50/50 : 50% CNC stocks + 50% pledged liquid ETF (the liquid ETF is
                               cash-equivalent collateral, so it unlocks the stock collateral
                               under the 50:50 rule -> maximum MIS buying power)
  V4  Liquid collateral + MIS: 100% pledged liquid ETF, flat overnight, alpha only from MIS
  V5  Adaptive split         : stock fraction = 0.6 x regime risk score, rest pledged liquid ETF

Protocol (prevents choosing parameters on the data used to judge them)
  1. All predictions are already out-of-sample (Phases 4/5).
  2. For each family a small parameter grid is run on the SELECTION window (first half of the
     combined OOS period); the best config by goal score is frozen.
  3. The frozen config is reported on the HOLDOUT window (second half) and on the full period.
  4. Every grid config's daily returns are saved for Phase 7's probability-of-backtest-
     overfitting (CSCV) test.

Realism
  * decisions at close t; CNC executed at OPEN t+1; whole shares on RAW prices when available
  * Flattrade statutory costs + spread + Rs 23.6 pledge fee per new name (+ DP/unpledge config)
  * bought shares become pledgeable collateral only after T+1 settlement
  * 50:50 cash-collateral rule; MIS buying power = 5 x usable margin
  * MIS: side-specific CAUSAL isotonic calibration of the model score to the stop-adjusted
    outcome; trade only when expected edge > round-trip cost + margin; fractional-Kelly sizing,
    per-name cap, daily MIS risk budget
  * overlay: CNC vol targeting on an RMT-cleaned covariance, regime tilt, drawdown brake
    (floor 25%), monthly circuit breaker
Output folder: phase6_portfolio_backtest/
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

import p1_data_costs as L
import p3_features as F3
import p4_ml_models as P4

plt = L.plt
PHASE = "phase6_portfolio_backtest"
STOPS = ("005", "010", "015")


@dataclass
class Params:
    name: str = "V3"
    stock_frac: float = 0.5
    cash_frac: float = 0.0           # share of the non-stock remainder kept as free cash
    adaptive: bool = False
    k: int = 4
    hold: int = 3
    buffer: int = 6
    vol_target: float = 0.15         # annualised vol target for the stock sleeve
    regime_weight: float = 0.4
    use_mis: bool = True
    mis_gross_max: float = 3.0
    mis_kelly: float = 0.25
    mis_max_pos: float = 1.0
    mis_daily_vol: float = 0.006
    hurdle_margin: float = 5e-4
    stop: str = "010"
    dd_soft: float = 0.08
    dd_hard: float = 0.18
    dd_floor: float = 0.25
    month_stop: float = -0.004
    cost_mult: float = 1.0           # stress multiplier on REALISED spreads (decisions use base costs)
    stop_slip_bps: float = 5.0       # extra slippage when a stop-loss is hit (stop-market fill)
    max_loss_per_trade: float = 0.002   # worst-case loss of one MIS trade (stop+costs) <= 0.2% of equity
    max_daily_mis_loss: float = 0.004   # worst-case loss of all MIS trades in a day <= 0.4% of equity
    month_floor: float = -0.005      # Goal-2 floor: MIS worst case may not push MTD below this
    integer_shares: bool = True


FAMILIES = {
    "V1 Goal-3 pledge-rotate (100% CNC)": Params("V1", stock_frac=1.0, cash_frac=0.0),
    "V2 Goal-4 80% CNC / 20% cash": Params("V2", stock_frac=0.8, cash_frac=1.0),
    "V3 50% CNC + 50% pledged liquid": Params("V3", stock_frac=0.5),
    "V4 Pledged liquid + MIS only": Params("V4", stock_frac=0.0),
    "V5 Adaptive regime split": Params("V5", stock_frac=0.6, adaptive=True),
}
SELECTION_FRAC = 0.5      # selection window = first half of the combined CNC+MIS window
GRID = {"hold": [2, 5, 10], "k": [4, 8], "use_mis": [True, False],
        "hurdle_margin": [5e-4, 1.5e-3], "vol_target": [0.15, 0.25]}
GRID_QUICK = {"hold": [2, 10], "k": [4], "use_mis": [True, False], "hurdle_margin": [1.5e-3], "vol_target": [0.15]}


def rmt_clean_cov(R: np.ndarray) -> np.ndarray:
    """Eigenvalue clipping (Laloux-Cizeau-Bouchaud-Potters 2000)."""
    T, N = R.shape
    sd = R.std(0, ddof=1) + 1e-12
    if N == 1:
        return np.array([[sd[0] ** 2]])
    C = np.nan_to_num(np.corrcoef(R.T))
    np.fill_diagonal(C, 1.0)
    w, V = np.linalg.eigh(C)
    lam = (1 + np.sqrt(N / T)) ** 2
    noise = w < lam
    if noise.any():
        w[noise] = w[noise].mean()
    Cc = V @ np.diag(w) @ V.T
    d = np.sqrt(np.clip(np.diag(Cc), 1e-12, None))
    return Cc / np.outer(d, d) * np.outer(sd, sd)


# =============================================================================
# inputs
# =============================================================================
def load_inputs(data):
    p4, p5, p3 = L.HERE / "phase4_ml_models", L.HERE / "phase5_deep_learning", L.HERE / "phase3_features"
    adopt = json.loads((p5 / "adoption.json").read_text()) if (p5 / "adoption.json").exists() else {}
    # (fix) the daily / intraday model is chosen by IC measured ONLY before the selection end (the
    # first half of the combined window). Previously it used the IC over the whole OOS period,
    # which includes the holdout and the phase-8 test window.
    I4 = pd.read_csv(p4 / "oos_pred_intraday_ret_to_exit.csv", parse_dates=["date"])
    end_all = pd.read_csv(p4 / "oos_pred_daily_fwd2_ex.csv", usecols=["date"], parse_dates=["date"])["date"].max()
    sel_end = I4["date"].min() + (end_all - I4["date"].min()) * SELECTION_FRAC - pd.Timedelta(days=5)

    def pick(frames: dict, target: str, key: str):
        best, best_t = None, -np.inf
        for (fname, colname), df_ in frames.items():
            d_ = df_[df_["date"] < sel_end].dropna(subset=[colname, target])
            if d_.empty:
                continue
            ic = P4.daily_ic(d_, colname, target, key)
            t = ic.mean() / (ic.std() + 1e-12) * np.sqrt(len(ic))
            if t > best_t:
                best, best_t = (fname, colname), t
        return best, best_t
    D4 = pd.read_csv(p4 / "oos_pred_daily_fwd2_ex.csv", parse_dates=["date"])
    dframes = {("p4", c): D4 for c in ("pred_ridge", "pred_lgbm", "pred_xgb") if c in D4}
    if (p5 / "oos_pred_daily_ensemble.csv").exists():
        D5 = pd.read_csv(p5 / "oos_pred_daily_ensemble.csv", parse_dates=["date"])
        dframes[("p5", "pred_ensemble")] = D5
    (dsrc, dcol), dt_ = pick(dframes, "fwd2_ex", "date")
    D = dframes[(dsrc, dcol)]
    I4["dk"] = I4["date"].dt.strftime("%Y-%m-%d") + I4["slot"].astype(str).str.zfill(4)
    iframes = {("p4", c): I4 for c in ("pred_ridge", "pred_lgbm", "pred_xgb") if c in I4}
    if (p5 / "oos_pred_intraday_ensemble.csv").exists():
        I5 = pd.read_csv(p5 / "oos_pred_intraday_ensemble.csv", parse_dates=["date"])
        I5["dk"] = I5["date"].dt.strftime("%Y-%m-%d") + I5["slot"].astype(str).str.zfill(4)
        iframes[("p5", "pred_ensemble")] = I5
    (isrc, icol), it_ = pick(iframes, "ret_to_exit", "dk")
    I = iframes[(isrc, icol)].drop(columns=["dk"])
    adopt = dict(adopt, chosen_daily=f"{dsrc}:{dcol}", chosen_daily_ic_t_pre_selection=round(float(dt_), 2),
                 chosen_intraday=f"{isrc}:{icol}", chosen_intraday_ic_t_pre_selection=round(float(it_), 2),
                 model_choice_window_end=str(sel_end.date()))
    print(f"[P6] models chosen on data before {sel_end.date()}: daily {dsrc}:{dcol} (t={dt_:.2f}), "
          f"intraday {isrc}:{icol} (t={it_:.2f})")
    I["slot"] = I["slot"].astype(str).str.zfill(4)
    extra = pd.read_csv(p3 / "intraday_features.csv", parse_dates=["date"], usecols=["date", "name", "slot", "prevday_vol_20"])
    extra["slot"] = extra["slot"].astype(str).str.zfill(4)
    I = I.drop(columns=[c for c in ("prevday_vol_20",) if c in I]).merge(extra, on=["date", "name", "slot"], how="left")
    print("[P6] causal side-specific calibration of MIS scores ...")
    I["neg_score"] = -I[icol]
    for s in STOPS:
        I[f"exp_long_{s}"] = P4.causal_calibrate(I, icol, f"long_sl{s}")
        I[f"exp_short_{s}"] = P4.causal_calibrate(I, "neg_score", f"short_sl{s}")
    I = I.dropna(subset=[f"exp_long_{STOPS[1]}"])
    reg = pd.read_csv(L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv", index_col=0, parse_dates=True)
    P = L.load_daily_panel(data)
    names = [c for c in P["adj_close"].columns if c != L.INDEX]
    return {"D": D, "dcol": dcol, "I": I, "icol": icol, "reg": reg, "P": P, "names": names,
            "liq": L.liquid_returns(data, P["adj_close"].index),
            "hs": L.half_spread_bps(P["adj_high"], P["adj_low"], P["turnover"]), "adopt": adopt,
            "Ig": {d: g for d, g in I.groupby("date")}, "sel_cache": {}}


# =============================================================================
# simulator
# =============================================================================
def simulate(inp: dict, prm: Params, start=None, end=None, log_trades=False):
    P, reg, liq = inp["P"], inp["reg"], inp["liq"]
    names = inp["names"]
    col = {n: j for j, n in enumerate(names)}
    O = P["adj_open"][names].to_numpy(); C = P["adj_close"][names].to_numpy()
    RO = P["raw_open"][names].reindex(columns=names).to_numpy()
    R = P["adj_close"][names].pct_change().to_numpy()
    all_dates = P["adj_close"].index
    # (fix) time-varying, causal half-spread: row t-1 (known before the open of t)
    HS = (inp["hs"].reindex(index=all_dates, columns=names).ffill().fillna(3.0).to_numpy() / 1e4
          if isinstance(inp["hs"], pd.DataFrame) else np.tile(inp["hs"].reindex(names).fillna(3.0).to_numpy() / 1e4,
                                                               (len(all_dates), 1)))
    key = (prm.k, prm.hold, prm.buffer)
    if key not in inp["sel_cache"]:
        inp["sel_cache"][key] = F3.cnc_weights_from_scores(inp["D"].set_index(["date", "name"])[inp["dcol"]],
                                                            k=prm.k, hold=prm.hold, buffer=prm.buffer)
    sel = inp["sel_cache"][key]
    lo = sel.index.min()
    if prm.use_mis:
        lo = max(lo, inp["I"]["date"].min())
    if start:
        lo = max(lo, pd.Timestamp(start))
    hi = min(all_dates.max(), pd.Timestamp(end)) if end else all_dates.max()
    di = np.where((all_dates >= lo) & (all_dates <= hi))[0]
    risk = reg["risk_score"].reindex(all_dates).ffill().fillna(0.5).to_numpy()
    liqr = liq.reindex(all_dates).fillna(0).to_numpy()
    buy_r, sell_r, mis_rt = L.cnc_rate("buy"), L.cnc_rate("sell"), L.mis_rate_roundtrip()
    pf, cap0 = L.CFG["pledge_fee"], L.CFG["capital"]
    pos = np.zeros(len(names)); settled = np.zeros(len(names), bool)
    cash = cap0 * prm.cash_frac * (1 - prm.stock_frac)
    liquid = cap0 - cash - pf
    peak = eq_prev = cap0
    month, m_start, halt = None, cap0, False
    rows, ctr, mtr = [], [], []
    for t in di:
        d = all_dates[t]
        if month != d.to_period("M"):
            month, m_start, halt = d.to_period("M"), eq_prev, False
        o, c, cp = O[t], C[t], C[t - 1]
        g = np.where(np.isfinite(o) & np.isfinite(cp) & (cp > 0), o / cp, 1.0)
        pos *= g
        eq_open = pos.sum() + cash + liquid
        dd = eq_open / peak - 1
        dd_mult = prm.dd_floor + (1 - prm.dd_floor) * float(np.clip((dd + prm.dd_hard) / (prm.dd_hard - prm.dd_soft), 0, 1))
        if eq_open / m_start - 1 <= prm.month_stop:
            halt = True
        hs_dec = HS[t - 1]                         # used for decisions
        hs = hs_dec * prm.cost_mult                # realised (stress-able)
        rs = risk[t - 1]
        reg_mult = 1 - prm.regime_weight * (1 - rs)
        # ---------------- CNC at the open (decided yesterday) ------------------
        sf = prm.stock_frac * (rs if prm.adaptive else 1.0)
        target = np.zeros(len(names))
        prev_d = all_dates[t - 1]
        if sf > 0 and prev_d in sel.index:
            row = sel.loc[prev_d]
            nm = [col[n] for n in row[row > 0].index if n in col and np.isfinite(o[col[n]])]
            if nm:
                win = R[max(t - 63, 1):t, nm]
                ok = np.isfinite(win).sum(0) >= 40
                nm = [j for j, k_ in zip(nm, ok) if k_]
                if nm:
                    win = np.nan_to_num(R[max(t - 63, 1):t, nm])
                    cov = rmt_clean_cov(win)
                    iv = 1 / np.sqrt(np.diag(cov))
                    w = iv / iv.sum()
                    pv = float(np.sqrt(w @ cov @ w * L.TRADING_DAYS))
                    scale = min(1.0, prm.vol_target / max(pv, 1e-6)) * reg_mult * dd_mult * (0.5 if halt else 1.0)
                    target[nm] = eq_open * sf * scale * w
        cost_cnc, pledges = 0.0, 0
        for j in np.nonzero((pos > 0) | (target > 0))[0]:
            cur, tgt = pos[j], target[j]
            if cur > 0 and tgt > 0 and abs(tgt - cur) < 0.03 * eq_open:
                continue
            px = RO[t, j]
            if prm.integer_shares and np.isfinite(px) and px > 0:
                tgt = np.floor(tgt / px) * px
            delta = tgt - cur
            if abs(delta) < 1:
                continue
            if delta > 0:
                cost = delta * (buy_r + hs[j]) + (pf if cur <= 0 else 0.0)
                pledges += int(cur <= 0)
            else:
                cost = -delta * (sell_r + hs[j]) + (L.CFG["dp_fee_per_sell"] + L.CFG["unpledge_fee"] if tgt <= 0 else 0.0)
            cost_cnc += cost
            if log_trades:
                ctr.append({"date": d, "name": names[j], "side": "BUY" if delta > 0 else "SELL",
                            "value": round(delta, 2), "cost": round(cost, 2)})
            if cur <= 0:
                settled[j] = False
            pos[j] = max(tgt, 0.0)
            liquid -= delta + cost + abs(delta) * L.CFG["liquid_cost_side"]
        # ---------------- MIS ---------------------------------------------------
        mis_pnl = mis_cost = mis_gross = 0.0
        n_mis = 0
        if prm.use_mis and not halt and d in inp["Ig"]:
            coll = float((pos * settled).sum() * (1 - L.CFG["haircut_floor"]))
            cap = L.mis_capacity(cash, max(liquid, 0.0), coll)
            gross_cap = min(cap, prm.mis_gross_max * eq_open) * dd_mult * reg_mult
            gd = inp["Ig"][d]
            el, es = gd[f"exp_long_{prm.stop}"].to_numpy(), gd[f"exp_short_{prm.stop}"].to_numpy()
            jj = np.array([col.get(n, -1) for n in gd["name"]])
            h2 = np.where(jj >= 0, 2 * hs_dec[np.clip(jj, 0, None)], 6e-4)
            h2_real = np.where(jj >= 0, 2 * hs[np.clip(jj, 0, None)], 6e-4 * prm.cost_mult)
            edge_l, edge_s = el - mis_rt - h2, es - mis_rt - h2
            side = np.where(edge_l >= edge_s, 1, -1)
            edge = np.maximum(edge_l, edge_s)
            m = edge > prm.hurdle_margin
            if m.any():
                cand = gd[m].assign(edge=edge[m], side=side[m], h2=h2_real[m]).sort_values("slot").drop_duplicates("name")
                sig = (cand["prevday_vol_20"].fillna(0.02).clip(0.005, 0.08) * 0.8).to_numpy()
                # loss budgets: worst case of a trade = stop + slippage + round-trip cost
                stop_v = int(prm.stop) / 1000 + prm.stop_slip_bps / 1e4
                wc = stop_v + mis_rt + cand["h2"].to_numpy()
                room = (eq_open / m_start - 1 - prm.month_floor) * m_start      # Rs left before the monthly floor
                gross_left = gross_cap
                var_left = (prm.mis_daily_vol * eq_open) ** 2
                loss_left = max(0.0, min(prm.max_daily_mis_loss * eq_open, 0.5 * room))
                notional = np.zeros(len(cand))
                slots_ = cand["slot"].astype(str).to_numpy()
                jc = np.array([col.get(n, -1) for n in cand["name"]])
                # (fix) size slot by slot in time order: a 09:30 trade only sees the budget, never the
                # later slots' candidates; positions are whole shares at the day's raw open
                for s_ in sorted(set(slots_)):
                    ix = np.where(slots_ == s_)[0]
                    nt = np.minimum(prm.mis_kelly * cand["edge"].to_numpy()[ix] / sig[ix] ** 2, prm.mis_max_pos) * eq_open
                    nt = np.minimum(nt, prm.max_loss_per_trade * eq_open / wc[ix])
                    if nt.sum() > gross_left:
                        nt *= gross_left / max(nt.sum(), 1e-9)
                    v = ((nt * sig[ix]) ** 2).sum()
                    if v > var_left:
                        nt *= np.sqrt(max(var_left, 0.0) / max(v, 1e-18))
                    worst = (nt * wc[ix]).sum()
                    if worst > loss_left:
                        nt *= loss_left / max(worst, 1e-9)
                    px = np.where(jc[ix] >= 0, RO[t, np.clip(jc[ix], 0, None)], np.nan)
                    nt = np.where(np.isfinite(px) & (px > 0), np.floor(nt / np.where(px > 0, px, 1)) * px, 0.0)
                    notional[ix] = nt
                    gross_left -= nt.sum(); var_left -= ((nt * sig[ix]) ** 2).sum(); loss_left -= (nt * wc[ix]).sum()
                sd = cand["side"].to_numpy()
                outc = np.where(sd > 0, cand[f"long_sl{prm.stop}"].to_numpy(), cand[f"short_sl{prm.stop}"].to_numpy())
                outc = np.where(outc <= -int(prm.stop) / 1000 + 1e-12, outc - prm.stop_slip_bps / 1e4, outc)
                costv = notional * (mis_rt + cand["h2"].to_numpy())
                ok = np.isfinite(outc) & (notional > 0)
                mis_pnl = float((notional * outc)[ok].sum() - costv[ok].sum())
                mis_cost, mis_gross, n_mis = float(costv[ok].sum()), float(notional[ok].sum()), int(ok.sum())
                if log_trades:
                    for (_, r), nt, s_, oc, cs in zip(cand.iterrows(), notional, sd, outc, costv):
                        if nt <= 0 or not np.isfinite(oc):
                            continue
                        mtr.append({"date": d, "name": r["name"], "slot": r["slot"], "side": "BUY" if s_ > 0 else "SELL",
                                    "notional": round(nt, 2), "exp_edge": round(r["edge"], 6),
                                    "realised_ret": round(oc, 6), "cost": round(cs, 2), "pnl": round(nt * oc - cs, 2)})
        # ---------------- close ---------------------------------------------------
        rc = np.where(np.isfinite(c) & np.isfinite(o) & (o > 0), c / o, 1.0)
        pos *= rc
        settled |= pos > 0
        liquid = liquid * (1 + liqr[t]) + mis_pnl
        eq = pos.sum() + cash + liquid
        peak = max(peak, eq)
        rows.append((d, eq, eq / eq_prev - 1, pos.sum() / eq, int((pos > 0).sum()), mis_pnl, mis_gross / eq, n_mis,
                     cost_cnc, mis_cost, pledges, eq / peak - 1, dd_mult, halt))
        eq_prev = eq
    res = pd.DataFrame(rows, columns=["date", "equity", "ret", "stock_exposure", "n_stocks", "mis_pnl", "mis_gross_x",
                                      "mis_trades", "cnc_cost", "mis_cost", "pledge_orders", "drawdown", "dd_mult",
                                      "month_halt"]).set_index("date")
    return res, pd.DataFrame(ctr), pd.DataFrame(mtr)


def robust_selection_score(r: pd.DataFrame, liq: pd.Series, label: str) -> dict:
    """Goal score on the selection window, taking the WORSE of (a) as-is and (b) with the single
    best MIS day neutralised (its return replaced by the liquid yield). Stops the selection from
    rewarding a configuration whose result hinges on one lucky trade."""
    a = L.scorecard(r["ret"], label, n_boot=500)
    if r["mis_pnl"].abs().sum() == 0:
        return a
    rr = r["ret"].copy()
    d = r["mis_pnl"].idxmax()
    rr.loc[d] = liq.get(d, 0.0)
    b = L.scorecard(rr, label, n_boot=500)
    return a if a.get("goal_score", -1) <= b.get("goal_score", -1) else b


def mis_trade_stats(mt: pd.DataFrame) -> dict:
    if mt is None or mt.empty:
        return {"mis_trades": 0}
    p = mt["pnl"].sort_values(ascending=False)
    tot = p.sum()
    return {"mis_trades": int(len(p)), "mis_pnl_rs": round(float(tot)), "hit_rate": round(float((p > 0).mean()), 3),
            "best_trade_rs": round(float(p.iloc[0])), "worst_trade_rs": round(float(p.iloc[-1])),
            "pnl_ex_best_trade_rs": round(float(tot - p.iloc[0])),
            "pnl_ex_best_5pct_rs": round(float(p.iloc[max(1, int(len(p) * 0.05)):].sum())),
            "avg_pnl_per_trade_rs": round(float(p.mean()), 1)}


def grid_configs(base: Params, grid: dict):
    keys = list(grid)
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = replace(base, **dict(zip(keys, vals)))
        if p.stock_frac == 0 and (p.hold != grid["hold"][0] or p.k != grid["k"][0] or p.vol_target != grid["vol_target"][0]):
            continue                      # stock params irrelevant for the MIS-only family
        if p.stock_frac == 0 and not p.use_mis:
            continue
        if not p.use_mis and p.hurdle_margin != grid["hurdle_margin"][0]:
            continue                      # (fix) hurdle_margin only matters for MIS -> exact duplicates
        out.append(p)
    return out


def tag(p: Params) -> str:
    return f"{p.name}|hold{p.hold}|k{p.k}|mis{int(p.use_mis)}|hm{p.hurdle_margin:g}|vt{p.vol_target:g}"


def main():
    ap = L.base_args("Phase 6: integrated portfolio backtest with selection/holdout")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    inp = load_inputs(data)
    mis_start = inp["I"]["date"].min()
    end = inp["P"]["adj_close"].index.max()
    mid = mis_start + (end - mis_start) * SELECTION_FRAC
    print(f"[P6] CNC model {inp['dcol']} | MIS model {inp['icol']} | combined OOS {mis_start.date()}..{end.date()}"
          f" | selection < {mid.date()} <= holdout")
    grid = GRID_QUICK if args.quick else GRID
    grid_rows, grid_rets, chosen = [], {}, {}
    for fam, base in FAMILIES.items():
        best_sc, best_p = -1, None
        for p in grid_configs(base, grid):
            r, _, _ = simulate(inp, p, start=mis_start)
            grid_rets[tag(p)] = r["ret"]
            sel_sc = robust_selection_score(r.loc[:mid], inp["liq"], tag(p))
            hol_sc = L.scorecard(r["ret"].loc[mid:], tag(p), n_boot=500)
            grid_rows.append({"family": fam, "config": tag(p), **{k: getattr(p, k) for k in grid},
                              "sel_goal_score": sel_sc.get("goal_score"), "sel_CAGR": sel_sc.get("CAGR"),
                              "hold_goal_score": hol_sc.get("goal_score"), "hold_CAGR": hol_sc.get("CAGR"),
                              "hold_maxdd": hol_sc.get("max_drawdown")})
            if sel_sc.get("goal_score", -1) > best_sc:
                best_sc, best_p = sel_sc["goal_score"], p
        chosen[fam] = best_p
        print(f"   {fam:<38} selected {tag(best_p)}  (selection score {best_sc})")
    gdf = pd.DataFrame(grid_rows)
    gdf.to_csv(out / "grid_results.csv", index=False)
    pd.DataFrame(grid_rets).to_csv(out / "grid_daily_returns.csv")
    (out / "selection_split.json").write_text(json.dumps({"combined_start": str(mis_start.date()), "selection_end": str(mid.date()),
                                                          "end": str(end.date())}, indent=2))

    # ---------- frozen configs: full-period + holdout --------------------------
    cards, hold_cards, series, daily_all = [], [], {}, {}
    for fam, p in chosen.items():
        r, ct, mt = simulate(inp, p, start=mis_start, log_trades=True)
        daily_all[fam], series[fam] = r, r["ret"]
        cards.append(L.scorecard(r["ret"], fam))
        hold_cards.append(L.scorecard(r["ret"].loc[mid:], fam + " [HOLDOUT]"))
        r.to_csv(out / f"daily_{p.name}.csv")
        for stale in (out / f"cnc_trades_{p.name}.csv", out / f"mis_trades_{p.name}.csv"):
            stale.unlink(missing_ok=True)
        if len(ct):
            ct.to_csv(out / f"cnc_trades_{p.name}.csv", index=False)
        if len(mt):
            mt.to_csv(out / f"mis_trades_{p.name}.csv", index=False)
    long_cards = []
    for fam in list(FAMILIES)[:3] + [list(FAMILIES)[4]]:
        q = replace(chosen[fam], use_mis=False)
        r, _, _ = simulate(inp, q)
        long_cards.append(L.scorecard(r["ret"], f"{fam} | CNC-only since {r.index[0].date()}"))
    pd.DataFrame(long_cards).to_csv(out / "long_history_cnc_only.csv", index=False)
    pd.DataFrame(hold_cards).to_csv(out / "holdout_scorecards.csv", index=False)

    best = L.save_scorecards(out, cards, PHASE, series=series,
                             notes=f"Configs frozen on selection window < {mid.date()}; see holdout_scorecards.csv")
    bfam = max(cards, key=lambda c: c.get("goal_score", -1))["strategy"]
    (out / "best_variant.json").write_text(json.dumps({"variant": bfam, "params": asdict(chosen[bfam]),
                                                       "chosen_per_family": {k: asdict(v) for k, v in chosen.items()}}, indent=2))
    gc = []
    for fam, r in daily_all.items():
        m = L.to_monthly(r["ret"])
        r12 = L.rolling_12m(r["ret"]).dropna()
        mtf = out / f"mis_trades_{chosen[fam].name}.csv"
        ts = mis_trade_stats(pd.read_csv(mtf) if mtf.exists() else None)
        gc.append({"variant": fam, "config": tag(chosen[fam]), **{k: v for k, v in ts.items() if k != "mis_trades"},
                   "max_dd": round(r["drawdown"].min(), 4),
                   "goal1_dd_ok": bool(r["drawdown"].min() >= L.CFG["goal_max_dd"]),
                   "min_trailing_12m": round(float(r12.min()), 4) if len(r12) else None,
                   "goal1_12m_ok": bool(len(r12) and r12.min() > L.CFG["goal_trailing_12m"]),
                   "months": len(m), "months_ge_-0.5%": int((m >= -0.005).sum()),
                   "months_in_band": int(((m >= -0.005) & (m <= 0.03)).sum()),
                   "goal2_all_months_ok": bool((m >= -0.005).all()),
                   "avg_stock_exposure": round(r["stock_exposure"].mean(), 3),
                   "avg_mis_gross_x": round(r["mis_gross_x"].mean(), 2), "mis_trades": int(r["mis_trades"].sum()),
                   "mis_pnl_rs": round(r["mis_pnl"].sum()), "pledge_orders": int(r["pledge_orders"].sum()),
                   "cnc_costs_rs": round(r["cnc_cost"].sum()), "mis_costs_rs": round(r["mis_cost"].sum())})
    gc = pd.DataFrame(gc)
    gc.to_csv(out / "goals_check.csv", index=False)
    pd.DataFrame({k: L.to_monthly(v) for k, v in series.items()}).to_csv(out / "monthly_returns.csv")

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for k, r in daily_all.items():
        ax[0].plot(r["equity"], label=k, lw=1.1)
        ax[1].plot(r["drawdown"] * 100, lw=1)
    for a in ax:
        a.axvline(mid, color="grey", ls=":", lw=1)
    ax[0].axhline(L.CFG["capital"], color="k", lw=0.6)
    ax[0].legend(fontsize=8); ax[0].set_title("Equity (Rs) | dotted line = selection/holdout split")
    ax[1].axhline(-20, color="red", ls="--"); ax[1].set_title("Drawdown % (Goal 1 limit -20%)")
    fig.tight_layout(); fig.savefig(out / "equity_drawdown.png", dpi=120); plt.close(fig)
    mb = L.to_monthly(series[bfam]) * 100
    fig, ax = plt.subplots(figsize=(12, 4.5))
    mb.plot(kind="bar", ax=ax, color=["tab:green" if x >= -0.5 else "tab:red" for x in mb])
    ax.axhline(-0.5, color="red", ls="--"); ax.axhline(3, color="green", ls="--")
    ax.set_title(f"Monthly returns % - {bfam}")
    fig.tight_layout(); fig.savefig(out / "monthly_returns_best.png", dpi=120); plt.close(fig)

    md = f"""# Phase 6 - Integrated portfolio backtest (CNC + pledge + MIS)

CNC model `{inp['dcol']}`, MIS model `{inp['icol']}` (side-specific causal calibration).
Combined OOS period {mis_start.date()} -> {end.date()}; configs selected on < {mid.date()}, judged on the holdout after it.
Capital Rs {L.CFG['capital']:,.0f}. Grid: {grid}

## Full-period scorecards of the frozen configs
{L.md_table(pd.DataFrame(cards)[L.SC_COLS])}

## HOLDOUT-only scorecards (the honest number)
{L.md_table(pd.DataFrame(hold_cards)[L.SC_COLS])}

## Mandate check
{L.md_table(gc)}

## Long-history context: CNC sleeve only (no MIS)
{L.md_table(pd.DataFrame(long_cards)[L.SC_COLS])}

**Best family:** {bfam} -> P(profitable 12m) {best.get('P_profitable_12m')}, predicted monthly {best.get('pred_monthly_mean')}
(5-95%: {best.get('pred_monthly_p05')} .. {best.get('pred_monthly_p95')}), P(all 12 months >= -0.5%)
{best.get('P_all_12_months_ge_floor')}, P(Goal 1 both) {best.get('P_goal1_both')}.
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    L.print_scorecard(best, PHASE)
    print(f"[P6] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
