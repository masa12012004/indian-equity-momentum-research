#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 7 - Is the edge real? Robustness, overfitting statistics, go / no-go,
FINAL_REPORT.md against the mandate.

Checks (Vols 6, 8, 9 of the strategy volumes)
---------------------------------------------
  * excess Sharpe over the random-pick null (same universe, same k, same costs);
    excess > 1.2 is treated as "too good to be true" -> look for a leak
  * Lo (2002) standard error of the Sharpe ratio (monthly, autocorrelation-aware)
  * Probability of Backtest Overfitting (CSCV, Bailey et al. 2017) on the whole variant grid
  * Deflated Sharpe Ratio with the TOTAL number of trials (grid + Optuna + target + DL)
  * CPCV 5th-percentile gate from phase 4
  * era table, train 70% / test 30% segments
  * stress: costs x2, 25 bp slippage floor, 50 bp stop slippage, liquid yield 4%, no regime layer
  * PSI drift of the model features (last 63 days vs train window) -> degradation protocol
  * stationary-bootstrap probabilities for every goal of the mandate
Output folder: phase7_robustness_report/  (+ FINAL_REPORT.md)
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import p1_data_costs as L
import p6_portfolio_backtest as E

plt = L.plt
PHASE = "phase7_robustness_report"
P6 = L.HERE / "phase6_portfolio_backtest"
CNC = "CNC-only long (buy side, delivery, idle cash 0%)"
SWEEP = "CNC + liquid-ETF sweep (selected)"


def pbo_cscv(M: pd.DataFrame, S: int = 8) -> dict:
    """M: (T periods x N variants) returns. Probability that the in-sample best is below the
    out-of-sample median (logit < 0)."""
    T = len(M) - len(M) % S
    blocks = np.array_split(np.arange(T), S)
    X = M.iloc[:T].to_numpy()
    logits = []
    for comb in itertools.combinations(range(S), S // 2):
        is_ = np.concatenate([blocks[i] for i in comb]); oos = np.setdiff1d(np.arange(T), is_)
        sr_is = X[is_].mean(0) / (X[is_].std(0) + 1e-12)
        sr_oos = X[oos].mean(0) / (X[oos].std(0) + 1e-12)
        best = int(np.argmax(sr_is))
        w = (sr_oos < sr_oos[best]).mean() + 0.5 / len(sr_oos)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"PBO": round(float((logits <= 0).mean()), 3), "n_combinations": len(logits), "n_variants": M.shape[1],
            "median_logit": round(float(np.median(logits)), 3)}


def main():
    ap = L.base_args("Strategy02 Phase 7: robustness and final report")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    lines = pd.read_csv(P6 / "lines_daily_returns.csv", index_col=0, parse_dates=True)
    grid = pd.read_csv(P6 / "grid_daily_returns.csv", index_col=0, parse_dates=True)
    bestv = json.loads((P6 / "best_variant.json").read_text())
    nulls = pd.read_csv(P6 / "null_draws.csv")
    gate4 = json.loads((L.HERE / "phase4_ml_models" / "lgbm_params.json").read_text())["gate"]
    adopt5 = json.loads((L.HERE / "phase5_deep_learning" / "adoption.json").read_text())
    oos_start = pd.Timestamp(bestv["test"][0]) if "test" in bestv else E.TEST[0]

    sel = lines[SWEEP].dropna(); cnc = lines[CNC].dropna()
    res = {}
    # ---- excess over the random null (Sharpe on monthly returns, same convention as null_draws)
    msr = lambda s: float(L.to_monthly(s).mean() / L.to_monthly(s).std() * np.sqrt(12))
    null_med = float(nulls["sharpe"].median())
    for lab, s in (("cnc_only", cnc), ("cnc_sweep", sel)):
        ex = msr(s) - null_med
        res[f"{lab}_sharpe_monthly"] = round(msr(s), 3)
        res[f"{lab}_excess_over_null"] = round(ex, 3)
        res[f"{lab}_null_percentile"] = round(float((nulls["sharpe"] < msr(s)).mean()), 3)
        res[f"{lab}_leak_flag_excess_gt_1.2"] = bool(ex > 1.2)
        res[f"{lab}_oos_sharpe_monthly"] = round(msr(s.loc[oos_start:]), 3)
    res["null_median_sharpe"] = round(null_med, 3)
    # same comparison restricted to the out-of-sample window, against the null's OOS distribution
    nd = pd.read_csv(P6 / "null_daily_returns.csv", index_col=0, parse_dates=True).loc[oos_start:]
    null_oos = np.array([msr(nd[c]) for c in nd.columns])
    null_oos_cagr = np.array([(1 + nd[c]).prod() ** (252 / len(nd)) - 1 for c in nd.columns])
    res["null_oos_median_sharpe"] = round(float(np.median(null_oos)), 3)
    res["null_oos_median_cagr"] = round(float(np.median(null_oos_cagr)), 4)
    for lab, s in (("cnc_only", cnc), ("cnc_sweep", sel)):
        so = s.loc[oos_start:]
        res[f"{lab}_oos_excess_over_null"] = round(msr(so) - float(np.median(null_oos)), 3)
        res[f"{lab}_oos_null_percentile_sharpe"] = round(float((null_oos < msr(so)).mean()), 3)
        cg = (1 + so).prod() ** (252 / len(so)) - 1
        res[f"{lab}_oos_null_percentile_cagr"] = round(float((null_oos_cagr < cg).mean()), 3)
    lo = L.lo_sharpe_se(L.to_monthly(sel.loc[oos_start:]))
    res["lo_sharpe_oos"] = lo
    # ---- PBO on the variant grid (monthly)
    Mg = (1 + grid).groupby(grid.index.to_period("M")).prod() - 1
    res["pbo"] = pbo_cscv(Mg.dropna(axis=1, how="all").fillna(0))
    # ---- DSR with every trial counted
    n_trials = int(grid.shape[1] + gate4.get("optuna_trials", 0) + adopt5.get("dl_trials", 0))
    sr_var = float((grid.mean() / grid.std()).var())
    res["dsr_full"] = L.deflated_sharpe(sel, n_trials, sr_var)
    res["dsr_oos"] = L.deflated_sharpe(sel.loc[oos_start:], n_trials, sr_var)
    res["cpcv_gate_phase4"] = gate4
    res["dl_adoption_phase5"] = adopt5

    # ---- stress tests on the selected configuration (re-simulated)
    print("[P7] stress tests ...")
    mk = E.Market(data)
    scores, meta = E.load_scores(mk)
    regime = pd.read_csv(L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv", index_col=0, parse_dates=True)
    bp = E.Params(**{k: v for k, v in bestv["params"].items()})
    start = sel.index[0]
    scen = {"base (selected, sweep)": bp, "costs x2": replace(bp, cost_mult=2.0),
            "slippage floor 25 bp": replace(bp, slip_floor_bps=25.0), "stop slippage 50 bp": replace(bp, stop_slip_bps=50.0),
            "liquid yield 4%": replace(bp, liquid_yield_override=0.04), "no regime layer": replace(bp, regime=False),
            "CNC-only, idle cash 0%": replace(bp, sweep=False), "CNC-only, costs x2": replace(bp, sweep=False, cost_mult=2.0)}
    srows, sser = [], {}
    for lab, p in scen.items():
        d, _ = E.run(mk, p, scores[p.score], meta if p.score == "final" else None, regime, start)
        sser[lab] = d["ret"]
        srows.append({"scenario": lab, **E.metrics(d["ret"]), **{f"oos_{k}": v for k, v in E.metrics(d["ret"].loc[oos_start:]).items()}})
    stress = pd.DataFrame(srows)
    stress.to_csv(out / "stress_tests.csv", index=False)

    # ---- PSI drift of model features (train window vs last 63 trading days)
    feats = pd.read_csv(L.HERE / "phase3_features" / "daily_features.csv", parse_dates=["date"])
    fl = json.loads((L.HERE / "phase3_features" / "feature_lists.json").read_text())
    tr = feats[feats["date"] <= pd.Timestamp(fl["train_end"])]
    last = feats[feats["date"] >= feats["date"].drop_duplicates().nlargest(63).min()]
    # stock features are cross-sectional ranks (uniform by construction) -> PSI on the RAW stock-level
    # inputs; market features are one value per day -> "outside the train 1-99% range" flags instead
    raw_cols = ["vol_63_raw", "atr14_raw", "amihud_raw"]
    psi = {c: round(L.psi(tr[c].dropna().to_numpy(), last[c].dropna().to_numpy()), 4) for c in raw_cols if c in feats}
    pd.Series(psi, name="psi").to_csv(out / "psi_drift.csv", index_label="feature")
    drift = {k: v for k, v in psi.items() if v > 0.25}
    trd = tr.drop_duplicates("date")
    cur = last.drop_duplicates("date")
    res["market_feature_out_of_range_share_last63d"] = {
        c: round(float(((cur[c] < trd[c].quantile(0.01)) | (cur[c] > trd[c].quantile(0.99))).mean()), 3)
        for c in fl["market_features"] if c in feats}
    res["psi_max"] = max(psi.values()); res["psi_drifted_features"] = drift
    res["degradation_protocol"] = "HALVE position size until drift resolves" if drift else "normal size"

    # ---- goal probabilities (bootstrap) on the OUT-OF-SAMPLE part and on the full window
    cards = []
    for lab in [CNC, SWEEP, "CNC pledged + MIS overlay (v1 intraday sleeve)", "NIFTY50 buy&hold",
                [c for c in lines if c.startswith("Random")][0], "EW NIFTY100 monthly", "Liquid ETF only"]:
        if lab in lines:
            cards.append(L.scorecard(lines[lab].loc[oos_start:], f"{lab} [OOS]"))
            cards.append(L.scorecard(lines[lab], f"{lab} [full]"))
    sc = pd.DataFrame(cards)
    best = L.save_scorecards(out, [c for c in cards if c["strategy"].endswith("[OOS]")], PHASE,
                             notes="Out-of-sample scorecards (TEST 30%).",
                             series={c: lines[c].loc[oos_start:] for c in lines})

    # ---- go / no-go
    oos_ex = res["cnc_sweep_oos_excess_over_null"]
    checks = {
        "OOS Sharpe above the random-null OOS median": oos_ex > 0,
        "OOS excess Sharpe over null > 1 Lo standard error": oos_ex > (lo.get("se") or 9),
        "OOS CAGR above the null's median CAGR": res["cnc_sweep_oos_null_percentile_cagr"] > 0.5,
        "excess over null below the 1.2 leak ceiling": not res["cnc_sweep_leak_flag_excess_gt_1.2"],
        "PBO < 0.5": res["pbo"]["PBO"] < 0.5,
        "Deflated Sharpe (full) > 0.90": res["dsr_full"]["DSR"] > 0.90,
        "survives costs x2 with CAGR > NIFTY50": float(stress.set_index("scenario").loc["costs x2", "CAGR"]) >
                                                  float(E.metrics(lines["NIFTY50 buy&hold"])["CAGR"]),
        "no feature drift (PSI <= 0.25)": not drift,
        "max drawdown better than -20% (full)": float(E.metrics(sel)["max_drawdown"]) > -0.20,
    }
    n_pass = sum(checks.values())
    significant = checks["OOS excess Sharpe over null > 1 Lo standard error"]
    verdict = ("GO - paper trade 3 months, then live at half size" if (n_pass >= len(checks) - 1 and significant) else
               "CONDITIONAL - the edge over random picking is not yet statistically proven; paper trade first"
               if n_pass >= len(checks) - 3 else "NO-GO")
    res["checks"] = {k: bool(v) for k, v in checks.items()}
    res["verdict"] = verdict
    (out / "robustness.json").write_text(json.dumps(res, indent=2, default=str))

    # ---- charts
    oos_lines = {k: v.loc[oos_start:] for k, v in lines.items()}
    L.multi_panel(oos_lines, out / "oos_only_all_lines.png",
                  f"Out-of-sample only ({oos_start.date()} .. end): every book from Rs 1,00,000", highlight=CNC)
    L.multi_panel(sser, out / "stress_all_lines.png", "Selected book under stress scenarios", highlight="CNC-only, idle cash 0%")

    prof = pd.read_csv(P6 / "profit_table.csv")
    era = pd.read_csv(P6 / "era_table.csv")
    m_sel = L.to_monthly(sel)
    g = {"Goal 1a  max DD better than -20%": E.metrics(sel)["max_drawdown"] > -0.20,
         "Goal 1b  trailing 12m > +20% at every point": bool(L.rolling_12m(sel).dropna().min() > 0.20),
         "Goal 2   every month inside [-0.5%, +3%]": bool(((m_sel >= -0.005) & (m_sel <= 0.03)).all())}
    sc_cols = ["strategy", "CAGR", "sharpe", "max_drawdown", "P_profitable_12m", "pred_monthly_mean", "P_month_ge_floor",
               "P_12m_gt_20pct", "P_maxdd_ok_12m", "P_goal1_both", "goal_score"]
    full_tab = prof[prof["window"] == "full"].drop(columns="window")
    seg = prof[prof["line"].isin([CNC, SWEEP, "NIFTY50 buy&hold"])]
    (out / "FINAL_REPORT.md").write_text(f"""# Strategy02 - FINAL REPORT

Capital Rs 1,00,000 - Flattrade CNC (delivery, long only) monthly swing book on NIFTY100.
Selected configuration (chosen on {bestv['selected_on']} only): **{bestv['params']['name']}**

## Verdict: **{verdict}**  ({n_pass}/{len(checks)} checks passed)
{L.md_table(pd.DataFrame({'check': list(checks), 'pass': [bool(v) for v in checks.values()]}))}

## Profits - what Rs 1,00,000 became ({sel.index[0].date()} .. {sel.index[-1].date()})
{L.md_table(full_tab, 4)}

## By window (train 70% = in-sample selection, test 30% = out-of-sample)
{L.md_table(seg, 4)}

## Probability of profitability / predicted profitability (stationary bootstrap, 12 months ahead)
{L.md_table(sc[[c for c in sc_cols if c in sc]], 4)}

## Overfitting statistics
* Random-pick null median Sharpe {null_med:.2f}; CNC-only Sharpe {res['cnc_only_sharpe_monthly']}
  (excess {res['cnc_only_excess_over_null']}, beats {res['cnc_only_null_percentile']*100:.0f}% of random books);
  sweep book {res['cnc_sweep_sharpe_monthly']} (excess {res['cnc_sweep_excess_over_null']}).
* OOS Sharpe {res['cnc_sweep_oos_sharpe_monthly']} with Lo standard error {lo.get('se')}.
* PBO {res['pbo']['PBO']} over {res['pbo']['n_variants']} variants; DSR full {res['dsr_full']['DSR']}, OOS {res['dsr_oos']['DSR']} with {n_trials} trials.
* CPCV gate (phase 4 LightGBM): 5th percentile IC {gate4.get('p05_ic')} -> pass = {gate4.get('gate_p05_gt_0')}.
* DL adoption: {adopt5.get('adopted')} (paired t {adopt5.get('paired_t')}).
* PSI drift max {res['psi_max']:.3f} -> {res['degradation_protocol']}.

## Stress tests
{L.md_table(stress[['scenario','final_value_rs','CAGR','sharpe','max_drawdown','oos_CAGR','oos_max_drawdown']], 4)}

## Mandate goals (historical, selected sweep book)
{L.md_table(pd.DataFrame({'goal': list(g), 'met': [bool(v) for v in g.values()]}))}
* Goal 3/4 (CNC pledged + MIS 5x, 80/20): the MIS sleeve from strategy/ did not survive costs there;
  the overlay line uses that sleeve's P&L where it exists and is shown for comparison only.

## Eras (total return)
{L.md_table(era, 3)}

## Caveats
* Universe = today's NIFTY100 (survivorship bias). That is why the random-pick null, not NIFTY50,
  is the fair hurdle. Size/liquidity features were excluded for the same reason.
* Raw prices exist only from 2023-10; earlier whole-share sizing uses adjusted prices.
* Taxes (STCG 20% / LTCG 12.5%) are not deducted.

![oos](oos_only_all_lines.png)
![stress](stress_all_lines.png)
""")
    (out / "report.md").write_text((out / "FINAL_REPORT.md").read_text())
    L.print_scorecard(best, PHASE)
    print(f"[P7] verdict: {verdict}  ({n_pass}/{len(checks)})")
    print(f"[P7] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
