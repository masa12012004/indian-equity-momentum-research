#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 7 - Robustness, probability of profitability, overfitting tests, final report.

  * Stationary-bootstrap forward simulation (10,000 one-year paths) of the selected strategy:
    probability of profitability, of every month >= -0.5%, of 12m return > 20%,
    of max drawdown better than -20%, predicted monthly return distribution, equity fan chart
  * Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation
    (Bailey, Borwein, Lopez de Prado & Zhu 2017) across every Phase-6 grid configuration
  * Deflated Sharpe Ratio with the number of configurations tried
  * Stress tests: 1.5x / 2x spreads, double pledge fee, DP fee charged, lower MIS leverage,
    lower liquid-ETF yield, delayed start
  * Local parameter sensitivity around the chosen configuration
  * Return attribution by regime and by calendar year
  * Phase-by-phase progression of probability of profitability / predicted profitability
  * FINAL_REPORT.md with mandate compliance and a deployment checklist
Output folder: phase7_robustness_report/
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import p1_data_costs as L
import p6_portfolio_backtest as P6

plt = L.plt
PHASE = "phase7_robustness_report"


def pbo_cscv(M: pd.DataFrame, S: int = 8) -> dict:
    """CSCV: split T into S blocks; for every half/half combination pick the in-sample best
    (by Sharpe) and record its out-of-sample relative rank. PBO = P(logit < 0)."""
    M = M.dropna(how="all").fillna(0.0)
    T, N = M.shape
    if N < 3 or T < S * 10:
        return {"PBO": None, "note": "not enough configs/data"}
    blocks = np.array_split(np.arange(T), S)
    lam, deg = [], []
    X = M.to_numpy()
    for comb in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        sr_is = X[is_idx].mean(0) / (X[is_idx].std(0) + 1e-12)
        sr_oos = X[oos_idx].mean(0) / (X[oos_idx].std(0) + 1e-12)
        b = int(np.argmax(sr_is))
        w = (sr_oos < sr_oos[b]).sum() / N + 0.5 / N            # relative rank in (0,1)
        lam.append(np.log(w / (1 - w)))
        deg.append((sr_is[b], sr_oos[b]))
    lam = np.array(lam)
    d = np.array(deg)
    slope = np.polyfit(d[:, 0], d[:, 1], 1)[0] if len(d) > 2 else np.nan
    return {"PBO": round(float((lam < 0).mean()), 4), "n_combinations": len(lam), "n_configs": N,
            "median_logit": round(float(np.median(lam)), 3),
            "IS_to_OOS_sharpe_slope": round(float(slope), 3),
            "mean_OOS_sharpe_of_IS_best_daily": round(float(d[:, 1].mean()), 4)}


def fan_chart(ret: pd.Series, path, n=10000):
    rng = np.random.default_rng(L.CFG["seed"])
    paths = L.stationary_bootstrap(ret.dropna().to_numpy(), n, 252, L.CFG["boot_block"], rng)
    eq = L.CFG["capital"] * np.cumprod(1 + paths, axis=1)
    q = np.percentile(eq, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 253)
    ax.fill_between(x, q[0], q[4], alpha=0.2, label="5-95%")
    ax.fill_between(x, q[1], q[3], alpha=0.35, label="25-75%")
    ax.plot(x, q[2], color="k", label="median")
    ax.axhline(L.CFG["capital"], color="grey", lw=0.8)
    ax.axhline(L.CFG["capital"] * 1.2, color="green", ls="--", label="+20% (Goal 1)")
    ax.set_xlabel("trading days ahead"); ax.set_ylabel("equity Rs"); ax.legend(); ax.set_title("12-month forward equity fan (bootstrap)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    months = (1 + paths[:, :252].reshape(n, 12, 21)).prod(axis=2) - 1
    return pd.DataFrame({"month_ahead": np.arange(1, 13),
                         "P_month_ge_-0.5%": (months >= -0.005).mean(0).round(3),
                         "P_month_in_band": ((months >= -0.005) & (months <= 0.03)).mean(0).round(3),
                         "median_month": np.median(months, 0).round(4),
                         "p05_equity_rs": np.percentile(eq[:, 20::21][:, :12], 5, axis=0).round(0),
                         "median_equity_rs": np.median(eq[:, 20::21][:, :12], axis=0).round(0),
                         "p95_equity_rs": np.percentile(eq[:, 20::21][:, :12], 95, axis=0).round(0)})


def main():
    ap = L.base_args("Phase 7: robustness and final report")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    p6 = L.HERE / "phase6_portfolio_backtest"
    bv = json.loads((p6 / "best_variant.json").read_text())
    split = json.loads((p6 / "selection_split.json").read_text())
    params = P6.Params(**bv["params"])
    print(f"[P7] selected strategy: {bv['variant']}  {P6.tag(params)}")
    inp = P6.load_inputs(data)
    base, _, _ = P6.simulate(inp, params, start=split["combined_start"])
    ret = base["ret"]

    # ---------------- alpha excluding the liquid-ETF carry ----------------------
    liq_r = inp["liq"].reindex(ret.index).fillna(0)
    alpha = ret - liq_r * (1 - base["stock_exposure"].shift(1).fillna(0))
    sc_alpha = L.scorecard(alpha, bv["variant"] + " [alpha ex liquid carry]", n_boot=5000)
    _, _, mt_base = P6.simulate(inp, params, start=split["combined_start"], log_trades=True)
    tstats = P6.mis_trade_stats(mt_base)
    (out / "mis_trade_stats.json").write_text(json.dumps(tstats, indent=2))
    G = pd.read_csv(p6 / "grid_daily_returns.csv", index_col=0, parse_dates=True)
    # Deflated Sharpe on the ALPHA (carry removed): a near-constant liquid yield would otherwise
    # produce an absurdly high Sharpe and a meaningless DSR.
    dsr = L.deflated_sharpe(alpha, n_trials=G.shape[1], sr_trials_var=float((G.mean() / G.std()).var()))

    # ---------------- forward probabilities ---------------------------------
    sc_full = L.scorecard(ret, bv["variant"], n_boot=10000)
    sc_hold = L.scorecard(ret.loc[split["selection_end"]:], bv["variant"] + " [holdout]", n_boot=10000)
    fan = fan_chart(ret, out / "forward_equity_fan.png")
    fan.to_csv(out / "forward_monthly_probabilities.csv", index=False)

    # ---------------- overfitting -------------------------------------------
    # rank configs on returns in EXCESS of the liquid carry (otherwise near-riskless carry dominates Sharpe)
    G_ex = G.sub(inp["liq"].reindex(G.index).fillna(0), axis=0)
    pbo = pbo_cscv(G_ex, S=6 if args.quick else 8)

    (out / "overfitting.json").write_text(json.dumps({"PBO_CSCV": pbo, "deflated_sharpe": dsr}, indent=2))

    # ---------------- stress tests ------------------------------------------
    stress = []
    scen = [("base", {}, {}), ("spreads x1.5", {"cost_mult": 1.5}, {}), ("spreads x2", {"cost_mult": 2.0}, {}),
            ("pledge fee x2", {}, {"pledge_fee": L.CFG["pledge_fee"] * 2}),
            ("DP fee Rs 15.93/sell", {}, {"dp_fee_per_sell": 13.5 * 1.18}),
            ("MIS leverage 3x", {}, {"mis_leverage": 3.0}), ("liquid yield 4%", {}, {"liquid_yield": 0.04}),
            ("stop slippage 15 bps", {"stop_slip_bps": 15.0}, {}),
            ("start +3 months", {"__start": 63}, {})]
    saved = dict(L.CFG)
    for lab, pmod, cmod in scen:
        L.CFG.update(cmod)
        if "liquid_yield" in cmod:
            inp["liq"] = L.liquid_returns(data, inp["P"]["adj_close"].index)
        st = split["combined_start"]
        pm = dict(pmod)
        if "__start" in pm:
            st = inp["P"]["adj_close"].index[inp["P"]["adj_close"].index.searchsorted(pd.Timestamp(st)) + pm.pop("__start")]
        r, _, _ = P6.simulate(inp, replace(params, **pm), start=st)
        c = L.scorecard(r["ret"], lab, n_boot=1000)
        stress.append({k: c.get(k) for k in ["strategy", "CAGR", "max_drawdown", "mean_monthly", "frac_months_ge_floor",
                                             "P_profitable_12m", "P_month_ge_floor", "goal_score"]})
        L.CFG.clear(); L.CFG.update(saved)
        inp["liq"] = L.liquid_returns(data, inp["P"]["adj_close"].index)
    stress = pd.DataFrame(stress)
    stress.to_csv(out / "stress_tests.csv", index=False)

    # ---------------- sensitivity -------------------------------------------
    sens = []
    grid_s = [("hurdle_margin", [3e-4, 1e-3, 1.5e-3, 2e-3]), ("max_loss_per_trade", [0.001, 0.002, 0.003]),
              ("dd_hard", [0.14, 0.18, 0.22]), ("month_stop", [-0.003, -0.004, -0.006])]
    if params.stock_frac > 0:            # stock-sleeve parameters only matter when there is a stock sleeve
        grid_s = [("vol_target", [0.10, 0.15, 0.20, 0.25]), ("hold", [3, 5, 10, 15]), ("k", [3, 4, 6, 8])] + grid_s
    for fld, vals in grid_s:
        for v in vals:
            r, _, _ = P6.simulate(inp, replace(params, **{fld: v}), start=split["combined_start"])
            c = L.scorecard(r["ret"], f"{fld}={v}", n_boot=500)
            sens.append({"param": fld, "value": v, "CAGR": c.get("CAGR"), "max_dd": c.get("max_drawdown"),
                         "frac_months_ge_floor": c.get("frac_months_ge_floor"), "goal_score": c.get("goal_score"),
                         "is_chosen": getattr(params, fld) == v})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity.csv", index=False)
    fig, axs = plt.subplots(3, 3, figsize=(13, 10))
    for ax in axs.ravel()[sens["param"].nunique():]:
        ax.axis("off")
    for ax, (fld, g) in zip(axs.ravel(), sens.groupby("param", sort=False)):
        ax.plot(g["value"].astype(float), g["goal_score"], "o-")
        ch = g[g["is_chosen"]]
        ax.plot(ch["value"].astype(float), ch["goal_score"], "r*", ms=14)
        ax.set_title(fld); ax.grid(alpha=0.3)
    fig.suptitle("Goal-score sensitivity (red star = chosen)")
    fig.tight_layout(); fig.savefig(out / "sensitivity.png", dpi=120); plt.close(fig)

    # ---------------- attribution -------------------------------------------
    reg = inp["reg"]
    att = pd.DataFrame({"ret": ret, "risk_on": reg["risk_on"].shift(1).reindex(ret.index)})
    by_reg = att.groupby("risk_on")["ret"].agg(["mean", "std", "count"])
    by_reg["ann_ret"], by_reg["ann_vol"] = by_reg["mean"] * 252, by_reg["std"] * np.sqrt(252)
    by_reg.to_csv(out / "attribution_by_regime.csv")
    by_year = ret.groupby(ret.index.year).apply(lambda r: (1 + r).prod() - 1).rename("return")
    by_year.to_csv(out / "attribution_by_year.csv")
    comp = pd.DataFrame({"mis_pnl_rs": base["mis_pnl"].sum(), "cnc_costs_rs": base["cnc_cost"].sum(),
                         "mis_costs_rs": base["mis_cost"].sum(), "pledge_orders": base["pledge_orders"].sum(),
                         "pledge_fees_rs": base["pledge_orders"].sum() * L.CFG["pledge_fee"],
                         "final_equity_rs": base["equity"].iloc[-1]}, index=[0])
    comp.to_csv(out / "pnl_components.csv", index=False)

    # ---------------- progression across phases ------------------------------
    prog = []
    for ph in L.PHASES[:-1]:
        f = L.HERE / ph / "scorecard.json"
        if f.exists():
            j = json.loads(f.read_text())
            b, nb = j.get("best", {}), j.get("best_new_this_phase", {})
            prog.append({"phase": ph, "best_strategy": b.get("strategy"), "P_profitable_12m": b.get("P_profitable_12m"),
                         "pred_monthly_mean": b.get("pred_monthly_mean"), "P_month_ge_floor": b.get("P_month_ge_floor"),
                         "P_all_12_months_ge_floor": b.get("P_all_12_months_ge_floor"),
                         "P_goal1_both": b.get("P_goal1_both"), "goal_score": b.get("goal_score"),
                         "new_idea_this_phase": nb.get("strategy"), "new_idea_goal_score": nb.get("goal_score")})
    prog.append({"phase": PHASE, "best_strategy": bv["variant"] + " (full OOS, 10k bootstrap)",
                 "P_profitable_12m": sc_full["P_profitable_12m"], "pred_monthly_mean": sc_full["pred_monthly_mean"],
                 "P_month_ge_floor": sc_full["P_month_ge_floor"], "P_all_12_months_ge_floor": sc_full["P_all_12_months_ge_floor"],
                 "P_goal1_both": sc_full["P_goal1_both"], "goal_score": sc_full["goal_score"],
                 "new_idea_this_phase": "holdout only", "new_idea_goal_score": sc_hold.get("goal_score")})
    prog = pd.DataFrame(prog)
    prog.to_csv(out / "phase_progression.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(prog))
    for colm, lab in [("P_profitable_12m", "P(profitable 12m)"), ("P_month_ge_floor", "P(month >= -0.5%)"),
                      ("P_goal1_both", "P(Goal 1 both)"), ("goal_score", "goal score")]:
        ax.plot(x, prog[colm].astype(float), "o-", label=lab)
    ax2 = ax.twinx()
    ax2.bar(x, prog["pred_monthly_mean"].astype(float) * 100, alpha=0.2, color="grey", label="pred. monthly %")
    ax2.set_ylabel("predicted mean monthly return %")
    ax.set_xticks(x); ax.set_xticklabels([p.replace("phase", "P").split("_")[0] for p in prog["phase"]])
    ax.set_ylim(0, 1.05); ax.legend(loc="upper left", fontsize=8); ax.set_title("Probability of profitability by phase")
    fig.tight_layout(); fig.savefig(out / "phase_progression.png", dpi=120); plt.close(fig)

    cards = [sc_full, sc_hold]
    best = L.save_scorecards(out, cards, PHASE, series={bv["variant"]: ret},
                             notes="Final: selected Phase-6 strategy, full OOS + holdout, 10k-path bootstrap.")
    L.plot_equity({bv["variant"]: ret}, out / "final_equity.png", "Selected strategy (OOS)", log=False)

    m = L.to_monthly(ret)
    r12 = L.rolling_12m(ret).dropna()
    verdict = {
        "Goal 1a max DD better than -20%": f"{'MET' if base['drawdown'].min() >= -0.2 else 'NOT MET'} (hist {base['drawdown'].min():.1%}; P {sc_full['P_maxdd_ok_12m']:.0%})",
        "Goal 1b trailing 12m > +20%": f"{'MET' if len(r12) and r12.min() > 0.2 else 'NOT MET'} (min trailing 12m {r12.min() if len(r12) else float('nan'):.1%}; P(12m>20%) {sc_full['P_12m_gt_20pct']:.0%})",
        "Goal 2 every month in [-0.5%, +3%]": f"{'MET' if ((m >= -0.005) & (m <= 0.03)).all() else 'NOT MET'} ({((m >= -0.005)).mean():.0%} of months >= -0.5%; P(all 12 next months) {sc_full['P_all_12_months_ge_floor']:.1%})",
        "Goal 3 CNC+pledge+MIS": "implemented (V1); 2-day holds lose to costs, hold length selected by data",
        "Goal 4 80/20": "implemented (V2)",
        "Goal 5 other distributions": "implemented (V3 pledged-liquid 50/50, V4 liquid+MIS, V5 adaptive)",
    }
    md = f"""# FINAL REPORT - Indian equity CNC + pledge + MIS strategy (Rs {L.CFG['capital']:,.0f}, Flattrade)

## Selected strategy
**{bv['variant']}** with `{P6.tag(params)}` (frozen on the selection window, {split['combined_start']} -> {split['selection_end']}).

| metric | full OOS | holdout only |
|---|---|---|
| CAGR | {sc_full['CAGR']:.2%} | {sc_hold.get('CAGR', float('nan')):.2%} |
| annual vol | {sc_full['ann_vol']:.2%} | {sc_hold.get('ann_vol', float('nan')):.2%} |
| max drawdown | {sc_full['max_drawdown']:.2%} | {sc_hold.get('max_drawdown', float('nan')):.2%} |
| mean monthly | {sc_full['mean_monthly']:.2%} | {sc_hold.get('mean_monthly', float('nan')):.2%} |
| months >= -0.5% | {sc_full['frac_months_ge_floor']:.0%} | {sc_hold.get('frac_months_ge_floor', float('nan')):.0%} |
| **probability of profitability (12m)** | **{sc_full['P_profitable_12m']:.1%}** | {sc_hold.get('P_profitable_12m', float('nan')):.1%} |
| **predicted mean monthly return** | **{sc_full['pred_monthly_mean']:.2%}** (5-95%: {sc_full['pred_monthly_p05']:.2%} .. {sc_full['pred_monthly_p95']:.2%}) | {sc_hold.get('pred_monthly_mean', float('nan')):.2%} |
| P(every one of next 12 months >= -0.5%) | {sc_full['P_all_12_months_ge_floor']:.1%} | {sc_hold.get('P_all_12_months_ge_floor', float('nan')):.1%} |
| P(12m return > 20%) | {sc_full['P_12m_gt_20pct']:.1%} | {sc_hold.get('P_12m_gt_20pct', float('nan')):.1%} |
| P(max DD better than -20%) | {sc_full['P_maxdd_ok_12m']:.1%} | {sc_hold.get('P_maxdd_ok_12m', float('nan')):.1%} |

## How much is real alpha vs. liquid-ETF carry?
The liquid ETF yield is an assumption (CFG liquid_yield = {L.CFG['liquid_yield']:.1%} unless LIQUIDBEES data is present).
Strategy return minus the carry on the non-stock capital:

| metric | alpha only |
|---|---|
| annualised alpha | {sc_alpha['CAGR']:.2%} |
| probability alpha > 0 over 12m | {sc_alpha['P_profitable_12m']:.1%} |
| deflated Sharpe ratio of the alpha | {dsr['DSR']} |

MIS trade statistics: {tstats}
If `pnl_ex_best_trade_rs` is small or negative, the MIS result depends on very few trades and should not be trusted yet.

## Mandate compliance
{chr(10).join(f'* **{k}:** {v}' for k, v in verdict.items())}

## Progression across phases
{L.md_table(prog)}

## Overfitting diagnostics
* PBO (CSCV over {pbo.get('n_configs')} configs): **{pbo.get('PBO')}** (< 0.5 means the selection process tends to pick configs that also do well out of sample)
* Deflated Sharpe (alpha ex carry): {dsr}

## Stress tests
{L.md_table(stress)}

## Return attribution
By year: {by_year.round(4).to_dict()}

P&L components (Rs): {comp.round(0).to_dict('records')[0]}

## Deployment checklist (daily)
1. **15:20-15:30** - update data (`iifl_fetch_nifty100.py`, incremental) -> `clean_nifty100.py` -> score today's close with the
   frozen model; compute target CNC basket (hysteresis k / hold / buffer from the config above).
2. **09:00-09:08 pre-open / 09:15 open** - place CNC buys/sells (DELIVERY). Never exit a CNC holding with a MIS sell.
3. **T+1 after purchase** - pledge new holdings (Rs 20 + GST per ISIN); keep the liquid ETF pledged permanently.
4. **09:30 / 11:00 / 13:00** - MIS decisions only when calibrated edge > cost hurdle; place SL-M stops at the configured %.
5. **15:10-15:15** - square off all MIS (before Flattrade auto square-off).
6. Risk overlay daily: vol target, drawdown brake, monthly circuit breaker (no new MIS after MTD <= {params.month_stop:.1%}).
7. Re-run phases 2->7 monthly (walk-forward refit); compare live fills vs model costs.

## Verify before going live (assumptions in p1_data_costs.CFG)
* Flattrade DP charge per sell and unpledge fee (set to 0 here; stress test above shows the impact).
* Pledge haircut per stock and that pledged collateral is usable for equity MIS under the 50:50 rule.
* Liquid ETF yield / haircut (download LIQUIDBEES 1D data to replace the yield proxy).
* Data caveats: survivorship bias (today's NIFTY100), intraday OOS covers only ~2 years.

Figures: final_equity.png, forward_equity_fan.png, phase_progression.png, sensitivity.png
"""
    (out / "FINAL_REPORT.md").write_text(md, encoding="utf-8")
    (out / "report.md").write_text(md, encoding="utf-8")
    L.print_scorecard(best, PHASE)
    print(f"PBO={pbo.get('PBO')}  DSR={dsr.get('DSR')}")
    print(f"[P7] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
