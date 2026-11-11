#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 10 - Research-driven improvements that can be tested with the current data.

Pre-registered ideas (parameters from the papers / index methodologies, not tuned):
  * Rebalance less often: monthly vs quarterly vs semi-annual (NSE momentum whitepaper 2026:
    semi-annual beat quarterly in most years for Nifty200 Momentum 30; fewer trades -> lower
    costs and more gains held > 12 months for the 12.5% LTCG rate instead of 20% STCG).
  * More names: k = 8 / 12 / 16 (idiosyncratic risk falls quickly with k).
  * Residual momentum (Blitz, Huij & Martens 2011) and a consensus score = average percentile of
    vol-normalised momentum, Frog-in-the-Pan momentum and residual momentum (signal ensemble,
    less dependent on one definition - Novy-Marx 2015 warns against picking the best signal).
  * Kept from phase 8: equal weight, no stop, monthly SMA200 trend exit, idle cash in LIQUIDBEES.
Everything is selected on TRAIN (first 70%) only, then TEST is reported.
Outputs: phase10_research_improvements/ (train_70pct.png, test_30pct.png, grid_results.csv,
         metrics.csv, report.md)
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

import p1_data_costs as L
import p6_portfolio_backtest as E
import p8_train_test_split as S8

plt = L.plt
PHASE = "phase10_research_improvements"


def main():
    ap = L.base_args("Strategy02 Phase 10: research-driven improvements")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    mk = E.Market(data)
    scores, meta = E.load_scores(mk)
    regime = pd.read_csv(L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv", index_col=0, parse_dates=True)
    split = json.loads((L.HERE / "phase8_train_test_split" / "split.json").read_text())
    tr_w = (pd.Timestamp(split["train"][0]), pd.Timestamp(split["train"][1]))
    te_w = (pd.Timestamp(split["test"][0]), pd.Timestamp(split["test"][1]))
    prev = E.Params(**{k: v for k, v in split["params"].items() if k in E.Params.__dataclass_fields__})

    grid = []
    for sc, k, rb in itertools.product(["mom_vn", "mom_fip", "resid_mom", "consensus"], [8, 12, 16], [1, 3, 6]):
        if args.quick and (k == 16 or rb == 6):
            continue
        grid.append(E.Params(name=f"{sc}|EW|k{k}|rebal{rb}m|trend_sma200", score=sc, opt="EW", k=k, stops=False,
                             trend_exit="sma200", rebal_months=rb))
    rows, G, logs = [], {}, {}
    for p in grid:
        d, _ = E.run(mk, p, scores[p.score], None, regime, tr_w[0])
        G[p.name] = d["ret"]; logs[p.name] = d
        a = L.scorecard(d["ret"].loc[tr_w[0]:tr_w[1]], p.name, n_boot=300)
        b = E.metrics(d["ret"].loc[te_w[0]:te_w[1]])
        rows.append({"variant": p.name, "train_goal_score": a["goal_score"], "train_sharpe": a["sharpe"],
                     "train_CAGR": a["CAGR"], "train_max_dd": a["max_drawdown"], "test_sharpe": b["sharpe"],
                     "test_CAGR": b["CAGR"], "test_max_dd": b["max_drawdown"], "costs_rs": round(float(d["cost"].sum()))})
        print(f"   {p.name:<34} train Sharpe {a['sharpe']:>6} CAGR {a['CAGR']:+.3f} | test Sharpe {b['sharpe']:>6} CAGR {b['CAGR']:+.3f}")
    gt = pd.DataFrame(rows).sort_values(["train_goal_score", "train_sharpe"], ascending=False)
    gt.to_csv(out / "grid_results.csv", index=False)
    goal_pick = gt.iloc[0]["variant"]
    gt["score"] = gt["variant"].str.split("|").str[0]
    gt["k"] = gt["variant"].str.split("|").str[2]
    gt["rebal"] = gt["variant"].str.split("|").str[3]
    # ROBUST selection rule (pre-specified, TRAIN only): for each design dimension pick the level
    # with the highest MEDIAN train Sharpe across all other settings ("plateau" choice - a level that
    # is good everywhere, not one lucky combination); ties within 0.02 go to the cheaper (lower-cost) level.
    pick = {}
    for dim in ("score", "k", "rebal"):
        m = gt.groupby(dim).agg(sh=("train_sharpe", "median"), cost=("costs_rs", "median")).sort_values("sh", ascending=False)
        top = m[m["sh"] >= m["sh"].iloc[0] - 0.02].sort_values("cost")
        pick[dim] = top.index[0]
    chosen = gt[(gt["score"] == pick["score"]) & (gt["k"] == pick["k"]) & (gt["rebal"] == pick["rebal"])].iloc[0]["variant"]
    bp = next(p for p in grid if p.name == chosen)
    print(f"[P10] robust choice on TRAIN (marginal medians): {chosen}   | goal-score pick: {goal_pick}")
    ablation = pd.concat({dim: gt.groupby(dim)[["train_sharpe", "test_sharpe", "test_CAGR", "costs_rs"]].median()
                          for dim in ("score", "k", "rebal")})
    ablation.round(3).to_csv(out / "ablation_medians.csv")

    CNC = "CNC-only (stocks + LIQUIDBEES when out) - phase 10 robust choice"
    lines = {}
    d, tr = E.run(mk, bp, scores[bp.score], None, regime, tr_w[0], log_trades=True)
    lines[CNC] = d["ret"]; tr.to_csv(out / "trades_cnc_only.csv", index=False); d.to_csv(out / "daily_cnc_only.csv")
    shade = (d["stock_exposure"] < 0.05)
    dp, _ = E.run(mk, prev, scores[prev.score], meta if prev.score == "final" else None, regime, tr_w[0])
    lines[f"Phase 8 book: {prev.name}"] = dp["ret"]
    if goal_pick != chosen:
        lines[f"goal-score pick (train): {goal_pick}"] = G[goal_pick]
    for lab, pred in [("best monthly (train)", lambda v: "rebal1m" in v), ("best quarterly (train)", lambda v: "rebal3m" in v),
                      ("best semi-annual (train)", lambda v: "rebal6m" in v), ("best consensus score (train)", lambda v: v.startswith("consensus")),
                      ("best residual momentum (train)", lambda v: v.startswith("resid_mom"))]:
        sub = gt[gt["variant"].apply(pred)]
        if len(sub) and sub.iloc[0]["variant"] != chosen:
            lines[f"{lab}: {sub.iloc[0]['variant']}"] = G[sub.iloc[0]["variant"]]
    # no-selection benchmark: capital split equally across ALL variants (avoids picking-the-best bias)
    lines["Average of all grid variants (no selection)"] = pd.DataFrame(G).mean(1)
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
                       f"Strategy02 phase 10 - {wn}: {w[0].date()} .. {w[1].date()} (robust choice on TRAIN: {chosen})",
                       highlight=CNC, band=band, shade=shade.reindex(dates).fillna(False).loc[w[0]:w[1]])
        for k, s in win.items():
            met.append(S8.metrics_row(s, k, wn))
    met = pd.DataFrame(met)
    met.to_csv(out / "metrics.csv", index=False)
    cols = ["window", "line", "final_value_rs", "CAGR", "sharpe", "max_drawdown", "worst_month", "P_profitable_12m", "pred_monthly_mean"]
    key = met[met["line"].isin([CNC, f"Phase 8 book: {prev.name}", "Average of all grid variants (no selection)",
                                "NIFTY50 buy&hold", "Liquid ETF only"])]
    abl_tab = ablation.reset_index().rename(columns={"level_0": "dimension", "level_1": "value"})
    (out / "chosen.json").write_text(json.dumps({"robust_choice_on_train": chosen, "goal_score_pick_on_train": goal_pick,
                                                 "rule": "per dimension, highest median TRAIN Sharpe; ties<0.02 -> lower cost",
                                                 "levels": pick, "params": asdict(bp)}, indent=2, default=str))
    (out / "report.md").write_text(f"""# Strategy02 - Phase 10: research-driven improvements

Robust choice on TRAIN only (per dimension, highest median train Sharpe): **{chosen}**
(the plain goal-score pick was {goal_pick}).

## Key lines
{L.md_table(key[cols], 4)}

## What each idea does (medians across the other settings)
{L.md_table(abl_tab, 3)}

## Grid (sorted by TRAIN goal score)
{L.md_table(gt.drop(columns=['score', 'k', 'rebal']), 3)}

![train](train_70pct.png)
![test](test_30pct.png)
""")
    print("\n" + "=" * 96)
    for _, r in key.iterrows():
        print(f"  {r['window']:<10} {r['line'][:56]:<56} Rs {r['final_value_rs']:>9,}  CAGR {r['CAGR']:+.3f}  Sharpe {r['sharpe']:>6}"
              f"  maxDD {r['max_drawdown']:+.3f}")
    print("=" * 96)
    print(ablation.round(3).to_string())
    print(f"[P10] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
