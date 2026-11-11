#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 11 - Edge builder: research-backed portfolio rules on top of the phase-10 book.

Base book (phase 10 robust choice): consensus momentum, equal weight, 16 names, quarterly
rebalance, monthly SMA200 trend exit, idle cash in LIQUIDBEES.

Ideas added (each has a published basis; parameters fixed in advance, NOT tuned):
  1. Industry momentum      Moskowitz & Grinblatt (1999): half of the score = the stock's sector 6-month
                            momentum rank (score "consensus_sector").
  2. Sector cap = 3         momentum clusters in one theme (PSU banks, defence); cap diversifies.
  3. Absolute momentum      Antonacci / Moskowitz-Ooi-Pedersen time-series momentum: only buy names
                            whose 12-month return beats the liquid ETF.
  4. Skip "hot" entries     Jegadeesh (1990) short-term reversal: do not open a position in a name whose
                            1-month return is in the top 10% (they tend to give some back).
  5. Breadth-confirmed exit leave stocks only if NIFTY < SMA200 AND < 50% of stocks are above their
                            own SMA200 (two signals -> fewer whipsaws).
  6. Gold as risk-off asset Indian dual momentum (Nifty / Gold / debt). Runs only if data_clean/1D has
                            GOLDBEES (download it with the IIFL fetcher: --symbols GOLDBEES LIQUIDBEES).
Selection: same pre-specified robust rule as phase 10 on TRAIN only (per idea, keep the level with
the higher MEDIAN train Sharpe across all other settings; ties within 0.02 -> cheaper / simpler).
Outputs: phase11_edge_builder/ (train_70pct.png, test_30pct.png, grid_results.csv, ablation.csv,
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
PHASE = "phase11_edge_builder"
DIMS = {"score": ["consensus", "consensus_sector"], "sector_cap": [0, 3], "abs_mom_filter": [False, True],
        "skip_hot": [False, True], "trend_exit": ["sma200", "sma200_breadth"]}
SIMPLER = {"score": "consensus", "sector_cap": 0, "abs_mom_filter": False, "skip_hot": False, "trend_exit": "sma200"}


def main():
    ap = L.base_args("Strategy02 Phase 11: edge builder")
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
    base = E.Params(name="phase10", score="consensus", opt="EW", k=16, stops=False, trend_exit="sma200", rebal_months=3)

    dims = dict(DIMS)
    if mk.gold is not None:
        dims["riskoff_asset"] = ["liquid", "gold_dual"]
        SIMPLER["riskoff_asset"] = "liquid"
    else:
        print("[P11] GOLDBEES not found in data_clean/1D -> gold risk-off idea skipped (download it to test)")
    keys = list(dims)
    rows, G = [], {}
    for vals in itertools.product(*[dims[k] for k in keys]):
        kw = dict(zip(keys, vals))
        name = "|".join(f"{k}={v}" for k, v in kw.items())
        p = replace(base, name=name, **kw)
        d, _ = E.run(mk, p, scores[p.score], None, regime, tr_w[0])
        G[name] = d["ret"]
        a = L.scorecard(d["ret"].loc[tr_w[0]:tr_w[1]], name, n_boot=300)
        b = E.metrics(d["ret"].loc[te_w[0]:te_w[1]])
        rows.append({"variant": name, **kw, "train_goal_score": a["goal_score"], "train_sharpe": a["sharpe"],
                     "train_CAGR": a["CAGR"], "train_max_dd": a["max_drawdown"], "test_sharpe": b["sharpe"],
                     "test_CAGR": b["CAGR"], "test_max_dd": b["max_drawdown"], "costs_rs": round(float(d["cost"].sum()))})
        print(f"   {name:<110} train Sh {a['sharpe']:>5}  test Sh {b['sharpe']:>6} CAGR {b['CAGR']:+.3f}")
    gt = pd.DataFrame(rows)
    gt.to_csv(out / "grid_results.csv", index=False)

    # robust per-idea selection on TRAIN medians (ties within 0.02 -> the simpler level)
    pick, abl = {}, []
    for k in keys:
        m = gt.groupby(k).agg(train_sharpe=("train_sharpe", "median"), test_sharpe=("test_sharpe", "median"),
                              test_CAGR=("test_CAGR", "median"), costs_rs=("costs_rs", "median"))
        best = m["train_sharpe"].idxmax()
        if m["train_sharpe"].max() - m.loc[SIMPLER[k], "train_sharpe"] < 0.02:
            best = SIMPLER[k]
        pick[k] = best
        for lv, r in m.iterrows():
            abl.append({"idea": k, "level": str(lv), **r.round(4).to_dict(), "adopted": lv == best})
    abl = pd.DataFrame(abl)
    abl.to_csv(out / "ablation.csv", index=False)
    chosen = "|".join(f"{k}={pick[k]}" for k in keys)
    bp = replace(base, name=chosen, **pick)
    print(f"[P11] robust choice on TRAIN: {chosen}")

    CNC = "CNC-only edge book (phase 11, chosen on train)"
    lines = {}
    d, tr = E.run(mk, bp, scores[bp.score], None, regime, tr_w[0], log_trades=True)
    lines[CNC] = d["ret"]; tr.to_csv(out / "trades_cnc_only.csv", index=False); d.to_csv(out / "daily_cnc_only.csv")
    shade = d["stock_exposure"] < 0.05
    d0, _ = E.run(mk, base, scores[base.score], None, regime, tr_w[0])
    lines["Phase 10 book (consensus k16 quarterly)"] = d0["ret"]
    lines["Average of all phase-11 variants (no selection)"] = pd.DataFrame(G).mean(1)
    l6 = pd.read_csv(L.HERE / "phase6_portfolio_backtest" / "lines_daily_returns.csv", index_col=0, parse_dates=True)
    for b_ in ["NIFTY50 buy&hold", "EW NIFTY100 monthly", "Liquid ETF only"] + [c for c in l6 if c.startswith("Random")]:
        lines[b_] = l6[b_]
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
                       f"Strategy02 phase 11 edge book - {wn}: {w[0].date()} .. {w[1].date()}",
                       highlight=CNC, band=band, shade=shade.reindex(dates).fillna(False).loc[w[0]:w[1]])
        for k, s_ in win.items():
            met.append(S8.metrics_row(s_, k, wn))
    met = pd.DataFrame(met)
    met.to_csv(out / "metrics.csv", index=False)
    cols = ["window", "line", "final_value_rs", "CAGR", "sharpe", "max_drawdown", "worst_month", "months_ge_-0.5%",
            "P_profitable_12m", "pred_monthly_mean"]
    key = met[met["line"].isin([CNC, "Phase 10 book (consensus k16 quarterly)", "Average of all phase-11 variants (no selection)",
                                "NIFTY50 buy&hold", "Liquid ETF only"])]
    (out / "chosen.json").write_text(json.dumps({"chosen_on_train": pick, "params": asdict(bp)}, indent=2, default=str))
    (out / "report.md").write_text(f"""# Strategy02 - Phase 11: edge builder

Robust choice on TRAIN only: **{chosen}**

## Key lines (rebased to Rs 1,00,000 at each window start)
{L.md_table(key[cols], 4)}

## Each idea: median across all other settings
{L.md_table(abl, 3)}

![train](train_70pct.png)
![test](test_30pct.png)
""")
    print("\n" + "=" * 100)
    for _, r in key.iterrows():
        print(f"  {r['window']:<10} {r['line'][:52]:<52} Rs {r['final_value_rs']:>9,}  CAGR {r['CAGR']:+.3f}  Sharpe {r['sharpe']:>6}"
              f"  maxDD {r['max_drawdown']:+.3f}")
    print("=" * 100)
    print(abl.to_string(index=False))
    print(f"[P11] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
