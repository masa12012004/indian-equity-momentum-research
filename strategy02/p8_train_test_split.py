#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 8 - Chronological 70% TRAIN / 30% TEST split.

Protocol
--------
  * Window = the period the strategy trades on walk-forward (out-of-sample) signals
    (phase 6: 2019-07-01 .. last date). The first 70% of its trading days is TRAIN, the
    remaining 30% is TEST. Nothing is shuffled.
  * Everything that is CHOSEN is chosen on TRAIN only: the configuration is re-selected from
    the full phase-6 grid (270 variants incl. the research scores and trend exits) by the mandate goal score on the TRAIN window, then frozen.
    (Feature signs / hyper-parameters were already fitted <= 2021-08-02, i.e. inside TRAIN;
    the model refits are walk-forward and only ever see past data.)
  * TEST is reported for the frozen configuration exactly as it was chosen.
  * Both windows come from ONE continuous run; each chart rebases every line to Rs 1,00,000 at the window start (positions held at the split carry over into TEST).

Outputs (phase8_train_test_split/)
  train_70pct.png, test_30pct.png   - every line in one chart (equity, drawdown, rolling return,
                                      monthly box plot); the CNC-only book is the thick black line
  split_overview.png                - full window with the split marked
  train_test_metrics.csv / .md, split.json, train_variant_ranking.csv
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

import p1_data_costs as L
import p6_portfolio_backtest as E

plt = L.plt
PHASE = "phase8_train_test_split"
TRAIN_FRAC = 0.70
P6 = L.HERE / "phase6_portfolio_backtest"
# Highlighted line: CNC delivery buys only. When the trend exit has the book out of stocks the money
# is parked in LIQUIDBEES (also a CNC delivery buy), so the line no longer goes flat.
CNC = "CNC-only long (stocks, LIQUIDBEES when out of stocks)"
IDLE = "same book, idle cash at 0% (goes flat when out of stocks)"


def split_panel(series: dict[str, pd.Series], path, title: str, highlight: str, band=None, shade: pd.Series | None = None):
    """Same layout as L.multi_panel but rebased at the window start and with a rolling window that
    fits short periods (min(252, n/3) days)."""
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    cap = L.CFG["capital"]
    cmap = plt.get_cmap("tab20")
    bench = ("NIFTY50", "EW NIFTY100", "Liquid", "Random")
    n_days = max(len(s.dropna()) for s in series.values())
    roll = int(min(252, max(21, n_days // 3)))
    for n, (k, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        lw, z = (2.8, 5) if k == highlight else (1.2, 2)
        col = "black" if k == highlight else cmap((2 * n) % 20 if n < 10 else (2 * n + 1) % 20)
        ls = "--" if any(b in k for b in bench) else "-"
        eq = cap * (1 + s).cumprod()
        ax[0, 0].plot(eq, label=f"{k}  ->  Rs {eq.iloc[-1]:,.0f}", lw=lw, zorder=z, color=col, ls=ls)
        ax[0, 1].plot((eq / eq.cummax() - 1) * 100, lw=lw * 0.7, zorder=z, color=col, ls=ls)
        ax[1, 0].plot((eq / eq.shift(roll) - 1) * 100, lw=lw * 0.7, zorder=z, color=col, ls=ls)
    if shade is not None and shade.any():
        # contiguous stretches where the highlighted book held no stocks (trend exit -> LIQUIDBEES)
        s_ = shade.astype(int)
        starts = s_.index[(s_.diff().fillna(s_.iloc[0]) == 1)]
        ends = s_.index[(s_.diff(-1).fillna(s_.iloc[-1]) == 1)]
        for k_, (a_, b_) in enumerate(zip(starts, ends)):
            for ax_ in (ax[0, 0], ax[0, 1]):
                ax_.axvspan(a_, b_, color="#f2c94c", alpha=0.18, lw=0,
                            label="book out of stocks (trend exit) - money in LIQUIDBEES" if (k_ == 0 and ax_ is ax[0, 0]) else None)
    if band is not None:
        lo_, hi_ = band
        ax[0, 0].fill_between(lo_.index, cap * (1 + lo_).cumprod(), cap * (1 + hi_).cumprod(),
                              color="grey", alpha=0.15, label="random-pick null 5-95%")
    ax[0, 0].set_title("Equity from Rs 1,00,000 (log)"); ax[0, 0].set_yscale("log")
    ax[0, 1].axhline(L.CFG["goal_max_dd"] * 100, color="red", ls="--", lw=1); ax[0, 1].set_title("Drawdown % (Goal 1: -20%)")
    ax[1, 0].axhline(0, color="k", lw=0.6)
    if roll == 252:
        ax[1, 0].axhline(L.CFG["goal_trailing_12m"] * 100, color="green", ls="--", lw=1)
    ax[1, 0].set_title(f"Rolling {roll}-day return %")
    mons = {k: L.to_monthly(v.dropna()) * 100 for k, v in series.items() if v.dropna().size}
    ax[1, 1].boxplot(list(mons.values()), vert=False)
    ax[1, 1].set_yticks(range(1, len(mons) + 1)); ax[1, 1].set_yticklabels([k[:40] for k in mons], fontsize=7)
    ax[1, 1].axvline(L.CFG["goal_month_floor"] * 100, color="red", ls="--"); ax[1, 1].axvline(L.CFG["goal_month_cap"] * 100, color="green", ls="--")
    ax[1, 1].set_title("Monthly returns % (band -0.5% .. +3%)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02 + 0.022 * ((len(lab) + 2) // 3), 1, 1)); fig.savefig(path, dpi=120); plt.close(fig)


def metrics_row(s: pd.Series, label: str, window: str) -> dict:
    s = s.dropna()
    m = L.to_monthly(s)
    sc = L.scorecard(s, label, n_boot=1000)
    return {"window": window, "line": label, "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
            "final_value_rs": round(L.CFG["capital"] * float((1 + s).prod())), "total_return": round(float((1 + s).prod() - 1), 4),
            "CAGR": sc.get("CAGR"), "sharpe": sc.get("sharpe"), "max_drawdown": sc.get("max_drawdown"),
            "worst_month": round(float(m.min()), 4), "months_positive": round(float((m > 0).mean()), 3),
            "months_ge_-0.5%": round(float((m >= -0.005).mean()), 3),
            "P_profitable_12m": sc.get("P_profitable_12m"), "pred_monthly_mean": sc.get("pred_monthly_mean"),
            "P_maxdd_ok_12m": sc.get("P_maxdd_ok_12m"), "goal_score": sc.get("goal_score")}


def main():
    ap = L.base_args("Strategy02 Phase 8: 70/30 chronological train/test split")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    grid = pd.read_csv(P6 / "grid_daily_returns.csv", index_col=0, parse_dates=True).dropna(how="all")
    lines6 = pd.read_csv(P6 / "lines_daily_returns.csv", index_col=0, parse_dates=True)
    nullD = pd.read_csv(P6 / "null_daily_returns.csv", index_col=0, parse_dates=True)
    dates = grid.index
    cut = int(len(dates) * TRAIN_FRAC)
    tr_w = (dates[0], dates[cut - 1]); te_w = (dates[cut], dates[-1])
    print(f"[P8] window {dates[0].date()} .. {dates[-1].date()}  ({len(dates)} days)")
    print(f"[P8] TRAIN {tr_w[0].date()} .. {tr_w[1].date()} ({cut} days)   TEST {te_w[0].date()} .. {te_w[1].date()} ({len(dates) - cut} days)")

    # ---- re-select the configuration on TRAIN only
    rank = []
    for c in grid.columns:
        sc = L.scorecard(grid[c].loc[tr_w[0]:tr_w[1]], c, n_boot=300)
        rank.append({"variant": c, "train_goal_score": sc["goal_score"], "train_sharpe": sc["sharpe"],
                     "train_CAGR": sc["CAGR"], "train_max_dd": sc["max_drawdown"],
                     "test_CAGR_(not used for choosing)": round(float((1 + grid[c].loc[te_w[0]:]).prod() ** (252 / (len(dates) - cut)) - 1), 4)})
    rank = pd.DataFrame(rank).sort_values(["train_goal_score", "train_sharpe"], ascending=False)
    rank.to_csv(out / "train_variant_ranking.csv", index=False)
    chosen = rank.iloc[0]["variant"]
    parts = chosen.split("|")
    sc_, opt, k, st = parts[:4]
    te = parts[4].replace("trend_", "") if len(parts) > 4 else "none"
    stops = {"stop_off": dict(stops=False), "stop_2atr12": dict(stops=True, atr_mult=2.0, stop_pct=0.12),
             "stop_3atr15": dict(stops=True, atr_mult=3.0, stop_pct=0.15)}[st]
    bp = E.Params(name=chosen, score=sc_, opt=opt, k=int(k[1:]), trend_exit=te, **stops)
    print(f"[P8] chosen on TRAIN: {chosen}")
    best_of = lambda pred: rank[rank["variant"].apply(pred)].iloc[0]["variant"]

    # ---- lines (the chosen config re-simulated; alternatives taken from the grid)
    mk = E.Market(data)
    scores, meta = E.load_scores(mk)
    regime = pd.read_csv(L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv", index_col=0, parse_dates=True)
    lines = {}
    for lab, p in {CNC: bp, IDLE: replace(bp, sweep=False)}.items():
        d, tr = E.run(mk, p, scores[p.score], meta if p.score == "final" else None, regime, dates[0], log_trades=True)
        lines[lab] = d["ret"]
        if lab == CNC:
            tr.to_csv(out / "trades_cnc_only.csv", index=False)
            out_of_stocks = (d["stock_exposure"] < 0.05).reindex(dates).fillna(False)
    for lab, pred in [("best MV weights (train)", lambda v: "|MV|" in v), ("best HRP weights (train)", lambda v: "|HRP|" in v),
                      ("best EW weights (train)", lambda v: "|EW|" in v), ("best ML-score variant (train)", lambda v: v.startswith("final")),
                      ("best no-stop variant (train)", lambda v: "stop_off" in v),
                      ("best without trend exit (train)", lambda v: v.endswith("trend_none")),
                      ("original strategy02 scores only (train)", lambda v: v.split("|")[0] in ("final", "composite") and v.endswith("trend_none"))]:
        v = best_of(pred)
        if v != chosen:
            lines[f"{lab}: {v}"] = grid[v]
    if "CNC pledged + MIS overlay (v1 intraday sleeve)" in lines6:
        lines["CNC pledged + MIS overlay (v1 sleeve, fixed config)"] = lines6["CNC pledged + MIS overlay (v1 intraday sleeve)"]
    for b in ("EW NIFTY100 monthly", "NIFTY50 buy&hold", "Liquid ETF only"):
        lines[b] = lines6[b]
    rnd = [c for c in lines6 if c.startswith("Random")]
    if rnd:
        lines[rnd[0]] = lines6[rnd[0]]
    lines = {k: v.reindex(dates) for k, v in lines.items()}
    pd.DataFrame(lines).to_csv(out / "lines_daily_returns.csv")

    rows = []
    for wname, w in (("TRAIN 70%", tr_w), ("TEST 30%", te_w)):
        win = {k: v.loc[w[0]:w[1]] for k, v in lines.items()}
        nd = nullD.loc[w[0]:w[1]]
        cg = (1 + nd).prod()
        lo_c, hi_c = cg.sort_values().index[int(0.05 * (len(cg) - 1))], cg.sort_values().index[int(0.95 * (len(cg) - 1))]
        fn = "train_70pct.png" if wname.startswith("TRAIN") else "test_30pct.png"
        split_panel(win, out / fn, f"Strategy02 {wname}: {w[0].date()} .. {w[1].date()} - rebased to Rs 1,00,000 at the window start "
                                   f"(config chosen on TRAIN only: {chosen})", highlight=CNC, band=(nd[lo_c], nd[hi_c]),
                    shade=out_of_stocks.loc[w[0]:w[1]])
        for k, s in win.items():
            rows.append(metrics_row(s, k, wname))
    met = pd.DataFrame(rows)
    met.to_csv(out / "train_test_metrics.csv", index=False)

    # overview with the split marked
    fig, ax = plt.subplots(figsize=(14, 6))
    for k in (CNC, IDLE, "NIFTY50 buy&hold", rnd[0] if rnd else None):
        if k:
            ax.plot(L.CFG["capital"] * (1 + lines[k].fillna(0)).cumprod(), label=k, lw=2.5 if k == CNC else 1.2,
                    color="black" if k == CNC else None, ls="--" if ("NIFTY" in k or "Random" in k) else "-")
    ax.axvline(te_w[0], color="red", lw=1.5); ax.axvspan(te_w[0], te_w[1], color="red", alpha=0.06)
    ax.text(tr_w[0], ax.get_ylim()[1] * 0.97, "  TRAIN 70%", va="top"); ax.text(te_w[0], ax.get_ylim()[1] * 0.97, "  TEST 30%", va="top", color="red")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Strategy02 - one continuous run, split marked (train left, test right)")
    fig.tight_layout(); fig.savefig(out / "split_overview.png", dpi=120); plt.close(fig)

    split = {"train_frac": TRAIN_FRAC, "train": [str(tr_w[0].date()), str(tr_w[1].date())], "train_days": cut,
             "test": [str(te_w[0].date()), str(te_w[1].date())], "test_days": len(dates) - cut,
             "chosen_on_train": chosen, "params": asdict(bp)}
    (out / "split.json").write_text(json.dumps(split, indent=2, default=str))
    key = met[met["line"].isin([CNC, IDLE, "NIFTY50 buy&hold"] + rnd[:1])]
    cols = ["window", "line", "final_value_rs", "total_return", "CAGR", "sharpe", "max_drawdown", "worst_month",
            "months_ge_-0.5%", "P_profitable_12m", "pred_monthly_mean"]
    (out / "report.md").write_text(f"""# Strategy02 - Phase 8: 70% train / 30% test

TRAIN {split['train'][0]} .. {split['train'][1]} ({cut} days) | TEST {split['test'][0]} .. {split['test'][1]} ({len(dates) - cut} days)

Configuration chosen on TRAIN only: **{chosen}** (then frozen for TEST).

## Key lines (rebased to Rs 1,00,000 at the start of each window)
{L.md_table(key[cols], 4)}

## All lines
{L.md_table(met[cols], 4)}

![train](train_70pct.png)
![test](test_30pct.png)
![overview](split_overview.png)
""")
    print("\n" + "=" * 78)
    for _, r in key.iterrows():
        print(f"  {r['window']:<10} {r['line'][:48]:<48} Rs {r['final_value_rs']:>9,}  CAGR {r['CAGR']:+.3f}  "
              f"maxDD {r['max_drawdown']:+.3f}  P(profit 12m) {r['P_profitable_12m']}  pred/month {r['pred_monthly_mean']}")
    print("=" * 78)
    print(f"[P8] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
