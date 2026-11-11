#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 8 (strategy v1) - Chronological 70% TRAIN / 30% TEST split.

Protocol
--------
  * Window = the combined CNC + pledge + MIS period of phase 6 (the period in which BOTH the
    daily and the intraday walk-forward predictions exist). First 70% of its trading days = TRAIN,
    remaining 30% = TEST. Nothing is shuffled.
  * The configuration is re-selected on TRAIN only from the whole phase-6 grid (every family,
    hold, k, MIS on/off, hurdle, vol target) by the mandate goal score, then frozen for TEST.
    Model refits are walk-forward, so they only ever see past data.
  * "CNC-only" = the best configuration with MIS switched off (use_mis = False) on TRAIN:
    buy-side delivery orders only, no intraday. (v1 still charges the pledge fee on new names.)
  * Both windows come from ONE continuous run, rebased to Rs 1,00,000 at each window start.

Outputs (phase8_train_test_split/)
  train_70pct.png, test_30pct.png, split_overview.png,
  train_test_metrics.csv, train_variant_ranking.csv, split.json, report.md
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import p1_data_costs as L

plt = L.plt
PHASE = "phase8_train_test_split"
TRAIN_FRAC = 0.70
P6 = L.HERE / "phase6_portfolio_backtest"
CNC = "CNC-only (no MIS), best on train"
FAM = {"V1": "V1 pledge-rotate 100% CNC", "V2": "V2 80% CNC / 20% cash", "V3": "V3 50% CNC + 50% pledged liquid",
       "V4": "V4 pledged liquid + MIS only", "V5": "V5 adaptive regime split"}


def split_panel(series, path, title, highlight, band=None):
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
    if band is not None:
        lo_, hi_ = band
        ax[0, 0].fill_between(lo_.index, cap * (1 + lo_).cumprod(), cap * (1 + hi_).cumprod(),
                              color="grey", alpha=0.15, label="random-pick null 5-95%")
    ax[0, 0].set_title("Equity from Rs 1,00,000")
    ax[0, 1].axhline(L.CFG["goal_max_dd"] * 100, color="red", ls="--", lw=1); ax[0, 1].set_title("Drawdown % (Goal 1: -20%)")
    ax[1, 0].axhline(0, color="k", lw=0.6); ax[1, 0].set_title(f"Rolling {roll}-day return %")
    mons = {k: L.to_monthly(v.dropna()) * 100 for k, v in series.items() if v.dropna().size}
    ax[1, 1].boxplot(list(mons.values()), vert=False)
    ax[1, 1].set_yticks(range(1, len(mons) + 1)); ax[1, 1].set_yticklabels([k[:40] for k in mons], fontsize=7)
    ax[1, 1].axvline(L.CFG["goal_month_floor"] * 100, color="red", ls="--"); ax[1, 1].axvline(L.CFG["goal_month_cap"] * 100, color="green", ls="--")
    ax[1, 1].set_title("Monthly returns % (band -0.5% .. +3%)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02 + 0.022 * ((len(lab) + 2) // 3), 1, 1)); fig.savefig(path, dpi=120); plt.close(fig)


def metrics_row(s, label, window):
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


def monthly_books(close, opn, start, k=None, n_draws=1, seed=11):
    """k=None: equal weight of every name (monthly buy-and-hold). k=int: random k names per month
    (the random-pick null). CNC statutory costs + 10 bp slippage on the traded fraction."""
    rng = np.random.default_rng(seed)
    stocks = [c for c in close.columns if c != L.INDEX]
    C, O = close[stocks].loc[start:], opn[stocks].loc[start:]
    R = C.pct_change().fillna(0.0).to_numpy()
    ON = (O / C.shift(1) - 1).fillna(0.0).to_numpy()
    dates = C.index; T = len(dates)
    rebal = np.where(np.r_[True, dates.month[1:] != dates.month[:-1]])[0]
    rt = L.cnc_rate("buy") + L.cnc_rate("sell") + 2 * 10e-4      # + 10 bp slippage per side
    avail = C.notna().to_numpy()
    G = np.ones((T, n_draws))
    for j in range(n_draws):
        held = None
        for a, b in zip(rebal, list(rebal[1:]) + [T]):
            if a + 1 >= T:
                break
            pool = np.where(avail[a])[0]
            pick = pool if k is None else rng.choice(pool, min(k, len(pool)), replace=False)
            turn = 1.0 if held is None else (0.05 if k is None else 1 - len(set(pick) & set(held)) / len(pick))
            if held is not None:
                G[a + 1, j] *= 1 + ON[a + 1, held].mean()
            held = pick
            oc = (1 + R[a + 1, pick]) / (1 + ON[a + 1, pick])
            G[a + 1, j] *= oc.mean() * (1 - turn * rt)
            if a + 2 < T:
                val = np.vstack([oc[None, :], np.cumprod(1 + R[a + 2:b + 1][:, pick], axis=0) * oc])
                G[a + 2:b + 1, j] *= val[1:].mean(1) / val[:-1].mean(1)
    return pd.DataFrame(G - 1, index=dates)


def main():
    ap = L.base_args("Phase 8: 70/30 chronological train/test split")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    grid = pd.read_csv(P6 / "grid_daily_returns.csv", index_col=0, parse_dates=True).dropna(how="all")
    dates = grid.index
    cut = int(len(dates) * TRAIN_FRAC)
    tr_w = (dates[0], dates[cut - 1]); te_w = (dates[cut], dates[-1])
    print(f"[P8] window {dates[0].date()} .. {dates[-1].date()}  ({len(dates)} days)")
    print(f"[P8] TRAIN {tr_w[0].date()} .. {tr_w[1].date()} ({cut} days)   TEST {te_w[0].date()} .. {te_w[1].date()} ({len(dates) - cut} days)")

    rank = []
    for c in grid.columns:
        sc = L.scorecard(grid[c].loc[tr_w[0]:tr_w[1]], c, n_boot=300)
        te = grid[c].loc[te_w[0]:]
        rank.append({"variant": c, "family": c.split("|")[0], "mis": "mis1" in c, "train_goal_score": sc["goal_score"],
                     "train_sharpe": sc["sharpe"], "train_CAGR": sc["CAGR"], "train_max_dd": sc["max_drawdown"],
                     "test_total_return_(not used for choosing)": round(float((1 + te).prod() - 1), 4)})
    rank = pd.DataFrame(rank).sort_values(["train_goal_score", "train_sharpe"], ascending=False)
    rank.to_csv(out / "train_variant_ranking.csv", index=False)
    chosen = rank.iloc[0]["variant"]
    cnc_v = rank[~rank["mis"]].iloc[0]["variant"]
    print(f"[P8] chosen on TRAIN: {chosen}   | CNC-only (MIS off) best on TRAIN: {cnc_v}")

    lines = {CNC: grid[cnc_v], f"Best overall on train: {chosen}": grid[chosen]}
    for f, lab in FAM.items():
        v = rank[rank["family"] == f].iloc[0]["variant"]
        if v not in (chosen, cnc_v):
            lines[f"{lab} (best on train)"] = grid[v]
    P = L.load_daily_panel(data)
    close, opn = P["adj_close"], P["adj_open"]
    lines["NIFTY50 buy&hold"] = close[L.INDEX].pct_change().fillna(0)
    lines["EW NIFTY100 monthly"] = monthly_books(close, opn, dates[0])[0]
    lines["Liquid ETF only"] = L.liquid_returns(data, close.index)
    nullD = monthly_books(close, opn, dates[0], k=4, n_draws=60 if args.quick else 200)
    lines["Random 4 names monthly (null median)"] = nullD[(1 + nullD).prod().sort_values().index[len(nullD.columns) // 2]]
    lines = {k: v.reindex(dates) for k, v in lines.items()}
    pd.DataFrame(lines).to_csv(out / "lines_daily_returns.csv")

    rows = []
    for wname, w in (("TRAIN 70%", tr_w), ("TEST 30%", te_w)):
        win = {k: v.loc[w[0]:w[1]] for k, v in lines.items()}
        nd = nullD.loc[w[0]:w[1]]
        cg = (1 + nd).prod().sort_values()
        band = (nd[cg.index[int(0.05 * (len(cg) - 1))]], nd[cg.index[int(0.95 * (len(cg) - 1))]])
        fn = "train_70pct.png" if wname.startswith("TRAIN") else "test_30pct.png"
        split_panel(win, out / fn, f"Strategy v1 {wname}: {w[0].date()} .. {w[1].date()} - rebased to Rs 1,00,000 at the window start "
                                   f"(configs chosen on TRAIN only)", highlight=CNC, band=band)
        for k, s in win.items():
            rows.append(metrics_row(s, k, wname))
    met = pd.DataFrame(rows)
    met.to_csv(out / "train_test_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    for k in (CNC, f"Best overall on train: {chosen}", "NIFTY50 buy&hold", "Random 4 names monthly (null median)"):
        ax.plot(L.CFG["capital"] * (1 + lines[k].fillna(0)).cumprod(), label=k, lw=2.5 if k == CNC else 1.2,
                color="black" if k == CNC else None, ls="--" if ("NIFTY" in k or "Random" in k) else "-")
    ax.axvline(te_w[0], color="red", lw=1.5); ax.axvspan(te_w[0], te_w[1], color="red", alpha=0.06)
    ax.text(tr_w[0], ax.get_ylim()[1], "  TRAIN 70%", va="top"); ax.text(te_w[0], ax.get_ylim()[1], "  TEST 30%", va="top", color="red")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Strategy v1 - one continuous run, split marked (train left, test right)")
    fig.tight_layout(); fig.savefig(out / "split_overview.png", dpi=120); plt.close(fig)

    split = {"train_frac": TRAIN_FRAC, "train": [str(tr_w[0].date()), str(tr_w[1].date())], "train_days": cut,
             "test": [str(te_w[0].date()), str(te_w[1].date())], "test_days": len(dates) - cut,
             "chosen_on_train": chosen, "cnc_only_chosen_on_train": cnc_v}
    (out / "split.json").write_text(json.dumps(split, indent=2))
    key = met[met["line"].isin([CNC, f"Best overall on train: {chosen}", "NIFTY50 buy&hold", "Random 4 names monthly (null median)"])]
    cols = ["window", "line", "final_value_rs", "total_return", "CAGR", "sharpe", "max_drawdown", "worst_month",
            "months_ge_-0.5%", "P_profitable_12m", "pred_monthly_mean"]
    (out / "report.md").write_text(f"""# Strategy v1 - Phase 8: 70% train / 30% test

TRAIN {split['train'][0]} .. {split['train'][1]} ({cut} days) | TEST {split['test'][0]} .. {split['test'][1]} ({len(dates) - cut} days)

Chosen on TRAIN only: **{chosen}**; CNC-only (MIS off): **{cnc_v}** (both frozen for TEST).

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
