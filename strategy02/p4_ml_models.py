#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRATEGY 02 / PHASE 4 - LightGBM ranking model on the 21-day horizon, validated with CPCV.

What changed versus strategy/p4 (Vols 3-4 of the strategy volumes)
-----------------------------------------------------------------
  * Target: triple-barrier y_bin (phase 3) with UNIQUENESS sample weights; model quality is
    judged on the ECONOMIC target y_rank (21-day residual return), not on accuracy.
  * Hyper-parameters chosen ONLY on the train window (<= 2021-08-02) with Optuna, scored by
    Combinatorial Purged CV: 6 blocks, 2 test blocks -> 15 splits (5 full backtest paths),
    35-day purge + 5-day embargo. The vols' default LightGBM set is trial 0.
    Gate: the 5th percentile of the 15 split-ICs must be > 0. The trial count is saved for DSR.
  * Walk-forward OOS predictions: refit every quarter on an expanding window, labels purged
    so that no training label ends after the refit date.
  * Rank on the RAW probability; the causal isotonic-calibrated probability is used only for
    sizing. A meta-labelling model (Lopez de Prado ch.3) estimates P(top-ranked pick beats the
    cross-section) -> meta_conf, also used only for sizing.
Output folder: phase4_ml_models/
"""
from __future__ import annotations

import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import p1_data_costs as L
import p3_features as F3

warnings.filterwarnings("ignore")
plt = L.plt
PHASE = "phase4_ml_models"
OOS_START = pd.Timestamp("2019-07-01")
PURGE_DAYS, EMBARGO_DAYS = 42, 5          # 21 trading days can span 38 calendar days (fix: was 35)
BASE_PARAMS = {"objective": "binary", "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31,
               "max_depth": 6, "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
               "colsample_bytree": 0.7, "reg_lambda": 5.0, "verbose": -1, "n_jobs": 2}

try:
    import lightgbm as lgb
except ImportError:            # pragma: no cover
    lgb = None
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:            # pragma: no cover
    optuna = None


def load_features():
    f = L.HERE / "phase3_features" / "daily_features.csv"
    df = pd.read_csv(f, parse_dates=["date"])
    meta = json.loads((L.HERE / "phase3_features" / "feature_lists.json").read_text())
    return df, meta


def fit(X, y, w, params):
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y, sample_weight=w)
    return m


def daily_ic(df: pd.DataFrame, col: str, target: str = "y_rank") -> pd.Series:
    d = df[["date", col, target]].dropna()
    return d.groupby("date").apply(lambda g: g[col].rank().corr(g[target].rank()) if len(g) >= 20 else np.nan).dropna()


def cpcv_splits(dates: np.ndarray, n_blocks=6, k_test=2):
    """Yields (train_mask, test_mask, test_blocks) over unique sorted dates with purge+embargo."""
    u = np.unique(dates)
    blocks = np.array_split(u, n_blocks)
    for test_ids in itertools.combinations(range(n_blocks), k_test):
        test_dates = np.concatenate([blocks[i] for i in test_ids])
        bad = np.zeros(len(dates), dtype=bool)
        for i in test_ids:
            lo = blocks[i][0] - np.timedelta64(PURGE_DAYS, "D")          # labels running into the block
            hi = blocks[i][-1] + np.timedelta64(EMBARGO_DAYS + PURGE_DAYS, "D")
            bad |= (dates >= lo) & (dates <= hi)
        yield ~bad, np.isin(dates, test_dates), test_ids


TARGET = "y_bin"          # set in main(): "y_bin" (triple barrier) or "y_xs" (beats the cross-section)


def cpcv_score(df: pd.DataFrame, feats: list[str], params: dict) -> pd.DataFrame:
    rows = []
    D = df["date"].to_numpy()
    for tr, te, ids in cpcv_splits(D):
        sub = df[tr & df[TARGET].notna().to_numpy()]
        m = fit(sub[feats], sub[TARGET].astype(int), sub["uniq_w"].fillna(sub["uniq_w"].median()), params)
        t = df[te].copy()
        t["p"] = m.predict_proba(t[feats])[:, 1]
        ic = daily_ic(t, "p")
        t["pct"] = t.groupby("date")["p"].rank(pct=True)
        spread = t[t["pct"] >= 0.9].groupby("date")["y_rank"].mean().mean()
        rows.append({"test_blocks": str(ids), "mean_ic": ic.mean(), "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic) / 21),
                     "top_decile_excess_21d": spread})
    return pd.DataFrame(rows)


def tune(train: pd.DataFrame, feats: list[str], n_trials: int):
    trials = []
    if optuna is None or n_trials <= 1:
        s = cpcv_score(train, feats, BASE_PARAMS)
        trials.append({"trial": 0, **{k: BASE_PARAMS[k] for k in ("learning_rate", "num_leaves", "min_child_samples",
                                                                   "colsample_bytree", "reg_lambda")},
                       "cpcv_mean_ic": s["mean_ic"].mean(), "cpcv_p05_ic": s["mean_ic"].quantile(0.05)})
        return dict(BASE_PARAMS), pd.DataFrame(trials), s

    cache = {}

    def objective(trial):
        p = dict(BASE_PARAMS)
        p.update({"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                  "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                  "min_child_samples": trial.suggest_int("min_child_samples", 50, 600, log=True),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                  "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30, log=True)})
        s = cpcv_score(train, feats, p)
        cache[trial.number] = (p, s)
        trials.append({"trial": trial.number, **trial.params, "cpcv_mean_ic": s["mean_ic"].mean(),
                       "cpcv_p05_ic": s["mean_ic"].quantile(0.05)})
        # objective = mean IC penalised by its dispersion across splits (robust choice)
        return s["mean_ic"].mean() - 0.5 * s["mean_ic"].std()

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial({k: BASE_PARAMS[k] for k in ("learning_rate", "num_leaves", "min_child_samples",
                                                  "colsample_bytree", "reg_lambda")})
    st.optimize(objective, n_trials=n_trials)
    p, s = cache[st.best_trial.number]
    return p, pd.DataFrame(trials), s


def walk_forward(df: pd.DataFrame, feats: list[str], params: dict, quick=False) -> pd.DataFrame:
    dates = np.sort(df["date"].unique())
    q_starts = pd.date_range(OOS_START, pd.Timestamp(dates[-1]) + pd.Timedelta(days=1), freq="QS")
    if quick:
        q_starts = q_starts[::2]
    outs = []
    for a, b in zip(q_starts, list(q_starts[1:]) + [pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)]):
        tr = df[(df["date"] <= a - pd.Timedelta(days=PURGE_DAYS)) & df[TARGET].notna()]
        te = df[(df["date"] >= a) & (df["date"] < b)].copy()
        if te.empty or len(tr) < 20000:
            continue
        m = fit(tr[feats], tr[TARGET].astype(int), tr["uniq_w"].fillna(tr["uniq_w"].median()), params)
        te["prob_raw"] = m.predict_proba(te[feats])[:, 1]
        te["refit"] = a
        outs.append(te)
        print(f"   refit {a.date()}  train rows {len(tr):>6}  predict {len(te):>5}")
    return pd.concat(outs, ignore_index=True)


def causal_isotonic(oos: pd.DataFrame) -> pd.Series:
    """Monthly refit; only predictions whose labels are already complete (date <= refit - 35d)."""
    out = pd.Series(np.nan, index=oos.index)
    months = pd.date_range(oos["date"].min(), oos["date"].max() + pd.Timedelta(days=31), freq="MS")
    for a, b in zip(months[:-1], months[1:]):
        hist = oos[(oos["date"] <= a - pd.Timedelta(days=PURGE_DAYS)) & oos["y_bin"].notna()]
        cur = (oos["date"] >= a) & (oos["date"] < b)
        if len(hist) < 3000:
            out[cur] = oos.loc[cur, "prob_raw"]
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(hist["prob_raw"], hist["y_bin"])
        out[cur] = iso.predict(oos.loc[cur, "prob_raw"])
    return out


def meta_label(oos: pd.DataFrame, params_small: dict) -> pd.Series:
    """P(y_rank > 0 | primary picked it in its top 20%), walk-forward quarterly."""
    oos = oos.copy()
    oos["pct_rank"] = oos.groupby("date")["prob_raw"].rank(pct=True)
    mf = ["pct_rank", "prob_raw", "vol_63", "beta_252", "mkt_vol_21", "dispersion_21", "p_hostile", "breadth_above_200"]
    mf = [c for c in mf if c in oos]
    out = pd.Series(0.5, index=oos.index)
    q = pd.date_range(oos["date"].min() + pd.Timedelta(days=365), oos["date"].max() + pd.Timedelta(days=92), freq="QS")
    for a, b in zip(q[:-1], q[1:]):
        hist = oos[(oos["date"] <= a - pd.Timedelta(days=PURGE_DAYS)) & (oos["pct_rank"] >= 0.8) & oos["y_rank"].notna()]
        cur = (oos["date"] >= a) & (oos["date"] < b)
        if len(hist) < 2000 or not cur.any():
            continue
        m = lgb.LGBMClassifier(**params_small)
        m.fit(hist[mf], (hist["y_rank"] > 0).astype(int), sample_weight=hist["uniq_w"].fillna(hist["uniq_w"].median()))
        out[cur] = m.predict_proba(oos.loc[cur, mf])[:, 1]
    return out


def main():
    ap = L.base_args("Strategy02 Phase 4: LightGBM 21-day ranking with CPCV")
    ap.add_argument("--trials", type=int, default=12, help="Optuna trials (each = 15 CPCV fits)")
    args = ap.parse_args()
    if lgb is None:
        raise SystemExit("pip install lightgbm optuna")
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    df, meta = load_features()
    feats = meta["model_features"]
    df = df.dropna(subset=feats[:4])
    train = df[(df["date"] <= F3.TRAIN_END - pd.Timedelta(days=PURGE_DAYS)) & df["y_bin"].notna()].reset_index(drop=True)
    print(f"[P4] {len(df)} rows, {len(feats)} features; train window rows {len(train)}")

    global TARGET
    df["y_xs"] = np.where(df["y_rank"].notna(), (df["y_rank"] > 0).astype(float), np.nan)
    train["y_xs"] = np.where(train["y_rank"].notna(), (train["y_rank"] > 0).astype(float), np.nan)
    # target choice is itself a trial: both scored by CPCV on the train window with the base params
    tgt_rows = []
    for tgt in ("y_bin", "y_xs"):
        TARGET = tgt
        s_ = cpcv_score(train, feats, BASE_PARAMS)
        tgt_rows.append({"target": tgt, "cpcv_mean_ic": s_["mean_ic"].mean(), "cpcv_sd_ic": s_["mean_ic"].std(),
                         "score": s_["mean_ic"].mean() - 0.5 * s_["mean_ic"].std()})
        print(f"   target {tgt}: CPCV mean IC {s_['mean_ic'].mean():.4f}  sd {s_['mean_ic'].std():.4f}")
    tgt_tab = pd.DataFrame(tgt_rows)
    tgt_tab.round(5).to_csv(out / "target_choice.csv", index=False)
    TARGET = tgt_tab.sort_values("score").iloc[-1]["target"]
    print(f"[P4] training target: {TARGET}")
    n_trials = 3 if args.quick else args.trials
    print(f"[P4] Optuna x CPCV on the train window ({n_trials} trials x 15 splits) ...")
    params, trials, cp = tune(train, feats, n_trials)
    trials.round(5).to_csv(out / "optuna_trials.csv", index=False)
    cp.round(5).to_csv(out / "cpcv_splits.csv", index=False)
    gate = {"cpcv_splits": len(cp), "cpcv_paths": 5, "mean_ic": round(float(cp["mean_ic"].mean()), 4),
            "p05_ic": round(float(cp["mean_ic"].quantile(0.05)), 4),
            "gate_p05_gt_0": bool(cp["mean_ic"].quantile(0.05) > 0),
            "mean_top_decile_excess_21d": round(float(cp["top_decile_excess_21d"].mean()), 4),
            "optuna_trials": int(len(trials)) + 2, "target": TARGET}
    (out / "lgbm_params.json").write_text(json.dumps({"params": params, "gate": gate}, indent=2, default=str))
    print(f"[P4] CPCV gate: {gate}")

    print("[P4] walk-forward quarterly refits ...")
    oos = walk_forward(df, feats, params, quick=args.quick)
    oos["prob_cal"] = causal_isotonic(oos)
    small = dict(BASE_PARAMS, n_estimators=150, num_leaves=8, min_child_samples=150, max_depth=4)
    oos["meta_conf"] = meta_label(oos, small)
    oos["pct_rank"] = oos.groupby("date")["prob_raw"].rank(pct=True)
    keep = ["date", "name", "prob_raw", "prob_cal", "meta_conf", "pct_rank", "y_rank", "y_bin", "fwd_ret",
            "vol_63_raw", "atr14_raw", "amihud_raw"]
    oos[keep].to_csv(out / "oos_pred_lgbm.csv", index=False, float_format="%.6g")

    # ---------------- evaluation on month-end decision dates (non-overlapping 21-day labels)
    me_dates = set(pd.Series(oos["date"].unique()).groupby(pd.Series(oos["date"].unique()).dt.to_period("M")).max())
    ev = oos[oos["date"].isin(me_dates)].copy()
    sgn = pd.read_csv(L.HERE / "phase3_features" / "ic_table.csv").set_index("feature")["ic_train_yrank"].apply(np.sign)
    ev["composite"] = sum(ev[c] * sgn[c] for c in meta["selected_stock_features"] if c in ev)
    comp_cols = {"LightGBM prob_raw": "prob_raw", "mom_12_1 (Level 1)": "mom_12_1", "composite (Level 1)": "composite"}
    rows, ics = [], {}
    for lab, col in comp_cols.items():
        if col not in ev:
            continue
        ic = daily_ic(ev, col)
        ics[lab] = ic
        ev["_pct"] = ev.groupby("date")[col].rank(pct=True)
        top = ev[ev["_pct"] >= 0.9].groupby("date")["y_rank"].mean()
        rows.append({"model": lab, "monthly_mean_ic": ic.mean(), "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                     "top_decile_excess_21d": top.mean(), "hit_rate_top_decile": (top > 0).mean(), "n_months": len(ic)})
    evt = pd.DataFrame(rows)
    a, b = ics.get("LightGBM prob_raw"), ics.get("composite (Level 1)")
    if a is not None and b is not None:
        d = (a - b).dropna()
        evt["paired_t_vs_composite"] = [round(float(d.mean() / d.std() * np.sqrt(len(d))), 3) if l == "LightGBM prob_raw" else np.nan
                                        for l in evt["model"]]
    evt.round(5).to_csv(out / "model_evaluation.csv", index=False)

    # calibration
    v = oos.dropna(subset=["y_bin"])
    bins = pd.qcut(v["prob_cal"].rank(method="first"), 10, labels=False)
    cal = v.groupby(bins).agg(pred=("prob_cal", "mean"), raw=("prob_raw", "mean"), obs=("y_bin", "mean"))
    cal.to_csv(out / "calibration_table.csv")
    brier = {"brier_raw": float(((v["prob_raw"] - v["y_bin"]) ** 2).mean()),
             "brier_cal": float(((v["prob_cal"] - v["y_bin"]) ** 2).mean()),
             "brier_base_rate": float(((v["y_bin"].mean() - v["y_bin"]) ** 2).mean())}
    (out / "calibration.json").write_text(json.dumps(brier, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(cal["raw"], cal["obs"], "o-", label="raw"); ax[0].plot(cal["pred"], cal["obs"], "s-", label="isotonic (causal)")
    ax[0].plot([0.3, 0.8], [0.3, 0.8], "k--", lw=0.8); ax[0].legend(); ax[0].set_title("Reliability (deciles)")
    ax[0].set_xlabel("predicted P(upper barrier first)"); ax[0].set_ylabel("observed")
    for lab, ic in ics.items():
        ax[1].plot(ic.rolling(12).mean(), label=lab)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].legend(fontsize=8); ax[1].set_title("12-month rolling rank IC vs y_rank")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "calibration_and_ic.png", dpi=120); plt.close(fig)

    # ---------------- candidates: monthly top-10 CNC books from OOS predictions
    P = L.load_daily_panel(data)
    close, opn = P["adj_close"], P["adj_open"]
    stocks = [c for c in close.columns if c != L.INDEX]
    liq = L.liquid_returns(data, close.index)
    hs = L.half_spread_bps(P["adj_high"][stocks], P["adj_low"][stocks], P["turnover"][stocks])
    me = L.month_end_mask(close.index)
    score = oos.pivot(index="date", columns="name", values="prob_raw").reindex(close.index)
    sm = (oos.assign(s=oos["pct_rank"] * (0.5 + oos["meta_conf"])).pivot(index="date", columns="name", values="s")
          .reindex(close.index))
    start = oos["date"].min()
    sims = {
        "LightGBM top10 monthly (CNC)": L.simulate_weights(F3.top_k_buffer(score, 10, 20, me), opn, close, liq,
                                                           pledge=False, half_spread=hs, rebal_mask=me)["ret"].loc[start:],
        "LightGBM x meta top10 monthly (CNC)": L.simulate_weights(F3.top_k_buffer(sm, 10, 20, me), opn, close, liq,
                                                                  pledge=False, half_spread=hs, rebal_mask=me)["ret"].loc[start:],
        "NIFTY50 buy&hold": close[L.INDEX].pct_change().fillna(0).loc[start:],
    }
    prev3 = L.find_series(PHASE, "Composite rank (selected features) top10 monthly (CNC)")
    if prev3 is not None:
        sims["Composite rank top10 (phase 3)"] = prev3.loc[start:]
    cards = [L.scorecard(v, k) for k, v in sims.items()]
    best = L.save_scorecards(out, cards, PHASE, notes="LightGBM on 21-day triple-barrier labels.", series=sims)
    L.multi_panel(sims, out / "ml_books_all_in_one.png", "Phase 4 - ML monthly books vs Level-1 (CNC only, OOS)",
                  highlight="LightGBM top10 monthly (CNC)")
    tab = pd.DataFrame(cards)[L.SC_COLS]
    (out / "report.md").write_text(f"""# Strategy02 - Phase 4: LightGBM on the 21-day horizon

Features ({len(feats)}): {', '.join(feats)}

## CPCV on the train window (6 blocks, 2 test -> 15 splits; purge {PURGE_DAYS}d, embargo {EMBARGO_DAYS}d)
```
{json.dumps(gate, indent=2)}
```
Chosen parameters: `{ {k: params[k] for k in ('learning_rate','num_leaves','min_child_samples','colsample_bytree','reg_lambda')} }`

## Walk-forward OOS evaluation (month-end decisions, from {OOS_START.date()})
{L.md_table(evt)}

Brier: {json.dumps({k: round(v, 5) for k, v in brier.items()})}

## Books (CNC only, monthly, keep while rank <= 20)
{L.md_table(tab)}

![cal](calibration_and_ic.png)
![books](ml_books_all_in_one.png)
""")
    L.print_scorecard(best, PHASE)
    print(f"[P4] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
