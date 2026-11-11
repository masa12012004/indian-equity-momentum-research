#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 4 - Machine learning: Ridge, LightGBM, XGBoost (+ calibrated classifier).

Validation is strictly out-of-sample:
  * expanding-window walk-forward, retrained every `retrain` days
  * purge + embargo of `horizon` days between train and test (overlapping targets)
  * hyper-parameters tuned with Optuna (if installed) on an inner, embargoed validation
    slice of the FIRST fold only, then frozen (no peeking at later data)
  * probabilities calibrated with isotonic regression on the inner validation slice
  * Deflated Sharpe Ratio corrects for the number of model/target trials

Daily (CNC) targets: fwd2_ex and fwd5_ex (2-day and 5-day open-to-open excess returns).
Intraday (MIS) targets: ret_to_exit (directional) and ret_to_exit_ex (cross-sectional).
Output folder: phase4_ml_models/
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

import p1_data_costs as L
import p3_features as F3

plt = L.plt
PHASE = "phase4_ml_models"
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None


def gauss_rank(s: pd.Series, by) -> pd.Series:
    r = s.groupby(by).rank(pct=True)
    n = s.groupby(by).transform("count")
    return pd.Series(norm.ppf((r * n - 0.5) / n), index=s.index)


def daily_ic(df: pd.DataFrame, pred: str, target: str, by="date") -> pd.Series:
    return df.groupby(by).apply(lambda g: spearmanr(g[pred], g[target])[0] if len(g) > 10 else np.nan,
                                include_groups=False).dropna()


LGB_DEFAULT = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=300,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
XGB_DEFAULT = dict(n_estimators=400, learning_rate=0.03, max_depth=5, min_child_weight=50,
                   subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist")


def tune_lgb(Xtr, ytr, Xva, yva, dva, n_trials):
    if optuna is None or lgb is None or n_trials <= 0:
        return dict(LGB_DEFAULT)

    def obj(trial):
        p = dict(LGB_DEFAULT)
        p.update(num_leaves=trial.suggest_int("num_leaves", 8, 64, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 50, 1000, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
                 n_estimators=trial.suggest_int("n_estimators", 100, 800, step=100))
        m = lgb.LGBMRegressor(**p).fit(Xtr, ytr)
        d = pd.DataFrame({"p": m.predict(Xva), "y": yva, "date": dva})
        return daily_ic(d, "p", "y").mean()

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    p = dict(LGB_DEFAULT)
    p.update(st.best_params)
    return p


def walk_forward(df: pd.DataFrame, feats, target, date_col, first_train_end, retrain, embargo, n_trials,
                 tag, classify=True):
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.DatetimeIndex(df[date_col].drop_duplicates().sort_values())
    df["y_rank"] = gauss_rank(df[target], df[date_col])
    starts = [d for d in dates[dates >= first_train_end][::retrain]]
    preds = []
    params = None
    for fi, ts in enumerate(starts):
        te_end = dates[min(dates.searchsorted(ts) + retrain, len(dates) - 1)]
        tr_end = dates[max(dates.searchsorted(ts) - embargo, 1)]
        tr = df[(df[date_col] < tr_end)].dropna(subset=[target])
        te = df[(df[date_col] >= ts) & (df[date_col] < te_end if te_end > ts else df[date_col] >= ts)]
        if fi == len(starts) - 1:
            te = df[df[date_col] >= ts]
        if len(tr) < 5000 or te.empty:
            continue
        Xtr, ytr = tr[feats].fillna(0).to_numpy(), tr["y_rank"].to_numpy()
        Xte = te[feats].fillna(0).to_numpy()
        out = te[[date_col, "name"] + ([c for c in ("slot",) if c in te] )].copy()
        # inner validation slice (last 15% of train dates, embargoed) for tuning + calibration
        tdates = pd.DatetimeIndex(tr[date_col].drop_duplicates().sort_values())
        vcut = tdates[int(len(tdates) * 0.85)]
        vemb = tdates[max(int(len(tdates) * 0.85) - embargo, 1)]
        itr, iva = tr[tr[date_col] < vemb], tr[tr[date_col] >= vcut]
        if params is None:
            params = tune_lgb(itr[feats].fillna(0).to_numpy(), itr["y_rank"].to_numpy(),
                              iva[feats].fillna(0).to_numpy(), iva[target].to_numpy(), iva[date_col].to_numpy(), n_trials)
        out["pred_ridge"] = Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte)
        if lgb is not None:
            out["pred_lgbm"] = lgb.LGBMRegressor(**params).fit(Xtr, ytr).predict(Xte)
        if xgb is not None:
            out["pred_xgb"] = xgb.XGBRegressor(**XGB_DEFAULT).fit(Xtr, ytr).predict(Xte)
        if classify and lgb is not None:
            clf = lgb.LGBMClassifier(**{k: v for k, v in params.items()}).fit(
                itr[feats].fillna(0).to_numpy(), (itr[target] > 0).astype(int))
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                clf.predict_proba(iva[feats].fillna(0).to_numpy())[:, 1], (iva[target] > 0).astype(int))
            out["prob_up"] = iso.predict(clf.predict_proba(Xte)[:, 1])
            out["prob_up_raw"] = clf.predict_proba(Xte)[:, 1]
        for c in [c for c in ("fwd2", "fwd2_ex", "fwd5", "fwd5_ex", "ret_to_exit", "ret_to_exit_ex",
                              "long_sl005", "long_sl010", "long_sl015", "short_sl005", "short_sl010",
                              "short_sl015", "vol_20_raw", "rv_sofar", "prevday_vol_20") if c in te]:
            out[c] = te[c].to_numpy()
        preds.append(out)
        print(f"   [{tag}] fold {fi+1}/{len(starts)} train<{tr_end.date()} test {ts.date()}..{te[date_col].max().date()} n={len(te)}")
    return pd.concat(preds, ignore_index=True), params


def evaluate(pred: pd.DataFrame, target: str, date_col: str, models) -> pd.DataFrame:
    rows = []
    for m in models:
        if m not in pred:
            continue
        ic = daily_ic(pred, m, target, date_col)
        dec = pred.groupby(date_col).apply(lambda g: g.loc[g[m] >= g[m].quantile(0.9), target].mean()
                                           - g.loc[g[m] <= g[m].quantile(0.1), target].mean(), include_groups=False)
        rows.append({"model": m, "target": target, "mean_ic": ic.mean(), "ic_ir": ic.mean() / ic.std(),
                     "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)), "hit_rate": (ic > 0).mean(),
                     "top_bottom_decile_spread": dec.mean(), "n_periods": len(ic)})
    return pd.DataFrame(rows)


def calibration_plot(pred, target, path):
    if "prob_up" not in pred:
        return {}
    y = (pred[target] > 0).astype(int)
    bins = pd.qcut(pred["prob_up_raw"], 10, duplicates="drop")
    raw = pred.groupby(bins, observed=True).apply(lambda g: pd.Series({"p": g["prob_up_raw"].mean(), "y": (g[target] > 0).mean()}), include_groups=False)
    bins2 = pd.qcut(pred["prob_up"].rank(method="first"), 10)
    cal = pred.groupby(bins2, observed=True).apply(lambda g: pd.Series({"p": g["prob_up"].mean(), "y": (g[target] > 0).mean()}), include_groups=False)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(raw["p"], raw["y"], "o-", label="raw LightGBM")
    ax.plot(cal["p"], cal["y"], "s-", label="isotonic-calibrated")
    ax.set_xlabel("predicted P(up)"); ax.set_ylabel("observed frequency"); ax.legend(); ax.set_title(f"Reliability: {target}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    brier_raw = float(((pred["prob_up_raw"] - y) ** 2).mean())
    brier_cal = float(((pred["prob_up"] - y) ** 2).mean())
    return {"brier_raw": round(brier_raw, 5), "brier_calibrated": round(brier_cal, 5),
            "base_rate_brier": round(float(y.mean() * (1 - y.mean())), 5)}


# ----------------------------------------------------------------------------
# Strategy construction from predictions
# ----------------------------------------------------------------------------
def cnc_from_pred(pred: pd.DataFrame, col: str, k=4, hold=3, buffer=6) -> pd.DataFrame:
    s = pred.set_index(["date", "name"])[col]
    return F3.cnc_weights_from_scores(s, k=k, hold=hold, buffer=buffer)


def causal_calibrate(pred: pd.DataFrame, col: str, target: str, lookback_days=250, refit_every=21,
                     min_rows=3000, date_col="date") -> pd.Series:
    """Map a model score to an EXPECTED RETURN (in return units) with isotonic regression fitted
    only on earlier out-of-sample rows (rolling `lookback_days`, refit every `refit_every` days).
    This makes cost-hurdle decisions possible without look-ahead."""
    dates = pd.DatetimeIndex(pred[date_col].drop_duplicates().sort_values())
    out = pd.Series(np.nan, index=pred.index)
    for i in range(0, len(dates), refit_every):
        d0 = dates[i]
        d1 = dates[min(i + refit_every, len(dates) - 1)] if i + refit_every < len(dates) else dates[-1] + pd.Timedelta(days=1)
        hist = pred[(pred[date_col] < d0) & (pred[date_col] >= d0 - pd.Timedelta(days=int(lookback_days * 1.45)))]
        hist = hist.dropna(subset=[col, target])
        if len(hist) < min_rows:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(hist[col], hist[target])
        m = (pred[date_col] >= d0) & (pred[date_col] < d1)
        out[m] = iso.predict(pred.loc[m, col])
    return out


def mis_from_pred(pred: pd.DataFrame, col: str, mode="neutral", n_side=2, gross=2.0, stop="010",
                  hurdle=None, slots=("0930", "1100", "1300")):
    """Daily return on capital of an MIS book. mode: 'neutral' (long top / short bottom)
    or 'directional' (trade only names whose |prediction| clears the cost hurdle)."""
    rt = L.mis_rate_roundtrip() + 2 * 2.2e-4
    hurdle = rt if hurdle is None else hurdle
    d = pred[pred["slot"].astype(str).str.zfill(4).isin(slots)].dropna(subset=[col, f"long_sl{stop}"])
    out = {}
    for day, g in d.groupby("date"):
        pnl, n = 0.0, 0
        # one position per name per day: use the earliest slot where it qualifies
        g = g.sort_values(["slot", col])
        if mode == "neutral":
            g0 = g[g["slot"] == g["slot"].min()]
            if len(g0) < 2 * n_side + 4:
                continue
            longs, shorts = g0.nlargest(n_side, col), g0.nsmallest(n_side, col)
            per = gross / (2 * n_side)
            pnl = per * ((longs[f"long_sl{stop}"] - rt).sum() + (shorts[f"short_sl{stop}"] - rt).sum())
        else:
            # (fix) fill slot by slot in time order: a 09:30 decision cannot know the 11:00/13:00 signals
            lsel, ssel, used = [], [], set()
            for _, gs in g.groupby("slot", sort=True):
                gs = gs[~gs["name"].isin(used)]
                if len(lsel) < n_side:
                    add = gs[gs[col] > hurdle].nlargest(n_side - len(lsel), col)
                    lsel.append(add); used |= set(add["name"])
                gs = gs[~gs["name"].isin(used)]
                if len(ssel) < n_side:
                    add = gs[gs[col] < -hurdle].nsmallest(n_side - len(ssel), col)
                    ssel.append(add); used |= set(add["name"])
            longs = pd.concat(lsel) if lsel else g.iloc[:0]
            shorts = pd.concat(ssel) if ssel else g.iloc[:0]
            n = len(longs) + len(shorts)
            if n == 0:
                out[day] = 0.0
                continue
            per = gross / max(2 * n_side, 1)
            pnl = per * ((longs[f"long_sl{stop}"] - rt).sum() + (shorts[f"short_sl{stop}"] - rt).sum())
        out[day] = pnl
    return pd.Series(out).sort_index()


def main():
    ap = L.base_args("Phase 4: ML models with purged walk-forward validation")
    ap.add_argument("--trials", type=int, default=25, help="Optuna trials (first fold only)")
    args = ap.parse_args()
    data = L.data_dir(args.data)
    out = L.out_dir(PHASE)
    t0 = time.time()
    p3 = L.HERE / "phase3_features"
    fl = json.loads((p3 / "feature_lists.json").read_text())
    print("[P4] loading phase-3 features ...")
    D = pd.read_csv(p3 / "daily_features.csv", parse_dates=["date"])
    I = pd.read_csv(p3 / "intraday_features.csv", parse_dates=["date"])
    I["slot"] = I["slot"].astype(str).str.zfill(4)
    for s in F3.SLOTS:
        I[f"slot_{s}"] = (I["slot"] == s).astype(int)
    dfeats = fl["daily_features"] + fl["daily_market_features"]
    ifeats = fl["intraday_features"] + ["mkt_ret_since_open", "dow"] + [f"slot_{s}" for s in F3.SLOTS]
    ifeats = [c for c in ifeats if c in I]
    trials = 0 if args.quick else args.trials
    retrain_d = 126 if args.quick else 63
    retrain_i = 63 if args.quick else 42

    results, preds_d = [], {}
    for target, emb in (("fwd2_ex", 3), ("fwd5_ex", 6)):
        print(f"[P4] daily walk-forward target={target}")
        pdd, params = walk_forward(D.dropna(subset=[target]), dfeats, target, "date", pd.Timestamp("2019-01-01"),
                                   retrain_d, emb, trials, f"daily-{target}")
        preds_d[target] = pdd
        ev = evaluate(pdd, target, "date", ["pred_ridge", "pred_lgbm", "pred_xgb", "prob_up"])
        results.append(ev)
        (out / f"lgbm_params_{target}.json").write_text(json.dumps(params, indent=2))
    idates = I["date"].drop_duplicates().sort_values()
    first_i = idates.iloc[min(250, len(idates) // 2)]
    preds_i = {}
    for target in ("ret_to_exit_ex", "ret_to_exit"):
        print(f"[P4] intraday walk-forward target={target}")
        pii, params = walk_forward(I.dropna(subset=[target]), ifeats, target, "date", first_i,
                                   retrain_i, 1, trials, f"intra-{target}")
        preds_i[target] = pii
        results.append(evaluate(pii.assign(dk=pii["date"].astype(str) + pii["slot"]), target, "dk",
                                ["pred_ridge", "pred_lgbm", "pred_xgb", "prob_up"]))
    res = pd.concat(results, ignore_index=True)
    res.to_csv(out / "model_evaluation.csv", index=False)
    print(res.to_string())

    calib = calibration_plot(preds_d["fwd2_ex"], "fwd2_ex", out / "calibration_daily.png")
    calib_i = calibration_plot(preds_i["ret_to_exit"], "ret_to_exit", out / "calibration_intraday.png")

    # save OOS predictions (csv)
    for t, p in preds_d.items():
        p.to_csv(out / f"oos_pred_daily_{t}.csv", index=False, float_format="%.6g")
    for t, p in preds_i.items():
        p.to_csv(out / f"oos_pred_intraday_{t}.csv", index=False, float_format="%.6g")

    # ---------------- strategies from the best model ------------------------
    P = L.load_daily_panel(data)
    hs = L.half_spread_bps(P["adj_high"], P["adj_low"], P["turnover"])
    liq = L.liquid_returns(data, P["adj_close"].index)
    regp = L.HERE / "phase2_regimes_stochastic" / "regimes_daily.csv"
    reg = pd.read_csv(regp, index_col=0, parse_dates=True) if regp.exists() else None

    dres = res[res["target"].isin(["fwd2_ex", "fwd5_ex"]) & (res["model"] != "prob_up")]
    bd = dres.sort_values("ic_t", ascending=False).iloc[0]
    W = cnc_from_pred(preds_d[bd["target"]], bd["model"], k=4, hold=2 if bd["target"] == "fwd2_ex" else 5, buffer=6)
    oos_start = W.index.min()
    sims, n_trials = {}, len(res)
    cnc = L.simulate_weights(W, P["adj_open"], P["adj_close"], liq, pledge=True, half_spread=hs)["ret"].loc[oos_start:]
    sims[f"CNC ML top-4 ({bd['model']}, {bd['target']})"] = cnc
    if reg is not None:
        tv = (0.10 / reg["garch_vol_fcst"]).clip(upper=1).reindex(W.index).ffill().fillna(0)
        sims[f"CNC ML top-4 vol-targeted ({bd['model']})"] = L.simulate_weights(
            W.mul(tv, axis=0), P["adj_open"], P["adj_close"], liq, pledge=True, half_spread=hs)["ret"].loc[oos_start:]
    ires = res[res["target"].isin(["ret_to_exit", "ret_to_exit_ex"]) & (res["model"] != "prob_up")]
    bi = ires.sort_values("ic_t", ascending=False).iloc[0]
    pi = preds_i[bi["target"]]
    mis_n = mis_from_pred(pi, bi["model"], mode="neutral", gross=2.0)
    pdir = preds_i["ret_to_exit"]
    mcol = bi["model"] if bi["model"] in pdir else "pred_lgbm"
    pdir = pdir.assign(exp_ret=causal_calibrate(pdir, mcol, "ret_to_exit"))
    mis_d = mis_from_pred(pdir.dropna(subset=["exp_ret"]), "exp_ret", mode="directional", gross=2.0,
                          hurdle=L.mis_rate_roundtrip() + 2 * 2.2e-4 + 3e-4)
    liq_i = liq.reindex(mis_n.index).fillna(0)
    sims[f"Liquid-pledge + MIS ML neutral ({bi['model']}, {bi['target']})"] = liq_i + mis_n
    sims["Liquid-pledge + MIS ML directional, cost-hurdle (causal isotonic)"] = liq.reindex(mis_d.index).fillna(0) + mis_d
    cards = [L.scorecard(s, k) for k, s in sims.items()]
    dsr = {k: L.deflated_sharpe(s, n_trials) for k, s in sims.items()}
    best = L.save_scorecards(out, cards, PHASE, series=sims,
                             notes=f"OOS walk-forward. {n_trials} model/target trials -> deflated Sharpe in dsr.json")
    (out / "dsr.json").write_text(json.dumps(dsr, indent=2))
    (out / "calibration.json").write_text(json.dumps({"daily_fwd2": calib, "intraday_ret_to_exit": calib_i}, indent=2))
    L.plot_equity(sims, out / "candidates_equity.png", "Phase 4 ML candidates (OOS)", log=False)

    md = f"""# Phase 4 - Machine learning (purged walk-forward, OOS only)

Models: Ridge, LightGBM{'' if lgb else ' (not installed)'}, XGBoost{'' if xgb else ' (not installed)'},
LightGBM classifier + isotonic calibration. Optuna trials on fold 1: {trials}.

## Out-of-sample model evaluation
{L.md_table(res)}

Best daily model: **{bd['model']} on {bd['target']}** (IC t = {bd['ic_t']:.2f}).
Best intraday model: **{bi['model']} on {bi['target']}** (IC t = {bi['ic_t']:.2f}).

## Probability calibration
Daily: {calib} | Intraday: {calib_i}

## Deflated Sharpe ratios ({n_trials} trials)
{L.md_table(pd.DataFrame(dsr).T.reset_index().rename(columns={'index': 'strategy'}))}

## Candidate scorecards (OOS)
{L.md_table(pd.DataFrame(cards)[L.SC_COLS])}

**Best overall (incl. carried):** {best.get('strategy')} | P(profitable 12m) {best.get('P_profitable_12m')}
| predicted monthly {best.get('pred_monthly_mean')} | goal score {best.get('goal_score')}
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    L.print_scorecard(best, PHASE)
    print(f"[P4] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
