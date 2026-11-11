#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_nifty100.py - Turn the raw IIFL downloads (data/) into a research-ready,
corporate-action-consistent dataset (data_clean/).

What it fixes
-------------
1. Price-scale mismatch. IIFL's 1D candles are back-adjusted for splits, bonuses,
   demergers and rights issues, but its intraday candles are the raw prices as
   traded. For every stock, the script compares each day's intraday high/low with
   the daily high/low and fits a piecewise-constant adjustment factor
   F(date) = raw / adjusted. The fit uses a rolling median plus segmentation, so
   one-day noise can't create a false corporate action, and it snaps to F = 1 when
   the two series already agree.
2. Bad bars. Bars with a zero, negative or missing price are dropped. Bars where
   high/low don't enclose open/close are repaired and flagged.
3. Bad daily ticks. A move of more than 40% that reverts to within 5% of the
   prior close within 3 days is removed (e.g. BEL, GAIL, TORNTPHARM on 2021-09-15).
4. Session tagging. Every intraday bar is tagged regular / closing (15:30 print) /
   pre_open / special (Muhurat etc.), so strategies can keep only regular bars.

Output schema (identical for every timeframe)
---------------------------------------------
    datetime                tz-aware Asia/Kolkata, bar OPEN time
    adj_open..adj_close     adjusted prices  -> use for returns, signals, features
    adj_volume              raw volume x factor (share count on today's basis)
    raw_open..raw_close     prices as actually traded -> use for tick size,
                            circuit limits, order prices, position sizing
    raw_volume
    adj_factor              raw / adjusted for that date (1.0 = no later corporate
                            action). NaN on daily rows older than the intraday
                            history, where the factor can't be inferred from this data.
    session                 regular | closing | pre_open | special (intraday only)
    repaired                True if high/low were repaired
    log_ret, true_range, hl_range_pct, gap_pct, turnover   (from adjusted prices;
    session_date, bar_in_session, vwap_session              backward-looking only)

    data_clean/meta/
        quality_report.csv              one row per instrument x timeframe
        corporate_actions_inferred.csv  ex-date + price ratio for each detected event
        adjustment_factors.csv          instrument, date, adj_factor (daily)

Usage
-----
    python clean_nifty100.py                     # data/ -> data_clean/ (parquet)
    python clean_nifty100.py --format csv
    python clean_nifty100.py --symbols RELIANCE,BAJFINANCE   # quick test

Re-run after every new download; it rebuilds data_clean/ from scratch in a few
minutes. Tip: refresh the daily files with
`python iifl_fetch_nifty100.py --timeframes 1D --full` from time to time, because a
new split makes IIFL re-adjust *all* past daily candles, and an incremental
download only adds new rows.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IST = "Asia/Kolkata"
TFS = ["1D", "1H", "15min", "5min", "1min"]
INTRA = {"1H": 60, "15min": 15, "5min": 5, "1min": 1}
OPEN_MIN, CLOSE_MIN = 9 * 60 + 15, 15 * 60 + 30
RAW = ["datetime", "open", "high", "low", "close", "volume"]
PX = ["open", "high", "low", "close"]

# --- tunables ---------------------------------------------------------------
SEG_TOL = 0.004          # |log ratio| change that counts as a new factor level (0.4%)
SEG_MIN_DAYS = 3         # a factor level must persist this many trading days
SPIKE_LOG = np.log(1.40) # daily bad-tick detector: >40% move ...
SPIKE_REVERT = 0.05      # ... that comes back within 5% of the prior close ...
SPIKE_MAX_LEN = 3        # ... within 3 bars


def grid(tf: str) -> np.ndarray:
    m = INTRA[tf]
    return np.array([OPEN_MIN + k * m for k in range(-(-(CLOSE_MIN - OPEN_MIN) // m))])


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def find_file(in_dir: Path, tf: str, name: str) -> Path | None:
    for ext in (".parquet", ".csv"):
        p = in_dir / tf / f"{name}{ext}"
        if p.exists():
            return p
    return None


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=RAW) if path.suffix == ".parquet" else pd.read_csv(path, usecols=RAW)
    dt = df["datetime"]
    if pd.api.types.is_datetime64_any_dtype(dt):
        dt = dt.dt.tz_localize(IST) if dt.dt.tz is None else dt.dt.tz_convert(IST)
    else:
        dt = pd.to_datetime(dt, utc=True, format="ISO8601").dt.tz_convert(IST)
    df["datetime"] = dt
    for c in PX + ["volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
              .sort_values("datetime").reset_index(drop=True))


def write(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if fmt == "parquet":
        out = df.copy()
        if "session_date" in out:
            out["session_date"] = pd.to_datetime(out["session_date"])
        out.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Cleaning steps
# ----------------------------------------------------------------------------
def clean_bars(df: pd.DataFrame, st: dict) -> pd.DataFrame:
    px = df[PX]
    bad = px.isna().any(axis=1) | (px <= 0).any(axis=1) | df["volume"].isna() | (df["volume"] < 0)
    st["dropped_bad_price"] = int(bad.sum())
    df = df[~bad].copy()
    hi, lo = df[PX].max(axis=1), df[PX].min(axis=1)
    rep = (df["high"] < hi - 1e-9) | (df["low"] > lo + 1e-9)
    st["repaired_hilo"] = int(rep.sum())
    df.loc[rep, "high"], df.loc[rep, "low"] = hi[rep], lo[rep]
    df["repaired"] = rep.values
    return df.reset_index(drop=True)


def remove_daily_spikes(d: pd.DataFrame, st: dict) -> pd.DataFrame:
    c = d["close"].to_numpy()
    bad = np.zeros(len(d), bool)
    i = 1
    while i < len(d):
        if abs(np.log(c[i] / c[i - 1])) > SPIKE_LOG:
            for j in range(i + 1, min(i + 1 + SPIKE_MAX_LEN, len(d))):
                if abs(c[j] / c[i - 1] - 1) < SPIKE_REVERT:
                    bad[i:j] = True
                    i = j
                    break
        i += 1
    st["daily_spikes_removed"] = int(bad.sum())
    st["daily_spike_dates"] = ";".join(d.loc[bad, "datetime"].dt.strftime("%Y-%m-%d"))
    return d[~bad].reset_index(drop=True)


def tag_session(df: pd.DataFrame, tf: str) -> pd.Series:
    m = (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute).to_numpy()
    s = np.full(len(df), "special", dtype=object)
    s[m < OPEN_MIN] = "pre_open"
    s[np.isin(m, grid(tf))] = "regular"
    s[(m >= CLOSE_MIN) & (m < 16 * 60)] = "closing"
    return pd.Series(s, index=df.index)


# ----------------------------------------------------------------------------
# Adjustment factor estimation
# ----------------------------------------------------------------------------
def _segments(lr: pd.Series) -> list[tuple[int, int, float]]:
    """Piecewise-constant fit of log(raw/adj). Returns [(start, end_excl, level)]."""
    sm = lr.rolling(5, center=True, min_periods=1).median().to_numpy()
    v = lr.to_numpy()
    segs, start = [], 0
    for t in range(1, len(sm)):
        level = np.median(sm[start:t])
        if abs(sm[t] - level) > SEG_TOL:
            segs.append([start, t])
            start = t
    segs.append([start, len(sm)])
    # merge segments shorter than SEG_MIN_DAYS into a neighbour
    merged = []
    for s in segs:
        if merged and (s[1] - s[0] < SEG_MIN_DAYS):
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < SEG_MIN_DAYS:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    out = []
    for a, b in merged:
        lvl = float(np.median(v[a:b]))
        if abs(lvl) < SEG_TOL:
            lvl = 0.0
        if out and abs(out[-1][2] - lvl) < SEG_TOL:   # merge equal neighbours
            out[-1] = (out[-1][0], b, out[-1][2])
        else:
            out.append((a, b, lvl))
    # Snap each level to a clean corporate-action ratio (2, 5, 1.5, 2.5, ...) when
    # it is within 0.15% of one; demerger ratios (e.g. 1.0571) stay as measured.
    snapped = []
    for a, b, lvl in out:
        r = np.exp(lvl)
        best = min((p / q for q in range(1, 6) for p in range(q, 41 * q)), key=lambda x: abs(np.log(r / x)))
        snapped.append((a, b, float(np.log(best)) if abs(np.log(r / best)) < 0.0015 else lvl))
    out = snapped
    # Refine each boundary: the rolling median can put the ex-date 1-2 days off
    # (e.g. a special price-discovery session on the ex-date). Pick the split
    # point that minimises absolute residuals on the raw daily ratios.
    for k in range(1, len(out)):
        a0, b0, l0 = out[k - 1]
        a1, b1, l1 = out[k]
        lo_, hi_ = max(a0 + 1, b0 - 3), min(b1 - 1, b0 + 3)
        best_b, best_cost = b0, np.inf
        for bb in range(lo_, hi_ + 1):
            cost = np.abs(v[lo_:bb] - l0).sum() + np.abs(v[bb:hi_ + 1] - l1).sum()
            if cost < best_cost - 1e-12:
                best_b, best_cost = bb, cost
        out[k - 1], out[k] = (a0, best_b, l0), (best_b, b1, l1)
    return out


def estimate_factors(daily: pd.DataFrame, intraday: dict, st: dict, name: str):
    parts = []
    for tf, df in intraday.items():
        reg = df[df["session"] == "regular"]
        if reg.empty:
            continue
        g = reg.groupby(reg["datetime"].dt.date)
        parts.append(pd.DataFrame({"h": g["high"].max(), "l": g["low"].min()}))
    if not parts or daily is None:
        return None, []
    agg = pd.concat(parts).groupby(level=0).median()
    dd = daily.set_index(daily["datetime"].dt.date)[["high", "low"]]
    j = agg.join(dd, how="inner")
    if j.empty:
        return None, []
    lr = np.log(pd.concat([j["h"] / j["high"], j["l"] / j["low"]], axis=1)).median(axis=1).sort_index()
    segs = _segments(lr)
    level = np.zeros(len(lr))
    for a, b, lvl in segs:
        level[a:b] = lvl
    fac = pd.Series(np.exp(level), index=lr.index, name="adj_factor")
    resid = np.abs(lr.to_numpy() - level)
    st["factor_days"] = len(lr)
    st["factor_levels"] = len(segs)
    st["residual_mismatch_days"] = int((resid > 0.005).sum())
    st["residual_mismatch_pct"] = round(100 * float((resid > 0.005).mean()), 3)
    st["latest_factor"] = round(float(fac.iloc[-1]), 5)
    events = []
    for k in range(1, len(segs)):
        events.append({"name": name, "ex_date": str(lr.index[segs[k][0]]),
                       "price_ratio": round(float(np.exp(segs[k - 1][2] - segs[k][2])), 4),
                       "factor_before": round(float(np.exp(segs[k - 1][2])), 4),
                       "factor_after": round(float(np.exp(segs[k][2])), 4)})
    st["events"] = len(events)
    return fac, events


def factor_for_dates(fac: pd.Series | None, dates: pd.Series, extrapolate_back: bool) -> np.ndarray:
    if fac is None:
        return np.ones(len(dates)) if extrapolate_back else np.full(len(dates), np.nan)
    f = pd.DataFrame({"d": pd.to_datetime(pd.Series(fac.index)).astype("datetime64[ns]"),
                      "f": fac.to_numpy()}).sort_values("d")
    q = pd.DataFrame({"d": pd.to_datetime(dates).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]").to_numpy(),
                      "i": np.arange(len(dates))})
    q = q.sort_values("d")
    back = pd.merge_asof(q, f, on="d", direction="backward")
    fwd = pd.merge_asof(q, f, on="d", direction="forward")
    res = back["f"].to_numpy(dtype=float, copy=True)
    before_first = np.isnan(res)
    # dates older than the first factor date: intraday -> use first level;
    # daily -> unknown (events before the intraday window are invisible here)
    res[before_first] = fwd["f"].to_numpy()[before_first] if extrapolate_back else np.nan
    out = np.empty(len(dates))
    out[q["i"].to_numpy()] = res
    return out


# ----------------------------------------------------------------------------
# Output assembly
# ----------------------------------------------------------------------------
def build(df: pd.DataFrame, factor: np.ndarray, intraday: bool, daily_is_adjusted: bool) -> pd.DataFrame:
    out = pd.DataFrame({"datetime": df["datetime"]})
    if daily_is_adjusted:                       # 1D: IIFL values are already adjusted
        for c in PX:
            out[f"adj_{c}"] = df[c].to_numpy()
        out["adj_volume"] = df["volume"].to_numpy()
        for c in PX:
            out[f"raw_{c}"] = df[c].to_numpy() * factor
        out["raw_volume"] = df["volume"].to_numpy() / factor
    else:                                       # intraday: IIFL values are raw
        for c in PX:
            out[f"adj_{c}"] = df[c].to_numpy() / factor
        out["adj_volume"] = df["volume"].to_numpy() * factor
        for c in PX:
            out[f"raw_{c}"] = df[c].to_numpy()
        out["raw_volume"] = df["volume"].to_numpy()
    out["adj_factor"] = factor
    if intraday:
        out["session"] = df["session"].to_numpy()
    out["repaired"] = df["repaired"].to_numpy()

    o, h, l, c, v = (out[f"adj_{k}"] for k in ("open", "high", "low", "close", "volume"))
    pc = c.shift(1)
    out["log_ret"] = np.log(c / pc)
    out["true_range"] = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    out["hl_range_pct"] = (h - l) / c
    out["gap_pct"] = o / pc - 1.0
    out["turnover"] = (h + l + c) / 3.0 * v            # ~ rupees traded (adjustment-invariant)
    if intraday:
        sd = out["datetime"].dt.date
        reg = out["session"] == "regular"
        out["session_date"] = sd
        out["bar_in_session"] = np.where(reg, reg.groupby(sd).cumsum() - 1, -1)
        tp_v = ((h + l + c) / 3.0 * v).where(reg, 0.0)
        cv = v.where(reg, 0.0).groupby(sd).cumsum()
        out["vwap_session"] = (tp_v.groupby(sd).cumsum() / cv.where(cv > 0)).where(reg)
    return out


def coverage(df: pd.DataFrame, tf: str, cal: pd.DatetimeIndex) -> tuple[int, int]:
    d = df["datetime"].dt.tz_localize(None).dt.normalize()
    exp_days = cal[(cal >= d.min()) & (cal <= d.max())]
    if tf == "1D":
        return len(exp_days), int(d.isin(exp_days).sum())
    reg = (df["session"] == "regular") & d.isin(exp_days)
    return len(exp_days) * len(grid(tf)), int(reg.sum())


# ----------------------------------------------------------------------------
# Per-instrument worker
# ----------------------------------------------------------------------------
def process(name: str, in_dir: str, out_dir: str, fmt: str, cal_vals) -> tuple[list, list, pd.DataFrame | None]:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    cal = pd.DatetimeIndex(cal_vals)
    reports, frames = [], {}
    for tf in TFS:
        p = find_file(in_dir, tf, name)
        st = {"name": name, "tf": tf}
        if p is None:
            st["status"] = "MISSING"
            reports.append(st)
            continue
        df = read_raw(p)
        st["rows_in"] = len(df)
        df = clean_bars(df, st)
        if tf == "1D":
            df = remove_daily_spikes(df, st)
        else:
            df["session"] = tag_session(df, tf)
        frames[tf] = (df, st)

    daily = frames.get("1D", (None, None))[0]
    intraday = {tf: frames[tf][0] for tf in INTRA if tf in frames}
    fst = {}
    fac, events = estimate_factors(daily, intraday, fst, name)

    for tf, (df, st) in frames.items():
        is_d = tf == "1D"
        f = factor_for_dates(fac, df["datetime"], extrapolate_back=not is_d)
        out = build(df, f, intraday=not is_d, daily_is_adjusted=is_d)
        write(out, out_dir / tf / f"{name}.{fmt}", fmt)
        exp, got = coverage(df, tf, cal)
        st.update(rows_out=len(out), first=str(out["datetime"].iloc[0].date()) if len(out) else "",
                  last=str(out["datetime"].iloc[-1].date()) if len(out) else "",
                  expected_bars=exp, regular_bars=got,
                  coverage_pct=round(100 * got / exp, 3) if exp else np.nan, status="OK", **fst)
        if not is_d:
            st["non_regular_bars"] = int((df["session"] != "regular").sum())
        reports.append(st)

    fac_df = None
    if fac is not None:
        fac_df = pd.DataFrame({"name": name, "date": [str(x) for x in fac.index], "adj_factor": fac.to_numpy()})
    return reports, events, fac_df


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Clean + corporate-action-adjust IIFL NIFTY 100 data",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=HERE / "data")
    ap.add_argument("--out-dir", type=Path, default=HERE / "data_clean")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--symbols", default=None, help="comma list (file names, e.g. M_M) to process only these")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)

    if args.format == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            print("[!] pyarrow not installed -> writing CSV (pip install pyarrow for parquet)")
            args.format = "csv"

    names = sorted({p.stem for p in (args.in_dir / "1D").glob("*.*") if p.suffix in (".csv", ".parquet")})
    if args.symbols:
        want = {s.strip().upper().replace("&", "_") for s in args.symbols.split(",")}
        names = [n for n in names if n.upper() in want]
    if not names:
        sys.exit(f"No 1D files found in {args.in_dir / '1D'}")

    # Trading calendar = union of daily dates across instruments
    cal = set()
    for n in names if len(names) < 5 else ["NIFTY50"] + names[:20]:
        p = find_file(args.in_dir, "1D", n)
        if p:
            cal |= set(read_raw(p)["datetime"].dt.tz_localize(None).dt.normalize())
    cal_vals = np.array(sorted(cal), dtype="datetime64[ns]")

    t0 = time.time()
    reports, events, facs = [], [], []
    print(f"[*] Cleaning {len(names)} instruments with {args.workers} workers -> {args.out_dir}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, n, str(args.in_dir), str(args.out_dir), args.format, cal_vals): n for n in names}
        for k, fut in enumerate(as_completed(futs), 1):
            n = futs[fut]
            try:
                r, e, f = fut.result()
                reports += r
                events += e
                if f is not None:
                    facs.append(f)
                ev = ", ".join(f"{x['ex_date']} x{x['price_ratio']}" for x in e)
                print(f"[{k:>3}/{len(names)}] {n:<12} " + (f"corporate actions: {ev}" if ev else "no corporate actions"))
            except Exception as exc:  # noqa: BLE001
                print(f"[!] {n} failed: {exc}")
                reports.append({"name": n, "status": f"ERROR: {exc}"})

    meta = args.out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    rep = pd.DataFrame(reports)
    rep.to_csv(meta / "quality_report.csv", index=False)
    pd.DataFrame(events, columns=["name", "ex_date", "price_ratio", "factor_before", "factor_after"]) \
        .sort_values(["name", "ex_date"]).to_csv(meta / "corporate_actions_inferred.csv", index=False)
    if facs:
        pd.concat(facs).to_csv(meta / "adjustment_factors.csv", index=False)
    inst = args.in_dir / "meta" / "instruments.csv"
    if inst.exists():
        pd.read_csv(inst).to_csv(meta / "instruments.csv", index=False)

    if "status" not in rep or not (rep["status"] == "OK").any():
        sys.exit("[!] No instrument processed successfully - see errors above.")
    ok = rep[rep["status"] == "OK"]
    print("\n================ QUALITY SUMMARY ================")
    summ = ok.groupby("tf").agg(files=("name", "count"), rows=("rows_out", "sum"),
                                expected=("expected_bars", "sum"), regular=("regular_bars", "sum"),
                                dropped=("dropped_bad_price", "sum"), repaired=("repaired_hilo", "sum"))
    summ["coverage_%"] = (100 * summ["regular"] / summ["expected"]).round(2)
    print(summ.reindex([t for t in TFS if t in summ.index]).to_string())
    one = ok.drop_duplicates("name")
    print(f"\nCorporate actions detected: {len(events)} across {len({e['name'] for e in events})} instruments")
    if "residual_mismatch_pct" in one:
        print(f"Intraday-vs-daily residual mismatch after adjustment: median "
              f"{one['residual_mismatch_pct'].median():.2f}% of days, worst "
              f"{one['residual_mismatch_pct'].max():.2f}% ({one.loc[one['residual_mismatch_pct'].idxmax(), 'name']})")
        bad_latest = one[(one["latest_factor"] - 1).abs() > 0.004]
        if len(bad_latest):
            print(f"[!] Latest factor != 1 for {list(bad_latest['name'])} - re-download 1D with --full")
    sp = ok[ok.get("daily_spikes_removed", 0) > 0] if "daily_spikes_removed" in ok else ok.iloc[:0]
    for _, r in sp.iterrows():
        print(f"Daily bad ticks removed: {r['name']} {r['daily_spike_dates']}")
    print(f"\nDone in {time.time() - t0:.0f}s. Reports in {meta}")


if __name__ == "__main__":
    main()
