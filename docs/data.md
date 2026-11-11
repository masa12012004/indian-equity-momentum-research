# Data and configuration

## Local directory layout

Raw and cleaned files are grouped by timeframe, with one CSV or Parquet file per instrument:

```text
data/
  1D/RELIANCE.parquet
  5min/RELIANCE.parquet
  meta/
data_clean/
  1D/RELIANCE.parquet
  1D/NIFTY50.parquet
  5min/RELIANCE.parquet
  meta/
```

Other supported timeframe directories are `1min`, `15min` and `1H`. The illustrative filenames above describe the layout; no actual data files are included. Required intraday coverage depends on the pipeline and phase. The full baseline includes intraday experiments.

Raw files require `datetime`, `open`, `high`, `low`, `close`, `volume`. The cleaner emits adjusted and raw price fields, volumes, adjustment factors, session tags and derived features. Daily research reads fields including `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_volume`, `raw_open`, `raw_close` and `turnover`. Preserve the cleaner output schema for downstream phases.

The raw datetime convention is Asia/Kolkata, bar-open time. Review signal timing and corporate-action adjustment assumptions before relying on results. The cleaner infers factors from daily/intraday comparisons; these are not an authoritative corporate-action database.

Both pipelines default to the repository-level `data_clean/` directory. Their shared argument parser supports `--data PATH`; `QUANT_DATA` is another supported override. Metadata under `meta/` can affect classifications; preserve authorized instrument metadata where required.

## Optional IIFL downloader

Copy `.env.example` to `.env` in the repository root and populate the app key, secret and registered redirect URL locally. The example contains no credentials. `IIFL_USER_SESSION` is optional. Do not put tokens on a command line that may be recorded in shell history.

The downloader supports browser authentication and writes `iifl_session.json`. Both the populated environment and cached session must remain private. It fetches market data and can refresh a universe list; review API access, usage limits and redistribution permissions yourself. Credentials and broker integration were not exercised during publication.

From the root, after configuring your own account, the supplied probe entry point is:

```powershell
.venv\Scripts\python iifl_fetch_nifty100.py --probe
```

This is a network operation with authentication, not an offline test. Alternatively, provide equivalent authorized raw files without using the downloader.

## Cleaning

```powershell
.venv\Scripts\python clean_nifty100.py
```

The default input is `data/` and output is `data_clean/`. Use `--in-dir` and `--out-dir` to override them. Cleaning can rebuild outputs, so preserve any data you need before rerunning. Keep all generated datasets local unless you have explicit redistribution rights.
