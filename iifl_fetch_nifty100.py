#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iifl_fetch_nifty100.py - Bulk multi-timeframe OHLCV downloader for the NIFTY 100
universe + NIFTY indices, from the IIFL Capital Open API (v1).

Default horizons
----------------
    1min  ->  6 months      (5-day request windows)
    5min  ->  3 years       (15-day windows)
    15min ->  3 years       (30-day windows)
    1H    ->  3 years       (60-day windows)
    1D    -> 10 years       (365-day windows)

Windows are requested NEWEST FIRST. When a run of consecutive windows comes back
empty / "something went wrong", the script concludes it has walked past IIFL's
intraday retention for that timeframe and stops early, recording the earliest
date actually available in the manifest. So you ALWAYS get the maximum the broker
serves, up to the horizon above - but the broker may serve less than you ask for.

What you get
------------
    data/
      1min/RELIANCE.parquet, ...    one file per symbol per timeframe
      5min/... 15min/... 1H/... 1D/...
      1min/NIFTY50.parquet ...      indices live next to the stocks
      meta/
        nifty100_constituents.csv   NSE constituent list (symbol, ISIN, industry)
        instruments.csv             symbol -> IIFL instrumentId mapping used
        manifest.csv                rows / first / last / QC flags per file
        quotes_YYYY-MM-DD.json      (optional, --quotes) raw market-quote snapshot

Columns
-------
    raw:     datetime (tz-aware, Asia/Kolkata), open, high, low, close, volume
    derived  (strictly backward-looking, safe as features; disable with --no-derived):
             log_ret, true_range, hl_range_pct, typical_price, turnover_est,
             gap_pct (open vs previous bar close),
             intraday only: session_date, bar_in_session, vwap_session
             (cumulative session VWAP using bars up to and including the current one)

Re-running is incremental: existing files are extended from their last timestamp.
Use --full to rebuild from scratch.

Credentials (.env next to this script)
--------------------------------------
    IIFL_APP_KEY=xxxxxxxx
    IIFL_APP_SECRET=xxxxxxxx
    IIFL_REDIRECT_URL=http://127.0.0.1:5050/

Dependencies
------------
    pip install requests pandas numpy pyarrow python-dotenv

Examples
--------
    python iifl_fetch_nifty100.py --probe                 # sanity check, 1 symbol, all TFs
    python iifl_fetch_nifty100.py --symbols RELIANCE,TCS  # small test
    python iifl_fetch_nifty100.py                         # everything
    python iifl_fetch_nifty100.py --timeframes 1D,1H      # subset of timeframes
    python iifl_fetch_nifty100.py --quotes                # + live quote snapshot
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.server
import io
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests
import urllib3.util.connection as _urllib3_conn

# IIFL whitelists only the IPv4 registered to your app; a request that leaves over
# IPv6 gets a bare HTTP 401 from historicaldata/orders. Force IPv4 everywhere.
_urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

HERE = Path(__file__).resolve().parent
IST = "Asia/Kolkata"


# ----------------------------------------------------------------------------
# .env loading
# ----------------------------------------------------------------------------
def _load_env_file():
    candidates = [HERE / ".env", Path.cwd() / ".env"]
    try:
        from dotenv import load_dotenv
        for p in candidates:
            if p.is_file():
                load_dotenv(p, override=False)
                return p
        return None
    except ImportError:
        pass
    for p in candidates:
        if not p.is_file():
            continue
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v
        return p
    return None


_load_env_file()
APP_KEY = os.getenv("IIFL_APP_KEY", "")
APP_SECRET = os.getenv("IIFL_APP_SECRET", "")
REDIRECT_URL = os.getenv("IIFL_REDIRECT_URL", "http://127.0.0.1:5050/")

BASE_URL = "https://api.iiflcapital.com/v1"
WEB_LOGIN = "https://markets.iiflcapital.com/"
TOKEN_FILE = HERE / "iifl_session.json"   # same format as iifl_fetch_ohlcv.py


# ----------------------------------------------------------------------------
# Timeframe configuration  (edit here if you want different horizons)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TF:
    label: str          # folder name
    interval: str       # IIFL interval string
    months: int         # lookback horizon
    window_days: int    # calendar days per API request
    max_barren: int     # consecutive empty windows => past broker retention
    intraday: bool


TIMEFRAMES = {
    "1min":  TF("1min",  "1 minute",    6,   5,   6, True),
    "5min":  TF("5min",  "5 minutes",   36,  15,  6, True),
    "15min": TF("15min", "15 minutes",  36,  30,  5, True),
    "1H":    TF("1H",    "60 minutes",  36,  60,  4, True),
    "1D":    TF("1D",    "1 day",       120, 365, 2, False),
}

# Index name -> names it might appear under in the IIFL contract masters.
INDEX_ALIASES = {
    "NIFTY50":  ["NIFTY 50", "NIFTY50", "NIFTY", "NIFTY 50 INDEX"],
    "NIFTY100": ["NIFTY 100", "NIFTY100", "NIFTY 100 INDEX"],
}
# F&O underlying symbol (used as a fallback to discover the index instrumentId).
INDEX_FNO_UNDERLYING = {"NIFTY50": "NIFTY"}

NIFTY100_CSV_URLS = [
    "https://niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
]

# Fallback ONLY if NSE's site can't be reached. Constituents change twice a year,
# so verify against https://www.niftyindices.com before relying on this list.
NIFTY100_FALLBACK = """
ABB ADANIENSOL ADANIENT ADANIGREEN ADANIPORTS ADANIPOWER AMBUJACEM APOLLOHOSP
ASIANPAINT AXISBANK BAJAJ-AUTO BAJAJFINSV BAJAJHLDNG BAJFINANCE BANKBARODA BEL
BHARTIARTL BOSCHLTD BPCL BRITANNIA CANBK CGPOWER CHOLAFIN CIPLA COALINDIA DABUR
DIVISLAB DLF DMART DRREDDY EICHERMOT ETERNAL GAIL GODREJCP GRASIM HAL HAVELLS
HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR HINDZINC ICICIBANK
ICICIGI INDHOTEL INDIGO INDUSINDBK INFY IOC IRFC ITC JINDALSTEL JIOFIN JSWENERGY
JSWSTEEL KOTAKBANK LICI LODHA LT LTIM M&M MARUTI MOTHERSON NAUKRI NESTLEIND NTPC
ONGC PFC PIDILITIND PNB POWERGRID RECLTD RELIANCE SBILIFE SBIN SHREECEM SHRIRAMFIN
SIEMENS SUNPHARMA SWIGGY TATACONSUM TATAMOTORS TATAPOWER TATASTEEL TCS TECHM TITAN
TORNTPHARM TRENT TVSMOTOR ULTRACEMCO UNITDSPR VBL VEDL WIPRO ZYDUSLIFE
""".split()


class IIFLAuthError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def months_ago(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + (d.month - 1) - months, 12)
    m += 1
    for day in (d.day, 30, 29, 28):
        try:
            return date(y, m, day)
        except ValueError:
            continue
    return date(y, m, 28)


class RateLimiter:
    """Global token-spacing limiter shared by all worker threads."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / max(per_second, 0.01)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _parse_json_response(resp: requests.Response):
    """IIFL sometimes returns stringified (even double-encoded) JSON."""
    text = resp.text.strip()
    if not text:
        raise RuntimeError(f"Empty response (HTTP {resp.status_code})")
    try:
        data = json.loads(text)
    except ValueError:
        raise RuntimeError(f"Unparseable response (HTTP {resp.status_code}): {text[:300]}")
    for _ in range(5):
        if not isinstance(data, str):
            break
        try:
            data = json.loads(data)
        except ValueError:
            break
    return data


def _is_bad_date_range(err: str) -> bool:
    e = err.lower()
    return "ec807" in e or "invalid date range" in e


# ---- candle extraction (shape-agnostic, ported from iifl_fetch_ohlcv.py) ----
_TS_KEYS = ("initialtimestamp", "timestamp", "time", "datetime", "date", "ts")
_WRAPPER_KEYS = ("candles", "data", "result", "candlesdata", "chartdata", "values", "candledata")


def _is_num(v) -> bool:
    if isinstance(v, (int, float)):
        return True
    try:
        float(str(v).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _candle_from_dict(d: dict):
    lower = {str(k).lower(): k for k in d}

    def pick(*names):
        for n in names:
            if n in lower:
                return d[lower[n]]
        return None

    o, c = pick("open"), pick("close")
    if o is None or c is None:
        return None
    return [pick(*_TS_KEYS), o, pick("high"), pick("low"), c,
            pick("volume", "vol", "tradedvolume", "totalvolumetraded")]


def _extract_candle_rows(node, depth=0) -> list:
    if depth > 8 or node is None:
        return []
    if isinstance(node, str):
        s = node.strip()
        if s[:1] in ("{", "[", '"'):
            try:
                return _extract_candle_rows(json.loads(s), depth + 1)
            except ValueError:
                pass
        return []
    if isinstance(node, dict):
        row = _candle_from_dict(node)
        if row:
            return [row]
        for key in _WRAPPER_KEYS:
            for k, v in node.items():
                if str(k).lower() == key:
                    rows = _extract_candle_rows(v, depth + 1)
                    if rows:
                        return rows
        for v in node.values():
            rows = _extract_candle_rows(v, depth + 1)
            if rows:
                return rows
        return []
    if isinstance(node, (list, tuple)):
        if not node:
            return []
        if all(isinstance(x, (list, tuple)) and len(x) >= 6
               and sum(_is_num(y) for y in x[1:6]) >= 4 for x in node):
            return [list(x[:6]) for x in node]
        rows = []
        for item in node:
            if isinstance(item, dict):
                r = _candle_from_dict(item)
                if r:
                    rows.append(r)
                    continue
            rows.extend(_extract_candle_rows(item, depth + 1))
        return rows
    return []


def _broker_error(node, depth=0):
    """Find an embedded {"status":"Not ok","message":...} anywhere in the reply."""
    if depth > 6 or node is None:
        return None
    if isinstance(node, str):
        try:
            return _broker_error(json.loads(node), depth + 1)
        except ValueError:
            return None
    if isinstance(node, dict):
        keys = {str(k).lower() for k in node}
        if ("status" in keys or "errorcode" in keys) and ("message" in keys or "errormessage" in keys):
            st = node.get("status", node.get("errorCode"))
            msg = node.get("message", node.get("errorMessage"))
            if str(st).lower() not in ("ok", "success", "0"):
                return f"IIFL API error {st}: {msg}"
        for v in node.values():
            f = _broker_error(v, depth + 1)
            if f:
                return f
    elif isinstance(node, (list, tuple)):
        for v in node:
            f = _broker_error(v, depth + 1)
            if f:
                return f
    return None


def _to_ist(raw: pd.Series) -> pd.Series:
    """Normalise IIFL timestamps (epoch s/ms or strings) to tz-aware IST."""
    s = pd.Series(raw)
    nums = pd.to_numeric(s, errors="coerce")
    if nums.notna().all() and len(nums):
        unit = "s" if nums.max() < 1e11 else "ms"
        return pd.to_datetime(nums, unit=unit, utc=True).dt.tz_convert(IST)
    parsed = None
    for fmt in ("ISO8601", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y",
                "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            parsed = pd.to_datetime(s, format=fmt, errors="raise")
            break
        except (ValueError, TypeError):
            continue
    if parsed is None:
        parsed = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(parsed.dt, "tz", None) is not None:
        return parsed.dt.tz_convert(IST)
    return parsed.dt.tz_localize(IST)   # naive strings from IIFL are exchange time


# ----------------------------------------------------------------------------
# IIFL client
# ----------------------------------------------------------------------------
class IIFLClient:
    def __init__(self, app_key: str, app_secret: str, redirect_url: str,
                 rps: float = 3.0, max_retries: int = 4):
        if not app_key or not app_secret:
            raise IIFLAuthError("IIFL_APP_KEY / IIFL_APP_SECRET missing - put them in .env "
                                "next to this script.")
        self.app_key, self.app_secret = app_key.strip(), app_secret.strip()
        self.redirect_url = redirect_url.strip()
        self.limiter = RateLimiter(rps)
        self.max_retries = max_retries
        self.token: str | None = None
        self._tls = threading.local()
        self.stop = threading.Event()       # set on auth failure -> all workers stop

    # ---- per-thread HTTP session ----
    def _http(self) -> requests.Session:
        s = getattr(self._tls, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
            self._tls.s = s
        return s

    # ---- login (browser SSO -> authCode -> checksum -> userSession) ----
    def login(self, force: bool = False, token: str | None = None) -> None:
        token = token or os.getenv("IIFL_USER_SESSION")
        if token:
            self.token = token
            log("[*] Using session token supplied via --token / IIFL_USER_SESSION.")
            return
        if not force:
            try:
                d = json.loads(TOKEN_FILE.read_text())
                if d.get("savedOn") == date.today().isoformat() and d.get("userSession"):
                    self.token = d["userSession"]
                    log("[*] Reusing today's cached session (iifl_session.json). --login forces a new one.")
                    return
            except (OSError, ValueError):
                pass

        url = f"{WEB_LOGIN}?v=1&appkey={self.app_key}&redirecturl={self.redirect_url}"
        print("\n" + "=" * 72 + "\nLog in to IIFL in the browser (credentials + OTP/TOTP):\n\n   "
              + url + "\n" + "=" * 72)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        auth_code, client_id = self._capture_auth_code()
        checksum = hashlib.sha256(f"{client_id}{auth_code}{self.app_secret}".encode()).hexdigest()
        r = self._http().post(f"{BASE_URL}/getusersession", json={"checkSum": checksum}, timeout=30)
        data = _parse_json_response(r)
        if not isinstance(data, dict) or not data.get("userSession"):
            raise IIFLAuthError(f"getusersession failed: {str(data)[:400]}")
        self.token = data["userSession"]
        TOKEN_FILE.write_text(json.dumps({"userSession": self.token, "clientId": client_id,
                                          "savedOn": date.today().isoformat()}))
        log("[+] Login OK - session cached for today.")

    def _capture_auth_code(self):
        p = urlparse(self.redirect_url)
        if p.scheme == "http" and p.hostname in ("localhost", "127.0.0.1"):
            port = p.port or 80
            log(f"[*] Waiting on {p.hostname}:{port} for the IIFL redirect (10 min timeout)...")
            captured, done = {}, threading.Event()

            class H(http.server.BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802
                    captured.update({k.lower(): v for k, v in parse_qs(urlparse(self.path).query).items()})
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<h3>IIFL login captured - you can close this tab.</h3>")
                    done.set()

                def log_message(self, *a):
                    pass

            srv = http.server.HTTPServer(("127.0.0.1", port), H)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            ok = done.wait(600)
            srv.shutdown()
            if ok:
                ac = captured.get("authcode") or captured.get("auth_code") or captured.get("code")
                cid = captured.get("clientid") or captured.get("client_id")
                if ac and cid:
                    return ac[0], cid[0]
        raw = input("\nPaste the FULL redirected URL (…authCode=XXX&clientId=YYY):\n> ").strip().strip("'\"")
        qs = {k.lower(): v for k, v in parse_qs(urlparse(raw).query if "?" in raw else raw).items()}
        ac = qs.get("authcode") or qs.get("auth_code") or qs.get("code")
        cid = qs.get("clientid") or qs.get("client_id")
        if not (ac and cid):
            raise IIFLAuthError("authCode / clientId not found in: " + raw)
        return ac[0], cid[0]

    # ---- transport ----
    def post(self, path: str, payload):
        if not self.token:
            raise IIFLAuthError("Not logged in")
        last = None
        for attempt in range(self.max_retries):
            if self.stop.is_set():
                raise IIFLAuthError("aborted (auth failure in another worker)")
            self.limiter.wait()
            try:
                r = self._http().post(BASE_URL + path, json=payload, timeout=90,
                                      headers={"Authorization": f"Bearer {self.token}"})
            except requests.RequestException as e:
                last = e
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code in (401, 403):
                self.stop.set()
                raise IIFLAuthError(
                    f"HTTP {r.status_code} on {path}: {r.text[:200] or '(empty body)'} - session expired "
                    "or IP not whitelisted. Re-run with --login (and check the app's whitelisted IPv4).")
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                # 5xx is often IIFL's "out of retention" crash - don't grind on it
                if r.status_code >= 500 and attempt >= 1:
                    break
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else min(2 ** (attempt + 1), 30))
                continue
            return _parse_json_response(r)
        raise RuntimeError(f"{path} failed: {last}")

    def candles(self, instrument_id, exchange: str, interval: str, f: date, t: date) -> list:
        data = self.post("/marketdata/historicaldata", {
            "exchange": exchange,
            "instrumentId": str(instrument_id),
            "interval": interval,
            "fromDate": f.strftime("%d-%b-%Y"),
            "toDate": t.strftime("%d-%b-%Y"),
        })
        if isinstance(data, dict):
            st = str(data.get("status", "")).lower()
            if st not in ("ok", "success"):
                raise RuntimeError(f"historicaldata: {json.dumps(data)[:300]}")
            data = data.get("result")
        err = _broker_error(data)
        if err:
            raise RuntimeError(err)
        return _extract_candle_rows(data)

    # ---- contract masters (public, cached per day) ----
    def contracts(self, exchange: str, meta_dir: Path) -> list[dict]:
        cache = meta_dir / f"iifl_contracts_{exchange}.json"
        if cache.exists() and date.fromtimestamp(cache.stat().st_mtime) == date.today():
            return json.loads(cache.read_text())
        log(f"[*] Downloading {exchange} contract master ...")
        r = self._http().get(f"{BASE_URL}/contractfiles/{exchange}.json", timeout=180)
        r.raise_for_status()
        data = _parse_json_response(r)
        if isinstance(data, dict):
            for k in ("result", "data", "contracts"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected {exchange} contract master shape")
        cache.write_text(json.dumps(data))
        return data


# ----------------------------------------------------------------------------
# Universe + instrument resolution
# ----------------------------------------------------------------------------
def load_nifty100(meta_dir: Path, refresh: bool) -> pd.DataFrame:
    cache = meta_dir / "nifty100_constituents.csv"
    if cache.exists() and not refresh and (time.time() - cache.stat().st_mtime) < 7 * 86400:
        return pd.read_csv(cache)
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Accept": "text/csv,*/*", "Referer": "https://www.niftyindices.com/"}
    for url in NIFTY100_CSV_URLS:
        try:
            r = requests.get(url, headers=hdrs, timeout=30)
            if r.ok and "Symbol" in r.text[:500]:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                df = df.rename(columns={"Symbol": "symbol", "Company Name": "company",
                                        "Industry": "industry", "ISIN Code": "isin", "Series": "series"})
                df["symbol"] = df["symbol"].str.strip().str.upper()
                df.to_csv(cache, index=False)
                log(f"[+] NIFTY 100 constituents: {len(df)} symbols from {url}")
                return df
        except Exception as e:  # noqa: BLE001
            log(f"[!] {url}: {e}")
    if cache.exists():
        log("[!] Using stale cached constituent list.")
        return pd.read_csv(cache)
    log("[!] Could not reach NSE - using the built-in FALLBACK list (may be out of date).")
    return pd.DataFrame({"symbol": NIFTY100_FALLBACK})


def _norm(v) -> str:
    return str(v or "").strip().upper()


def build_equity_map(rows: list[dict]) -> tuple[dict, dict]:
    """symbol -> instrumentId and isin -> instrumentId for NSE EQ-series rows."""
    by_sym, by_isin = {}, {}
    for r in rows:
        low = {str(k).lower(): v for k, v in r.items()}
        iid = low.get("instrumentid")
        if iid is None:
            continue
        series = _norm(low.get("series"))
        sym = _norm(low.get("tradingsymbol") or low.get("underlyinginstrumentsymbol"))
        sym = sym.removesuffix("-EQ")
        prio = 0 if series == "EQ" else 1
        if sym and (sym not in by_sym or prio < by_sym[sym][1]):
            by_sym[sym] = (iid, prio)
        isin = _norm(low.get("isin") or low.get("isincode"))
        if isin and (isin not in by_isin or prio < by_isin[isin][1]):
            by_isin[isin] = (iid, prio)
    return ({k: v[0] for k, v in by_sym.items()}, {k: v[0] for k, v in by_isin.items()})


def resolve_index(client: IIFLClient, name: str, meta_dir: Path, end: date):
    """Find (exchange, instrumentId) for an index. Tries several contract masters,
    then the F&O underlying id, then probes which exchange code serves candles."""
    aliases = {a.upper() for a in INDEX_ALIASES.get(name, [name])}
    cands: list[tuple[str, str]] = []
    for exch in ("INDICES", "NSEINDEX", "NSEEQ"):
        try:
            rows = client.contracts(exch, meta_dir)
        except Exception:  # noqa: BLE001 - that master may not exist
            continue
        for r in rows:
            low = {str(k).lower(): v for k, v in r.items()}
            names = {_norm(low.get(k)) for k in ("tradingsymbol", "underlyinginstrumentsymbol",
                                                  "formattedinstrumentname", "name", "symbol",
                                                  "instrumentname")}
            if names & aliases and low.get("instrumentid") is not None:
                series = _norm(low.get("series")) + _norm(low.get("instrumenttype"))
                if exch == "NSEEQ" and "EQ" in series and "IND" not in series:
                    continue
                cands.append((exch, str(low["instrumentid"])))
    und = INDEX_FNO_UNDERLYING.get(name)
    if und:
        try:
            for r in client.contracts("NSEFO", meta_dir):
                low = {str(k).lower(): v for k, v in r.items()}
                if _norm(low.get("underlyinginstrumentsymbol")) != und:
                    continue
                for k, v in low.items():
                    if "underlying" in k and ("id" in k or "token" in k) and "symbol" not in k and v:
                        cands.append(("NSEFO-underlying", str(v)))
                        break
                if any(c[0] == "NSEFO-underlying" for c in cands):
                    break
        except Exception as e:  # noqa: BLE001
            log(f"[!] NSEFO master unavailable for index discovery: {e}")

    seen = []
    for c in cands:
        if c[1] not in [s[1] for s in seen]:
            seen.append(c)
    # Probe a recent 1D window across plausible exchange codes.
    for src, iid in seen[:6]:
        for exch in ("NSEEQ", "INDICES", "NSEINDEX"):
            try:
                rows = client.candles(iid, exch, "1 day", end - timedelta(days=12), end - timedelta(days=1))
            except IIFLAuthError:
                raise
            except Exception:  # noqa: BLE001
                continue
            if rows:
                log(f"[+] {name}: instrumentId {iid} on exchange {exch} (found via {src})")
                return exch, iid
    log(f"[!] Could not resolve {name}. Candidates tried: {seen[:6]}. "
        f"Pass it explicitly: --index-id {name}=<instrumentId> [--index-exchange NSEEQ]")
    return None


# ----------------------------------------------------------------------------
# Fetching one (instrument, timeframe)
# ----------------------------------------------------------------------------
def fetch_range(client: IIFLClient, iid, exch: str, tf: TF, start: date, end: date, tag: str):
    """Newest-first windows with retention cut-off. Returns (rows, earliest_date_with_data)."""
    windows, cur = [], start
    while cur <= end:
        w_end = min(cur + timedelta(days=tf.window_days - 1), end)
        windows.append((cur, w_end))
        cur = w_end + timedelta(days=1)
    windows.reverse()

    rows, barren, oldest = [], 0, None
    for w_from, w_to in windows:
        got, bad_range = [], False
        for attempt in range(3):
            try:
                got = client.candles(iid, exch, tf.interval, w_from, w_to)
                bad_range = False
                break
            except IIFLAuthError:
                raise
            except RuntimeError as e:
                bad_range = _is_bad_date_range(str(e))
                if bad_range and w_to > w_from and attempt < 2:
                    w_to -= timedelta(days=1)     # IIFL's "today" lags ours
                    continue
                break
        if bad_range:
            continue
        if got:
            rows.extend(got)
            oldest, barren = w_from, 0
        else:
            barren += 1
            if barren >= tf.max_barren:
                break
    return rows, oldest


def to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = _to_ist(df["datetime"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.dropna(subset=["datetime", "close"])
              .drop_duplicates("datetime", keep="last")
              .sort_values("datetime").reset_index(drop=True))


def add_derived(df: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    """Backward-looking only (no look-ahead): each value uses bars <= current."""
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    pc = c.shift(1)
    df["log_ret"] = np.log(c / pc)
    df["true_range"] = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["hl_range_pct"] = (h - l) / c
    df["typical_price"] = (h + l + c) / 3.0
    df["turnover_est"] = df["typical_price"] * v
    df["gap_pct"] = o / pc - 1.0
    if intraday:
        sd = df["datetime"].dt.date
        df["session_date"] = sd
        df["bar_in_session"] = df.groupby(sd).cumcount()
        cum_v = v.groupby(sd).cumsum()
        df["vwap_session"] = (df["turnover_est"].groupby(sd).cumsum() / cum_v.where(cum_v > 0))
    return df


def qc(df: pd.DataFrame, intraday: bool) -> dict:
    if df.empty:
        return {}
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    out = {
        "qc_bad_hilo": int(((h < np.maximum(o, c)) | (l > np.minimum(o, c))).sum()),
        "qc_nonpos_px": int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "qc_neg_vol": int((v < 0).sum()),
        "qc_zero_vol": int((v == 0).sum()),
    }
    if intraday:
        t = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
        out["qc_outside_session"] = int(((t < 9 * 60 + 15) | (t > 15 * 60 + 30)).sum())
        # Large jumps in |log_ret| (> 20%) inside a session usually = bad tick / split
        lr = np.log(c / c.shift(1)).abs()
        same_day = df["datetime"].dt.date == df["datetime"].dt.date.shift(1)
        out["qc_jumps_gt20pct"] = int(((lr > 0.2) & same_day).sum())
    return out


def _path(out_dir: Path, tf: TF, name: str, fmt: str) -> Path:
    safe = name.replace("&", "_").replace("/", "_")
    return out_dir / tf.label / f"{safe}.{fmt}"


def _read(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        df["datetime"] = _to_ist(df["datetime"]) if df["datetime"].dtype == object \
            else pd.to_datetime(df["datetime"]).dt.tz_convert(IST)
        return df[["datetime", "open", "high", "low", "close", "volume"]]
    except Exception as e:  # noqa: BLE001
        log(f"[!] could not read {p} ({e}) - refetching from scratch")
        return None


def _write(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    if p.suffix == ".parquet":
        out = df.copy()
        if "session_date" in out:
            out["session_date"] = pd.to_datetime(out["session_date"])
        out.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, p)


def run_job(client, name, kind, iid, exch, tf: TF, end: date, args) -> dict:
    p = _path(args.out_dir, tf, name, args.format)
    start = months_ago(end, tf.months)
    rec = {"name": name, "kind": kind, "timeframe": tf.label, "instrument_id": iid,
           "exchange": exch, "requested_from": start.isoformat(), "file": str(p)}
    old = None if args.full else _read(p)
    fetch_from = start
    if old is not None and len(old):
        fetch_from = max(start, old["datetime"].iloc[-1].date())   # 1-day overlap
    t0 = time.time()
    rows, oldest = fetch_range(client, iid, exch, tf, fetch_from, end, f"{name}/{tf.label}")
    new = to_frame(rows) if rows else None
    parts = [d for d in (old, new) if d is not None and len(d)]
    if not parts:
        rec.update(status="EMPTY", rows=0)
        log(f"[!] {name:<12} {tf.label:<5} no data")
        return rec
    df = (pd.concat(parts).drop_duplicates("datetime", keep="last")
            .sort_values("datetime").reset_index(drop=True))
    df = df[df["datetime"].dt.date >= start] if not args.keep_older else df
    # IIFL occasionally returns placeholder bars with a 0 (or missing) price. They
    # poison log returns (-inf/+inf), true range and VWAP, so drop them here. This
    # also cleans files written by earlier runs, since old rows pass through here.
    px = df[["open", "high", "low", "close"]]
    bad_px = (px.isna() | (px <= 0)).any(axis=1)
    rec["dropped_zero_px_bars"] = int(bad_px.sum())
    if bad_px.any():
        df = df[~bad_px].reset_index(drop=True)
    if df.empty:
        rec.update(status="EMPTY", rows=0)
        return rec
    if not args.no_derived:
        df = add_derived(df, tf.intraday)
    _write(df, p)
    rec.update(status="OK", rows=len(df), new_rows=0 if new is None else len(new),
               first=str(df["datetime"].iloc[0]), last=str(df["datetime"].iloc[-1]),
               retention_limited=bool(old is None and oldest and oldest > start + timedelta(days=tf.window_days)),
               seconds=round(time.time() - t0, 1), **qc(df, tf.intraday))
    log(f"[+] {name:<12} {tf.label:<5} {len(df):>8} rows  {df['datetime'].iloc[0]:%Y-%m-%d} -> "
        f"{df['datetime'].iloc[-1]:%Y-%m-%d %H:%M}  (+{rec['new_rows']}, {rec['seconds']}s)"
        + (f"  [dropped {rec['dropped_zero_px_bars']} zero-price bars]" if rec["dropped_zero_px_bars"] else ""))
    return rec


def snapshot_quotes(client: IIFLClient, instruments: list[tuple[str, str, str]], meta_dir: Path):
    """Best-effort live quote snapshot (LTP, OHLC, 52w hi/lo, circuits, depth... whatever
    IIFL returns). Saved raw - field names are not normalised."""
    out = []
    for i in range(0, len(instruments), 50):
        batch = [{"exchange": e, "instrumentId": str(iid)} for _, e, iid in instruments[i:i + 50]]
        try:
            out.append(client.post("/marketdata/marketquotes", batch))
        except IIFLAuthError:
            raise
        except Exception as e:  # noqa: BLE001
            log(f"[!] marketquotes failed ({e}); skipping snapshot.")
            return
    p = meta_dir / f"quotes_{date.today():%Y-%m-%d}.json"
    p.write_text(json.dumps(out, indent=1))
    log(f"[+] Quote snapshot saved to {p}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="NIFTY 100 + NIFTY index multi-timeframe OHLCV from IIFL",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    ap.add_argument("--timeframes", default="1min,5min,15min,1H,1D",
                    help=f"subset of {list(TIMEFRAMES)}")
    ap.add_argument("--symbols", default=None, help="comma list to override the NIFTY 100 universe")
    ap.add_argument("--limit", type=int, default=None, help="only first N stocks (testing)")
    ap.add_argument("--indices", default="NIFTY50,NIFTY100", help="indices to fetch ('' for none)")
    ap.add_argument("--index-id", action="append", default=[],
                    help="override index id, e.g. --index-id NIFTY50=26000 (repeatable)")
    ap.add_argument("--index-exchange", default="NSEEQ", help="exchange code used with --index-id")
    ap.add_argument("--end", default=None, help="end date YYYY-MM-DD (default today)")
    ap.add_argument("--workers", type=int, default=4, help="parallel download threads")
    ap.add_argument("--rps", type=float, default=3.0, help="global API requests per second")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--full", action="store_true", help="ignore existing files, refetch everything")
    ap.add_argument("--keep-older", action="store_true",
                    help="don't trim rows older than the horizon from existing files")
    ap.add_argument("--no-derived", action="store_true", help="save raw OHLCV only")
    ap.add_argument("--quotes", action="store_true", help="also save a live market-quote snapshot")
    ap.add_argument("--refresh-universe", action="store_true", help="re-download the NIFTY 100 list")
    ap.add_argument("--login", action="store_true", help="force a fresh browser login")
    ap.add_argument("--token", default=None, help="use this userSession JWT directly")
    ap.add_argument("--probe", action="store_true",
                    help="quick check: 1 stock + NIFTY50, last ~10 days, every timeframe")
    args = ap.parse_args(argv)

    if args.format == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            log("[!] pyarrow not installed -> falling back to CSV (pip install pyarrow for parquet)")
            args.format = "csv"

    tfs = [TIMEFRAMES[t.strip()] for t in args.timeframes.split(",") if t.strip()]
    end = date.fromisoformat(args.end) if args.end else date.today()
    args.out_dir = args.out_dir.resolve()
    meta = args.out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    client = IIFLClient(APP_KEY, APP_SECRET, REDIRECT_URL, rps=args.rps)
    client.login(force=args.login, token=args.token)

    # ---- universe ----
    if args.symbols:
        uni = pd.DataFrame({"symbol": [s.strip().upper() for s in args.symbols.split(",") if s.strip()]})
    else:
        uni = load_nifty100(meta, args.refresh_universe)
    if args.probe:
        uni = uni.head(1)
    elif args.limit:
        uni = uni.head(args.limit)

    by_sym, by_isin = build_equity_map(client.contracts("NSEEQ", meta))
    instruments, missing = [], []
    for _, r in uni.iterrows():
        sym = str(r["symbol"]).upper()
        isin = _norm(r.get("isin")) if "isin" in uni.columns else ""
        iid = by_sym.get(sym) or (by_isin.get(isin) if isin else None)
        if iid is None:
            missing.append(sym)
        else:
            instruments.append((sym, "NSEEQ", str(iid), "stock"))
    if missing:
        log(f"[!] Not found in NSEEQ master (renamed/delisted?): {missing}")

    # ---- indices ----
    overrides = dict(s.split("=", 1) for s in args.index_id)
    idx_names = [s.strip().upper() for s in args.indices.split(",") if s.strip()]
    if args.probe:
        idx_names = idx_names[:1]
    for name in idx_names:
        if name in overrides:
            instruments.insert(0, (name, args.index_exchange, overrides[name], "index"))
            continue
        res = resolve_index(client, name, meta, end)
        if res:
            instruments.insert(0, (name, res[0], res[1], "index"))

    pd.DataFrame(instruments, columns=["name", "exchange", "instrument_id", "kind"]).merge(
        uni.rename(columns={"symbol": "name"}), on="name", how="left"
    ).to_csv(meta / "instruments.csv", index=False)
    log(f"[*] {len(instruments)} instruments x {len(tfs)} timeframes = {len(instruments) * len(tfs)} jobs, "
        f"{args.workers} workers @ {args.rps} req/s -> {args.out_dir}")

    if args.probe:
        tfs = [TF(t.label, t.interval, 0, t.window_days, 2, t.intraday) for t in tfs]
        for name, exch, iid, kind in instruments:
            for tf in tfs:
                try:
                    rows = client.candles(iid, exch, tf.interval, end - timedelta(days=10), end - timedelta(days=1))
                    log(f"[probe] {name:<10} {tf.label:<5} {len(rows)} candles"
                        + (f"  first={rows[0][:6]}" if rows else ""))
                except Exception as e:  # noqa: BLE001
                    log(f"[probe] {name:<10} {tf.label:<5} ERROR {str(e)[:150]}")
        return

    # ---- download ----
    jobs = [(n, k, iid, ex, tf) for (n, ex, iid, k) in instruments for tf in tfs]
    records, done = [], 0
    manifest = meta / "manifest.csv"
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(run_job, client, n, k, iid, ex, tf, end, args): (n, tf.label)
                for (n, k, iid, ex, tf) in jobs}
        try:
            for fut in as_completed(futs):
                n, tfl = futs[fut]
                done += 1
                try:
                    records.append(fut.result())
                except IIFLAuthError as e:
                    log(f"[FATAL] {e}")
                    records.append({"name": n, "timeframe": tfl, "status": "AUTH_ERROR", "error": str(e)})
                except Exception as e:  # noqa: BLE001
                    log(f"[!] {n} {tfl} failed: {e}")
                    records.append({"name": n, "timeframe": tfl, "status": "ERROR", "error": str(e)[:300]})
                if done % 25 == 0:
                    log(f"[*] progress {done}/{len(jobs)}")
                    pd.DataFrame(records).to_csv(manifest, index=False)
        except KeyboardInterrupt:
            client.stop.set()
            log("[!] Interrupted - finished files are kept; re-run to resume.")
            pool.shutdown(wait=False, cancel_futures=True)
    pd.DataFrame(records).to_csv(manifest, index=False)

    if args.quotes and not client.stop.is_set():
        snapshot_quotes(client, [(n, e, i) for (n, e, i, _) in instruments], meta)

    m = pd.DataFrame(records)
    print("\nSummary (rows per timeframe):")
    if "rows" in m:
        print(m.pivot_table(index="timeframe", values="rows", aggfunc=["sum", "count"]).to_string())
        lim = m[m["retention_limited"].fillna(False).astype(bool)] if "retention_limited" in m else m.iloc[:0]
        if len(lim):
            print("\nRetention-limited (broker served less history than requested):")
            print(lim.groupby("timeframe")["first"].min().to_string())
    bad = m[m["status"] != "OK"] if "status" in m else m
    if len(bad):
        print(f"\n{len(bad)} jobs not OK - see {manifest}")
    print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except (IIFLAuthError, ValueError) as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
