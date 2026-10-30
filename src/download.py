"""Download daily ETF prices from Yahoo Finance and freeze them.

Yahoo recalculates adjusted prices after every dividend, so downloads made on
different days never match exactly. The prices are therefore downloaded once,
saved together with an MD5 checksum, and every later step reads that frozen
copy. A changed file is detected by its checksum.
"""

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# Yahoo column names and the names used in this project.
COLUMN_NAMES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
    "Capital Gains": "capital_gains",
}

# Column order of the long price table.
COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close",
           "volume", "dividends", "splits", "capital_gains"]

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
ACTION_COLUMNS = ["dividends", "splits", "capital_gains"]


def _to_long(raw, tickers):
    """Reshape the wide yfinance result into one row per date and ticker.

    yfinance returns two column levels, field and ticker, for example
    ("Adj Close", "SPY"). Their order differs between yfinance versions, so
    the ticker level is identified by its contents rather than its position.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected two column levels (field, ticker) from yfinance.")

    level_values = [set(raw.columns.get_level_values(i)) for i in range(2)]
    ticker_level = 0 if level_values[0] & set(tickers) else 1

    frames = []
    for ticker in tickers:
        if ticker not in level_values[ticker_level]:
            continue
        one = raw.xs(ticker, axis=1, level=ticker_level).copy()
        one = one.rename(columns=COLUMN_NAMES)
        one["ticker"] = ticker
        one["date"] = one.index
        frames.append(one)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    long = pd.concat(frames, ignore_index=True)

    # Corporate action columns are only present when an action occurred.
    for col in ACTION_COLUMNS:
        if col not in long.columns:
            long[col] = 0.0
        long[col] = long[col].fillna(0.0)

    # Plain calendar dates, without time zones.
    dates = pd.to_datetime(long["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    long["date"] = dates.dt.normalize()

    # yfinance pads each ticker with empty rows on days it has no data.
    long = long.dropna(subset=PRICE_COLUMNS + ["volume"], how="all")

    return long[COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def download_prices(tickers=None, start=None, end=None, retries=3, pause=10):
    """Daily prices, volume and corporate actions for all tickers.

    auto_adjust=False keeps both Close (the traded price) and Adj Close
    (adjusted for splits and dividends). Returns are computed from Adj Close;
    Close is used only in the quality checks.

    Yahoo occasionally drops tickers when it rate-limits a request. Tickers
    that come back empty are requested again, up to `retries` times, with a
    longer pause before each attempt.
    """
    import yfinance as yf

    tickers = list(tickers or config.ALL_TICKERS)
    start = start or config.START
    end = end or config.END

    collected = []
    missing = tickers
    for attempt in range(1, retries + 1):
        raw = yf.download(
            missing,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            group_by="column",
            multi_level_index=True,
            progress=False,
            threads=True,
        )
        long = _to_long(raw, missing)

        # A ticker counts as received only if it has adjusted prices.
        received = set(long.loc[long["adj_close"].notna(), "ticker"])
        collected.append(long[long["ticker"].isin(received)])
        missing = [t for t in missing if t not in received]

        print(f"Attempt {attempt}: received {len(received)} tickers, "
              f"{len(missing)} still missing {missing if missing else ''}")
        if not missing:
            break
        time.sleep(pause * attempt)

    if missing:
        raise RuntimeError(
            f"No data for {missing} after {retries} attempts, most likely "
            "because of Yahoo's rate limit."
        )

    prices = pd.concat(collected, ignore_index=True)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def _md5(path):
    """MD5 checksum of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _library_versions():
    """Versions of the libraries that produced the download."""
    versions = {"python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__}
    try:
        import yfinance as yf
        versions["yfinance"] = yf.__version__
    except ImportError:
        versions["yfinance"] = "not installed"
    return versions


def freeze(prices, raw_dir, start=None, end=None, overwrite=False):
    """Save the download as the raw price file and its information file.

    The information file records the download time, the source settings, the
    row counts per ticker, the library versions and the MD5 checksum of the
    price file. An existing frozen download is kept unless overwrite=True.
    Returns the paths of both files.
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / config.FILES["raw"]
    info_path = raw_dir / config.FILES["raw_info"]

    if csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"{csv_path.name} already exists; pass overwrite=True to replace it."
        )

    out = prices.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False)

    per_ticker = {
        ticker: {"rows": int(len(group)),
                 "first_date": group["date"].min().strftime("%Y-%m-%d"),
                 "last_date": group["date"].max().strftime("%Y-%m-%d")}
        for ticker, group in prices.groupby("ticker")
    }
    info = {
        "file": csv_path.name,
        "md5": _md5(csv_path),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Yahoo Finance via yfinance, auto_adjust=False, actions=True",
        "start": start or config.START,
        "end_exclusive": end or config.END,
        "tickers": sorted(prices["ticker"].unique().tolist()),
        "n_rows": int(len(prices)),
        "per_ticker": per_ticker,
        "versions": _library_versions(),
    }
    info_path.write_text(json.dumps(info, indent=2))

    print(f"Frozen {info['n_rows']:,} rows for {len(info['tickers'])} tickers")
    print(f"  data: {csv_path}")
    print(f"  md5:  {info['md5']}")
    return csv_path, info_path


def is_frozen(raw_dir):
    """Whether a frozen download exists in `raw_dir`."""
    return (Path(raw_dir) / config.FILES["raw"]).exists()


def load_frozen(raw_dir, verify=True):
    """Load the frozen download as (prices, info).

    With verify=True the checksum of the price file is recomputed and compared
    with the recorded one; a mismatch means the file changed after freezing
    and raises an error.
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / config.FILES["raw"]
    info = json.loads((raw_dir / config.FILES["raw_info"]).read_text())

    if verify:
        actual = _md5(csv_path)
        if actual != info["md5"]:
            raise ValueError(
                f"Checksum mismatch for {csv_path.name}: expected {info['md5']}, "
                f"got {actual}. The file was changed after freezing."
            )

    prices = pd.read_csv(csv_path, parse_dates=["date"], dtype={"ticker": str})
    print(f"Loaded {csv_path.name}: {len(prices):,} rows, "
          f"{prices['ticker'].nunique()} tickers, checksum "
          f"{'verified' if verify else 'not checked'}")
    return prices, info
