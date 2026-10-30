"""Quality checks and cleaning of the frozen daily prices.

Every check reports PASS, WARN or FAIL. WARN marks an issue that the cleaning
step handles or that is worth a single review; FAIL marks data that cannot be
used as it is.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.download import ACTION_COLUMNS, PRICE_COLUMNS


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _row(check, ticker, status, detail):
    """One line of the quality report."""
    return {"check": check, "ticker": ticker, "status": status, "detail": detail}


def _first_dates(index, n=5):
    """Short text list of the first n dates, for report details."""
    shown = ", ".join(pd.Index(index[:n]).strftime("%Y-%m-%d"))
    return shown + (" ..." if len(index) > n else "")


def reference_calendar(prices):
    """Split all dates into real trading days and stray dates.

    A date counts as a trading day if at least half of the tickers have a
    price on it. Dates where only one or two tickers have a row are almost
    always Yahoo artifacts (for example a row on a market holiday), so they
    are returned separately and removed during cleaning.
    """
    has_price = prices.dropna(subset=["adj_close"])
    counts = has_price.groupby("date")["ticker"].nunique()
    n_tickers = prices["ticker"].nunique()
    calendar = counts.index[counts >= n_tickers / 2].sort_values()
    stray = counts.index[counts < n_tickers / 2].sort_values()
    return pd.DatetimeIndex(calendar), pd.DatetimeIndex(stray)


def _runs(mask):
    """Lengths of consecutive True runs in a boolean Series."""
    values = mask.to_numpy()
    if not values.any():
        return np.array([], dtype=int)
    edges = np.diff(np.concatenate([[0], values.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return ends - starts


def run_quality_checks(prices, tickers=None):
    """Quality report of the raw prices.

    Table-level checks: required columns, tickers present, duplicate rows,
    stray dates and the trading calendar. Per-ticker checks: date order,
    coverage of the download window, missing trading days, empty values,
    positive prices, high/low range, large daily moves, the Close/Adj Close
    ratio, zero-volume days and stale prices.

    Returns a DataFrame with columns check, ticker, status, detail; ticker
    "ALL" marks table-level checks.
    """
    tickers = list(tickers or config.ALL_TICKERS)
    rows = []

    # Table-level checks.
    required = ["date", "ticker"] + PRICE_COLUMNS + ["volume"] + ACTION_COLUMNS
    absent = [c for c in required if c not in prices.columns]
    rows.append(_row("columns", "ALL", "FAIL" if absent else "PASS",
                     f"missing columns {absent}" if absent else "all columns present"))
    if absent:
        return pd.DataFrame(rows)

    missing_tickers = sorted(set(tickers) - set(prices["ticker"]))
    rows.append(_row("tickers present", "ALL", "FAIL" if missing_tickers else "PASS",
                     f"missing {missing_tickers}" if missing_tickers
                     else f"all {len(tickers)} tickers present"))

    n_dupes = int(prices.duplicated(["date", "ticker"]).sum())
    rows.append(_row("duplicate rows", "ALL", "WARN" if n_dupes else "PASS",
                     f"{n_dupes} duplicate (date, ticker) rows, the last copy is kept"
                     if n_dupes else "no duplicates"))

    calendar, stray = reference_calendar(prices)
    rows.append(_row("stray dates", "ALL", "WARN" if len(stray) else "PASS",
                     f"{len(stray)} dates with data for few tickers, dropped in cleaning: "
                     f"{_first_dates(stray)}" if len(stray) else "none"))
    rows.append(_row("calendar", "ALL", "PASS",
                     f"{len(calendar):,} trading days from "
                     f"{calendar[0]:%Y-%m-%d} to {calendar[-1]:%Y-%m-%d}"))

    expected_first = pd.Timestamp(config.EXPECTED_FIRST_DAY)
    last_day = calendar[-1]

    # Per-ticker checks.
    for ticker in tickers:
        df = prices[prices["ticker"] == ticker]
        if df.empty:
            continue

        # Rows in date order, as downloaded.
        in_order = df["date"].is_monotonic_increasing
        rows.append(_row("date order", ticker, "PASS" if in_order else "WARN",
                         "increasing" if in_order else "not sorted, fixed in cleaning"))

        df = df.drop_duplicates(["date"], keep="last").sort_values("date").set_index("date")
        df = df[~df.index.isin(stray)]

        # Coverage of the full download window.
        first, last = df.index.min(), df.index.max()
        rows.append(_row("coverage start", ticker,
                         "PASS" if first <= expected_first else "FAIL",
                         f"first date {first:%Y-%m-%d}"))
        rows.append(_row("coverage end", ticker,
                         "PASS" if last == last_day else "FAIL",
                         f"last date {last:%Y-%m-%d} (calendar ends {last_day:%Y-%m-%d})"))

        # Missing trading days inside the ticker's range.
        span = calendar[(calendar >= first) & (calendar <= last)]
        absent_mask = pd.Series(~span.isin(df.index[df["adj_close"].notna()]), index=span)
        runs = _runs(absent_mask)
        if len(runs) == 0:
            status, detail = "PASS", "no missing trading days"
        elif runs.max() > config.FFILL_LIMIT:
            status = "FAIL"
            detail = (f"{int(absent_mask.sum())} missing days, longest gap {int(runs.max())} "
                      f"days: {_first_dates(span[absent_mask.to_numpy()])}")
        else:
            status = "WARN"
            detail = (f"{int(absent_mask.sum())} isolated missing days, filled from the "
                      f"previous day: {_first_dates(span[absent_mask.to_numpy()])}")
        rows.append(_row("missing days", ticker, status, detail))

        # Empty values in rows that do exist.
        nulls = df[PRICE_COLUMNS + ["volume"]].isna().sum()
        nulls = nulls[nulls > 0]
        rows.append(_row("empty values", ticker, "WARN" if len(nulls) else "PASS",
                         ", ".join(f"{c}: {n}" for c, n in nulls.items()) if len(nulls)
                         else "none"))

        # Prices must be positive.
        bad = df[(df["adj_close"] <= 0) | (df["close"] <= 0)]
        rows.append(_row("positive prices", ticker, "FAIL" if len(bad) else "PASS",
                         f"{len(bad)} rows with price <= 0: {_first_dates(bad.index)}"
                         if len(bad) else "all prices positive"))

        # High and low must bracket open and close.
        eps = 1e-6
        ohlc_bad = df[(df["low"] > df[["open", "close"]].min(axis=1) + eps) |
                      (df["high"] < df[["open", "close"]].max(axis=1) - eps)]
        rows.append(_row("high/low range", ticker, "WARN" if len(ohlc_bad) else "PASS",
                         f"{len(ohlc_bad)} rows where high/low do not contain open/close "
                         f"(not used by features): {_first_dates(ohlc_bad.index)}"
                         if len(ohlc_bad) else "consistent"))

        # Large one-day moves in adjusted prices.
        ret = df["adj_close"].pct_change(fill_method=None)
        big = ret[ret.abs() > config.MAX_DAILY_MOVE]
        rows.append(_row("large moves", ticker, "WARN" if len(big) else "PASS",
                         "check against news: " + ", ".join(
                             f"{d:%Y-%m-%d} {r:+.1%}" for d, r in big.items())
                         if len(big) else f"no daily move above {config.MAX_DAILY_MOVE:.0%}"))

        # The Close / Adj Close ratio changes only around dividends, capital
        # gains distributions and splits; a change on any other day points to
        # a faulty adjustment.
        ratio = df["adj_close"] / df["close"]
        jump = (ratio / ratio.shift(1) - 1).abs() > config.ADJ_RATIO_TOL
        action_day = (df[ACTION_COLUMNS].fillna(0) != 0).any(axis=1)
        near_action = (action_day | action_day.shift(1, fill_value=False)
                       | action_day.shift(-1, fill_value=False))
        unexplained = jump & ~near_action
        n_actions = int(action_day.sum())
        rows.append(_row("adjustment ratio", ticker,
                         "WARN" if unexplained.any() else "PASS",
                         f"{int(unexplained.sum())} ratio changes away from any of the "
                         f"{n_actions} action dates: {_first_dates(df.index[unexplained])}"
                         if unexplained.any()
                         else f"ratio changes only around the {n_actions} action dates"))

        # Days without trading volume, which the volume bias feature treats
        # separately.
        n_zero = int((df["volume"] == 0).sum())
        rows.append(_row("zero volume", ticker, "WARN" if n_zero else "PASS",
                         f"{n_zero} days with volume 0: {_first_dates(df.index[df['volume'] == 0])}"
                         if n_zero else "none"))

        # Stale prices: the same close for many days in a row.
        same = df["close"].diff().eq(0)
        stale_runs = _runs(same) + 1
        longest = int(stale_runs.max()) if len(stale_runs) else 1
        rows.append(_row("stale prices", ticker,
                         "WARN" if longest >= config.STALE_RUN else "PASS",
                         f"longest run of identical closes: {longest} days"))

    return pd.DataFrame(rows)


def summarize(report):
    """Print status counts and every row that is not PASS."""
    counts = report["status"].value_counts().reindex(["PASS", "WARN", "FAIL"], fill_value=0)
    print(f"PASS {counts['PASS']}   WARN {counts['WARN']}   FAIL {counts['FAIL']}")
    issues = report[report["status"] != "PASS"].copy()
    if issues.empty:
        print("All checks passed.")
    else:
        issues["order"] = issues["status"].map(STATUS_ORDER)
        issues = issues.sort_values(["order", "check", "ticker"], ascending=[False, True, True])
        with pd.option_context("display.max_colwidth", 140, "display.width", 200):
            print(issues.drop(columns="order").to_string(index=False))
    return "FAIL" if counts["FAIL"] else ("WARN" if counts["WARN"] else "PASS")


def clean_prices(prices, tickers=None):
    """Clean daily price table.

    Stray dates are removed, the last copy of a duplicate row is kept, and
    every ticker is placed on the common trading calendar. A missing price is
    filled from the previous trading day, for at most FFILL_LIMIT days in a
    row; filled rows are marked filled=True and keep an empty volume, so no
    volume is invented. Zero-volume days are marked zero_volume=True. Any
    adjusted price still missing after filling raises an error.

    Returns (clean, fill_log), where fill_log lists every filled price.
    """
    tickers = list(tickers or config.ALL_TICKERS)
    calendar, stray = reference_calendar(prices)

    df = prices[~prices["date"].isin(stray)]
    df = df.drop_duplicates(["date", "ticker"], keep="last")

    frames, logs = [], []
    for ticker in tickers:
        one = df[df["ticker"] == ticker].set_index("date").sort_index()
        span = calendar[calendar >= one.index.min()]
        one = one.reindex(span)
        one.index.name = "date"

        was_missing = one["adj_close"].isna()
        one[PRICE_COLUMNS] = one[PRICE_COLUMNS].ffill(limit=config.FFILL_LIMIT)
        one[ACTION_COLUMNS] = one[ACTION_COLUMNS].fillna(0.0)
        one["ticker"] = ticker
        one["filled"] = was_missing & one["adj_close"].notna()
        one.loc[one["filled"], "volume"] = np.nan
        one["zero_volume"] = one["volume"].eq(0)

        still_missing = one.index[one["adj_close"].isna()]
        if len(still_missing):
            raise ValueError(
                f"{ticker}: {len(still_missing)} prices still missing after filling, "
                f"for example {_first_dates(still_missing)}."
            )

        if one["filled"].any():
            logs.append(pd.DataFrame({"ticker": ticker,
                                      "date": one.index[one["filled"]],
                                      "action": "price filled from previous trading day"}))
        frames.append(one.reset_index())

    clean = pd.concat(frames, ignore_index=True)
    clean["volume"] = clean["volume"].astype("Float64")
    order = ["date", "ticker"] + PRICE_COLUMNS + ["volume"] + ACTION_COLUMNS + ["filled", "zero_volume"]
    clean = clean[order].sort_values(["ticker", "date"]).reset_index(drop=True)
    fill_log = (pd.concat(logs, ignore_index=True) if logs
                else pd.DataFrame(columns=["ticker", "date", "action"]))
    return clean, fill_log


def month_end_dates(clean):
    """Last trading day of every month in the clean table."""
    dates = pd.Series(clean["date"].unique()).sort_values()
    return pd.DatetimeIndex(dates.groupby(dates.dt.to_period("M")).max().to_numpy())


def check_month_ends(clean):
    """Check that every ticker has an adjusted price at every month-end.

    The monthly table samples prices at month-ends, so a gap would remove a
    whole row from it.
    """
    ends = month_end_dates(clean)
    wide = clean.pivot(index="date", columns="ticker", values="adj_close").reindex(ends)
    rows = []
    for ticker in wide.columns:
        gaps = wide.index[wide[ticker].isna()]
        rows.append(_row("month-end prices", ticker, "FAIL" if len(gaps) else "PASS",
                         f"missing at {_first_dates(gaps)}" if len(gaps)
                         else f"all {len(ends)} month-ends from {ends[0]:%Y-%m} to {ends[-1]:%Y-%m}"))
    return pd.DataFrame(rows)


def coverage_table(clean):
    """One row per ticker: dates, row counts, filled and zero-volume days."""
    universe = {t: ("A and B" if t in config.ETF_A and t in config.ETF_B
                    else "A" if t in config.ETF_A else "B")
                for t in clean["ticker"].unique()}
    table = clean.groupby("ticker").agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        trading_days=("date", "size"),
        filled_days=("filled", "sum"),
        zero_volume_days=("zero_volume", "sum"),
        dividend_days=("dividends", lambda s: int((s > 0).sum())),
    )
    table.insert(0, "universe", table.index.map(universe))
    growth = clean.groupby("ticker")["adj_close"].agg(["first", "last"])
    table["total_return"] = (growth["last"] / growth["first"] - 1).round(3)
    return table.reset_index()


def monthly_returns(daily, date_col="date", price_col="adj_close"):
    """Month-end to month-end returns from a daily price series."""
    s = daily.set_index(date_col)[price_col].sort_index()
    return s.groupby(s.index.to_period("M")).last().pct_change().dropna()


def cross_check(clean, other_csv, ticker, date_col="Date", price_col="Close"):
    """Compare one ticker's monthly returns with a second data source.

    `other_csv` is a daily price file for the same ticker from another
    provider with adjusted prices, such as Stooq. Returns one report row:
    PASS when every monthly return agrees within CROSS_CHECK_TOL.
    """
    other = pd.read_csv(other_csv, parse_dates=[date_col])
    ours = monthly_returns(clean[clean["ticker"] == ticker])
    theirs = monthly_returns(other, date_col=date_col, price_col=price_col)
    both = pd.concat({"yahoo": ours, "other": theirs}, axis=1).dropna()
    diff = (both["yahoo"] - both["other"]).abs()
    worst = diff.idxmax()
    n_bad = int((diff > config.CROSS_CHECK_TOL).sum())
    status = "PASS" if n_bad == 0 else "WARN"
    detail = (f"{len(both)} months compared, {n_bad} differ by more than "
              f"{config.CROSS_CHECK_TOL:.1%}; largest gap {diff.max():.3%} in {worst}")
    return _row("cross-check", ticker, status, detail)


def save_clean(clean, fill_log, processed_dir, reports_dir):
    """Save the clean daily prices as Parquet and CSV, and the fill log."""
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    path = processed_dir / config.FILES["daily"]
    clean.to_parquet(path, index=False)
    clean.to_csv(path.with_suffix(".csv"), index=False)
    fill_log.to_csv(reports_dir / config.FILES["filled"], index=False)
    print(f"Saved {len(clean):,} rows to {path}")
    return path


def load_clean(processed_dir):
    """Load the clean daily prices."""
    return pd.read_parquet(Path(processed_dir) / config.FILES["daily"])
