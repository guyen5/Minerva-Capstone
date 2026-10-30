"""Technical indicators and the monthly and daily data tables.

Seven indicators are computed for every ETF on every trading day from
adjusted close prices, so dividends never appear as price moves. Two tables
are built from them:

- the monthly table: one row per universe, ETF and month-end t, holding the
  features known at t and the return from t to the next month-end;
- the daily table: one row per universe, ETF and trading day d, holding the
  features known at d and the return over the next FORWARD_DAYS trading days.

A row's return is known only on the date it ends, so a row can enter a
training set only after that date.
"""

import numpy as np
import pandas as pd

from src import config


def _sma(series, window):
    """Simple moving average that needs a full window before giving a value."""
    return series.rolling(window, min_periods=window).mean()


def _ema(series, span):
    """Exponential moving average as used by MACD (alpha = 2 / (span + 1)).

    adjust=False gives the usual recursive definition:
    ema_t = alpha * x_t + (1 - alpha) * ema_(t-1).
    min_periods hides the first `span` values while the average settles.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_rsi(price, window=None):
    """Relative Strength Index with Wilder smoothing (alpha = 1 / window).

    RSI = 100 - 100 / (1 + average gain / average loss), between 0 and 100.
    Above 70 is usually read as overbought, below 30 as oversold.
    Edge cases: no losses in the window gives 100, no gains gives 0, and a
    completely flat window gives 50.
    """
    window = window or config.RSI_WINDOW
    change = price.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi.where(avg_gain.notna() & avg_loss.notna())


def _one_ticker_features(df):
    """Seven indicators for one ticker. df is sorted by date, indexed by date."""
    p = df["adj_close"]
    v = df["volume"].astype("float64")
    out = pd.DataFrame(index=df.index)

    # 1. Momentum: log return over the last 21 trading days (about one month).
    out["momentum"] = np.log(p / p.shift(config.RET_WINDOW))

    # 2. Trend: how far the 10-day average sits above the 50-day average.
    #    The paper uses a raw 10-day SMA, which is in dollars and cannot be
    #    compared across ETFs; this ratio keeps the idea and removes the scale.
    out["trend"] = _sma(p, config.SMA_SHORT) / _sma(p, config.SMA_LONG) - 1

    # 3. Price bias: today's price relative to its 10-day average.
    out["price_bias"] = p / _sma(p, config.SMA_SHORT) - 1

    # 4. RSI(14) with Wilder smoothing.
    out["rsi"] = wilder_rsi(p)

    # 5. MACD difference (histogram) divided by price to remove the scale.
    macd = _ema(p, config.MACD_FAST) - _ema(p, config.MACD_SLOW)
    signal = macd.ewm(span=config.MACD_SIGNAL, adjust=False,
                      min_periods=config.MACD_SIGNAL).mean()
    out["macd"] = (macd - signal) / p

    # 6. Bollinger width: band width (mean +/- 2 standard deviations) divided
    #    by the middle band, i.e. 4 * std20 / SMA20. A volatility measure.
    #    Population standard deviation (ddof=0), as in Bollinger's definition.
    std = p.rolling(config.BOLL_WINDOW, min_periods=config.BOLL_WINDOW).std(ddof=0)
    out["bollinger"] = 4 * std / _sma(p, config.BOLL_WINDOW)

    # 7. Volume bias: today's volume relative to its 10-day average.
    #    Undefined (missing) when today's volume is 0 or unknown, because a
    #    day without trades carries no information about trading interest.
    vol_avg = _sma(v, config.SMA_SHORT)
    vb = v / vol_avg - 1
    out["volume_bias"] = vb.where((v > 0) & (vol_avg > 0))

    return out


def daily_features(clean):
    """Compute the seven features for every ticker on every trading day.

    Input: the clean daily prices, one row per ticker and day.
    Output: date, ticker and the seven feature columns.
    """
    frames = []
    for ticker, df in clean.groupby("ticker", sort=True):
        df = df.sort_values("date").set_index("date")
        feats = _one_ticker_features(df)
        feats["ticker"] = ticker
        frames.append(feats.reset_index())
    daily = pd.concat(frames, ignore_index=True)
    return daily[["date", "ticker"] + config.FEATURES]


def month_ends(clean):
    """Last trading day of each month, from the shared trading calendar."""
    dates = pd.Series(pd.to_datetime(clean["date"].unique())).sort_values()
    return pd.DatetimeIndex(dates.groupby(dates.dt.to_period("M")).max().to_numpy())


def _universe_rows(daily, prices_me, ends, universe, tickers):
    """Monthly rows for one universe: features at t, return from t to the next month-end."""
    feats = daily[daily["ticker"].isin(tickers) & daily["date"].isin(ends)]

    # Month-end prices of this universe, one column per ticker.
    px = prices_me[tickers]

    # Simple return from this month-end to the next one.
    next_ret = px.shift(-1) / px - 1

    long = next_ret.stack().rename("next_return")
    long.index.names = ["date", "ticker"]
    long = long.reset_index()

    rows = feats.merge(long, on=["date", "ticker"], how="left")
    rows.insert(1, "universe", universe)
    return rows


def build_panel(clean, daily=None):
    """Monthly table: one row per universe, ETF and month-end.

    SPY and XLK appear once in each universe, with identical values.
    Columns: date, universe, ticker, month (the holding month whose return
    the row measures), period, the seven features, volume_filled and
    next_return. The newest month-end has features but no return yet.
    """
    daily = daily if daily is not None else daily_features(clean)
    ends = month_ends(clean)

    prices = clean.pivot(index="date", columns="ticker", values="adj_close")
    prices_me = prices.reindex(ends)

    parts = [_universe_rows(daily, prices_me, ends, "ETF_A", config.ETF_A),
             _universe_rows(daily, prices_me, ends, "ETF_B", config.ETF_B)]
    panel = pd.concat(parts, ignore_index=True)
    panel = panel[panel["date"] >= pd.Timestamp(config.PANEL_START)]

    # Holding month: the month whose return the row measures.
    panel.insert(3, "month",
                 (panel["date"].dt.to_period("M") + 1).astype(str))

    # A missing volume bias is flagged and set to 0, meaning volume at its
    # usual level. The fill uses no future information and keeps the row
    # usable by models that cannot handle missing values.
    panel["volume_filled"] = panel["volume_bias"].isna()
    panel["volume_bias"] = panel["volume_bias"].fillna(0.0)

    # Part of the sample each row belongs to, by holding month.
    first_test = pd.Period(config.TEST_FIRST_MONTH, "M")
    last_test = pd.Period(config.TEST_LAST_MONTH, "M")
    hold = pd.PeriodIndex(panel["month"], freq="M")
    panel["period"] = np.select(
        [panel["next_return"].isna(), hold < first_test, hold <= last_test],
        ["latest", "history", "test"],
        default="after test period",
    )

    order = (["date", "universe", "ticker", "month", "period"]
             + config.FEATURES + ["volume_filled", "next_return"])
    panel = panel[order].sort_values(["universe", "date", "ticker"])
    return panel.reset_index(drop=True)


def _check(name, ok, detail):
    """One line of a validation report."""
    return {"check": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def validate_panel(panel, clean, n_truncation_tests=8, seed=0):
    """Validation report of the monthly table.

    Covers row counts, unique keys, complete and finite features, the RSI and
    Bollinger ranges, the unlabelled newest month and an independent
    recomputation of the returns. The look-ahead test recomputes the features
    at a few random month-ends t from prices cut off at t; any difference
    from the table means a feature used data from after t.
    """
    rows = []
    n_months = panel["date"].nunique()
    sizes = {"ETF_A": len(config.ETF_A), "ETF_B": len(config.ETF_B)}

    for uni, k in sizes.items():
        n = int((panel["universe"] == uni).sum())
        rows.append(_check(f"row count {uni}", n == n_months * k,
                           f"{n} rows = {n_months} month-ends x {k} ETFs"))

    dupes = int(panel.duplicated(["date", "universe", "ticker"]).sum())
    rows.append(_check("unique rows", dupes == 0, f"{dupes} duplicate keys"))

    n_na = int(panel[config.FEATURES].isna().sum().sum())
    rows.append(_check("features complete", n_na == 0,
                       f"{n_na} missing feature values"))

    rsi_ok = panel["rsi"].between(0, 100).all()
    rows.append(_check("rsi range", rsi_ok, "all values between 0 and 100"))
    bw_ok = (panel["bollinger"] > 0).all()
    rows.append(_check("bollinger range", bw_ok, "all values above 0"))
    finite = np.isfinite(panel[config.FEATURES].to_numpy()).all()
    rows.append(_check("finite values", finite, "no infinite feature values"))

    labeled = panel[panel["next_return"].notna()]

    newest = panel["date"].max()
    unl = panel[panel["date"] == newest]["next_return"].isna().all()
    rows.append(_check("newest month unlabeled", unl,
                       f"{newest:%Y-%m-%d} has features but no label"))

    # Independent recomputation of the returns from month-end prices.
    px = clean.pivot(index="date", columns="ticker", values="adj_close")
    sample = labeled.sample(min(200, len(labeled)), random_state=seed)
    ends = month_ends(clean)
    errs = []
    for _, r in sample.iterrows():
        nxt = ends[ends.get_loc(r["date"]) + 1]
        errs.append(abs(px.at[nxt, r["ticker"]] / px.at[r["date"], r["ticker"]] - 1
                        - r["next_return"]))
    rows.append(_check("returns recomputed", max(errs) < 1e-12,
                       f"{len(errs)} random rows, largest difference {max(errs):.1e}"))

    # Look-ahead test. The volume bias is compared before its 0 fill.
    rng = np.random.default_rng(seed)
    dates = rng.choice(np.sort(panel["date"].unique())[:-1], n_truncation_tests,
                       replace=False)
    worst = 0.0
    for t in dates:
        t = pd.Timestamp(t)
        cut = daily_features(clean[clean["date"] <= t])
        cut = cut[cut["date"] == t].set_index("ticker")[config.FEATURES]
        full = (panel[(panel["date"] == t)]
                .drop_duplicates("ticker").set_index("ticker"))
        full_vals = full[config.FEATURES].copy()
        full_vals.loc[full["volume_filled"], "volume_bias"] = np.nan
        a = cut.loc[full_vals.index].to_numpy(dtype=float)
        b = full_vals.to_numpy(dtype=float)
        if (np.isnan(a) != np.isnan(b)).any():
            worst = np.inf
        else:
            both = ~np.isnan(a)
            worst = max(worst, float(np.abs(a[both] - b[both]).max()))
    rows.append(_check("no look-ahead", worst < 1e-10,
                       f"features at {len(dates)} random month-ends recomputed from "
                       f"data cut at that date; largest difference {worst:.1e}"))

    return pd.DataFrame(rows)


def _universe_daily_rows(daily, prices, universe, tickers, horizon):
    """Daily rows for one universe: features at d, target from d to d + horizon."""
    px = prices[tickers]
    calendar = px.index

    # Forward return over the next `horizon` trading days, and the day on
    # which it becomes known.
    fwd = px.shift(-horizon) / px - 1
    label_date = pd.Series(calendar, index=calendar).shift(-horizon)

    # Return relative to the universe average on the same days. With
    # weights summing to one, only these differences affect a portfolio.
    rel = fwd.sub(fwd.mean(axis=1), axis=0)

    long = pd.DataFrame({"fwd_return": fwd.stack(future_stack=True),
                         "fwd_rel_return": rel.stack(future_stack=True)})
    long.index.names = ["date", "ticker"]
    long = long.reset_index()
    long["label_date"] = long["date"].map(label_date)

    feats = daily[daily["ticker"].isin(tickers)]
    rows = feats.merge(long, on=["date", "ticker"], how="left")
    rows.insert(1, "universe", universe)
    return rows


def build_daily_table(clean, daily=None, horizon=None):
    """Daily table: one row per universe, ETF and trading day.

    A nine-month training window holds only nine month-end rows per ETF, so
    models are trained on daily rows instead: each trading day's features
    paired with the return over the next `horizon` trading days. Neighbouring
    rows share most of their return period, which the training windows and
    the bootstrap account for.

    Columns: date, universe, ticker, month_end, the seven features,
    volume_filled, fwd_return, fwd_rel_return (return minus the universe
    average on the same days) and label_date (the day the return is known).
    Rows start at PANEL_START; the last `horizon` days have no return yet.
    """
    horizon = horizon or config.FORWARD_DAYS
    daily = daily if daily is not None else daily_features(clean)
    prices = clean.pivot(index="date", columns="ticker", values="adj_close").sort_index()

    parts = [_universe_daily_rows(daily, prices, "ETF_A", config.ETF_A, horizon),
             _universe_daily_rows(daily, prices, "ETF_B", config.ETF_B, horizon)]
    table = pd.concat(parts, ignore_index=True)
    table = table[table["date"] >= pd.Timestamp(config.PANEL_START)]

    ends = month_ends(clean)
    table.insert(3, "month_end", table["date"].isin(ends))

    # Missing volume bias is flagged and set to 0, as in the monthly table.
    table["volume_filled"] = table["volume_bias"].isna()
    table["volume_bias"] = table["volume_bias"].fillna(0.0)

    order = (["date", "universe", "ticker", "month_end"] + config.FEATURES
             + ["volume_filled", "fwd_return", "fwd_rel_return", "label_date"])
    table = table[order].sort_values(["universe", "date", "ticker"])
    return table.reset_index(drop=True)


def validate_daily(table, clean, panel=None, n_samples=200, seed=0, horizon=None):
    """Validation report of the daily table.

    The features come from the same function as the monthly table, which
    has its own look-ahead test, so these checks centre on the return: it is
    recomputed independently, its label date always follows the row date,
    relative returns sum to zero, and at month-ends the features equal those
    of the monthly table.
    """
    horizon = horizon or config.FORWARD_DAYS
    rows = []
    sizes = {"ETF_A": len(config.ETF_A), "ETF_B": len(config.ETF_B)}
    n_days = table["date"].nunique()
    for uni, k in sizes.items():
        n = int((table["universe"] == uni).sum())
        rows.append(_check(f"row count {uni}", n == n_days * k,
                           f"{n:,} rows = {n_days:,} trading days x {k} ETFs"))

    dupes = int(table.duplicated(["date", "universe", "ticker"]).sum())
    rows.append(_check("unique rows", dupes == 0, f"{dupes} duplicate keys"))

    n_na = int(table[config.FEATURES].isna().sum().sum())
    rows.append(_check("features complete", n_na == 0, f"{n_na} missing feature values"))

    labeled = table[table["fwd_return"].notna()]
    unlabeled_days = table.loc[table["fwd_return"].isna(), "date"].nunique()
    rows.append(_check("newest days unlabeled", unlabeled_days == horizon,
                       f"the last {unlabeled_days} trading days have no target yet"))

    later = (labeled["label_date"] > labeled["date"]).all()
    rows.append(_check("label after row date", bool(later),
                       "every target becomes known after its row date"))

    # Relative returns sum to zero across each universe on each day.
    sums = labeled.groupby(["universe", "date"])["fwd_rel_return"].sum().abs().max()
    rows.append(_check("relative returns sum to 0", sums < 1e-12,
                       f"largest absolute daily sum {sums:.1e}"))

    # Independent recomputation of the target from the price table.
    px = clean.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    cal = px.index
    sample = labeled.sample(min(n_samples, len(labeled)), random_state=seed)
    errs = []
    for _, r in sample.iterrows():
        i = cal.get_loc(r["date"])
        errs.append(abs(px.iat[i + horizon, px.columns.get_loc(r["ticker"])]
                        / px.iat[i, px.columns.get_loc(r["ticker"])] - 1 - r["fwd_return"]))
        if cal[i + horizon] != r["label_date"]:
            errs.append(np.inf)
    rows.append(_check("target recomputed", max(errs) < 1e-12,
                       f"{len(sample)} random rows, largest difference {max(errs):.1e}"))

    if panel is not None:
        me = table[table["month_end"]].set_index(["date", "universe", "ticker"])[config.FEATURES]
        pm = panel.set_index(["date", "universe", "ticker"])[config.FEATURES]
        common = me.index.intersection(pm.index)
        diff = float((me.loc[common] - pm.loc[common]).abs().max().max())
        rows.append(_check("month-ends match monthly table", diff < 1e-12 and len(common) == len(pm),
                           f"{len(common):,} month-end rows, largest difference {diff:.1e}"))

    return pd.DataFrame(rows)
