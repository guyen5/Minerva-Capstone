"""Definitions of every saved column.

Each entry gives a column's table, label, type, unit, meaning, formula,
allowed range, when it may be empty and the function that produces it.
check_table() compares a saved table against its entries, and to_frame()
exports the list as the variables file, so the code and the documentation
share one source.
"""

import pandas as pd


def _v(table, column, label, kind, unit, meaning, formula="", lo=None, hi=None,
       empty="never", code=""):
    """One variable entry."""
    return dict(table=table, column=column, label=label, kind=kind, unit=unit,
                meaning=meaning, formula=formula, lo=lo, hi=hi, empty=empty, code=code)


D, M = "Daily Prices", "Monthly Data"
DF, DW, C = "Daily Features", "Daily Training Windows", "Covariances"
B = "Benchmark Months"
F, R = "PTO Forecasts", "Replication Months"

ENTRIES = [
    _v(D, "date", "Date", "date", "trading day", "NYSE trading day", code="cleaning.clean_prices"),
    _v(D, "ticker", "Ticker", "text", "", "ETF symbol (16 ETFs)", code="cleaning.clean_prices"),
    _v(D, "open", "Open", "float", "USD", "First trade of the day, not adjusted", lo=0, code="download.download_prices"),
    _v(D, "high", "High", "float", "USD", "Highest trade of the day, not adjusted", lo=0, code="download.download_prices"),
    _v(D, "low", "Low", "float", "USD", "Lowest trade of the day, not adjusted", lo=0, code="download.download_prices"),
    _v(D, "close", "Close", "float", "USD", "Last trade of the day, not adjusted. Only used in quality checks", lo=0, code="download.download_prices"),
    _v(D, "adj_close", "Adjusted close", "float", "USD", "Close adjusted for dividends and splits. All returns and features use this", lo=0, code="download.download_prices"),
    _v(D, "volume", "Volume", "float", "shares", "Shares traded that day", lo=0, empty="only on filled days", code="download.download_prices"),
    _v(D, "dividends", "Dividend", "float", "USD per share", "Cash paid on the ex-dividend date, else 0", lo=0, code="download.download_prices"),
    _v(D, "splits", "Split", "float", "ratio", "Split ratio on split days, else 0", lo=0, code="download.download_prices"),
    _v(D, "capital_gains", "Capital gains", "float", "USD per share", "Capital gains paid that day, else 0", lo=0, code="download.download_prices"),
    _v(D, "filled", "Filled", "bool", "", "Price copied from the previous day", code="cleaning.clean_prices"),
    _v(D, "zero_volume", "Zero volume", "bool", "", "No shares traded that day", code="cleaning.clean_prices"),

    _v(M, "date", "Decision date", "date", "month-end", "Last trading day of the month; features are measured here", code="features.build_panel"),
    _v(M, "universe", "Universe", "text", "", "ETF_A (primary, 8 ETFs) or ETF_B (robustness, 10 ETFs)", code="features.build_panel"),
    _v(M, "ticker", "Ticker", "text", "", "ETF symbol; one-hot encoded as a model input", code="features.build_panel"),
    _v(M, "month", "Holding month", "text", "YYYY-MM", "The month whose return the row measures", code="features.build_panel"),
    _v(M, "period", "Period", "text", "", "history (before the test period), test, after test period, or latest (no return yet)", code="features.build_panel"),
    _v(M, "momentum", "Momentum", "float", "log return", "Return over the last month", "ln(P today / P 21 days ago)", lo=-2, hi=2, code="features._one_ticker_features"),
    _v(M, "trend", "Trend", "float", "ratio", "10-day average relative to 50-day average", "SMA10 / SMA50 - 1", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(M, "price_bias", "Price bias", "float", "ratio", "Price relative to its 10-day average", "P / SMA10 - 1", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(M, "rsi", "RSI", "float", "0 to 100", "Balance of recent gains and losses", "100 - 100 / (1 + avg gain / avg loss), Wilder smoothing, 14 days", lo=0, hi=100, code="features.wilder_rsi"),
    _v(M, "macd", "MACD", "float", "share of price", "Whether short-term momentum is speeding up or slowing", "(MACD line - signal line) / P, with EMA 12, 26, 9", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(M, "bollinger", "Bollinger width", "float", "share of price", "Recent volatility relative to price", "4 x std20 / SMA20", lo=0, hi=2, code="features._one_ticker_features"),
    _v(M, "volume_bias", "Volume bias", "float", "ratio", "Trading volume relative to its 10-day average", "V / SMA10(V) - 1, set to 0 on zero-volume days", lo=-1, code="features._one_ticker_features"),
    _v(M, "volume_filled", "Volume filled", "bool", "", "Volume bias was set to 0 because nothing traded that day", code="features.build_panel"),
    _v(M, "next_return", "Next-month return", "float", "simple return", "Return over the holding month, earned by a portfolio formed at the decision date", "P next month-end / P this month-end - 1", lo=-1, empty="latest month only", code="features._universe_rows"),

    _v(DF, "date", "Date", "date", "trading day", "Trading day; features are measured at its close", code="features.build_daily_table"),
    _v(DF, "universe", "Universe", "text", "", "ETF_A (main, 8 ETFs) or ETF_B (robustness, 10 ETFs)", code="features.build_daily_table"),
    _v(DF, "ticker", "Ticker", "text", "", "ETF symbol; one-hot encoded as a model input", code="features.build_daily_table"),
    _v(DF, "month_end", "Month-end", "bool", "", "True on the last trading day of a month, the decision dates", code="features.build_daily_table"),
    _v(DF, "momentum", "Momentum", "float", "log return", "Return over the last month", "ln(P today / P 21 days ago)", lo=-2, hi=2, code="features._one_ticker_features"),
    _v(DF, "trend", "Trend", "float", "ratio", "10-day average relative to 50-day average", "SMA10 / SMA50 - 1", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(DF, "price_bias", "Price bias", "float", "ratio", "Price relative to its 10-day average", "P / SMA10 - 1", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(DF, "rsi", "RSI", "float", "0 to 100", "Balance of recent gains and losses", "100 - 100 / (1 + avg gain / avg loss), Wilder smoothing, 14 days", lo=0, hi=100, code="features.wilder_rsi"),
    _v(DF, "macd", "MACD", "float", "share of price", "Whether short-term momentum is speeding up or slowing", "(MACD line - signal line) / P, with EMA 12, 26, 9", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(DF, "bollinger", "Bollinger width", "float", "share of price", "Recent volatility relative to price", "4 x std20 / SMA20", lo=0, hi=2, code="features._one_ticker_features"),
    _v(DF, "volume_bias", "Volume bias", "float", "ratio", "Trading volume relative to its 10-day average", "V / SMA10(V) - 1, set to 0 on zero-volume days", lo=-1, code="features._one_ticker_features"),
    _v(DF, "volume_filled", "Volume filled", "bool", "", "Volume bias was set to 0 because nothing traded that day", code="features.build_daily_table"),
    _v(DF, "fwd_return", "Forward return", "float", "simple return", "Return over the next 21 trading days", "P in 21 trading days / P today - 1", lo=-1, empty="last 21 trading days only", code="features.build_daily_table"),
    _v(DF, "fwd_rel_return", "Forward relative return", "float", "simple return", "Forward return minus the universe average on the same days", "fwd_return - mean of fwd_return across the universe", empty="last 21 trading days only", code="features.build_daily_table"),
    _v(DF, "label_date", "Label date", "date", "trading day", "Day the forward return becomes known; a row trains a model only from this day on", "trading day 21 days after date", empty="last 21 trading days only", code="features.build_daily_table"),

    _v(DW, "month", "Test month", "text", "YYYY-MM", "Month being predicted", code="windows.daily_windows"),
    _v(DW, "decision_date", "Decision date", "date", "month-end", "When the model is refit and predicts", code="windows.daily_windows"),
    _v(DW, "train_start", "Training start", "date", "trading day", "First daily row in the window", code="windows.daily_windows"),
    _v(DW, "fit_end", "Fit end", "date", "trading day", "Last row used for fitting during tuning; its target is known before validation starts", code="windows.daily_windows"),
    _v(DW, "validation_start", "Validation start", "date", "trading day", "First row of the validation slice", code="windows.daily_windows"),
    _v(DW, "train_end", "Training end", "date", "trading day", "Newest row whose target is known on the decision date", code="windows.daily_windows"),
    _v(DW, "train_months", "Window months", "int", "months", "Length of the whole window, validation included", lo=1, code="windows.daily_windows"),
    _v(DW, "validation_months", "Validation months", "int", "months", "Length of the validation slice", lo=1, code="windows.daily_windows"),
    _v(DW, "window", "Window", "text", "", "Name of the window in config.DAILY_WINDOWS; 9+3 is the window of Wang and Hasuike", code="windows.daily_windows"),

    _v(C, "date", "Decision date", "date", "month-end", "Month-end at which the matrix is estimated", code="covariance.build_covariances"),
    _v(C, "universe", "Universe", "text", "", "ETF_A or ETF_B", code="covariance.build_covariances"),
    _v(C, "method", "Method", "text", "", "sample (with a small ridge on the diagonal, as in the paper) or ledoit_wolf", code="covariance.build_covariances"),
    _v(C, "row", "Row ETF", "text", "", "ETF of the matrix row", code="covariance.build_covariances"),
    _v(C, "col", "Column ETF", "text", "", "ETF of the matrix column", code="covariance.build_covariances"),
    _v(C, "value", "Covariance", "float", "monthly return squared", "Covariance of the two ETFs' returns, daily estimate x 21", "from the daily returns ending at the decision date", code="covariance.build_covariances"),
    _v(C, "shrinkage", "Shrinkage", "float", "0 to 1", "Ledoit-Wolf intensity toward the scaled identity; 0 for the sample method", lo=0, hi=1, code="covariance.ledoit_wolf"),
    _v(C, "window_start", "Window start", "date", "trading day", "First day of the return window", code="covariance.build_covariances"),

    _v(B, "strategy", "Strategy", "text", "", "1/N, minimum variance or classic Markowitz", code="backtest.run"),
    _v(B, "universe", "Universe", "text", "", "ETF_A or ETF_B", code="backtest.run"),
    _v(B, "lam", "Risk aversion", "float", "", "lambda in the optimizer; empty for 1/N and minimum variance", lo=0, empty="1/N and minimum variance", code="backtest.mean_variance"),
    _v(B, "date", "Decision date", "date", "month-end", "When the portfolio is chosen and traded", code="backtest.run"),
    _v(B, "month", "Holding month", "text", "YYYY-MM", "Month the return is earned", code="backtest.run"),
    _v(B, "gross", "Gross return", "float", "simple return", "Portfolio return before fees", "target weights x ETF returns", lo=-1, code="backtest.run"),
    _v(B, "fee", "Fee", "float", "share of portfolio", "Trading cost charged at the decision date", "0.005 x sum |target - pre-trade weights|", lo=0, code="backtest.run"),
    _v(B, "net", "Net return", "float", "simple return", "Return after the fee", "gross - fee", lo=-1, code="backtest.run"),
    _v(B, "turnover", "Turnover", "float", "share of portfolio", "Half the value traded; a full switch from one ETF to another is 1", "sum |target - pre-trade weights| / 2", lo=0, hi=1, code="backtest.run"),
    _v(B, "held", "ETFs held", "int", "count", "ETFs with a target weight above 0.0001%", lo=1, code="backtest.run"),
    _v(B, "effective_n", "Effective number of ETFs", "float", "count", "How many equal holdings the portfolio is worth", "1 / sum of squared weights", lo=1, code="backtest.run"),

    _v(F, "date", "Decision date", "date", "month-end", "When the model is refit and predicts", code="forecast.build_forecasts"),
    _v(F, "month", "Holding month", "text", "YYYY-MM", "Month being predicted", code="forecast.build_forecasts"),
    _v(F, "universe", "Universe", "text", "", "ETF_A or ETF_B", code="forecast.build_forecasts"),
    _v(F, "ticker", "Ticker", "text", "", "ETF symbol", code="forecast.build_forecasts"),
    _v(F, "forecast", "Forecast", "float", "simple return", "Predicted return over the holding month; the expected return of PTO", "pooled ridge on the daily rows of the training window", code="forecast.forecast_month"),
    _v(F, "alpha", "Ridge penalty", "float", "", "Penalty chosen on the validation slice", lo=0, code="forecast.forecast_month"),
    _v(F, "validation_mse", "Validation error", "float", "squared return", "Mean squared error of the chosen model on the validation slice", lo=0, code="forecast.forecast_month"),
    _v(F, "fit_rows", "Fit rows", "int", "rows", "Daily rows fitted while tuning", lo=1, code="forecast.forecast_month"),
    _v(F, "validation_rows", "Validation rows", "int", "rows", "Daily rows in the validation slice", lo=1, code="forecast.forecast_month"),
]

# Replication Months has the same columns as Benchmark Months, with lam always set.
_R_MEANING = {"strategy": "Classic Markowitz or PTO", "lam": "lambda in the optimizer"}
ENTRIES += [dict(e, table=R, meaning=_R_MEANING.get(e["column"], e["meaning"]),
                 empty="never" if e["column"] == "lam" else e["empty"])
            for e in ENTRIES if e["table"] == B]


def table_entries(table):
    """Entries of one table, in column order."""
    return [e for e in ENTRIES if e["table"] == table]


def to_frame():
    """All entries as a DataFrame (saved as Variables.csv)."""
    return pd.DataFrame(ENTRIES)


_KIND_CHECK = {
    "date": pd.api.types.is_datetime64_any_dtype,
    "text": lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
    "float": pd.api.types.is_float_dtype,
    "int": pd.api.types.is_integer_dtype,
    "bool": pd.api.types.is_bool_dtype,
}


def check_table(df, table):
    """Validation report of a saved table against its entries: columns, types, ranges, empties."""
    entries = table_entries(table)
    expected = [e["column"] for e in entries]
    same = list(df.columns) == expected
    rows = [{"check": "columns", "column": "all", "status": "PASS" if same else "FAIL",
             "detail": "same columns and order as Variables" if same
             else f"table {list(df.columns)} vs Variables {expected}"}]

    for e in entries:
        name = e["column"]
        if name not in df.columns:
            continue
        s = df[name]
        problems = []
        if not _KIND_CHECK[e["kind"]](s):
            problems.append(f"type {s.dtype}, expected {e['kind']}")
        n_na = int(s.isna().sum())
        if e["empty"] == "never" and n_na:
            problems.append(f"{n_na} empty values")
        values = s.dropna()
        if e["kind"] in ("float", "int") and len(values):
            # The 1e-9 tolerance absorbs floating-point rounding at the bounds.
            if e["lo"] is not None and values.min() < e["lo"] - 1e-9:
                problems.append(f"min {values.min():.4g} below {e['lo']}")
            if e["hi"] is not None and values.max() > e["hi"] + 1e-9:
                problems.append(f"max {values.max():.4g} above {e['hi']}")
        detail = "; ".join(problems) if problems else (
            f"{e['kind']}, {n_na} empty" + (
                f", range {values.min():.4g} to {values.max():.4g}"
                if e["kind"] in ("float", "int") and len(values) else ""))
        rows.append({"check": "column", "column": name,
                     "status": "FAIL" if problems else "PASS", "detail": detail})
    return pd.DataFrame(rows)
