"""Tests of the data pipeline on simulated prices.

yfinance is replaced by a function that returns simulated prices in the
format Yahoo uses, so no internet connection is needed. The simulated data
contains known problems (a missing day, zero-volume days, a price spike)
that the quality checks are expected to catch.

Run from the repository root with: python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cleaning, config, covariance, download, features, variables, windows  # noqa: E402

DATES = pd.bdate_range("2009-01-02", "2026-09-30")


def fake_yahoo(tickers, start=None, end=None, **kwargs):
    """Simulated yf.download output with columns (field, ticker) and daily rows."""
    rng = np.random.default_rng(0)
    fields = ["Adj Close", "Capital Gains", "Close", "Dividends", "High", "Low",
              "Open", "Stock Splits", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    df = pd.DataFrame(np.nan, index=pd.DatetimeIndex(DATES, name="Date"), columns=cols)
    n = len(DATES)
    for t in tickers:
        close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        div = np.zeros(n)
        div[60::63] = 0.3
        factor = np.ones(n)
        for i in np.where(div > 0)[0]:
            factor[:i] *= 1 - div[i] / close[i - 1]
        opn = close * (1 + rng.normal(0, 0.002, n))
        df[("Close", t)] = close
        df[("Adj Close", t)] = close * factor
        df[("Open", t)] = opn
        df[("High", t)] = np.maximum(opn, close) * 1.003
        df[("Low", t)] = np.minimum(opn, close) * 0.997
        df[("Volume", t)] = rng.integers(100_000, 10_000_000, n).astype(float)
        df[("Dividends", t)] = div
        df[("Stock Splits", t)] = 0.0
        df[("Capital Gains", t)] = 0.0
    if "JPXN" in tickers:
        df.loc[DATES[100], (slice(None), "JPXN")] = np.nan     # one missing day
        df.loc[DATES[200:203], ("Volume", "JPXN")] = 0          # zero volume
    if "DBC" in tickers:
        df.loc[DATES[500], ("Adj Close", "DBC")] *= 1.4         # price spike
    return df


@pytest.fixture(scope="module")
def raw(monkeypatch_module):
    import yfinance as yf
    monkeypatch_module.setattr(yf, "download", fake_yahoo)
    return download.download_prices(pause=0)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def clean(raw):
    return cleaning.clean_prices(raw)[0]


@pytest.fixture(scope="module")
def panel(clean):
    return features.build_panel(clean)


def test_download_shape(raw):
    assert raw["ticker"].nunique() == 16
    assert list(raw.columns) == download.COLUMNS


def test_quality_checks_catch_planted_problems(raw):
    report = cleaning.run_quality_checks(raw)
    warn = report[report["status"] == "WARN"]
    assert ("missing days", "JPXN") in set(zip(warn["check"], warn["ticker"]))
    assert ("zero volume", "JPXN") in set(zip(warn["check"], warn["ticker"]))
    assert ("large moves", "DBC") in set(zip(warn["check"], warn["ticker"]))
    assert not (report["status"] == "FAIL").any()


def test_freeze_roundtrip(raw, tmp_path):
    csv_path, _ = download.freeze(raw, tmp_path)
    loaded, meta = download.load_frozen(tmp_path)
    assert len(loaded) == len(raw) == meta["n_rows"]
    with pytest.raises(FileExistsError):
        download.freeze(raw, tmp_path)
    csv_path.write_text(csv_path.read_text() + "tampered")
    with pytest.raises(ValueError):
        download.load_frozen(tmp_path)


def test_features_match_hand_calculation(clean):
    daily = features.daily_features(clean)
    spy = clean[clean["ticker"] == "SPY"].set_index("date")
    p, v = spy["adj_close"], spy["volume"].astype(float)
    i = 3000
    row = daily[(daily["ticker"] == "SPY") & (daily["date"] == p.index[i])].iloc[0]

    def ema(x, span):
        a, out = 2 / (span + 1), [x[0]]
        for z in x[1:]:
            out.append(a * z + (1 - a) * out[-1])
        return np.array(out)

    ch = p.diff().iloc[1:i + 1].to_numpy()
    g, lo = np.clip(ch, 0, None), np.clip(-ch, 0, None)
    ag, al = g[0], lo[0]
    for k in range(1, len(ch)):
        ag, al = ag + (g[k] - ag) / 14, al + (lo[k] - al) / 14
    x = p.iloc[:i + 1].to_numpy()
    macd = ema(x, 12) - ema(x, 26)

    hand = {
        "momentum": np.log(p.iloc[i] / p.iloc[i - 21]),
        "trend": p.iloc[i - 9:i + 1].mean() / p.iloc[i - 49:i + 1].mean() - 1,
        "price_bias": p.iloc[i] / p.iloc[i - 9:i + 1].mean() - 1,
        "rsi": 100 - 100 / (1 + ag / al),
        "macd": (macd[-1] - ema(macd, 9)[-1]) / x[-1],
        "bollinger": 4 * p.iloc[i - 19:i + 1].std(ddof=0) / p.iloc[i - 19:i + 1].mean(),
        "volume_bias": v.iloc[i] / v.iloc[i - 9:i + 1].mean() - 1,
    }
    for name, value in hand.items():
        assert row[name] == pytest.approx(value, abs=1e-9), name


def test_panel_passes_all_checks(panel, clean):
    report = features.validate_panel(panel, clean)
    assert (report["status"] == "PASS").all(), report[report["status"] != "PASS"]
    assert (variables.check_table(panel, "Monthly Data")["status"] == "PASS").all()
    assert (variables.check_table(clean, "Daily Prices")["status"] == "PASS").all()


def test_zero_volume_month_end_is_flagged(clean):
    c = clean.copy()
    d = features.month_ends(c)[80]
    c.loc[(c["ticker"] == "JPXN") & (c["date"] == d), "volume"] = 0
    p = features.build_panel(c)
    row = p[(p["ticker"] == "JPXN") & (p["date"] == d)].iloc[0]
    assert row["volume_bias"] == 0 and row["volume_filled"]


def test_data_dirs_creates_the_four_folders(tmp_path):
    dirs = config.data_dirs(tmp_path)
    assert set(dirs) == {"raw", "processed", "reports", "archive"}
    assert all(path.is_dir() for path in dirs.values())


@pytest.fixture(scope="module")
def daily_table(clean):
    return features.build_daily_table(clean)


def test_daily_table_passes_all_checks(daily_table, clean, panel):
    report = features.validate_daily(daily_table, clean, panel)
    assert (report["status"] == "PASS").all(), report[report["status"] != "PASS"]
    assert (variables.check_table(daily_table, "Daily Features")["status"] == "PASS").all()


def test_daily_target_is_21_days_ahead(daily_table, clean):
    px = clean.pivot(index="date", columns="ticker", values="adj_close")
    row = daily_table[(daily_table["universe"] == "ETF_A")
                      & (daily_table["ticker"] == "EEM")].iloc[500]
    i = px.index.get_loc(row["date"])
    assert row["label_date"] == px.index[i + 21]
    assert row["fwd_return"] == pytest.approx(px["EEM"].iloc[i + 21] / px["EEM"].iloc[i] - 1)


def test_daily_windows(daily_table):
    for name in config.DAILY_WINDOWS:
        cal = windows.daily_windows(daily_table, name)
        assert len(cal) == 81
        report = windows.check_daily_windows(cal, daily_table)
        assert (report["status"] == "PASS").all(), report
        assert (variables.check_table(cal, "Daily Training Windows")["status"] == "PASS").all()


def test_daily_windows_leave_no_overlap(daily_table):
    cal = windows.daily_windows(daily_table, "9+3")
    step = cal.iloc[0]
    fit, val, test = windows.daily_rows_for(daily_table, step, "ETF_A")
    assert fit["label_date"].max() < val["date"].min()
    assert val["label_date"].max() <= step["decision_date"]
    assert len(test) == len(config.ETF_A) and test["month_end"].all()
    # A 9+3 window spans about 12 months of trading days; the simulated
    # calendar has no holidays.
    days = daily_table.loc[daily_table["universe"] == "ETF_A", "date"]
    n = days[(days >= step["train_start"]) & (days <= step["train_end"])].nunique()
    assert 240 <= n <= 265


def test_ledoit_wolf_matches_sklearn(clean):
    sk = pytest.importorskip("sklearn.covariance")
    r = covariance.daily_returns(clean)[config.ETF_A].iloc[-220:]
    mine, k = covariance.ledoit_wolf(r)
    ref, k_ref = sk.ledoit_wolf(r.to_numpy())
    assert k == pytest.approx(k_ref)
    assert np.abs(mine - ref).max() < 1e-15


def test_covariances_pass_all_checks(clean):
    ends = features.month_ends(clean)
    dates = ends[(ends >= "2019-06-01") & (ends <= "2020-06-30")]
    table = covariance.build_covariances(clean, dates)
    report = covariance.validate_covariances(table, clean)
    assert (report["status"] == "PASS").all(), report[report["status"] != "PASS"]
    assert (variables.check_table(table, "Covariances")["status"] == "PASS").all()
    m = covariance.matrix(table, dates[0], "ETF_A")
    assert list(m.index) == config.ETF_A


def test_backtest_on_simulated_data(clean, panel):
    from src import backtest
    returns = backtest.holding_returns(panel, "ETF_A")
    assert len(returns) == 81
    cov = backtest.covariance_lookup(covariance.build_covariances(clean, returns.index),
                                     "ETF_A", "sample")
    mu = backtest.historical_means(clean, returns.index, "ETF_A")
    months, weights = backtest.run(backtest.mean_variance(mu, cov, 10), returns)
    report = backtest.check_accounting(months, weights, returns)
    assert (report["status"] == "PASS").all(), report


def test_forecasts_on_simulated_data(daily_table):
    from src import forecast
    cal = windows.daily_windows(daily_table, "9+3").iloc[:6]
    f = forecast.build_forecasts(daily_table, cal, "ETF_A", config.ETF_A)
    report = forecast.check_forecasts(f, daily_table, cal, "ETF_A", config.ETF_A)
    assert (report["status"] == "PASS").all(), report
