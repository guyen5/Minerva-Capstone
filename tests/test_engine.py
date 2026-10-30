"""Tests of the optimizer, the backtest accounting and the ridge forecasts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest, optimizer  # noqa: E402


def two_months():
    """A +10% then 0%; B 0% then +10%."""
    idx = pd.to_datetime(["2019-12-31", "2020-01-31"])
    return pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.10]}, index=idx)


def test_month_by_hand():
    """Hold A in month 1, switch to B in month 2, starting from 50/50.

    Month 1: trade 0.5 into A and 0.5 out of B, value traded 1.0,
    turnover 0.5, fee 0.005, gross 10%, net 9.5%.
    Month 2: sell all of A and buy all of B, value traded 2.0,
    turnover 1.0, fee 0.01, gross 10%, net 9%.
    """
    plan = {pd.Timestamp("2019-12-31"): [1, 0], pd.Timestamp("2020-01-31"): [0, 1]}
    months, weights = backtest.run(lambda d, w: np.array(plan[d], float), two_months())
    assert months["turnover"].tolist() == pytest.approx([0.5, 1.0])
    assert months["fee"].tolist() == pytest.approx([0.005, 0.01])
    assert months["net"].tolist() == pytest.approx([0.095, 0.09])
    m = backtest.metrics(months)
    assert m["ann_return"] == pytest.approx(12 * 0.0925)
    assert m["turnover"] == pytest.approx(0.75)
    check = backtest.check_accounting(months, weights, two_months())
    assert (check["status"] == "PASS").all()


def test_weights_drift_between_trades():
    """A 50/50 portfolio that is not rebalanced drifts to 55/50 over 105."""
    r = two_months()
    months, weights = backtest.run(lambda d, w: w, r)
    before = weights[weights["date"] == r.index[1]]["before"].to_numpy()
    assert before == pytest.approx([0.55 / 1.05, 0.50 / 1.05])
    assert months["turnover"].tolist() == pytest.approx([0, 0])


def test_optimizer_checks_pass():
    report = optimizer.check_optimizer(draws=50)
    assert (report["status"] == "PASS").all(), report


def test_risk_aversion_spreads_the_portfolio():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 8)) * 0.03
    sigma = a @ a.T + np.eye(8) * 1e-3
    mu = rng.normal(0.01, 0.02, 8)
    held = [(optimizer.solve(mu, sigma, lam) > 1e-6).sum() for lam in (0.1, 50)]
    assert held[0] <= held[1]
    vols = [np.sqrt(w @ sigma @ w) for w in (optimizer.solve(mu, sigma, lam) for lam in (0.1, 1, 10, 50))]
    assert all(x >= y - 1e-12 for x, y in zip(vols, vols[1:]))


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    from src import forecast
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 10))
    y = x @ rng.normal(size=10) * 0.01 + rng.normal(size=300) * 0.05
    for alpha in (0.1, 100.0, 1e5):
        b0, b = forecast.ridge_fit(x, y, [alpha])[0]
        sk = Ridge(alpha=alpha).fit(x, y)
        assert b0 == pytest.approx(sk.intercept_, abs=1e-10)
        assert b == pytest.approx(sk.coef_, abs=1e-10)


def test_large_penalty_gives_equal_forecasts():
    """With a huge penalty every ETF gets the same forecast, the mean return."""
    from src import forecast
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 5))
    y = rng.normal(0.01, 0.05, 200)
    b0, b = forecast.ridge_fit(x, y, [1e12])[0]
    assert np.abs(b).max() < 1e-9
    assert b0 == pytest.approx(y.mean())
