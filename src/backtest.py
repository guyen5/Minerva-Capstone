"""Monthly backtest: decisions, drift, turnover, fees and performance.

Timing. At each decision date t (a month-end) the strategy chooses target
weights using data up to the close of t. The portfolio trades at the close
of t and earns the return from t to the next month-end. During the month the
weights drift with prices, so the portfolio held just before the next trade
differs from the target.

Accounting, for one month with target w, pre-trade weights w_before and
ETF returns r:
    traded     = sum |w - w_before|
    turnover   = traded / 2          (a full switch from one ETF to another = 1)
    fee        = FEE * traded        (0.5% of the value bought and sold)
    gross      = w'r
    net        = gross - fee
    next w_before = w * (1 + r) / (1 + gross)

The first decision starts from equal weights, held before the test begins.
"""

import numpy as np
import pandas as pd

from src import config, covariance, optimizer


# Table 3 of Wang and Hasuike (2026): the two baselines on both universes.
# Key (universe, strategy, lam); value (annual return, annual volatility,
# Sharpe ratio, maximum drawdown, monthly turnover). MVO is classic Markowitz.
PAPER_TABLE3 = {
    ("ETF_A", "MVO", 0.1): (0.1622, 0.2479, 0.6545, -0.3115, 0.1938),
    ("ETF_A", "MVO", 1.0): (0.1039, 0.2360, 0.4404, -0.3115, 0.2481),
    ("ETF_A", "MVO", 10.0): (0.0307, 0.1078, 0.2846, -0.2126, 0.1421),
    ("ETF_A", "MVO", 20.0): (0.0194, 0.0785, 0.2475, -0.1698, 0.0907),
    ("ETF_A", "MVO", 50.0): (0.0153, 0.0659, 0.2330, -0.1558, 0.0446),
    ("ETF_A", "PTO", 0.1): (-0.0410, 0.2046, -0.2002, -0.4113, 0.8310),
    ("ETF_A", "PTO", 1.0): (-0.0410, 0.2046, -0.2002, -0.4113, 0.8310),
    ("ETF_A", "PTO", 10.0): (-0.0410, 0.2046, -0.2002, -0.4113, 0.8310),
    ("ETF_A", "PTO", 20.0): (-0.0410, 0.2046, -0.2002, -0.4113, 0.8310),
    ("ETF_A", "PTO", 50.0): (-0.0330, 0.2031, -0.1627, -0.4113, 0.8376),
    ("ETF_B", "MVO", 0.1): (0.1636, 0.2763, 0.5920, -0.3115, 0.2022),
    ("ETF_B", "MVO", 1.0): (0.1575, 0.2701, 0.5830, -0.3115, 0.2305),
    ("ETF_B", "MVO", 10.0): (0.0493, 0.1788, 0.2756, -0.2695, 0.2480),
    ("ETF_B", "MVO", 20.0): (0.0522, 0.1651, 0.3164, -0.2545, 0.1580),
    ("ETF_B", "MVO", 50.0): (0.0542, 0.1606, 0.3373, -0.2610, 0.0918),
    ("ETF_B", "PTO", 0.1): (0.0929, 0.2422, 0.3834, -0.3443, 0.9014),
    ("ETF_B", "PTO", 1.0): (0.0929, 0.2422, 0.3834, -0.3443, 0.9014),
    ("ETF_B", "PTO", 10.0): (0.0971, 0.2423, 0.4005, -0.3443, 0.8956),
    ("ETF_B", "PTO", 20.0): (0.0968, 0.2423, 0.3996, -0.3443, 0.8968),
    ("ETF_B", "PTO", 50.0): (0.0963, 0.2418, 0.3982, -0.3443, 0.9018),
}


def paper_rows(strategy="MVO"):
    """Table 3 rows of one baseline as a DataFrame."""
    cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover"]
    rows = [dict(universe=u, lam=lam, **dict(zip(cols, v)))
            for (u, s, lam), v in PAPER_TABLE3.items() if s == strategy]
    return pd.DataFrame(rows)


def tickers_of(universe):
    """Tickers of a universe, in their fixed order."""
    return config.ETF_A if universe == "ETF_A" else config.ETF_B


def holding_returns(panel, universe):
    """Return of each ETF over each test month, indexed by decision date."""
    p = panel[(panel["universe"] == universe) & (panel["period"] == "test")]
    r = p.pivot(index="date", columns="ticker", values="next_return")
    return r[tickers_of(universe)].sort_index()


def historical_means(clean, dates, universe, days=None, scale=None):
    """Mean daily return over the `days` ending at each date, times `scale`.

    These are the expected returns of classic Markowitz, on the same monthly
    scale as the covariance. The window defaults to config.MU_DAYS.
    """
    days = days or config.MU_DAYS
    scale = scale or config.MONTH_DAYS
    rets = covariance.daily_returns(clean)[tickers_of(universe)]
    out = {}
    for d in pd.DatetimeIndex(dates):
        window = rets.loc[:d].iloc[-days:]
        out[d] = window.mean() * scale
    return pd.DataFrame(out).T


def covariance_lookup(cov_table, universe, method):
    """Dictionary date -> covariance matrix (NumPy, universe ticker order)."""
    t = cov_table[(cov_table["universe"] == universe) & (cov_table["method"] == method)]
    tickers = tickers_of(universe)
    return {d: g.pivot(index="row", columns="col", values="value").loc[tickers, tickers].to_numpy()
            for d, g in t.groupby("date")}


# Each strategy is a function (date, w_before) -> target weights.

def equal_weight(n):
    """1/N, rebalanced back to equal weights every month."""
    return lambda date, w_before: np.full(n, 1 / n)


def minimum_variance(cov):
    """Lowest-variance long-only portfolio; ignores returns and trading costs."""
    return lambda date, w_before: optimizer.solve(np.zeros(len(w_before)), cov[date], 1.0, 0.0)


def mean_variance(mu, cov, lam, kappa=None):
    """Eq. 30 with expected returns mu (DataFrame, dates x tickers).

    Historical means give classic Markowitz, model forecasts give PTO, and
    Black-Litterman means give robust PTO. kappa defaults to the fee.
    """
    kappa = config.FEE if kappa is None else kappa
    return lambda date, w_before: optimizer.solve(mu.loc[date].to_numpy(), cov[date],
                                                  lam, kappa, w_before)


def run(strategy, returns, fee=None, start=None):
    """Run one strategy over the test months.

    returns: DataFrame of holding returns (decision dates x tickers).
    Returns (months, weights):
        months: one row per test month with gross and net return, fee,
                turnover and the number of ETFs held;
        weights: one row per month and ETF with the pre-trade and target weight.
    """
    fee = config.FEE if fee is None else fee
    tickers = list(returns.columns)
    n = len(tickers)
    w_before = np.full(n, 1 / n) if start is None else np.asarray(start, dtype=float)
    month_rows, weight_rows = [], []
    for date, r in returns.iterrows():
        r = r.to_numpy()
        target = np.asarray(strategy(date, w_before), dtype=float)
        traded = np.abs(target - w_before).sum()
        gross = float(target @ r)
        cost = fee * traded
        month_rows.append({
            "date": date, "month": str(date.to_period("M") + 1),
            "gross": gross, "fee": cost, "net": gross - cost,
            "turnover": traded / 2, "held": int((target > 1e-6).sum()),
            "effective_n": 1 / float((target ** 2).sum()),
        })
        for i, t in enumerate(tickers):
            weight_rows.append({"date": date, "ticker": t,
                                "before": w_before[i], "target": target[i]})
        w_before = target * (1 + r) / (1 + gross)
    return pd.DataFrame(month_rows), pd.DataFrame(weight_rows)


def metrics(months, label=None):
    """Annual return and volatility, Sharpe ratio, drawdown and turnover.

    Annual return is 12 times the mean monthly net return, volatility the
    monthly standard deviation times sqrt(12), and the Sharpe ratio their
    ratio with no risk-free rate, as in Wang and Hasuike (their Sharpe ratios
    equal return divided by volatility in every row of Table 3).
    """
    net = months["net"]
    wealth = (1 + net).cumprod()
    out = {
        "months": len(net),
        "ann_return": net.mean() * 12,
        "ann_vol": net.std(ddof=1) * np.sqrt(12),
        "max_drawdown": float((wealth / wealth.cummax().clip(lower=1) - 1).min()),
        "turnover": months["turnover"].mean(),
        "fee_per_year": months["fee"].mean() * 12,
        "effective_n": months["effective_n"].mean(),
        "cagr": float(wealth.iloc[-1] ** (12 / len(net)) - 1),
    }
    out["sharpe"] = out["ann_return"] / out["ann_vol"]
    s = pd.Series(out)
    if label is not None:
        s.name = label
    return s


def check_accounting(months, weights, returns, fee=None):
    """Validation report of a backtest, recomputed from weights and returns alone.

    Covers valid weights, drift between months, turnover, fee and net return,
    each against an independent calculation.
    """
    fee = config.FEE if fee is None else fee
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    tgt = weights.pivot(index="date", columns="ticker", values="target")[returns.columns]
    bef = weights.pivot(index="date", columns="ticker", values="before")[returns.columns]
    m = months.set_index("date")

    sums = tgt.sum(axis=1)
    add("valid weights", bool(np.allclose(sums, 1) and (tgt.to_numpy() >= 0).all()),
        f"{len(tgt)} months, weights non-negative and summing to 1")

    gross = (tgt * returns).sum(axis=1)
    drifted = tgt.mul(1 + returns).div(1 + gross, axis=0).shift(1).iloc[1:]
    err = float((drifted - bef.iloc[1:]).abs().to_numpy().max())
    add("drift", err < 1e-12, f"pre-trade weights equal last month's drifted target (largest difference {err:.1e})")

    traded = (tgt - bef).abs().sum(axis=1)
    e1 = float((m["turnover"] - traded / 2).abs().max())
    e2 = float((m["fee"] - fee * traded).abs().max())
    e3 = float((m["net"] - (gross - fee * traded)).abs().max())
    add("turnover", e1 < 1e-12, f"half the value traded (largest difference {e1:.1e})")
    add("fee", e2 < 1e-12, f"{fee:.1%} of the value traded (largest difference {e2:.1e})")
    add("net return", e3 < 1e-12, f"gross return minus fee (largest difference {e3:.1e})")
    return pd.DataFrame(rows)
