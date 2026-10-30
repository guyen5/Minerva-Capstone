"""Covariance matrices of ETF returns at each decision date.

Each month-end t gets two estimates from the COV_DAYS daily returns ending
at t, scaled to a monthly horizon by MONTH_DAYS:

- sample: the sample covariance plus COV_RIDGE on the diagonal, as in
  Wang and Hasuike;
- ledoit_wolf: the sample covariance shrunk toward a scaled identity matrix
  with the data-chosen intensity of Ledoit and Wolf (2004). Closely related
  ETFs such as SPY and VTI make the sample matrix poorly conditioned, and
  shrinkage improves its conditioning.

The estimates are stored as one long table: date, universe, method, row,
col, value, the shrinkage intensity and the first day of the window.
"""

import numpy as np
import pandas as pd

from src import config


def daily_returns(clean):
    """Simple daily returns from adjusted closes, one column per ETF."""
    px = clean.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return px.pct_change().iloc[1:]


def sample_cov(r, ridge=None):
    """Sample covariance (ddof = 1) plus a small ridge on the diagonal."""
    ridge = config.COV_RIDGE if ridge is None else ridge
    x = np.asarray(r, dtype=float)
    s = np.cov(x, rowvar=False, ddof=1)
    return s + ridge * np.eye(s.shape[0])


def ledoit_wolf(r):
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity.

    Returns (matrix, intensity). The formulas are those of
    sklearn.covariance.ledoit_wolf, which uses the 1/n sample covariance.
    """
    x = np.asarray(r, dtype=float)
    x = x - x.mean(axis=0)
    n, p = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / p
    target = mu * np.eye(p)
    d2 = np.sum((s - target) ** 2)
    b2_bar = sum(np.sum((np.outer(row, row) - s) ** 2) for row in x) / n ** 2
    b2 = min(b2_bar, d2)
    intensity = 0.0 if d2 == 0 else b2 / d2
    return intensity * target + (1 - intensity) * s, intensity


def build_covariances(clean, dates, days=None, scale=None):
    """Long table of both estimates for both universes at every date in `dates`.

    The window is the `days` daily returns ending at the date (inclusive), so
    no return after the decision date is used.
    """
    days = days or config.COV_DAYS
    scale = scale or config.MONTH_DAYS
    rets = daily_returns(clean)
    out = []
    for d in pd.DatetimeIndex(dates):
        window = rets.loc[:d].iloc[-days:]
        if len(window) < days:
            continue
        for uni, tickers in (("ETF_A", config.ETF_A), ("ETF_B", config.ETF_B)):
            w = window[tickers]
            lw, k = ledoit_wolf(w)
            for method, m, shrink in (("sample", sample_cov(w), 0.0), ("ledoit_wolf", lw, k)):
                m = m * scale
                for i, a in enumerate(tickers):
                    for j, b in enumerate(tickers):
                        out.append((d, uni, method, a, b, m[i, j], shrink, window.index[0]))
    cols = ["date", "universe", "method", "row", "col", "value", "shrinkage", "window_start"]
    return pd.DataFrame(out, columns=cols)


def matrix(table, date, universe, method="ledoit_wolf"):
    """One covariance matrix as a square DataFrame, in the universe's ticker order."""
    tickers = config.ETF_A if universe == "ETF_A" else config.ETF_B
    t = table[(table["date"] == pd.Timestamp(date)) & (table["universe"] == universe)
              & (table["method"] == method)]
    return t.pivot(index="row", columns="col", values="value").loc[tickers, tickers]


def validate_covariances(table, clean, n_samples=3, seed=0):
    """Validation report: complete, symmetric, positive definite, no look-ahead, recomputed."""
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    keys = table.drop_duplicates(["date", "universe", "method"])
    counts = table.groupby(["date", "universe", "method"]).size()
    expected = {"ETF_A": len(config.ETF_A) ** 2, "ETF_B": len(config.ETF_B) ** 2}
    ok = all(n == expected[u] for (_, u, _), n in counts.items())
    add("complete matrices", ok, f"{len(keys):,} matrices, every one N x N")

    worst_sym, min_eig = 0.0, np.inf
    for (d, u, m), g in table.groupby(["date", "universe", "method"]):
        a = g.pivot(index="row", columns="col", values="value").to_numpy()
        worst_sym = max(worst_sym, float(np.abs(a - a.T).max()))
        min_eig = min(min_eig, float(np.linalg.eigvalsh(a).min()))
    add("symmetric", worst_sym < 1e-15, f"largest asymmetry {worst_sym:.1e}")
    add("positive definite", min_eig > 0, f"smallest eigenvalue {min_eig:.2e}")

    lw = keys[keys["method"] == "ledoit_wolf"]["shrinkage"]
    add("shrinkage between 0 and 1", bool(lw.between(0, 1).all()),
        f"intensity from {lw.min():.3f} to {lw.max():.3f}, median {lw.median():.3f}")

    rets = daily_returns(clean)
    ends_before = all(rets.loc[:d].index[-config.COV_DAYS] == s
                      for d, s in keys[["date", "window_start"]].drop_duplicates().itertuples(index=False))
    add("no look-ahead", ends_before,
        f"every window is the {config.COV_DAYS} returns ending at the decision date")

    rng = np.random.default_rng(seed)
    dates = rng.choice(keys["date"].unique(), min(n_samples, keys["date"].nunique()), replace=False)
    worst = 0.0
    for d in dates:
        d = pd.Timestamp(d)
        w = rets.loc[:d].iloc[-config.COV_DAYS:][config.ETF_A]
        direct = (w.cov().to_numpy() + config.COV_RIDGE * np.eye(w.shape[1])) * config.MONTH_DAYS
        stored = matrix(table, d, "ETF_A", "sample").to_numpy()
        worst = max(worst, float(np.abs(stored - direct).max()))
    add("sample recomputed", worst < 1e-12,
        f"{len(dates)} random dates checked against pandas .cov(), largest difference {worst:.1e}")
    return pd.DataFrame(rows)
