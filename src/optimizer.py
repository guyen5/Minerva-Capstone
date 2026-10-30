"""The mean-variance problem solved at every decision date.

    maximize   mu'w - lam * w'Sigma w - kappa * ||w - w_prev||_1
    subject to sum(w) = 1, w >= 0

This is Eq. 30 of Wang and Hasuike. mu is the expected monthly return of
each ETF, Sigma the monthly covariance, w_prev the portfolio held before
trading (last month's weights after drifting with returns), lam the risk
aversion and kappa the trading penalty. All strategies share this problem
and differ only in mu, Sigma and kappa.
"""

import numpy as np
import cvxpy as cp
import pandas as pd

_CACHE = {}
TINY = 1e-7       # weights below this are set to 0 after solving


def _problem(n):
    """Problem for n assets, compiled once and reused with new parameter values."""
    if n not in _CACHE:
        w = cp.Variable(n)
        mu = cp.Parameter(n)
        root = cp.Parameter((n, n))          # lam * Sigma = root' root
        w_prev = cp.Parameter(n)
        kappa = cp.Parameter(nonneg=True)
        # Helper variables keep every parameter in a linear term, so cvxpy
        # compiles the problem once and later solves are fast.
        y = cp.Variable(n)                   # y = root w: risk is ||y||^2
        z = cp.Variable(n)                   # z = w - w_prev: the trades
        objective = mu @ w - cp.sum_squares(y) - kappa * cp.norm1(z)
        prob = cp.Problem(cp.Maximize(objective),
                          [cp.sum(w) == 1, w >= 0, y == root @ w, z == w - w_prev])
        _CACHE[n] = (prob, w, mu, root, w_prev, kappa)
    return _CACHE[n]


def _root(sigma):
    """A matrix R with R'R = Sigma (Cholesky, with a tiny ridge if needed)."""
    s = np.asarray(sigma, dtype=float)
    s = (s + s.T) / 2
    try:
        return np.linalg.cholesky(s).T
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(s + 1e-12 * np.eye(len(s))).T


def solve(mu, sigma, lam, kappa=0.0, w_prev=None):
    """Optimal weights for one decision date, as a NumPy array summing to 1."""
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    w_prev = np.full(n, 1 / n) if w_prev is None else np.asarray(w_prev, dtype=float)
    prob, w, p_mu, p_root, p_prev, p_kappa = _problem(n)
    p_mu.value, p_prev.value = mu, w_prev
    p_root.value = np.sqrt(float(lam)) * _root(sigma)
    p_kappa.value = float(kappa)
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                   tol_feas=1e-10)
    except cp.SolverError:
        prob.solve(solver=cp.SCS, eps=1e-9)
    if w.value is None:
        raise RuntimeError(f"optimizer failed: {prob.status}")
    out = np.clip(w.value, 0, None)
    out[out < TINY] = 0
    # Trades smaller than TINY are solver noise around "do not trade".
    keep = np.abs(out - w_prev) < TINY
    out[keep] = w_prev[keep]
    return out / out.sum()


def objective(w, mu, sigma, lam, kappa, w_prev):
    """Value of the objective for given weights."""
    w = np.asarray(w)
    return float(mu @ w - lam * w @ sigma @ w - kappa * np.abs(w - w_prev).sum())


def check_optimizer(n=8, draws=200, seed=0):
    """Validation report on random problems of realistic size.

    - constraints: weights are non-negative and sum to 1;
    - optimal: no random portfolio, single-ETF portfolio or the previous
      portfolio scores higher;
    - no-trade band: with a very large trading penalty the portfolio stays put;
    - corner: with almost no risk aversion and no penalty, everything goes into
      the ETF with the highest mu;
    - minimum variance: with equal mu and no penalty, the solution matches the
      closed-form minimum-variance portfolio when that has no negative weights.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    worst_c, worst_gap = 0.0, -np.inf
    for _ in range(draws):
        a = rng.normal(size=(n, n)) * 0.03
        sigma = a @ a.T + np.diag(rng.uniform(1e-4, 4e-3, n))
        mu = rng.normal(0.01, 0.03, n)
        lam = rng.choice([0.1, 1, 10, 20, 50])
        kappa = rng.choice([0, 0.005, 0.02, 0.1])
        prev = rng.dirichlet(np.ones(n))
        w = solve(mu, sigma, lam, kappa, prev)
        worst_c = max(worst_c, abs(w.sum() - 1), -w.min())
        best = objective(w, mu, sigma, lam, kappa, prev)
        rivals = [rng.dirichlet(np.ones(n) * c) for c in (0.2, 1, 5) for _ in range(30)]
        rivals += list(np.eye(n)) + [prev]
        gap = max(objective(r, mu, sigma, lam, kappa, prev) for r in rivals) - best
        worst_gap = max(worst_gap, gap)
    add("constraints", worst_c < 1e-9, f"{draws} random problems, largest violation {worst_c:.1e}")
    add("optimal", worst_gap < 1e-8,
        f"no rival portfolio scores higher (largest excess {max(worst_gap, 0):.0e}, "
        "within solver tolerance)")

    prev = rng.dirichlet(np.ones(n))
    w = solve(rng.normal(0.01, 0.03, n), sigma, 1, 10.0, prev)
    add("no-trade band", np.abs(w - prev).max() < 1e-9,
        "with kappa = 10 the portfolio is left unchanged")

    mu = rng.normal(0.01, 0.03, n)
    w = solve(mu, sigma, 1e-6, 0.0)
    add("corner", w[np.argmax(mu)] > 0.999, "with lam near 0, all weight in the ETF with the highest mu")

    d = np.diag(rng.uniform(1e-3, 4e-3, n))
    inv = np.linalg.inv(d)
    closed = inv.sum(axis=1) / inv.sum()
    w = solve(np.zeros(n), d, 1.0, 0.0)
    add("minimum variance", np.abs(w - closed).max() < 1e-6,
        f"matches the closed form, largest difference {np.abs(w - closed).max():.1e}")
    return pd.DataFrame(rows)
