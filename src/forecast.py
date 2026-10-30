"""Pooled ridge forecasts of next-month returns, used as the PTO expected returns.

At each decision date, one ridge regression is fitted on the daily rows of
the training window and predicts each ETF's return over the coming month:

- inputs: the seven features, standardized with the mean and spread of the
  rows being fitted, plus a 0/1 column per ticker (the paper's "ticker
  information"); one model is shared by all ETFs of a universe;
- target: the forward return over FORWARD_DAYS trading days;
- penalty: chosen from config.RIDGE_ALPHAS by squared error on the
  validation slice, fitting only on rows whose targets are known before
  validation starts; the chosen model is then refit on the whole window.

The ridge is solved in closed form with an unpenalized intercept, and
matches scikit-learn's Ridge.
"""

import numpy as np
import pandas as pd

from src import config, windows


def _design(rows, tickers, center=None, scale=None):
    """Feature matrix: standardized features, then one 0/1 column per ticker."""
    x = rows[config.FEATURES].to_numpy(dtype=float)
    if center is None:
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
    x = (x - center) / scale
    dummies = (rows["ticker"].to_numpy()[:, None] == np.asarray(tickers)[None, :]).astype(float)
    return np.hstack([x, dummies]), center, scale


def ridge_fit(x, y, alphas):
    """Ridge coefficients for each penalty: list of (intercept, beta)."""
    xm, ym = x.mean(axis=0), y.mean()
    xc, yc = x - xm, y - ym
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    uty = u.T @ yc
    out = []
    for a in alphas:
        beta = vt.T @ (s / (s ** 2 + a) * uty)
        out.append((ym - xm @ beta, beta))
    return out


def forecast_month(table, step, universe, tickers, alphas=None, target="fwd_return", month_end_only=False):
    """Forecasts for one decision date, with the penalty chosen on validation.

    With month_end_only=True, only the month-end rows of the window (one row
    per ETF and month) are used instead of every trading day.
    Returns (forecasts, chosen penalty, validation error, fit rows,
    validation rows).
    """
    alphas = config.RIDGE_ALPHAS if alphas is None else alphas
    fit, val, test = windows.daily_rows_for(table, step, universe)
    if month_end_only:
        fit, val = fit[fit["month_end"]], val[val["month_end"]]

    x_fit, c, s = _design(fit, tickers)
    x_val, _, _ = _design(val, tickers, c, s)
    models = ridge_fit(x_fit, fit[target].to_numpy(), alphas)
    mse = [np.mean((val[target].to_numpy() - b0 - x_val @ b) ** 2) for b0, b in models]
    best = int(np.argmin(mse))

    full = pd.concat([fit, val])
    x_full, c, s = _design(full, tickers)
    b0, b = ridge_fit(x_full, full[target].to_numpy(), [alphas[best]])[0]
    x_test, _, _ = _design(test, tickers, c, s)
    pred = pd.Series(b0 + x_test @ b, index=test["ticker"].to_numpy())
    return pred.reindex(tickers), alphas[best], float(mse[best]), len(fit), len(val)


def build_forecasts(table, calendar, universe, tickers, alphas=None, month_end_only=False):
    """Forecasts for every decision date in a window calendar (one universe).

    Returns a long table: date, month, universe, ticker, forecast, alpha,
    validation_mse, fit_rows, validation_rows.
    """
    rows = []
    for _, step in calendar.iterrows():
        pred, alpha, mse, n_fit, n_val = forecast_month(table, step, universe, tickers, alphas,
                                                        month_end_only=month_end_only)
        for t, f in pred.items():
            rows.append({"date": step["decision_date"], "month": step["month"],
                         "universe": universe, "ticker": t, "forecast": float(f),
                         "alpha": alpha, "validation_mse": mse,
                         "fit_rows": n_fit, "validation_rows": n_val})
    return pd.DataFrame(rows)


def check_forecasts(forecasts, table, calendar, universe, tickers, n_samples=3, seed=0):
    """Validation report: complete, no look-ahead, and equal to scikit-learn's Ridge."""
    from sklearn.linear_model import Ridge

    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    f = forecasts[forecasts["universe"] == universe]
    add("complete", len(f) == len(calendar) * len(tickers) and f["forecast"].notna().all(),
        f"{len(calendar)} months x {len(tickers)} ETFs, no missing forecast")

    label = table.drop_duplicates("date").set_index("date")["label_date"]
    ok = all(label.loc[s["train_start"]:s["train_end"]].max() <= s["decision_date"]
             for _, s in calendar.iterrows())
    add("no look-ahead", bool(ok), "every training target is known on its decision date")

    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in rng.choice(len(calendar), min(n_samples, len(calendar)), replace=False):
        step = calendar.iloc[int(i)]
        fit, val, test = windows.daily_rows_for(table, step, universe)
        full = pd.concat([fit, val])
        alpha = f[f["date"] == step["decision_date"]]["alpha"].iloc[0]
        x, c, s = _design(full, tickers)
        model = Ridge(alpha=alpha).fit(x, full["fwd_return"].to_numpy())
        xt, _, _ = _design(test, tickers, c, s)
        sk = pd.Series(model.predict(xt), index=test["ticker"].to_numpy()).reindex(tickers)
        ours = f[f["date"] == step["decision_date"]].set_index("ticker")["forecast"].reindex(tickers)
        worst = max(worst, float((sk - ours).abs().max()))
    add("matches scikit-learn", worst < 1e-8,
        f"{n_samples} random months refit with sklearn Ridge, largest difference {worst:.1e}")
    return pd.DataFrame(rows)


def forecast_quality(forecasts, panel):
    """How well forecasts match the realized holding-month returns.

    Per universe: out-of-sample R^2 against a forecast of zero, the mean
    monthly rank correlation across ETFs (information coefficient), the share
    of months whose top-forecast ETF beat the universe average, and the
    cross-sectional spread of forecasts relative to that of realized returns.
    """
    realized = panel[["date", "universe", "ticker", "next_return"]]
    m = forecasts.merge(realized, on=["date", "universe", "ticker"])
    out = []
    for u, g in m.groupby("universe"):
        r2 = 1 - ((g["next_return"] - g["forecast"]) ** 2).sum() / (g["next_return"] ** 2).sum()
        ic = g.groupby("date").apply(lambda x: x["forecast"].rank().corr(x["next_return"].rank()))
        top = g.loc[g.groupby("date")["forecast"].idxmax()].set_index("date")["next_return"]
        hit = (top > g.groupby("date")["next_return"].mean()).mean()
        spread = g.groupby("date")["forecast"].std().mean() / g.groupby("date")["next_return"].std().mean()
        out.append({"universe": u, "oos_r2": r2, "mean_ic": ic.mean(), "ic_t_stat":
                    ic.mean() / ic.std() * np.sqrt(len(ic)), "top_pick_beats_average": hit,
                    "spread_ratio": spread})
    return pd.DataFrame(out).set_index("universe")
