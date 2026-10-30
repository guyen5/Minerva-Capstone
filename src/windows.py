"""Walk-forward training windows on the daily table.

At each decision date t (a month-end), a model is refit on daily rows whose
returns are already known on t and predicts the coming holding month. A row
dated d is usable at t only when its label_date is on or before t.

A window of M months ends at the newest usable row, and its last V months
form the validation slice. Rows used for fitting must have their returns
known before the validation slice starts; this leaves a gap of FORWARD_DAYS
trading days, so no fitted return overlaps the validation period. The
window sizes are listed in config.DAILY_WINDOWS.
"""

import pandas as pd

from src import config


def daily_windows(table, name, total_months=None, val_months=None):
    """Window calendar: one row per test month with the dates that bound its rows.

    `name` is a key of config.DAILY_WINDOWS, or any label when total_months
    and val_months are given.
    Columns: month, decision_date, train_start, fit_end, validation_start,
    train_end, train_months, validation_months, window.
    """
    if total_months is None:
        total_months, val_months = config.DAILY_WINDOWS[name]
    t = table.drop_duplicates("date")[["date", "label_date", "month_end"]].sort_values("date")
    dates = pd.DatetimeIndex(t["date"])
    label = pd.Series(t["label_date"].to_numpy(), index=dates)

    first = pd.Period(config.TEST_FIRST_MONTH, "M")
    last = pd.Period(config.TEST_LAST_MONTH, "M")
    decisions = [d for d in t.loc[t["month_end"], "date"]
                 if first <= d.to_period("M") + 1 <= last]

    rows = []
    for d in decisions:
        usable = dates[(label <= d).to_numpy()]
        train_end = usable[-1]
        start_cut = train_end - pd.DateOffset(months=total_months)
        val_cut = train_end - pd.DateOffset(months=val_months)
        train_start = usable[usable > start_cut][0]
        validation_start = usable[usable > val_cut][0]
        # Training rows whose targets are known before validation starts.
        fit = usable[(usable >= train_start)
                     & (label.reindex(usable) < validation_start).to_numpy()]
        rows.append({
            "month": str(d.to_period("M") + 1),
            "decision_date": d,
            "train_start": train_start,
            "fit_end": fit[-1],
            "validation_start": validation_start,
            "train_end": train_end,
            "train_months": total_months,
            "validation_months": val_months,
            "window": name,
        })
    return pd.DataFrame(rows)


def daily_rows_for(table, step, universe):
    """Fit, validation and test rows of one universe for one calendar row.

    fit: rows from train_start to fit_end, whose returns are known before
    validation starts. validation: rows from validation_start to train_end.
    test: the month-end rows at the decision date.
    """
    p = table[table["universe"] == universe]
    fit = p[(p["date"] >= step["train_start"]) & (p["date"] <= step["fit_end"])]
    val = p[(p["date"] >= step["validation_start"]) & (p["date"] <= step["train_end"])]
    test = p[p["date"] == step["decision_date"]]
    return fit, val, test


def check_daily_windows(calendar, table):
    """Validation report of a window calendar: no fit or validation row sees the test month."""
    label = table.drop_duplicates("date").set_index("date")["label_date"]
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    known = (label.reindex(calendar["train_end"]).to_numpy()
             <= calendar["decision_date"].to_numpy()).all()
    add("targets known at decision", bool(known),
        "the newest training row's target is known on the decision date")
    gap = (label.reindex(calendar["fit_end"]).to_numpy()
           < calendar["validation_start"].to_numpy()).all()
    add("gap before validation", bool(gap),
        "every fit target is known before the validation slice starts")
    order = ((calendar["train_start"] <= calendar["fit_end"])
             & (calendar["fit_end"] < calendar["validation_start"])
             & (calendar["validation_start"] <= calendar["train_end"])
             & (calendar["train_end"] < calendar["decision_date"])).all()
    add("dates in order", bool(order), "train start < fit end < validation < train end < decision")
    first, last = calendar["month"].iloc[0], calendar["month"].iloc[-1]
    add("test period", first == config.TEST_FIRST_MONTH and last == config.TEST_LAST_MONTH
        and calendar["month"].is_unique, f"{len(calendar)} test months, {first} to {last}")
    return pd.DataFrame(rows)
