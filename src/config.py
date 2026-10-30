"""Project settings shared by every module and notebook.

Tickers, sample dates, feature windows, experiment grids, the location of the
data folder and the names of the files saved in it are all defined here, so
each value is set in one place only.
"""

from pathlib import Path


# Asset universes from Wang and Hasuike (2026). ETF_A is the main universe and
# ETF_B the robustness universe; SPY and XLK belong to both.
ETF_A = ["SPY", "VTI", "XLK", "EFA", "EEM", "JPXN", "AGG", "DBC"]
ETF_B = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI"]
ALL_TICKERS = sorted(set(ETF_A) | set(ETF_B))

# Price history requested from Yahoo Finance. END is exclusive, so the last
# trading day is 30 September 2026. The year before 2010 gives every
# indicator a full history at the first monthly observation.
START = "2009-01-01"
END = "2026-10-01"
EXPECTED_FIRST_DAY = "2009-01-02"
YFINANCE_VERSION = "1.7.0"

# Data quality thresholds.
MAX_DAILY_MOVE = 0.25        # one-day price moves above this are flagged
FFILL_LIMIT = 1              # longest run of missing days that may be filled
STALE_RUN = 5                # this many identical closes in a row are flagged
ADJ_RATIO_TOL = 1e-4         # tolerance on Close / Adj Close changes
CROSS_CHECK_TOL = 0.001      # tolerance on monthly returns against a second source

# First month-end in the monthly table. Each row dated t holds the features
# known at t and the return earned from t to the next month-end.
PANEL_START = "2010-01-01"

# Indicator windows, in trading days.
RET_WINDOW = 21
SMA_SHORT = 10
SMA_LONG = 50
RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLL_WINDOW = 20

# Model inputs, in a fixed column order.
FEATURES = ["momentum", "trend", "price_bias", "rsi", "macd", "bollinger", "volume_bias"]

# Test period, given as the holding months whose returns are earned.
TEST_FIRST_MONTH = "2020-01"
TEST_LAST_MONTH = "2026-09"

# Forecast horizon of the daily training rows: each trading day is paired with
# the return over the next 21 trading days (about one month).
FORWARD_DAYS = 21

# Daily training windows, as name: (total months, validation months).
# "9+3" is the window of Wang and Hasuike (9 months to fit, 3 to tune).
DAILY_WINDOWS = {"9+3": (12, 3), "36": (36, 12), "60": (60, 12)}

# Covariance estimation: daily returns over COV_DAYS trading days, scaled to a
# monthly horizon by MONTH_DAYS. COV_RIDGE is added to the diagonal of the
# sample estimate, as in the paper.
COV_DAYS = 220
COV_RIDGE = 1e-6
MONTH_DAYS = 21

# Expected returns of classic Markowitz: the mean daily return over the last
# MU_DAYS trading days, scaled to a monthly horizon. MU_DAYS_GRID holds the
# windows compared when this setting was chosen.
MU_DAYS = 126
MU_DAYS_GRID = (21, 42, 63, 126, 220)

# Portfolio and experiment settings.
FEE = 0.005                                          # cost per unit of value traded
LAMBDA_GRID = (0.1, 1, 10, 20, 50)                   # risk aversion values in the paper
KAPPA_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)     # trading penalties in the optimizer
TAU_GRID = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)  # Black-Litterman prior uncertainty
BOOTSTRAP_REFITS = 100                               # refits per forecast for standard errors
BOOTSTRAP_BLOCK_MONTHS = 3                           # months per resampled block
SELECTION_PERIOD = ("2015-01", "2019-12")            # holding months used to choose settings
SECOND_TEST_PERIOD = ("2011-01", "2019-12")          # second, earlier test period

# Last holding month of the comparison window used against the paper. The
# paper's turnover figures are multiples of 1/70, which points to a test of
# about 70 months.
PAPER_LAST_MONTH = "2025-10"

# Ridge penalties for the pooled PTO forecasts. RIDGE_ALPHAS is the grid
# searched on the validation slice (0.1 to 1e6, 15 values); PTO_ALPHA_GRID
# holds the fixed penalties compared in the shrinkage analysis.
RIDGE_ALPHAS = tuple(float(f"{10 ** (k / 2):.3g}") for k in range(-2, 13))
PTO_ALPHA_GRID = (0.1, 10.0, 1000.0, 10000.0, 100000.0, 1000000.0)

# Location of the data folder in Google Drive, as mounted in Colab.
DRIVE_ROOT = Path("/content/drive/MyDrive/Capstone Main - Guyen")

# Names of the files saved in the data folder.
FILES = {
    "raw": "Raw Prices.csv",
    "raw_info": "Raw Prices Info.json",
    "daily": "Daily Prices.parquet",
    "quality": "Quality Report.csv",
    "filled": "Filled Prices.csv",
    "coverage": "Coverage.csv",
    "monthly": "Monthly Data.parquet",
    "monthly_info": "Monthly Data Info.json",
    "checks": "Monthly Data Checks.csv",
    "variables": "Variables.csv",
    "daily_table": "Daily Features.parquet",
    "daily_info": "Daily Features Info.json",
    "daily_windows": "Daily Training Windows.parquet",
    "daily_checks": "Daily Data Checks.csv",
    "covariances": "Covariances.parquet",
    "benchmark_months": "Benchmark Months.parquet",
    "benchmark_results": "Benchmark Results.csv",
    "window_diagnostic": "Markowitz Window Diagnostic.csv",
    "forecasts": "PTO Forecasts.parquet",
    "replication_months": "Replication Months.parquet",
    "replication_results": "Replication Results.csv",
    "replication_checks": "Replication Checks.csv",
    "penalty_diagnostic": "Ridge Penalty Diagnostic.csv",
}


def data_dirs(root=None):
    """Paths of the raw, processed, reports and archive folders.

    `root` defaults to DRIVE_ROOT; tests pass a temporary folder instead.
    The project folder itself must already exist: if it is missing, Drive is
    not mounted or belongs to a different account, and creating empty folders
    would lead to a fresh price download. The four data folders inside it are
    created if needed.
    """
    if root is None:
        root = DRIVE_ROOT
        if not root.exists():
            raise FileNotFoundError(
                f"{root} not found. Mount Google Drive with the account that "
                "holds the project folder.")
    root = Path(root)
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "reports": root / "data" / "reports",
        "archive": root / "data" / "archive",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
