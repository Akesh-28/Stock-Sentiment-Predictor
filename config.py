"""
config.py
=========
Single source of truth for the entire project: paths, ticker universe,
date ranges, model hyperparameters, and feature column definitions.

Nothing in src/ or app.py should hard-code a ticker list, a file path, or a
hyperparameter — everything is imported from here. This is what lets the
same pipeline run identically from `main.py`, `app.py`, and the test suite.
"""
from __future__ import annotations

import datetime
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

# Created on import so every entrypoint (main.py, app.py, pytest) can assume
# these directories exist without duplicating mkdir calls everywhere.
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_STORE_PATH = DATA_DIR / "processed_features.parquet"
MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"
METRICS_PATH = DATA_DIR / "metrics.json"
FEATURE_IMPORTANCE_PLOT_PATH = DATA_DIR / "feature_importance.png"

# =========================================================
# UNIVERSE & DATE RANGE
# =========================================================
TICKERS: list[str] = ["AAPL", "NVDA", "MSFT"]

START_DATE: str = "2023-01-01"
# Default end date is "today" so the pipeline stays reproducible when run on
# a fixed historical START_DATE but never goes stale. Override via CLI flag.
END_DATE: str = datetime.date.today().isoformat()

# =========================================================
# SENTIMENT MODEL (FinBERT)
# =========================================================
FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
SENTIMENT_BATCH_SIZE: int = 32

# Mock headline bank used when no live news API key is configured.
# Swap `get_headlines_for_ticker` in src/data_pipeline.py for a real
# NewsAPI / Kaggle ingestion call to replace this in production.
MOCK_HEADLINES: dict[str, list[str]] = {
    "AAPL": [
        "Apple reports record quarterly revenue driven by iPhone sales",
        "Tech stocks face regulatory pressure",
    ],
    "NVDA": [
        "Nvidia announces new AI chip architecture exceeding expectations",
        "Semiconductor supply constraints impact growth",
    ],
    "MSFT": [
        "Microsoft expands cloud partnership and AI integration",
        "Quarterly operating margins hold steady",
    ],
}
DEFAULT_HEADLINE = "Company releases annual performance update"

# =========================================================
# TECHNICAL INDICATOR WINDOWS
# =========================================================
RSI_WINDOW: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
SMA_WINDOW: int = 20
ATR_WINDOW: int = 14
VOLATILITY_WINDOW: int = 20

# =========================================================
# MODEL TRAINING
# =========================================================
TRAIN_TEST_RATIO: float = 0.80
CV_SPLITS: int = 5
RANDOM_STATE: int = 42

XGB_PARAMS: dict = dict(
    n_estimators=100,
    max_depth=3,          # shallow trees to resist noisy financial signal
    learning_rate=0.03,   # slow learning rate for stability
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1.0,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)

# Feature/target schema shared by data_pipeline (producer) and
# model_pipeline (consumer) — change it once, both sides stay in sync.
FEATURE_COLS: list[str] = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_diff",
    "sma_ratio_20",
    "atr_14",
    "log_ret",
    "volatility_20",
    "sent_pos_lag1",
    "sent_neg_lag1",
    "sent_neu_lag1",
]
TARGET_COL: str = "target"

# =========================================================
# STREAMLIT DASHBOARD
# =========================================================
DASHBOARD_TICKER_OPTIONS: list[str] = TICKERS
DEFAULT_LOOKBACK_DAYS: int = 180
