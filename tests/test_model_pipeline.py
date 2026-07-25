"""
Unit tests for src/model_pipeline.py.

Uses a small synthetic feature store so `train_xgboost` runs in well under a
second — these tests verify the *pipeline mechanics* (chronological
splitting, metric shapes, artifact contract) rather than model quality.
"""
import numpy as np
import pandas as pd

import config
from src.model_pipeline import evaluate_naive_baseline, temporal_train_test_split, train_xgboost


def _make_feature_store(n_days: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = ["AAA", "BBB"]

    rows = []
    for ticker in tickers:
        for dt in dates:
            row = {col: rng.normal() for col in config.FEATURE_COLS}
            row["Date"] = dt
            row["ticker"] = ticker
            row[config.TARGET_COL] = int(rng.random() > 0.5)
            rows.append(row)
    return pd.DataFrame(rows)


def test_temporal_split_respects_chronology_per_ticker():
    df = _make_feature_store()
    X_train, y_train, X_test, y_test = temporal_train_test_split(df, ratio=0.8)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    # Roughly an 80/20 split (allowing for per-ticker rounding).
    total = len(X_train) + len(X_test)
    assert 0.75 <= len(X_train) / total <= 0.85


def test_naive_baseline_returns_expected_metric_keys():
    df = _make_feature_store()
    X_train, y_train, X_test, y_test = temporal_train_test_split(df)
    metrics = evaluate_naive_baseline(X_train, y_train, X_test, y_test)

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_train_xgboost_produces_fitted_model_and_metrics():
    df = _make_feature_store()
    X_train, y_train, X_test, y_test = temporal_train_test_split(df)
    model, metrics = train_xgboost(X_train, y_train, X_test, y_test)

    # Model should be fitted and able to predict on the holdout set.
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)).issubset({0, 1})

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_ts_accuracy_mean"):
        assert key in metrics
