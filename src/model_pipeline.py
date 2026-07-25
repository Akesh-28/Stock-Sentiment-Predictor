"""
src/model_pipeline.py
======================
Consolidated model training pipeline. Replaces `model_training.py`.

Stages:
    1. load_feature_store()          -> read + validate the parquet feature store
    2. temporal_train_test_split()   -> chronological, per-ticker split (no shuffling)
    3. evaluate_naive_baseline()     -> majority-class DummyClassifier benchmark
    4. train_xgboost()               -> TimeSeriesSplit CV + final fit + holdout eval
    5. compute_feature_importance()  -> gain-based importances + bar chart
    6. save_artifacts()              -> model.joblib + metrics.json + importance plot

Design notes
------------
* Model is persisted with `joblib` (not the XGBoost-native `.json` format
  the original script used) so `app.py` can load it with one generic
  `joblib.load()` call regardless of which sklearn-compatible estimator
  produced it — this keeps the dashboard decoupled from the model's internal
  serialization format.
* All hyperparameters and the feature/target schema live in config.py, so
  training and inference can never silently drift out of sync.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

import config

logger = logging.getLogger(__name__)


# =========================================================
# 1. LOAD + VALIDATE
# =========================================================
def load_feature_store(path: Path = config.FEATURE_STORE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature store not found at {path}. Run `python main.py --stage data` first."
        )

    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

    null_counts = df[config.FEATURE_COLS + [config.TARGET_COL]].isnull().sum().sum()
    if null_counts != 0:
        raise ValueError(f"Found {null_counts} unexpected null values in the feature store.")

    logger.info(
        "Loaded feature store: %s rows, date range %s -> %s",
        len(df), df["Date"].min().date(), df["Date"].max().date(),
    )
    return df


# =========================================================
# 2. CHRONOLOGICAL SPLIT
# =========================================================
def temporal_train_test_split(
    df: pd.DataFrame, ratio: float = config.TRAIN_TEST_RATIO
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split each ticker's series chronologically (no shuffling) to avoid
    temporal leakage, then recombine.
    """
    train_parts, test_parts = [], []
    for _, group in df.groupby("ticker"):
        group = group.sort_values("Date")
        split_idx = int(len(group) * ratio)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts).sort_values("Date").reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values("Date").reset_index(drop=True)

    X_train, y_train = train_df[config.FEATURE_COLS], train_df[config.TARGET_COL]
    X_test, y_test = test_df[config.FEATURE_COLS], test_df[config.TARGET_COL]

    logger.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))
    return X_train, y_train, X_test, y_test


# =========================================================
# 3. NAIVE BASELINE
# =========================================================
def evaluate_naive_baseline(X_train, y_train, X_test, y_test) -> dict:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)
    y_prob = dummy.predict_proba(X_test)[:, 1]

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }
    logger.info("Naive baseline -> accuracy=%.4f roc_auc=%.4f", metrics["accuracy"], metrics["roc_auc"])
    return metrics


# =========================================================
# 4. XGBOOST TRAINING
# =========================================================
def train_xgboost(X_train, y_train, X_test, y_test) -> Tuple[XGBClassifier, dict]:
    model = XGBClassifier(**config.XGB_PARAMS)

    tss = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    cv_scores = []
    for train_idx, val_idx in tss.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        cv_scores.append(accuracy_score(y_train.iloc[val_idx], preds))
    logger.info("%d-fold TimeSeriesSplit mean accuracy: %.4f", config.CV_SPLITS, np.mean(cv_scores))

    # Final fit on the full training set, evaluated on the untouched holdout.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "cv_ts_accuracy_mean": float(np.mean(cv_scores)),
    }
    logger.info(
        "XGBoost holdout -> accuracy=%.4f precision=%.4f recall=%.4f roc_auc=%.4f",
        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["roc_auc"],
    )
    return model, metrics


# =========================================================
# 5. FEATURE IMPORTANCE
# =========================================================
def compute_feature_importance(model: XGBClassifier, save_plot: bool = True) -> dict:
    importances = dict(zip(config.FEATURE_COLS, [float(v) for v in model.feature_importances_]))
    ranked = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.barh(list(reversed(ranked.keys())), list(reversed(ranked.values())), color="skyblue")
        plt.title("XGBoost Feature Importance (Gain)")
        plt.xlabel("Relative Gain Score")
        plt.tight_layout()
        plt.savefig(config.FEATURE_IMPORTANCE_PLOT_PATH)
        plt.close()
        logger.info("Saved feature importance plot to %s", config.FEATURE_IMPORTANCE_PLOT_PATH)

    return ranked


# =========================================================
# 6. PERSIST ARTIFACTS
# =========================================================
def save_artifacts(model: XGBClassifier, report: dict) -> None:
    joblib.dump(model, config.MODEL_PATH)
    logger.info("Saved trained model to %s", config.MODEL_PATH)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(report, f, indent=4)
    logger.info("Saved metrics report to %s", config.METRICS_PATH)


# =========================================================
# ORCHESTRATION
# =========================================================
def run_training_pipeline() -> dict:
    df = load_feature_store()
    X_train, y_train, X_test, y_test = temporal_train_test_split(df)

    naive_metrics = evaluate_naive_baseline(X_train, y_train, X_test, y_test)
    model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    importances = compute_feature_importance(model)

    report = {
        "naive_baseline": naive_metrics,
        "xgboost_model": xgb_metrics,
        "feature_importance_gain": importances,
    }
    save_artifacts(model, report)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Naive Baseline Accuracy: {naive_metrics['accuracy']:.4f}")
    print(f"XGBoost Test Accuracy:   {xgb_metrics['accuracy']:.4f}")
    print(f"XGBoost Test ROC-AUC:    {xgb_metrics['roc_auc']:.4f}")
    print("-" * 50)
    print("TOP 5 FEATURES BY GAIN:")
    for feat, score in list(importances.items())[:5]:
        print(f"  - {feat:15s}: {score:.5f}")
    print("=" * 50)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_training_pipeline()
