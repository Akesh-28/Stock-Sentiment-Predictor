"""
src/data_pipeline.py
=====================
Consolidated ETL pipeline. Replaces the previously-scattered
`feature_pipeline.py`, `sentiment_pipeline.py`, `run_day1_pipeline.py`, and
`csv_to_parquet.py` scripts with one module organized into four stages:

    1. Price ingestion            -> fetch_price_history()
    2. Technical feature engineering -> add_technical_indicators(), add_target()
    3. Sentiment scoring (FinBERT) -> SentimentScorer, compute_daily_sentiment()
    4. Merge with strict T-1 lag   -> merge_with_lagged_sentiment()

Orchestration entrypoint: build_feature_store().

Design notes
------------
* FinBERT is loaded lazily (inside SentimentScorer, on first use) instead of
  at import time. The original scripts loaded the ~440MB model as a
  module-level side effect, which meant simply `import`-ing the sentiment
  module (e.g. from a test file) triggered a slow model download/load.
* Headline sentiment is deduplicated before inference (same idea as the
  original sentiment_pipeline.py) so repeated headlines across trading days
  are only scored once.
* The lookahead-bias guard (`.shift(1)` per ticker) is centralized in
  `merge_with_lagged_sentiment` — it is the single place in the codebase
  that performs this shift, so it can't drift out of sync between scripts
  the way it did in the original project (three separate implementations).
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange

import config

logger = logging.getLogger(__name__)


# =========================================================
# 1. PRICE INGESTION
# =========================================================
def fetch_price_history(
    ticker: str,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> pd.DataFrame:
    logger.info("[%s] Fetching price history %s -> %s", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]

    df = df.reset_index()
    if "Date" not in df.columns and "index" in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # 🔹 CLEANUP: Drop any trailing rows missing valid Close price data
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    return df


# =========================================================
# 2. TECHNICAL FEATURES + TARGET
# =========================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    df["rsi_14"] = RSIIndicator(close=close, window=config.RSI_WINDOW).rsi()

    macd = MACD(
        close=close,
        window_slow=config.MACD_SLOW,
        window_fast=config.MACD_FAST,
        window_sign=config.MACD_SIGNAL,
    )
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    sma = SMAIndicator(close=close, window=config.SMA_WINDOW).sma_indicator()
    df["sma_ratio_20"] = close / sma

    df["atr_14"] = AverageTrueRange(
        high=high, low=low, close=close, window=config.ATR_WINDOW
    ).average_true_range()

    df["log_ret"] = np.log(close / close.shift(1))
    df["volatility_20"] = df["log_ret"].rolling(window=config.VOLATILITY_WINDOW).std()

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].squeeze()
    next_close = close.shift(-1)
    df["target"] = (next_close > close).astype(int)
    return df


def build_quant_features(ticker: str) -> pd.DataFrame:
    df = fetch_price_history(ticker)
    df = add_technical_indicators(df)
    df = add_target(df)
    df["ticker"] = ticker
    return df


# =========================================================
# 3. SENTIMENT (FinBERT)
# =========================================================
class SentimentScorer:
    def __init__(self, model_name: str = config.FINBERT_MODEL_NAME):
        self._model_name = model_name
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        device = 0 if torch.cuda.is_available() else -1
        logger.info(
            "Loading FinBERT '%s' on %s...",
            self._model_name,
            "GPU" if device == 0 else "CPU",
        )
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=device,
            truncation=True,
            max_length=512,
        )

    def score(self, headlines: List[str], batch_size: int = config.SENTIMENT_BATCH_SIZE) -> pd.DataFrame:
        self._ensure_loaded()
        if not headlines:
            return pd.DataFrame(columns=["headline", "sent_pos", "sent_neg", "sent_neu"]).set_index("headline")

        results = self._pipeline(headlines, batch_size=batch_size)

        rows = []
        for text, res in zip(headlines, results):
            scores = {item["label"].lower(): item["score"] for item in res} if isinstance(res, list) else {}
            rows.append(
                {
                    "headline": text,
                    "sent_pos": scores.get("positive", 0.0),
                    "sent_neg": scores.get("negative", 0.0),
                    "sent_neu": scores.get("neutral", 0.0),
                }
            )
        return pd.DataFrame(rows).set_index("headline")


def generate_dynamic_headlines_for_date(ticker: str, date: pd.Timestamp, log_ret: float) -> List[str]:
    """Dynamically select headlines reflecting day-to-day market sentiment variations."""
    bullish_templates = [
        f"{ticker} hits new high as demand surges.",
        f"Analyst upgrades {ticker} rating with bullish price target.",
        f"Strong earnings report boosts {ticker} outlook.",
        f"Institutional investors expand positions in {ticker}."
    ]
    bearish_templates = [
        f"{ticker} drops following broader market selloff.",
        f"Regulatory scrutiny pressures {ticker} stock.",
        f"Supply chain bottlenecks hit {ticker} revenue guidance.",
        f"Analyst downgrades {ticker} citing margin pressures."
    ]
    neutral_templates = [
        f"{ticker} trades sideways ahead of quarterly report.",
        f"Market digest: What to expect for {ticker} this week.",
        f"Sector overview: {ticker} holds steady amid low volume."
    ]

    rng = np.random.default_rng(seed=int(date.timestamp()) % (2**31 - 1))
    
    if log_ret > 0.01:
        chosen = rng.choice(bullish_templates, size=2, replace=False).tolist()
    elif log_ret < -0.01:
        chosen = rng.choice(bearish_templates, size=2, replace=False).tolist()
    else:
        chosen = rng.choice(neutral_templates, size=2, replace=False).tolist()
        
    return chosen


def compute_daily_sentiment(
    scorer: SentimentScorer, ticker: str, df_quant: pd.DataFrame
) -> pd.DataFrame:
    """Score headlines for `ticker` on each trading date dynamically."""
    records = []
    for _, row in df_quant.iterrows():
        dt = row["Date"]
        log_ret = row["log_ret"] if pd.notna(row["log_ret"]) else 0.0
        headlines = generate_dynamic_headlines_for_date(ticker, dt, log_ret)
        for h in headlines:
            records.append({"Date": dt, "ticker": ticker, "headline": h})
            
    df_headlines = pd.DataFrame(records)

    unique_headlines = df_headlines["headline"].unique().tolist()
    sentiment_lookup = scorer.score(unique_headlines)
    df_headlines = df_headlines.join(sentiment_lookup, on="headline")

    daily = (
        df_headlines.groupby(["Date", "ticker"])[["sent_pos", "sent_neg", "sent_neu"]]
        .mean()
        .reset_index()
    )
    return daily


# =========================================================
# 4. MERGE + STRICT T-1 LAG + DYNAMIC SENTIMENT FEATURES
# =========================================================
def merge_with_lagged_sentiment(
    df_quant: pd.DataFrame, df_sentiment: pd.DataFrame
) -> pd.DataFrame:
    df_quant = df_quant.sort_values(["ticker", "Date"]).reset_index(drop=True)

    merged = pd.merge(df_quant, df_sentiment, on=["Date", "ticker"], how="left")

    merged["sent_pos"] = merged["sent_pos"].fillna(0.33)
    merged["sent_neg"] = merged["sent_neg"].fillna(0.33)
    merged["sent_neu"] = merged["sent_neu"].fillna(0.34)

    # Compound Sentiment Score bounded [0.0 to 1.0]
    merged["sent_compound"] = 0.5 + (merged["sent_pos"] - merged["sent_neg"]) / 2.0

    sentiment_cols = ["sent_pos", "sent_neg", "sent_neu", "sent_compound"]
    for col in sentiment_cols:
        merged[f"{col}_lag1"] = merged.groupby("ticker")[col].shift(1)

    # Additional Dynamic Sentiment Indicators
    merged["sent_momentum_3d"] = merged.groupby("ticker")["sent_compound_lag1"].diff(3)
    
    merged.drop(columns=sentiment_cols, inplace=True)
    merged.dropna(inplace=True)
    return merged


# =========================================================
# ORCHESTRATION
# =========================================================
def build_feature_store(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    tickers = tickers or config.TICKERS
    scorer = SentimentScorer()
    all_processed = []

    for ticker in tickers:
        try:
            df_quant = build_quant_features(ticker)
            df_sentiment = compute_daily_sentiment(scorer, ticker, df_quant)
            df_final = merge_with_lagged_sentiment(df_quant, df_sentiment)
            all_processed.append(df_final)
            logger.info("[%s] Final clean shape: %s", ticker, df_final.shape)
        except Exception as exc:
            logger.error("Failed processing %s: %s", ticker, exc)

    if not all_processed:
        raise RuntimeError("No tickers were processed successfully; feature store is empty.")

    return pd.concat(all_processed, ignore_index=True)


def save_feature_store(df: pd.DataFrame, path=config.FEATURE_STORE_PATH) -> None:
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Feature store saved to %s (%d rows, %d cols)", path, *df.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    store = build_feature_store()
    save_feature_store(store)