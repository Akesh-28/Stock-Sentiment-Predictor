"""
Unit tests for src/data_pipeline.py.

These tests operate entirely on synthetic in-memory DataFrames so they run
fast, offline, and deterministically in CI — no yfinance/network/FinBERT
calls involved. Network-dependent behavior (fetch_price_history) and
model-dependent behavior (SentimentScorer) are intentionally out of scope
here; they're better covered by a manual smoke test (see README "Testing &
Verification").
"""
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import add_target, add_technical_indicators, merge_with_lagged_sentiment


def _make_price_df(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, size=n_days))
    high = close + rng.uniform(0, 2, size=n_days)
    low = close - rng.uniform(0, 2, size=n_days)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n_days),
        }
    )


def test_add_technical_indicators_creates_expected_columns():
    df = add_technical_indicators(_make_price_df())
    expected = {"rsi_14", "macd_line", "macd_signal", "macd_diff", "sma_ratio_20", "atr_14", "log_ret", "volatility_20"}
    assert expected.issubset(df.columns)


def test_add_technical_indicators_rsi_is_bounded():
    df = add_technical_indicators(_make_price_df())
    rsi = df["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_add_target_is_binary_and_shifted_correctly():
    df = _make_price_df(n_days=10)
    df["Close"] = [10, 12, 11, 15, 14, 14, 20, 5, 5, 5]
    df = add_target(df)

    assert set(df["target"].dropna().unique()).issubset({0, 1})
    # Close goes 10 -> 12 (up), so target at row 0 must be 1.
    assert df.loc[0, "target"] == 1
    # Close goes 12 -> 11 (down), so target at row 1 must be 0.
    assert df.loc[1, "target"] == 0
    # Last row has no "next day" -> next_close is NaN -> target compares False -> 0
    assert df.loc[9, "target"] == 0


def test_merge_with_lagged_sentiment_prevents_lookahead():
    df_quant = add_target(add_technical_indicators(_make_price_df(n_days=40)))
    df_quant["ticker"] = "TEST"
    df_quant = df_quant.dropna(subset=["rsi_14"]).reset_index(drop=True)

    df_sentiment = pd.DataFrame(
        {
            "Date": df_quant["Date"],
            "ticker": "TEST",
            "sent_pos": np.linspace(0, 1, len(df_quant)),
            "sent_neg": np.linspace(1, 0, len(df_quant)),
            "sent_neu": 0.0,
        }
    )

    merged = merge_with_lagged_sentiment(df_quant, df_sentiment)

    # The lag-1 sentiment on any given row must equal the raw sentiment
    # value one row earlier in df_sentiment, never the same-day value.
    raw_lookup = df_sentiment.set_index("Date")["sent_pos"]
    sample_row = merged.iloc[5]
    prior_date = df_quant["Date"].iloc[df_quant.index[df_quant["Date"] == sample_row["Date"]][0] - 1]
    assert sample_row["sent_pos_lag1"] == pytest.approx(raw_lookup[prior_date])

    # No unlagged sentiment columns should remain (they'd invite lookahead misuse).
    assert "sent_pos" not in merged.columns
    assert "sent_pos_lag1" in merged.columns


def test_merge_with_lagged_sentiment_no_nulls_in_output():
    df_quant = add_target(add_technical_indicators(_make_price_df(n_days=40)))
    df_quant["ticker"] = "TEST"
    df_sentiment = pd.DataFrame(
        {
            "Date": df_quant["Date"],
            "ticker": "TEST",
            "sent_pos": 0.5,
            "sent_neg": 0.3,
            "sent_neu": 0.2,
        }
    )
    merged = merge_with_lagged_sentiment(df_quant, df_sentiment)
    assert merged.isnull().sum().sum() == 0
