"""
app.py
======
Streamlit dashboard entrypoint. Run with:

    streamlit run app.py

Two operating modes, chosen automatically:

  * LIVE MODE  - if `data/models/xgboost_model.joblib` exists (i.e. you've
    run `python main.py --stage train`), the dashboard loads the real
    trained XGBoost model and scores the latest available feature row with
    it. Sentiment inputs are read from the persisted feature store (the
    output of the real FinBERT pipeline) rather than re-running FinBERT on
    every page load, which would be far too slow for an interactive app.

  * DEMO MODE  - if no trained artifact is found yet, the dashboard falls
    back to a lightweight synthetic sentiment + rule-based signal generator
    so the UI is still fully explorable before you've run the training
    pipeline. This preserves the original prototype's standalone demo
    behavior.

Technical indicators (RSI/MACD/SMA/ATR/volatility) are always computed live
from real price data via `src.data_pipeline`, in both modes.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from src.data_pipeline import add_technical_indicators, fetch_price_history

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Market Sentiment & Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# CACHED DATA LAYER
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_and_indicators(ticker: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    df = fetch_price_history(ticker, start=str(start), end=str(end))
    df = add_technical_indicators(df)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_feature_store_slice(ticker: str) -> pd.DataFrame | None:
    if not config.FEATURE_STORE_PATH.exists():
        return None
    store = pd.read_parquet(config.FEATURE_STORE_PATH)
    store["Date"] = pd.to_datetime(store["Date"])
    return store[store["ticker"] == ticker].sort_values("Date")


@st.cache_resource(show_spinner="Loading trained model...")
def load_trained_model():
    if not config.MODEL_PATH.exists():
        return None
    import joblib

    return joblib.load(config.MODEL_PATH)


@st.cache_data(ttl=1800, show_spinner=False)
def generate_demo_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Generates dynamic time-varying synthetic sentiment tied to price returns."""
    df = df.copy()
    returns = df["Close"].pct_change().fillna(0)
    
    # Base sentiment around 0.50 with mean-reverting random walk + price return correlation
    noise = np.random.default_rng(42).normal(0, 0.08, size=len(df))
    sentiment_raw = 0.50 + (returns * 2.5) + noise
    sentiment_score = np.clip(sentiment_raw, 0.05, 0.95)
    
    df["Sentiment_Score"] = sentiment_score
    df["Pos_Sentiment"] = np.clip(df["Sentiment_Score"] * 0.7, 0, 1)
    df["Neu_Sentiment"] = 0.20
    df["Neg_Sentiment"] = np.clip(1.0 - (df["Pos_Sentiment"] + df["Neu_Sentiment"]), 0, 1)
    return df


def demo_predict_next_day(recent_prices: pd.Series, recent_sentiment: float) -> tuple[str, float]:
    price_return = (recent_prices.iloc[-1] - recent_prices.iloc[-5]) / recent_prices.iloc[-5]
    signal_score = (price_return * 0.4) + (recent_sentiment * 0.6)
    confidence = float(np.clip(0.50 + abs(signal_score - 0.5) * 0.8, 0.52, 0.94))
    direction = "UP" if signal_score >= 0.45 else "DOWN"
    return direction, confidence


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.selectbox("Select Asset Ticker", options=config.DASHBOARD_TICKER_OPTIONS, index=0)

col_s1, col_s2 = st.sidebar.columns(2)
start_d = col_s1.date_input(
    "Start Date", value=datetime.date.today() - datetime.timedelta(days=config.DEFAULT_LOOKBACK_DAYS)
)
end_d = col_s2.date_input("End Date", value=datetime.date.today())

rolling_window = st.sidebar.slider("Sentiment Moving Average (Days)", min_value=1, max_value=14, value=5)

st.sidebar.divider()

model = load_trained_model()
if model is not None:
    st.sidebar.success("🟢 Live Mode — trained XGBoost model loaded")
else:
    st.sidebar.warning("🟡 Demo Mode — no trained model found yet")
    st.sidebar.caption("Run `python main.py --stage all` to train a real model and switch to Live Mode.")

# ==========================================
# DATA LOADING
# ==========================================
with st.spinner("Fetching market data and computing indicators..."):
    price_df = load_price_and_indicators(ticker, start_d, end_d)

if price_df.empty:
    st.error(f"No price data retrieved for **{ticker}**. Please adjust your date range or ticker.")
    st.stop()

feature_store_slice = load_feature_store_slice(ticker)
using_live_sentiment = feature_store_slice is not None and not feature_store_slice.empty

if using_live_sentiment:
    processed_df = pd.merge(
        price_df,
        feature_store_slice[["Date", "sent_pos_lag1", "sent_neg_lag1", "sent_neu_lag1", "sent_compound_lag1"]],
        on="Date",
        how="left",
    )
    # Correct fillna using ffill() function directly to fix DeprecationWarning
    processed_df[["sent_pos_lag1", "sent_neg_lag1", "sent_neu_lag1", "sent_compound_lag1"]] = (
        processed_df[["sent_pos_lag1", "sent_neg_lag1", "sent_neu_lag1", "sent_compound_lag1"]].ffill().bfill()
    )
    
    processed_df["Sentiment_Score"] = processed_df["sent_compound_lag1"]
    processed_df["Pos_Sentiment"] = processed_df["sent_pos_lag1"]
    processed_df["Neg_Sentiment"] = processed_df["sent_neg_lag1"]
    processed_df["Neu_Sentiment"] = processed_df["sent_neu_lag1"]
else:
    processed_df = generate_demo_sentiment(price_df)

processed_df["Sentiment_MA"] = processed_df["Sentiment_Score"].rolling(window=rolling_window, min_periods=1).mean()

# ==========================================
# INFERENCE
# ==========================================
latest_close = float(processed_df["Close"].iloc[-1])
prev_close = float(processed_df["Close"].iloc[-2]) if len(processed_df) > 1 else latest_close
price_change_pct = ((latest_close - prev_close) / prev_close) * 100
latest_sentiment = float(processed_df["Sentiment_Score"].iloc[-1])

if model is not None and using_live_sentiment:
    latest_row = processed_df.dropna(subset=config.FEATURE_COLS).iloc[[-1]]
    if not latest_row.empty:
        X_latest = latest_row[config.FEATURE_COLS]
        proba_up = float(model.predict_proba(X_latest)[0, 1])
        direction = "UP" if proba_up >= 0.5 else "DOWN"
        confidence = proba_up if direction == "UP" else 1 - proba_up
    else:
        direction, confidence = demo_predict_next_day(processed_df["Close"], latest_sentiment)
else:
    direction, confidence = demo_predict_next_day(processed_df["Close"], latest_sentiment)

# ==========================================
# MAIN DASHBOARD UI
# ==========================================
st.title("📈 Stock Price & Sentiment Intelligence Platform")
st.markdown(f"Sentiment-augmented predictive signals for **{ticker}**.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close Price", f"${latest_close:.2f}", delta=f"{price_change_pct:+.2f}%")
c2.metric(
    "Avg Sentiment (lag-1)" if using_live_sentiment else "Avg Sentiment (demo)",
    f"{latest_sentiment:.2f} / 1.0",
    delta="Positive" if latest_sentiment > 0.55 else "Neutral/Negative",
    delta_color="normal" if latest_sentiment > 0.55 else "inverse",
)
c3.metric(
    "Next-Day Predicted Signal",
    f"FORWARD: {direction}",
    delta=f"Confidence: {confidence * 100:.1f}%",
    delta_color="normal" if direction == "UP" else "inverse",
)
c4.metric("Total Trade Bars Analyzed", f"{len(processed_df)} Days", delta="Data Fresh")

st.divider()

tab_analytics, tab_breakdown, tab_raw, tab_perf = st.tabs(
    ["📊 Price vs. Sentiment Trend", "📰 Sentiment Breakdown", "📋 Data Explorer", "🧪 Model Performance"]
)

# ------------------------------------------
# TAB 1: DUAL-AXIS CHART
# ------------------------------------------
with tab_analytics:
    st.subheader("Asset Price Trend vs. Rolling Sentiment Indicator")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{ticker} Closing Price", "Rolling Sentiment Score (0.0 to 1.0)"),
        row_heights=[0.65, 0.35],
    )
    fig.add_trace(
        go.Scatter(x=processed_df["Date"], y=processed_df["Close"], name="Close Price ($)",
                    line=dict(color="#1f77b4", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=processed_df["Date"], y=processed_df["Sentiment_MA"],
                    name=f"{rolling_window}-Day Sentiment MA", line=dict(color="#2ca02c", width=2, dash="dash"),
                    fill="tozeroy", fillcolor="rgba(44, 160, 44, 0.1)"),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_layout(
        height=550, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Index", range=[0, 1], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 2: SENTIMENT BREAKDOWN
# ------------------------------------------
with tab_breakdown:
    st.subheader("Distribution of News Sentiment")
    fig_sent = go.Figure()
    fig_sent.add_trace(go.Bar(x=processed_df["Date"], y=processed_df["Pos_Sentiment"], name="Positive", marker_color="#2ca02c"))
    fig_sent.add_trace(go.Bar(x=processed_df["Date"], y=processed_df["Neu_Sentiment"], name="Neutral", marker_color="#7f7f7f"))
    fig_sent.add_trace(go.Bar(x=processed_df["Date"], y=processed_df["Neg_Sentiment"], name="Negative", marker_color="#d62728"))
    fig_sent.update_layout(barmode="stack", height=400, xaxis_title="Date", yaxis_title="Proportion",
                            legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_sent, use_container_width=True)

# ------------------------------------------
# TAB 3: RAW DATA EXPLORER
# ------------------------------------------
with tab_raw:
    st.subheader("Aligned Model Dataset")
    display_cols = ["Date", "Close", "Volume", "Sentiment_Score", "Sentiment_MA"]
    st.dataframe(
        processed_df[display_cols].sort_values(by="Date", ascending=False),
        use_container_width=True,
    )

# ------------------------------------------
# TAB 4: MODEL PERFORMANCE
# ------------------------------------------
with tab_perf:
    st.subheader("Trained Model Evaluation")
    if config.METRICS_PATH.exists():
        import json

        with open(config.METRICS_PATH) as f:
            report = json.load(f)

        m1, m2 = st.columns(2)
        m1.write("**Naive Baseline**")
        m1.json(report["naive_baseline"])
        m2.write("**XGBoost Model**")
        m2.json(report["xgboost_model"])

        if config.FEATURE_IMPORTANCE_PLOT_PATH.exists():
            st.image(str(config.FEATURE_IMPORTANCE_PLOT_PATH), caption="Feature Importance (Gain)")
    else:
        st.info("No metrics found yet. Run `python main.py --stage train` to train a model and populate this tab.")