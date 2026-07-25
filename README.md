# 📈 Stock Sentiment Predictor

A production-structured ML pipeline that predicts next-day stock price
direction by combining **technical indicators** (RSI, MACD, SMA ratio, ATR,
volatility) with **FinBERT-scored news sentiment**, trained with an
**XGBoost classifier** and served through an interactive **Streamlit
dashboard**.

---

## Architecture

```
                ┌────────────────┐      ┌──────────────────┐
                │   yfinance     │      │  News headlines   │
                │  (OHLCV data)  │      │ (mock / NewsAPI)   │
                └───────┬────────┘      └─────────┬─────────┘
                        │                          │
                        ▼                          ▼
              add_technical_indicators     FinBERT SentimentScorer
                        │                          │
                        └───────────┬──────────────┘
                                    ▼
                merge_with_lagged_sentiment (strict T-1 shift)
                                    │
                                    ▼
                    data/processed_features.parquet
                                    │
                                    ▼
                  TimeSeriesSplit CV  →  XGBoost training
                                    │
                                    ▼
              data/models/xgboost_model.joblib + metrics.json
                                    │
                                    ▼
                         Streamlit dashboard (app.py)
```

> *(Replace this block with a rendered diagram image, e.g.
> `docs/architecture.png`, once you have one — this ASCII version is a
> placeholder.)*

### Directory layout

```
stock-sentiment-predictor/
├── app.py                    # Streamlit dashboard entrypoint
├── main.py                   # CLI orchestrator (data / train / all)
├── config.py                 # All constants, paths, hyperparameters
├── requirements.txt
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      # ETL: prices, indicators, sentiment, lag-merge
│   └── model_pipeline.py     # Split, train, evaluate, persist model
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   └── test_model_pipeline.py
└── data/                     # Generated at runtime (git-ignored)
    ├── processed_features.parquet
    ├── metrics.json
    ├── feature_importance.png
    └── models/
        └── xgboost_model.joblib
```

---

## Setup

```bash
git clone <your-repo-url>
cd stock-sentiment-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python 3.10+. FinBERT (`torch` + `transformers`) will
download ~440MB of model weights on first run — a GPU is optional but not
required for this project's headline volume.

---

## Running the pipeline

### 1. Full pipeline (data + training) in one command

```bash
python main.py --stage all
```

### 2. Or run stages independently

```bash
# Fetch prices, compute technical indicators, score sentiment, save feature store
python main.py --stage data --tickers AAPL NVDA MSFT --start 2023-01-01 --end 2026-07-24

# Train XGBoost, evaluate against a naive baseline, save model + metrics
python main.py --stage train
```

Outputs after a successful run:

| File | Description |
|---|---|
| `data/processed_features.parquet` | Model-ready feature store (technical + T-1 lagged sentiment) |
| `data/models/xgboost_model.joblib` | Trained classifier |
| `data/metrics.json` | Naive baseline vs. XGBoost metrics + feature importances |
| `data/feature_importance.png` | Gain-based feature importance chart |

### 3. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard auto-detects whether `data/models/xgboost_model.joblib`
exists:

* **🟢 Live Mode** — trained model found → real inference on the latest
  feature row, using persisted FinBERT sentiment.
* **🟡 Demo Mode** — no trained model yet → dashboard still fully renders
  using live price data + a labeled synthetic sentiment/signal generator,
  so the UI is explorable before you've trained anything.

---

## Testing & verification

Run the automated test suite (fast, offline, no network or model downloads):

```bash
pytest tests/ -v
```

What it checks:
- Technical indicators produce the expected columns and stay in valid
  ranges (e.g. RSI bounded 0–100).
- The binary target is constructed correctly relative to next-day close.
- **Lookahead-bias guard**: `sent_*_lag1` columns always reflect the *prior*
  day's sentiment, never same-day, and unlagged sentiment columns never leak
  into the final feature store.
- The chronological train/test split respects each ticker's date ordering
  and produces the expected split ratio.
- The XGBoost training routine returns a fitted, predictable model with all
  expected metric keys.

**Manual end-to-end smoke test** (exercises the network/FinBERT paths the
unit tests intentionally skip):

```bash
python main.py --stage all --tickers AAPL --start 2024-01-01 --end 2026-07-24
streamlit run app.py
```
Confirm the sidebar shows "🟢 Live Mode" and the *Model Performance* tab
populates with real metrics.

---

## Deploying the dashboard (Streamlit Community Cloud)

1. Push this repository to GitHub (the `.gitignore` already excludes
   generated data/model artifacts, so commit the code only).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `app.py` on your default branch.
3. Set Python version to match your local environment (3.10+).
4. Since `data/` is git-ignored, the deployed app will start in **Demo
   Mode**. To ship a trained model with the app, either:
   - Add a startup step (e.g. a Streamlit Cloud "Secrets" + init script)
     that runs `python main.py --stage all` once on deploy, or
   - Manually commit a pre-trained `data/models/xgboost_model.joblib` and
     `data/processed_features.parquet` for demo purposes (remove them from
     `.gitignore` first).
5. If you wire up a real news API for `get_headlines_for_ticker` in
   `src/data_pipeline.py`, add the API key via Streamlit's **Secrets**
   manager rather than a committed `.env` file.

---

## Design decisions (refactor summary)

- **One pipeline, one config.** All hyperparameters, paths, tickers, and
  the feature/target schema live in `config.py`. Previously these were
  duplicated (and had drifted out of sync) across `feature_pipeline.py`,
  `sentiment_pipeline.py`, and `run_day1_pipeline.py`.
- **De-duplicated the FinBERT loading logic.** The original
  `sentiment_pipeline.py` file contained its entire contents duplicated
  twice and loaded the model as a module-level side effect. It's now a
  single `SentimentScorer` class that lazy-loads the model on first use.
- **Centralized the lookahead-bias guard.** The `.shift(1)` lag logic
  previously existed in three near-identical copies across
  `run_day1_pipeline.py` and `sentiment_pipeline.py`. It's now one function,
  `merge_with_lagged_sentiment`, covered by a dedicated regression test.
- **Separated ETL from UI.** `app.py` no longer contains its own
  copy-pasted technical-indicator or sentiment logic — it imports the same
  `src.data_pipeline` functions used by the training pipeline, so the
  dashboard and the model are guaranteed to see identically-computed
  features.
- **Live/Demo mode instead of permanently-synthetic data.** The original
  `app.py` always simulated sentiment and used a dummy in-memory model.
  The refactored dashboard uses the *real* trained model and *real*
  FinBERT sentiment when available, and only falls back to a clearly
  labeled synthetic mode when no training run has happened yet — so no
  functionality was dropped, but the dashboard no longer silently pretends
  simulated numbers are real.
- **`joblib` instead of XGBoost's native JSON format.** Keeps `app.py`'s
  model-loading code generic (`joblib.load(...)`) and decoupled from the
  specific serialization format of whichever sklearn-compatible model you
  swap in later.
- **Removed redundant/one-off scripts.** `csv_to_parquet.py`,
  `verify_pipeline.py`, `verify_day2.py`, and `test_setup.py` were
  exploratory/ad-hoc checks duplicating logic now covered by
  `tests/test_data_pipeline.py`, `tests/test_model_pipeline.py`, and the
  `load_feature_store` validation in `model_pipeline.py`.
