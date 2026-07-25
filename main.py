"""
main.py
=======
Single end-to-end orchestration entrypoint.

Usage
-----
    python main.py --stage data              # fetch + engineer features + sentiment -> parquet
    python main.py --stage train              # train + evaluate model -> joblib + metrics.json
    python main.py --stage all                # data, then train
    python main.py --stage all --tickers AAPL NVDA --start 2024-01-01 --end 2026-07-24

Flags
-----
    --stage      {data, train, all}   (default: all)
    --tickers    space-separated ticker list (default: config.TICKERS)
    --start      YYYY-MM-DD           (default: config.START_DATE)
    --end        YYYY-MM-DD           (default: config.END_DATE)
"""
from __future__ import annotations

import argparse
import logging

import config
from src.data_pipeline import build_feature_store, save_feature_store
from src.model_pipeline import run_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock Sentiment Predictor pipeline runner")
    parser.add_argument("--stage", choices=["data", "train", "all"], default="all")
    parser.add_argument("--tickers", nargs="+", default=config.TICKERS)
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=config.END_DATE)
    return parser.parse_args()


def run_data_stage(tickers: list[str], start: str, end: str) -> None:
    # Runtime overrides are applied on the config module so every downstream
    # function (which reads config.START_DATE / config.END_DATE) stays in sync
    # without needing every function signature to accept these as arguments.
    config.START_DATE = start
    config.END_DATE = end
    logger.info("=== STAGE: DATA (tickers=%s, %s -> %s) ===", tickers, start, end)
    store = build_feature_store(tickers)
    save_feature_store(store)


def run_train_stage() -> None:
    logger.info("=== STAGE: TRAIN ===")
    run_training_pipeline()


def main() -> None:
    args = parse_args()

    if args.stage in ("data", "all"):
        run_data_stage(args.tickers, args.start, args.end)

    if args.stage in ("train", "all"):
        run_train_stage()

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
