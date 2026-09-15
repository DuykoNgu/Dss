"""Small, time-series-safe RF/XGB tuning search."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import config
from src.data import clean_index_data, clean_ohlcv_data
from src.features import FEATURE_COLUMNS, build_features_and_labels, calculate_technical_indicators
from src.models.validation import walk_forward_validate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_features(symbol: str) -> pd.DataFrame:
    index = clean_index_data(pd.read_csv(os.path.join(DATA_DIR, "index", "VNINDEX.csv")))
    raw = pd.read_csv(os.path.join(DATA_DIR, "stocks", f"{symbol}.csv"))
    clean = clean_ohlcv_data(raw)
    return build_features_and_labels(calculate_technical_indicators(clean), index)


def candidate_params() -> list[tuple[str, dict, dict]]:
    rf_base = dict(config.RF_PARAMS)
    xgb_base = dict(config.XGB_PARAMS)
    return [
        ("baseline", rf_base, xgb_base),
        (
            "regularized",
            {**rf_base, "max_depth": 6, "min_samples_leaf": 30},
            {**xgb_base, "max_depth": 3, "learning_rate": 0.03, "n_estimators": 400},
        ),
        (
            "responsive",
            {**rf_base, "max_depth": 12, "min_samples_leaf": 10},
            {**xgb_base, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 250},
        ),
    ]


def tune_symbol(symbol: str) -> list[dict]:
    features = load_features(symbol)
    rows = []
    for name, rf_params, xgb_params in candidate_params():
        result = walk_forward_validate(
            features,
            feature_columns=FEATURE_COLUMNS,
            rf_params=rf_params,
            xgb_params=xgb_params,
        )
        aggregate = [row for row in result["metrics"] if row["fold"] == "aggregate"]
        for row in aggregate:
            rows.append({"symbol": symbol, "candidate": name, **row})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune RF/XGBoost bằng walk-forward.")
    parser.add_argument("--symbols", default="FPT")
    args = parser.parse_args()
    rows = []
    for symbol in [item.strip().upper() for item in args.symbols.split(",") if item.strip()]:
        rows.extend(tune_symbol(symbol))
    output = os.path.join(BASE_DIR, "reports", "tuning_results.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Tuning report written to {output}")


if __name__ == "__main__":
    main()
