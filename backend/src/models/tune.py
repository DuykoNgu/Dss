"""Small, time-series-safe RF/XGB tuning search."""

from __future__ import annotations

import argparse
import os

import pandas as pd

import config
from src.features import FEATURE_COLUMNS
from src.models.validation import walk_forward_validate
from src.pipeline import load_featured, load_market_index, parse_symbols


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


def tune_symbol(symbol: str, index: pd.DataFrame) -> list[dict]:
    features = load_featured(symbol, index)
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
    index = load_market_index()
    rows = []
    for symbol in parse_symbols(args.symbols):
        rows.extend(tune_symbol(symbol, index))
    output = os.path.join(config.REPORT_DIR, "tuning_results.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Tuning report written to {output}")


if __name__ == "__main__":
    main()
