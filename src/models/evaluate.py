"""Generate baseline label and walk-forward model reports from cached data."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.data import clean_index_data, clean_ohlcv_data
from src.features import (
    EXTENDED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_features_and_labels,
    calculate_technical_indicators,
)
from src.models.validation import walk_forward_validate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_DIR = os.path.join(BASE_DIR, "reports")


def get_symbols(raw: str, limit: int) -> list[str]:
    if raw:
        symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    else:
        with open(os.path.join(DATA_DIR, "symbols.json")) as file:
            symbols = json.load(file)
    return symbols[:limit] if limit > 0 else symbols


def load_features(
    symbol: str,
    index: pd.DataFrame,
    label_strategy: str,
    feature_set: str,
) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "stocks", f"{symbol}.csv")
    raw = pd.read_csv(path)
    clean = clean_ohlcv_data(raw)
    indicated = calculate_technical_indicators(clean)
    return build_features_and_labels(
        indicated,
        index,
        label_strategy=label_strategy,
        feature_set=feature_set,
    )


def write_reports(symbols: list[str], label_strategy: str, feature_set: str) -> None:
    index = clean_index_data(pd.read_csv(os.path.join(DATA_DIR, "index", "VNINDEX.csv")))
    label_rows = []
    metric_rows = []
    confusion_rows = []

    for symbol in symbols:
        features = load_features(symbol, index, label_strategy, feature_set)
        feature_columns = EXTENDED_FEATURE_COLUMNS if feature_set == "extended" else FEATURE_COLUMNS
        labels = features["label"].dropna().value_counts().reindex([-1, 0, 1], fill_value=0)
        trainable_rows = len(features.dropna(subset=feature_columns + ["label"]))
        label_rows.append({
            "symbol": symbol,
            "rows": len(features),
            "trainable_rows": trainable_rows,
            "sell": int(labels[-1]),
            "hold": int(labels[0]),
            "buy": int(labels[1]),
        })

        result = walk_forward_validate(features, feature_columns=feature_columns)
        for metric in result["metrics"]:
            row = {
                "symbol": symbol,
                "label_strategy": label_strategy,
                "feature_set": feature_set,
                **metric,
            }
            matrix = row.pop("confusion_matrix")
            metric_rows.append(row)
            if metric["fold"] == "aggregate":
                for actual, values in enumerate(matrix):
                    for predicted, count in enumerate(values):
                        confusion_rows.append({
                            "symbol": symbol,
                            "model": metric["model"],
                            "actual": actual,
                            "predicted": predicted,
                            "count": count,
                        })
        print(f"{symbol}: {result['folds']} folds, {result['rows']} usable rows")

    output_dir = os.path.join(REPORT_DIR, f"{label_strategy}_{feature_set}")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(label_rows).to_csv(os.path.join(output_dir, "label_distribution.csv"), index=False)
    pd.DataFrame(metric_rows).to_csv(os.path.join(output_dir, "walk_forward_metrics.csv"), index=False)
    pd.DataFrame(confusion_rows).to_csv(os.path.join(output_dir, "confusion_matrix.csv"), index=False)
    print(f"Reports written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá baseline ML theo time-series.")
    parser.add_argument("--symbols", default="", help="VD: FPT,HPG")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--label-strategy",
        choices=["fixed", "volatility", "triple_barrier"],
        default="fixed",
    )
    parser.add_argument("--feature-set", choices=["baseline", "extended"], default="baseline")
    args = parser.parse_args()
    write_reports(get_symbols(args.symbols, args.limit), args.label_strategy, args.feature_set)


if __name__ == "__main__":
    main()
