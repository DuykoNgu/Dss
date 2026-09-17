"""Generate baseline label and walk-forward model reports from cached data."""

from __future__ import annotations

import argparse
import os

import pandas as pd

import config
from src.features import EXTENDED_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.models.validation import walk_forward_validate
from src.pipeline import cached_symbols, load_featured, load_market_index, parse_symbols


def write_reports(symbols: list[str], label_strategy: str, feature_set: str,
                  horizon: int = config.ML_FORWARD_DAYS) -> None:
    index = load_market_index()
    label_rows = []
    metric_rows = []
    confusion_rows = []

    for symbol in symbols:
        features = load_featured(symbol, index, label_strategy, feature_set, forward_days=horizon)
        if features.empty:
            print(f"{symbol}: bỏ qua, thiếu dữ liệu")
            continue
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

        result = walk_forward_validate(features, gap=horizon, feature_columns=feature_columns)
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

    suffix = "" if horizon == config.ML_FORWARD_DAYS else f"_h{horizon}"
    output_dir = os.path.join(config.REPORT_DIR, f"{label_strategy}_{feature_set}{suffix}")
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
        choices=["fixed", "volatility", "excess", "triple_barrier"],
        default="fixed",
    )
    parser.add_argument("--feature-set", choices=["baseline", "extended"], default="baseline")
    parser.add_argument("--horizon", type=int, default=config.ML_FORWARD_DAYS,
                        help="Tầm nhìn nhãn (phiên); cũng là purge gap của walk-forward")
    args = parser.parse_args()
    symbols = parse_symbols(args.symbols) or cached_symbols()
    symbols = symbols[: args.limit] if args.limit > 0 else symbols
    write_reports(symbols, args.label_strategy, args.feature_set, args.horizon)


if __name__ == "__main__":
    main()
