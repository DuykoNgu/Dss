"""Các bước nạp dữ liệu dùng chung cho main, backtest, evaluate và tune."""

from __future__ import annotations

import json
import os

import pandas as pd

import config
from src.data import clean_index_data, clean_ohlcv_data
from src.features import build_features_and_labels, calculate_technical_indicators


def parse_symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def cached_symbols() -> list[str]:
    """Danh sách mã đã đồng bộ: symbols.json, không có thì quét thư mục CSV."""
    manifest = os.path.join(config.DATA_DIR, "symbols.json")
    if os.path.exists(manifest):
        with open(manifest) as file:
            return json.load(file)
    if not os.path.isdir(config.STOCKS_DIR):
        return []
    return sorted(name[:-4] for name in os.listdir(config.STOCKS_DIR) if name.endswith(".csv"))


def load_market_index() -> pd.DataFrame:
    if not os.path.exists(config.INDEX_PATH):
        return pd.DataFrame()
    return clean_index_data(pd.read_csv(config.INDEX_PATH))


def load_featured(
    symbol: str,
    index_df: pd.DataFrame,
    label_strategy: str = "fixed",
    feature_set: str = "baseline",
    forward_days: int = config.ML_FORWARD_DAYS,
) -> pd.DataFrame:
    """CSV cache -> clean -> indicators -> features + label. Thiếu/hỏng dữ liệu thì trả rỗng."""
    path = os.path.join(config.STOCKS_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    clean = clean_ohlcv_data(pd.read_csv(path))
    indicated = calculate_technical_indicators(clean)
    if indicated is None or "sma_200" not in indicated.columns:
        return pd.DataFrame()
    return build_features_and_labels(
        indicated,
        index_df,
        forward_days=forward_days,
        label_strategy=label_strategy,
        feature_set=feature_set,
    )
