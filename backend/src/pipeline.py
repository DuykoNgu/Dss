"""Các bước nạp dữ liệu dùng chung cho main, backtest, evaluate và tune."""

from __future__ import annotations

import pandas as pd

import config
from src.data import clean_index_data, clean_ohlcv_data
from src.data import store
from src.features import build_features_and_labels, calculate_technical_indicators


def parse_symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def cached_symbols() -> list[str]:
    return store.current_symbols()


def load_market_index() -> pd.DataFrame:
    return clean_index_data(store.read_bars(store.INDEX_SYMBOL))


def load_featured(
    symbol: str,
    index_df: pd.DataFrame,
    label_strategy: str = "fixed",
    feature_set: str = "baseline",
    forward_days: int = config.ML_FORWARD_DAYS,
) -> pd.DataFrame:
    """SQLite -> clean -> indicators -> features + label. Thiếu dữ liệu thì trả rỗng."""
    clean = clean_ohlcv_data(store.read_bars(symbol))
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
