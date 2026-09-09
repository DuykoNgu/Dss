"""Phase 4: Feature engineering (19 features tương đối) + gán nhãn T+5."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import ta

import config

FEATURE_COLUMNS = [
    "price_vs_sma50", "price_vs_sma200", "sma50_vs_sma200",
    "macd_hist", "macd_hist_slope",
    "rsi", "stoch_k", "stoch_d", "williams_r",
    "bb_position", "bb_width", "atr_pct",
    "vol_ratio", "obv_slope",
    "return_1d", "return_5d", "return_20d",
    "vnindex_vs_sma50", "vnindex_return_5d",
]

EPS = 1e-9


def build_features_and_labels(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
    forward_days: int = config.ML_FORWARD_DAYS,
    threshold: float = config.ML_PROFIT_THRESHOLD,
) -> pd.DataFrame:
    """Mọi feature chỉ dùng quá khứ/hiện tại. Riêng label nhìn tương lai (chỉ để train)."""
    df = stock_df.copy()
    close = df["close"]

    df["price_vs_sma50"] = (close - df["sma_50"]) / (df["sma_50"].abs() + EPS) * 100
    df["price_vs_sma200"] = (close - df["sma_200"]) / (df["sma_200"].abs() + EPS) * 100
    df["sma50_vs_sma200"] = (df["sma_50"] - df["sma_200"]) / (df["sma_200"].abs() + EPS) * 100
    df["macd_hist_slope"] = df["macd_hist"] - df["macd_hist"].shift(3)
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + EPS)
    df["obv_slope"] = (df["obv"] - df["obv"].shift(5)) / (df["obv"].shift(5).abs() + EPS)
    for n in (1, 5, 20):
        df[f"return_{n}d"] = close.pct_change(n) * 100

    df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
    df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
    df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))

    if index_df is not None and not index_df.empty:
        idx = index_df.copy()
        idx["vnindex_sma50"] = ta.trend.sma_indicator(idx["indexValue"], window=50)
        idx["vnindex_vs_sma50"] = (
            (idx["indexValue"] - idx["vnindex_sma50"]) / (idx["vnindex_sma50"].abs() + EPS) * 100
        )
        idx["vnindex_return_5d"] = idx["indexValue"].pct_change(5) * 100
        df = pd.merge(df, idx[["time", "vnindex_vs_sma50", "vnindex_return_5d"]],
                      on="time", how="left")
        df[["vnindex_vs_sma50", "vnindex_return_5d"]] = (
            df[["vnindex_vs_sma50", "vnindex_return_5d"]].ffill().fillna(0)
        )
    else:
        df["vnindex_vs_sma50"] = 0.0
        df["vnindex_return_5d"] = 0.0

    # Nhãn dùng tương lai -> NaN ở `forward_days` dòng cuối (Phase 5 phải drop, cấm làm feature)
    df["future_return"] = close.shift(-forward_days) / close - 1
    df["label"] = float("nan")
    df.loc[df["future_return"] >= threshold, "label"] = 1
    df.loc[df["future_return"] <= -threshold, "label"] = -1
    df.loc[df["future_return"].abs() < threshold, "label"] = 0
    return df
