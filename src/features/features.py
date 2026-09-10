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

EXTENDED_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "return_3d", "return_10d", "return_60d", "rsi_slope",
    "volume_zscore_20", "vnindex_return_20d",
    "vnindex_volatility_20d", "relative_strength_5d",
]

EPS = 1e-9


def build_features_and_labels(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
    forward_days: int = config.ML_FORWARD_DAYS,
    threshold: float = config.ML_PROFIT_THRESHOLD,
    label_strategy: str = "fixed",
    feature_set: str = "baseline",
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

    df["return_3d"] = close.pct_change(3) * 100
    df["return_10d"] = close.pct_change(10) * 100
    df["return_60d"] = close.pct_change(60) * 100
    df["rsi_slope"] = df["rsi"] - df["rsi"].shift(5)
    volume_mean = df["volume"].rolling(20).mean()
    volume_std = df["volume"].rolling(20).std()
    df["volume_zscore_20"] = (df["volume"] - volume_mean) / (volume_std + EPS)

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
        idx["vnindex_return_20d"] = idx["indexValue"].pct_change(20) * 100
        idx["vnindex_volatility_20d"] = idx["indexValue"].pct_change().rolling(20).std() * 100
        df = pd.merge(df, idx[["time", "vnindex_vs_sma50", "vnindex_return_5d",
                               "vnindex_return_20d", "vnindex_volatility_20d"]],
                      on="time", how="left")
        market_columns = ["vnindex_vs_sma50", "vnindex_return_5d",
                          "vnindex_return_20d", "vnindex_volatility_20d"]
        df[market_columns] = df[market_columns].ffill().fillna(0)
    else:
        df["vnindex_vs_sma50"] = 0.0
        df["vnindex_return_5d"] = 0.0
        df["vnindex_return_20d"] = 0.0
        df["vnindex_volatility_20d"] = 0.0

    df["relative_strength_5d"] = df["return_5d"] - df["vnindex_return_5d"]

    # Nhãn dùng tương lai -> NaN ở `forward_days` dòng cuối (Phase 5 phải drop, cấm làm feature)
    df["future_return"] = close.shift(-forward_days) / close - 1
    df["label"] = _build_labels(df, forward_days, threshold, label_strategy)
    if feature_set not in {"baseline", "extended"}:
        raise ValueError("feature_set phải là 'baseline' hoặc 'extended'")
    return df


def _build_labels(
    df: pd.DataFrame,
    forward_days: int,
    threshold: float,
    strategy: str,
) -> pd.Series:
    """Build alternative labels using future data only for the target."""
    future_return = df["future_return"]
    labels = pd.Series(float("nan"), index=df.index)

    if strategy == "fixed":
        buy = future_return >= threshold
        sell = future_return <= -threshold
    elif strategy == "volatility":
        dynamic_threshold = (df["atr_pct"] / 100.0 * 1.5).clip(lower=threshold)
        buy = future_return >= dynamic_threshold
        sell = future_return <= -dynamic_threshold
    elif strategy == "triple_barrier":
        buy = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)
        for position in range(len(df) - forward_days):
            entry = float(df["close"].iloc[position])
            upper = entry * (1 + threshold)
            lower = entry * (1 - threshold)
            window = df.iloc[position + 1:position + forward_days + 1]
            hit_buy = window["high"] >= upper
            hit_sell = window["low"] <= lower
            first_buy = hit_buy.idxmax() if hit_buy.any() else None
            first_sell = hit_sell.idxmax() if hit_sell.any() else None
            if first_buy is not None and (first_sell is None or first_buy < first_sell):
                buy.iloc[position] = True
            elif first_sell is not None and (first_buy is None or first_sell < first_buy):
                sell.iloc[position] = True
    else:
        raise ValueError("label_strategy phải là fixed, volatility hoặc triple_barrier")

    labels.loc[buy] = 1
    labels.loc[sell] = -1
    labels.loc[future_return.notna() & ~buy & ~sell] = 0
    return labels
