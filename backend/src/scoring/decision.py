"""Tổng hợp Rule + ML thành tín hiệu khuyến nghị."""

from __future__ import annotations

import pandas as pd

import config
from src.features import FEATURE_COLUMNS
from src.models import load_ml_models, model_is_stale, predict_ml_score, train_ml_models
from src.scoring.scoring import calculate_rule_based_score

POOLED_MODEL_KEY = "POOLED"

SIGNAL_ORDER = (
    (config.SCORE_STRONG_BUY, "🟢 MUA MẠNH"),
    (config.SCORE_BUY, "🟡 MUA"),
    (config.SCORE_HOLD, "⚪ GIỮ"),
    (config.SCORE_SELL, "🟠 BÁN"),
)


def score_to_signal(total: float) -> str:
    for threshold, label in SIGNAL_ORDER:
        if total >= threshold:
            return label
    return "🔴 BÁN MẠNH"


def generate_decision(symbol: str, latest: pd.Series, parts: dict) -> dict:
    """parts: rule, ml, reasons, ml_available. Trả dict cho bảng Terminal."""
    total = parts["rule"] * config.WEIGHT_RULE_BASED + parts["ml"] * config.WEIGHT_ML_MODEL
    reasons = parts["reasons"][:2] if parts["reasons"] else ["Trung tính"]
    return {
        "symbol": symbol,
        "time": latest.get("time"),
        "price": float(latest.get("close", 0) or 0),
        "total_score": total,
        "rule_score": parts["rule"],
        "ml_score": parts["ml"],
        "ml_available": parts.get("ml_available", True),
        "signal": score_to_signal(total),
        "primary_reason": "; ".join(reasons),
    }


def get_models(key: str, frame: pd.DataFrame, force_retrain: bool = False) -> tuple:
    """Dùng model cache nếu còn mới; thiếu, cũ hoặc có yêu cầu retrain thì train lại."""
    rf, xgb = (None, None) if force_retrain else load_ml_models(key)
    if (rf is None or xgb is None or model_is_stale(rf, frame["time"].max())
            or model_is_stale(xgb, frame["time"].max())
            or rf.data_end_ != xgb.data_end_):
        rf, xgb = train_ml_models(frame, key)
    return rf, xgb


def generate_decisions(
    featured: dict[str, pd.DataFrame],
    force_retrain: bool = False,
    pooled: bool = config.ML_POOLED,
) -> list[dict]:
    frames = {symbol: df for symbol, df in featured.items() if df is not None and not df.empty}
    if not frames:
        return []
    if pooled:
        pooled_models = get_models(POOLED_MODEL_KEY, pd.concat(frames.values(), ignore_index=True),
                                   force_retrain)

    rows = []
    for symbol, df in frames.items():
        rf, xgb = pooled_models if pooled else get_models(symbol, df, force_retrain)
        ml_available = rf is not None and not df.iloc[-1][FEATURE_COLUMNS].isna().any()
        rule_score, reasons = calculate_rule_based_score(df)
        rows.append(generate_decision(
            symbol,
            df.iloc[-1],
            {
                "rule": rule_score,
                "ml": predict_ml_score(df.iloc[[-1]], rf, xgb),
                "reasons": reasons,
                "ml_available": ml_available,
            },
        ))
    rows.sort(key=lambda r: r["total_score"], reverse=True)
    return rows
