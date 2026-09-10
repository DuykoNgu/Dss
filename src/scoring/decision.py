"""Tổng hợp Rule + ML thành tín hiệu khuyến nghị."""

from __future__ import annotations

import pandas as pd

import config
from src.models import load_ml_models, predict_ml_score, train_ml_models
from src.scoring.scoring import calculate_rule_based_score

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
    """parts: rule, ml, reasons. Trả dict 7 trường cho bảng Terminal."""
    total = parts["rule"] * config.WEIGHT_RULE_BASED + parts["ml"] * config.WEIGHT_ML_MODEL
    reasons = parts["reasons"][:2] if parts["reasons"] else ["Trung tính"]
    return {
        "symbol": symbol,
        "price": float(latest.get("close", 0) or 0),
        "total_score": total,
        "rule_score": parts["rule"],
        "ml_score": parts["ml"],
        "signal": score_to_signal(total),
        "primary_reason": "; ".join(reasons),
    }


def generate_decisions(
    featured: dict[str, pd.DataFrame],
    force_retrain: bool = False,
) -> list[dict]:
    """Load model cache trước, train khi thiếu hoặc có yêu cầu retrain."""
    rows = []
    for symbol, df in featured.items():
        if df is None or df.empty:
            continue
        if force_retrain:
            rf, xgb = train_ml_models(df, symbol)
        else:
            rf, xgb = load_ml_models(symbol)
            if rf is None or xgb is None:
                rf, xgb = train_ml_models(df, symbol)
        ml_score = predict_ml_score(df.iloc[[-1]], rf, xgb)
        rule_score, reasons = calculate_rule_based_score(df)
        rows.append(generate_decision(
            symbol,
            df.iloc[-1],
            {"rule": rule_score, "ml": ml_score, "reasons": reasons},
        ))
    rows.sort(key=lambda r: r["total_score"], reverse=True)
    return rows
