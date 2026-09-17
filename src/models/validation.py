"""Expanding walk-forward validation with a purge gap for T+5 labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.features import FEATURE_COLUMNS
from src.models.metrics import classification_metrics
from src.models.ml_models import fit_models


def _baseline_predictions(validation: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return simple, deterministic predictions for model comparisons."""
    momentum = validation["return_5d"].to_numpy()
    threshold = config.BASELINE_MOMENTUM_THRESHOLD
    momentum_predictions = np.select(
        [momentum >= threshold, momentum <= -threshold],
        [2, 0],
        default=1,
    ).astype(int)
    return {
        "BaselineHold": np.ones(len(validation), dtype=int),
        "BaselineMomentum": momentum_predictions,
    }


def walk_forward_validate(
    df: pd.DataFrame,
    initial_fraction: float = 0.5,
    validation_fraction: float = 0.1,
    gap: int = config.ML_FORWARD_DAYS,
    feature_columns: list[str] | None = None,
    rf_params: dict | None = None,
    xgb_params: dict | None = None,
) -> dict:
    """Evaluate RF/XGB over expanding folds ordered strictly by time.

    ``gap`` purges samples whose forward-looking label overlaps the validation
    window. Returned predictions are out-of-fold and safe for ensemble tests.
    Đây là chẩn đoán phân loại (argmax); chất lượng điểm DSS dùng thật được đo
    bằng backtest (so sánh rule/ml/blend, danh mục, rank IC).
    """
    columns = feature_columns or FEATURE_COLUMNS
    usable = df.dropna(subset=columns + ["label"]).copy()
    usable["label"] = usable["label"].map({-1: 0, 0: 1, 1: 2}).astype(int)
    total = len(usable)
    initial = max(config.MIN_TRAIN_ROWS, int(total * initial_fraction))
    validation_size = max(1, int(total * validation_fraction))

    fold_rows = []
    predictions = []
    ensemble_predictions = []
    baseline_predictions = []
    fold = 0
    train_end = initial
    while train_end + gap + validation_size <= total:
        validation_start = train_end + gap
        validation_end = validation_start + validation_size
        train = usable.iloc[:train_end]
        validation = usable.iloc[validation_start:validation_end]
        y_true = validation["label"]
        baselines = _baseline_predictions(validation)
        for model, baseline_pred in baselines.items():
            fold_rows.append({
                "fold": fold,
                "model": model,
                **classification_metrics(y_true, baseline_pred),
            })
            baseline_predictions.extend([
                {"fold": fold, "model": model, "y_true": int(y), "y_pred": int(p)}
                for y, p in zip(y_true, baseline_pred)
            ])

        if train["label"].nunique() < 3:
            fold += 1
            train_end += validation_size
            continue

        rf, xgb = fit_models(train[columns], train["label"], rf_params, xgb_params)
        rf_pred = rf.predict(validation[columns])
        xgb_pred = xgb.predict(validation[columns])
        rf_proba = rf.predict_proba(validation[columns])
        xgb_proba = xgb.predict_proba(validation[columns])
        ensemble_pred = np.argmax((rf_proba + xgb_proba) / 2.0, axis=1)
        fold_rows.extend([
            {"fold": fold, "model": "RF", **classification_metrics(y_true, rf_pred)},
            {"fold": fold, "model": "XGB", **classification_metrics(y_true, xgb_pred)},
            {"fold": fold, "model": "Ensemble50", **classification_metrics(y_true, ensemble_pred)},
        ])
        predictions.extend([
            {"fold": fold, "model": "RF", "y_true": int(y), "y_pred": int(p)}
            for y, p in zip(y_true, rf_pred)
        ])
        predictions.extend([
            {"fold": fold, "model": "XGB", "y_true": int(y), "y_pred": int(p)}
            for y, p in zip(y_true, xgb_pred)
        ])
        ensemble_predictions.extend([
            {"fold": fold, "model": "Ensemble50", "y_true": int(y), "y_pred": int(p)}
            for y, p in zip(y_true, ensemble_pred)
        ])
        fold += 1
        train_end += validation_size

    if not fold_rows:
        return {"metrics": [], "predictions": [], "folds": 0, "rows": total}

    aggregate = []
    prediction_frame = pd.DataFrame(predictions + ensemble_predictions + baseline_predictions)
    for model, group in prediction_frame.groupby("model"):
        aggregate.append({
            "fold": "aggregate",
            "model": model,
            **classification_metrics(group["y_true"], group["y_pred"]),
        })

    return {
        "metrics": fold_rows + aggregate,
        "predictions": predictions,
        "folds": fold,
        "rows": total,
    }
