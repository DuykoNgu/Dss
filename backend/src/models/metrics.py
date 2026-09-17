"""Metrics for imbalanced, multi-class time-series models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
CLASS_IDS = [0, 1, 2]


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    """Return metrics that do not hide minority-class performance."""
    true = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    matrix = confusion_matrix(true, pred, labels=CLASS_IDS)

    precision = precision_score(true, pred, labels=CLASS_IDS, average=None, zero_division=0)
    recall = recall_score(true, pred, labels=CLASS_IDS, average=None, zero_division=0)
    f1 = f1_score(true, pred, labels=CLASS_IDS, average=None, zero_division=0)

    present_classes = np.unique(true)
    balanced_accuracy = recall_score(
        true,
        pred,
        labels=present_classes,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(f1_score(true, pred, labels=CLASS_IDS, average="macro", zero_division=0)),
        "sell_precision": float(precision[0]),
        "sell_recall": float(recall[0]),
        "sell_f1": float(f1[0]),
        "hold_precision": float(precision[1]),
        "hold_recall": float(recall[1]),
        "hold_f1": float(f1[1]),
        "buy_precision": float(precision[2]),
        "buy_recall": float(recall[2]),
        "buy_f1": float(f1[2]),
        "confusion_matrix": matrix.tolist(),
    }


def format_metrics(metrics: dict) -> str:
    """Format the most important metrics for CLI logs."""
    return (
        f"Accuracy={metrics['accuracy']:.1%} "
        f"BalancedAcc={metrics['balanced_accuracy']:.1%} "
        f"MacroF1={metrics['macro_f1']:.3f} "
        f"BUY(P={metrics['buy_precision']:.1%},R={metrics['buy_recall']:.1%}) "
        f"SELL(P={metrics['sell_precision']:.1%},R={metrics['sell_recall']:.1%})"
    )
