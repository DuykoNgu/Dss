"""Phase 5: Train Random Forest + XGBoost (walk-forward), ensemble predict."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

import config
from src.features import FEATURE_COLUMNS
from src.models.metrics import classification_metrics, format_metrics

LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASSES = [0, 1, 2]
BUY_CLASS = 2  # nhãn MUA sau khi map
NEUTRAL_SCORE = 50.0  # fallback khi thiếu model hoặc feature NaN
BALANCED_PRIOR = 1.0 / len(CLASSES)


def train_ml_models(
    df: pd.DataFrame,
    symbol: str,
    save: bool = True,
    verbose: bool = True,
) -> tuple:
    """Train RF + XGB trên 80% đầu (quá khứ), đánh giá 20% cuối. Lưu .pkl.

    Cả 2 model train với trọng số class cân bằng (giống RF class_weight='balanced',
    XGB dùng sample_weight từ compute_class_weight) để không "lười" đoán GIỮ.
    Xác suất output bị lệch prior cân bằng (1/3) nên predict_ml_score sẽ hiệu
    chỉnh lại theo prior thật của tập train.

    save=False, verbose=False dùng cho backtest (retrain nhiều lần, không ghi đè model).
    """
    train_df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    if len(train_df) < config.MIN_TRAIN_ROWS:
        if verbose:
            print(f"[{symbol}] Bỏ qua: chỉ có {len(train_df)} dòng, cần {config.MIN_TRAIN_ROWS}.")
        return None, None

    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"].map(LABEL_TO_CLASS)
    valid_labels = y.notna()
    X, y = X.loc[valid_labels], y.loc[valid_labels].astype(int)

    split = int(len(X) * (1 - config.TEST_SIZE_RATIO))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if not set(CLASSES).issubset(y_train.unique()):
        if verbose:
            print(f"[{symbol}] Bỏ qua: tập train thiếu một hoặc nhiều class.")
        return None, None

    weights = compute_class_weight(class_weight="balanced", classes=np.array(CLASSES), y=y_train)
    sample_weight = weights[y_train.values]

    rf = RandomForestClassifier(**config.RF_PARAMS).fit(X_train, y_train)
    xgb = XGBClassifier(**config.XGB_PARAMS).fit(X_train, y_train, sample_weight=sample_weight)
    if verbose:
        rf_metrics = classification_metrics(y_test, rf.predict(X_test))
        xgb_metrics = classification_metrics(y_test, xgb.predict(X_test))
        print(f"[{symbol}] RF test ({len(X_test)} dòng): {format_metrics(rf_metrics)}")
        print(f"[{symbol}] XGB test ({len(X_test)} dòng): {format_metrics(xgb_metrics)}")

    # Prior thật của tập train để hiệu chỉnh xác suất lúc predict (lưu kèm model)
    class_priors = y_train.value_counts(normalize=True).reindex(CLASSES, fill_value=0.0).to_dict()
    rf.class_priors_ = class_priors
    xgb.class_priors_ = class_priors

    if save:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(rf, os.path.join(config.MODEL_DIR, f"{symbol}_rf.pkl"))
        joblib.dump(xgb, os.path.join(config.MODEL_DIR, f"{symbol}_xgb.pkl"))
    return rf, xgb


def load_ml_models(symbol: str) -> tuple:
    """Nạp cặp model đã train, thiếu thì trả (None, None)."""
    rf_path = os.path.join(config.MODEL_DIR, f"{symbol}_rf.pkl")
    xgb_path = os.path.join(config.MODEL_DIR, f"{symbol}_xgb.pkl")
    if not (os.path.exists(rf_path) and os.path.exists(xgb_path)):
        return None, None
    return joblib.load(rf_path), joblib.load(xgb_path)


def _correct_priors(proba: np.ndarray, class_priors: dict) -> np.ndarray:
    """Hiệu chỉnh xác suất từ prior cân bằng (1/3) về prior thật của tập train."""
    corrected = np.zeros(len(CLASSES))
    for c in CLASSES:
        true_prior = class_priors.get(c, BALANCED_PRIOR)
        corrected[c] = proba[c] * (true_prior / BALANCED_PRIOR)
    total = corrected.sum()
    return corrected / total if total > 0 else np.full(len(CLASSES), BALANCED_PRIOR)


def predict_ml_score(row: pd.DataFrame, rf, xgb) -> float:
    """ML Score = P(MUA) đã hiệu chỉnh prior * 100. NaN/thiếu model -> 50."""
    if rf is None or xgb is None:
        return NEUTRAL_SCORE

    latest = row[FEATURE_COLUMNS].iloc[[0]]
    if latest.isnull().values.any():
        return NEUTRAL_SCORE

    avg_proba = (rf.predict_proba(latest)[0] + xgb.predict_proba(latest)[0]) / 2.0
    class_priors = getattr(rf, "class_priors_", None) or getattr(xgb, "class_priors_", None)
    if class_priors:
        avg_proba = _correct_priors(avg_proba, class_priors)
    return float(avg_proba[BUY_CLASS] * 100)
