"""Phase 5: Train Random Forest + XGBoost, ensemble predict."""

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
SELL_CLASS, BUY_CLASS = 0, 2
NEUTRAL_SCORE = 50.0  # ML trung tính, cùng thang với Rule Score; dùng khi thiếu model/feature
BALANCED_PRIOR = 1.0 / len(CLASSES)


def fit_models(x: pd.DataFrame, y: pd.Series, rf_params: dict | None = None,
               xgb_params: dict | None = None) -> tuple:
    """Fit RF + XGB với trọng số class cân bằng để model không "lười" đoán GIỮ.

    RF dùng class_weight trong params, XGB dùng sample_weight. y phải có đủ 3 class.
    """
    weights = compute_class_weight(class_weight="balanced", classes=np.array(CLASSES), y=y)
    rf = RandomForestClassifier(**(rf_params or config.RF_PARAMS)).fit(x, y)
    xgb = XGBClassifier(**(xgb_params or config.XGB_PARAMS))
    xgb.fit(x, y, sample_weight=weights[y.to_numpy()])
    return rf, xgb


def _print_holdout_metrics(train_df: pd.DataFrame, y: pd.Series, symbol: str) -> None:
    """Metric tham khảo: train model phụ trên 80% thời gian đầu (purge T+5), test 20% cuối."""
    dates = np.sort(train_df["time"].unique())
    split = int(len(dates) * (1 - config.TEST_SIZE_RATIO))
    purge_start = dates[max(0, split - config.ML_FORWARD_DAYS)]
    in_train = (train_df["time"] < purge_start).to_numpy()
    in_test = (train_df["time"] >= dates[split]).to_numpy()
    if not in_test.any() or not set(CLASSES).issubset(y[in_train].unique()):
        print(f"[{symbol}] Không đủ dữ liệu để tính metric holdout.")
        return
    x = train_df[FEATURE_COLUMNS]
    rf, xgb = fit_models(x[in_train], y[in_train])
    for name, model in (("RF", rf), ("XGB", xgb)):
        metrics = classification_metrics(y[in_test], model.predict(x[in_test]))
        print(f"[{symbol}] {name} holdout ({in_test.sum()} dòng): {format_metrics(metrics)}")


def train_ml_models(
    df: pd.DataFrame,
    symbol: str,
    save: bool = True,
    verbose: bool = True,
) -> tuple:
    """Train RF + XGB trên TOÀN BỘ dòng có label để model dùng cả dữ liệu mới nhất.

    verbose=True in thêm metric holdout 20% cuối (model phụ, không lưu) để tham khảo;
    đánh giá nghiêm túc dùng walk-forward (evaluate) và backtest.
    Xác suất output bị lệch về prior cân bằng (1/3) nên model lưu kèm prior thật
    của tập train để predict hiệu chỉnh lại. save=False dùng cho backtest.
    """
    train_df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    if "time" in train_df.columns:
        train_df = train_df.sort_values("time", kind="stable")
    if len(train_df) < config.MIN_TRAIN_ROWS:
        if verbose:
            print(f"[{symbol}] Bỏ qua: chỉ có {len(train_df)} dòng, cần {config.MIN_TRAIN_ROWS}.")
        return None, None

    y = train_df["label"].map(LABEL_TO_CLASS)
    train_df, y = train_df.loc[y.notna()], y.loc[y.notna()].astype(int)
    if not set(CLASSES).issubset(y.unique()):
        if verbose:
            print(f"[{symbol}] Bỏ qua: dữ liệu train thiếu một hoặc nhiều class.")
        return None, None

    if verbose and "time" in train_df.columns:
        _print_holdout_metrics(train_df, y, symbol)

    rf, xgb = fit_models(train_df[FEATURE_COLUMNS], y)
    class_priors = y.value_counts(normalize=True).reindex(CLASSES, fill_value=0.0).to_dict()
    data_end = df["time"].max() if "time" in df.columns else None
    for model in (rf, xgb):
        model.class_priors_ = class_priors
        model.feature_columns_ = list(FEATURE_COLUMNS)
        model.data_end_ = data_end

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


def model_is_stale(model, data_end: pd.Timestamp) -> bool:
    """Model thiếu metadata, khác feature schema, hoặc cũ hơn dữ liệu quá MODEL_MAX_AGE_DAYS."""
    trained_until = getattr(model, "data_end_", None)
    if trained_until is None or getattr(model, "feature_columns_", None) != FEATURE_COLUMNS:
        return True
    return data_end - trained_until > pd.Timedelta(days=config.MODEL_MAX_AGE_DAYS)


def predict_ml_scores(rows: pd.DataFrame, rf, xgb) -> np.ndarray:
    """ML Score = 50 + 50 x (P(MUA) - P(BÁN)), xác suất ensemble đã hiệu chỉnh prior.

    50 là trung tính (cùng thang với Rule Score), 100 = chắc chắn MUA, 0 = chắc chắn BÁN.
    Thiếu model hoặc dòng có feature NaN -> 50.
    """
    scores = np.full(len(rows), NEUTRAL_SCORE)
    if rf is None or xgb is None or rows.empty:
        return scores
    features = rows[FEATURE_COLUMNS]
    valid = features.notna().all(axis=1).to_numpy()
    if not valid.any():
        return scores

    proba = (rf.predict_proba(features[valid]) + xgb.predict_proba(features[valid])) / 2.0
    class_priors = getattr(rf, "class_priors_", None) or getattr(xgb, "class_priors_", None)
    if class_priors:
        # Train cân bằng kéo xác suất về 1/3 -> nhân ngược theo prior thật rồi chuẩn hóa
        proba = proba * np.array([class_priors.get(c, BALANCED_PRIOR) / BALANCED_PRIOR for c in CLASSES])
        proba = proba / proba.sum(axis=1, keepdims=True)
    scores[valid] = NEUTRAL_SCORE + 50.0 * (proba[:, BUY_CLASS] - proba[:, SELL_CLASS])
    return scores


def predict_ml_score(row: pd.DataFrame, rf, xgb) -> float:
    """ML Score cho dòng đầu tiên của `row`."""
    return float(predict_ml_scores(row.iloc[[0]], rf, xgb)[0])
