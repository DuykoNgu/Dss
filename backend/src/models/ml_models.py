"""Phase 5: Train Random Forest + XGBoost, ensemble predict."""

import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
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
DEFAULT_MODEL_SPEC = {"label_strategy": "fixed", "horizon": config.ML_FORWARD_DAYS,
                      "feature_set": "baseline"}


def model_config() -> dict:
    return {
        "data_store": "sqlite-validated-v1",
        "features": list(FEATURE_COLUMNS),
        "rf": dict(config.RF_PARAMS),
        "xgb": dict(config.XGB_PARAMS),
        "profit_threshold": config.ML_PROFIT_THRESHOLD,
        "label_cost_rate": config.ML_LABEL_COST_RATE,
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }


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
    spec = df.attrs.get("model_spec", DEFAULT_MODEL_SPEC)
    for model in (rf, xgb):
        model.class_priors_ = class_priors
        model.feature_columns_ = list(FEATURE_COLUMNS)
        model.data_end_ = data_end
        model.spec_ = spec
        model.config_ = model_config()

    if save:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        destination = os.path.join(config.MODEL_DIR, f"{symbol}_bundle.pkl")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=config.MODEL_DIR, suffix=".tmp", delete=False) as output:
                temporary = output.name
            joblib.dump({"rf": rf, "xgb": xgb}, temporary, compress=3)
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        for suffix in ("rf", "xgb"):
            legacy = os.path.join(config.MODEL_DIR, f"{symbol}_{suffix}.pkl")
            if os.path.exists(legacy):
                os.unlink(legacy)
    return rf, xgb


def load_ml_models(symbol: str) -> tuple:
    """Nạp cặp model đã train, thiếu thì trả (None, None)."""
    path = os.path.join(config.MODEL_DIR, f"{symbol}_bundle.pkl")
    if not os.path.exists(path):
        return None, None
    try:
        bundle = joblib.load(path)
        return bundle["rf"], bundle["xgb"]
    except Exception as error:
        print(f"Không nạp được model {symbol}: {error}", flush=True)
        return None, None


def model_is_stale(model, data_end: pd.Timestamp, spec: dict | None = None) -> bool:
    """Model thiếu metadata, khác feature schema, hoặc cũ hơn dữ liệu quá MODEL_MAX_AGE_DAYS."""
    trained_until = getattr(model, "data_end_", None)
    if (trained_until is None or getattr(model, "config_", None) != model_config()
            or getattr(model, "spec_", None) != (spec or DEFAULT_MODEL_SPEC)):
        return True
    return data_end - trained_until > pd.Timedelta(days=config.MODEL_MAX_AGE_DAYS)


def _class_signals(features: pd.DataFrame, rf, xgb) -> np.ndarray:
    signals = (rf.predict_proba(features) + xgb.predict_proba(features)) / 2.0
    class_priors = getattr(rf, "class_priors_", None) or getattr(xgb, "class_priors_", None)
    if class_priors:
        signals *= np.array([class_priors.get(c, BALANCED_PRIOR) / BALANCED_PRIOR for c in CLASSES])
        signals /= signals.sum(axis=1, keepdims=True)
    return signals


def predict_ml_scores(rows: pd.DataFrame, rf, xgb) -> np.ndarray:
    """Điểm = 50 + 50 × (tín hiệu lớp vượt chỉ số - lớp kém chỉ số)."""
    scores = np.full(len(rows), NEUTRAL_SCORE)
    if rf is None or xgb is None or rows.empty:
        return scores
    features = rows[FEATURE_COLUMNS]
    valid = features.notna().all(axis=1).to_numpy()
    if not valid.any():
        return scores
    signals = _class_signals(features[valid], rf, xgb)
    scores[valid] = NEUTRAL_SCORE + 50.0 * (signals[:, BUY_CLASS] - signals[:, SELL_CLASS])
    return scores


def explain_ml_score(row: pd.DataFrame, rf, xgb) -> dict | None:
    """Trả các thành phần số học tạo ra điểm của một phiên đã chốt."""
    if rf is None or xgb is None or row.empty or row[FEATURE_COLUMNS].isna().any(axis=None):
        return None
    sell, hold, buy = _class_signals(row[FEATURE_COLUMNS].iloc[[0]], rf, xgb)[0]
    return {
        "below": round(float(sell) * 100, 2),
        "neutral": round(float(hold) * 100, 2),
        "above": round(float(buy) * 100, 2),
        "score": round(float(NEUTRAL_SCORE + 50 * (buy - sell)), 1),
    }


def predict_ml_score(row: pd.DataFrame, rf, xgb) -> float:
    """ML Score cho dòng đầu tiên của `row`."""
    return float(predict_ml_scores(row.iloc[[0]], rf, xgb)[0])
