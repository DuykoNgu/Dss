"""Phase 5: Train Random Forest + XGBoost, ensemble predict."""

import os
import sys
import tempfile
from dataclasses import asdict, dataclass

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


@dataclass(frozen=True)
class TrainingWeights:
    class_balance: str | None = "balanced"
    half_life_days: float | None = None

    def __post_init__(self):
        if self.class_balance not in (None, "balanced"):
            raise ValueError("class_balance phải là None hoặc balanced")
        if self.half_life_days is not None and (
            not np.isfinite(self.half_life_days) or self.half_life_days <= 0
        ):
            raise ValueError("half_life_days phải hữu hạn và > 0")

    def temporal_weights(self, dates: pd.Series) -> np.ndarray:
        if self.half_life_days is None:
            return np.ones(len(dates))
        age = (dates.max() - dates).dt.total_seconds() / 86400
        weights = np.exp2(-age.to_numpy() / self.half_life_days)
        return weights / weights.mean()


def model_config() -> dict:
    return {
        "data_store": "sqlite-validated-v1",
        "training_semantics": "aligned-label-end-shared-weights-v2",
        "features": list(FEATURE_COLUMNS),
        "rf": dict(config.RF_PARAMS),
        "xgb": dict(config.XGB_PARAMS),
        "profit_threshold": config.ML_PROFIT_THRESHOLD,
        "label_cost_rate": config.ML_LABEL_COST_RATE,
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }


def fit_models(x: pd.DataFrame, y: pd.Series, rf_params: dict | None = None,
               xgb_params: dict | None = None,
               sample_weight: np.ndarray | None = None) -> tuple:
    """RF và XGB dùng cùng trọng số mẫu; chỉ hiệu chỉnh phần trọng số lớp khi predict."""
    rf_options = {**config.RF_PARAMS, **(rf_params or {})}
    balance = rf_options.pop("class_weight", None)
    temporal = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if temporal.shape != (len(y),) or not np.isfinite(temporal).all() or (temporal <= 0).any():
        raise ValueError("sample_weight phải dương, hữu hạn và có cùng số dòng với y")
    class_weights = compute_class_weight(
        class_weight=balance, classes=np.array(CLASSES), y=y, sample_weight=temporal
    )
    weights = temporal * class_weights[y.to_numpy()]
    weights /= weights.mean()
    rf = RandomForestClassifier(**rf_options).fit(x, y, sample_weight=weights)
    xgb = XGBClassifier(**{**config.XGB_PARAMS, **(xgb_params or {})})
    xgb.fit(x, y, sample_weight=weights)
    priors = np.bincount(y, weights=temporal, minlength=len(CLASSES)) / temporal.sum()
    for model in (rf, xgb):
        model.class_priors_ = dict(enumerate(priors))
        model.class_weight_correction_ = 1.0 / class_weights
    return rf, xgb


def _print_holdout_metrics(train_df: pd.DataFrame, y: pd.Series, symbol: str,
                           weighting: TrainingWeights) -> None:
    """Metric tham khảo trên 20% cuối, purge theo ngày kết thúc nhãn."""
    dates = np.sort(train_df["time"].unique())
    split = int(len(dates) * (1 - config.TEST_SIZE_RATIO))
    horizon = train_df.attrs.get("model_spec", DEFAULT_MODEL_SPEC)["horizon"]
    purge_start = dates[max(0, split - horizon)]
    in_train = (train_df["time"] < purge_start).to_numpy()
    in_test = (train_df["time"] >= dates[split]).to_numpy()
    if "label_end" in train_df:
        in_train &= (train_df["label_end"] < dates[split]).to_numpy()
    if not in_test.any() or not set(CLASSES).issubset(y[in_train].unique()):
        print(f"[{symbol}] Không đủ dữ liệu để tính metric holdout.")
        return
    x = train_df[FEATURE_COLUMNS]
    rf, xgb = fit_models(
        x[in_train], y[in_train], rf_params={"class_weight": weighting.class_balance},
        sample_weight=weighting.temporal_weights(train_df.loc[in_train, "time"]),
    )
    for name, model in (("RF", rf), ("XGB", xgb)):
        metrics = classification_metrics(y[in_test], model.predict(x[in_test]))
        print(f"[{symbol}] {name} holdout ({in_test.sum()} dòng): {format_metrics(metrics)}")


def train_ml_models(
    df: pd.DataFrame,
    symbol: str,
    save: bool = True,
    verbose: bool = True,
    weighting: TrainingWeights = TrainingWeights(),
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
        _print_holdout_metrics(train_df, y, symbol, weighting)

    dates = train_df["time"] if "time" in train_df else pd.Series(pd.Timestamp("1970-01-01"), index=train_df.index)
    rf, xgb = fit_models(
        train_df[FEATURE_COLUMNS], y,
        rf_params={"class_weight": weighting.class_balance},
        sample_weight=weighting.temporal_weights(dates),
    )
    data_end = df["time"].max() if "time" in df.columns else None
    spec = df.attrs.get("model_spec", DEFAULT_MODEL_SPEC)
    for model in (rf, xgb):
        model.weighting_ = asdict(weighting)
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
    correction = getattr(rf, "class_weight_correction_", None)
    class_priors = getattr(rf, "class_priors_", None) or getattr(xgb, "class_priors_", None)
    if correction is None and class_priors:
        correction = np.array([class_priors.get(c, BALANCED_PRIOR) / BALANCED_PRIOR for c in CLASSES])
    if correction is not None:
        signals *= correction
        signals /= signals.sum(axis=1, keepdims=True)
    return signals


def predict_ml_probabilities(rows: pd.DataFrame, rf, xgb) -> np.ndarray:
    """Tín hiệu ba lớp sau hiệu chỉnh; NaN đánh dấu thiếu model hoặc feature."""
    signals = np.full((len(rows), len(CLASSES)), np.nan)
    if rf is None or xgb is None or rows.empty:
        return signals
    features = rows[FEATURE_COLUMNS]
    valid = features.notna().all(axis=1).to_numpy()
    if valid.any():
        signals[valid] = _class_signals(features[valid], rf, xgb)
    return signals


def predict_ml_scores(rows: pd.DataFrame, rf, xgb) -> np.ndarray:
    """Điểm = 50 + 50 × (tín hiệu lớp vượt chỉ số - lớp kém chỉ số)."""
    signals = predict_ml_probabilities(rows, rf, xgb)
    return np.nan_to_num(NEUTRAL_SCORE + 50 * (signals[:, BUY_CLASS] - signals[:, SELL_CLASS]),
                         nan=NEUTRAL_SCORE)


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
