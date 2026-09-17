"""
config.py — Cấu hình toàn hệ thống DSS (VN30 Scope)
"""
import os
from datetime import datetime, timedelta

# ── 1. Thời gian lấy dữ liệu ──
# 8 năm = giới hạn lịch sử nến ngày của vnstock bản community
LOOKBACK_YEARS = 8
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")

# ── 2. Machine Learning ──
ML_FORWARD_DAYS = 5
ML_PROFIT_THRESHOLD = 0.03
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
MIN_TRAIN_ROWS = 100
BASELINE_MOMENTUM_THRESHOLD = 1.0  # percentage points over the last 5 sessions

# Random Forest ("phòng thủ"): 200 cây độc lập, lá >=20 mẫu chống overfit,
# balanced vì nhãn GIỮ thường chiếm đa số
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# XGBoost ("tấn công"): cây nông + học chậm để khỏi overfit dữ liệu nhiễu
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
}

# True: 1 cặp model học chung dữ liệu mọi mã; False: mỗi mã 1 cặp model riêng
ML_POOLED = False
# Model cache cũ hơn dữ liệu quá N ngày (hoặc khác feature schema) sẽ tự train lại
MODEL_MAX_AGE_DAYS = 7

# ── 3. Trọng số quyết định ──
WEIGHT_RULE_BASED = 0.60
WEIGHT_ML_MODEL = 0.40

# ── 4. Ngưỡng tín hiệu (0-100) ──
SCORE_STRONG_BUY = 75
SCORE_BUY = 60
SCORE_HOLD = 40
SCORE_SELL = 25

# ── 5. Backtest ──
BACKTEST_MONTHS = 12
BACKTEST_RETRAIN_DAYS = 20      # Walk-forward: train lại model mỗi N phiên
BACKTEST_FEE_RATE = 0.0015      # Phí môi giới mỗi chiều (0.15%)
BACKTEST_TAX_RATE = 0.001       # Thuế bán chứng khoán (0.1%)
BACKTEST_SLIPPAGE_RATE = 0.001  # Trượt giá giả định mỗi chiều (0.1%)
BACKTEST_PORTFOLIO_SLOTS = 5    # Backtest danh mục: chia vốn thành N phần bằng nhau
# T+2: cổ phiếu mua phiên T về tài khoản chiều T+2 -> lệnh bán tại giá mở cửa sớm nhất là T+3
MIN_DAYS_BEFORE_OPEN_SELL = 3
ML_LABEL_COST_RATE = (
    2 * BACKTEST_FEE_RATE + BACKTEST_TAX_RATE + 2 * BACKTEST_SLIPPAGE_RATE
)

# ── 6. Đường dẫn ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STOCKS_DIR = os.path.join(DATA_DIR, "stocks")
INDEX_PATH = os.path.join(DATA_DIR, "index", "VNINDEX.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# ── 7. vnstock API ──
# Số nến tối đa mỗi lần tải: ~260 phiên/năm, dư một chút để không cắt mất đầu kỳ
DATA_COUNT = LOOKBACK_YEARS * 260
