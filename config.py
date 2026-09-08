"""
config.py — Cấu hình toàn hệ thống DSS (VN30 Scope)
"""
import os
from datetime import datetime, timedelta

# ── 1. Thời gian lấy dữ liệu ──
LOOKBACK_YEARS = 3
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")

# ── 2. Machine Learning ──
ML_FORWARD_DAYS = 5
ML_PROFIT_THRESHOLD = 0.03
TEST_SIZE_RATIO = 0.2

# ── 3. Trọng số quyết định ──
WEIGHT_RULE_BASED = 0.60
WEIGHT_ML_MODEL = 0.40

# ── 4. Ngưỡng tín hiệu (0-100) ──
SCORE_STRONG_BUY = 75
SCORE_BUY = 60
SCORE_HOLD = 40
SCORE_SELL = 25

# ── 5. Backtest ──
BACKTEST_MONTHS = 6

# ── 6. Đường dẫn ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── 7. vnstock API ──
DATA_COUNT = 1000