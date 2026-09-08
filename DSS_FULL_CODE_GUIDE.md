# 📈 HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH MUA BÁN CỔ PHIẾU (STOCK DSS — VN30)
> **Tài liệu toàn diện & Mã nguồn chi tiết 7 Phase**  
> Tuân thủ chính xác theo bản thiết kế `implementation_plan.md` (đã phê duyệt)

**Scope:** Rổ VN30 (tự động) · Giao dịch ngắn hạn T+5 · Phân tích kỹ thuật + ML · Có Backtest

---

## 📑 MỤC LỤC
1. [Tổng quan luồng hoạt động](#1-tổng-quan-luồng-hoạt-động)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt môi trường (requirements.txt)](#3-cài-đặt-môi-trường-requirementstxt)
4. [File Cấu hình (config.py)](#4-file-cấu-hình-configpy)
5. [Phase 1: Thu thập dữ liệu VN30 tự động (src/data_fetcher.py)](#5-phase-1-thu-thập-dữ-liệu-vn30-tự-động)
6. [Phase 2: Làm sạch dữ liệu (src/data_cleaner.py)](#6-phase-2-làm-sạch-dữ-liệu)
7. [Phase 3: Chỉ báo kỹ thuật (src/indicators.py)](#7-phase-3-chỉ-báo-kỹ-thuật)
8. [Phase 4: Feature Engineering & Labeling (src/features.py)](#8-phase-4-feature-engineering--labeling)
9. [Phase 5: Machine Learning Models (src/ml_models.py)](#9-phase-5-machine-learning-models)
10. [Phase 6: Scoring & Decision Engine (src/scoring.py & src/decision.py)](#10-phase-6-scoring--decision-engine)
11. [Phase 7: Backtesting (src/backtester.py)](#11-phase-7-backtesting)
12. [File thực thi chính (main.py & backtest_runner.py)](#12-file-thực-thi-chính)
13. [Hướng dẫn chạy & Kiểm thử](#13-hướng-dẫn-chạy--kiểm-thử)

---

## 1. Tổng quan luồng hoạt động

```
[Tự động quét rổ VN30]
         │
         ▼
[Phase 1: vnstock API] ────> Lấy 3 năm OHLCV (30 mã) + VNINDEX
         │
         ▼
[Phase 2: Data Cleaner] ───> Xử lý Null, Chuẩn hóa, Flag Outlier
         │
         ▼
[Phase 3: Indicators] ─────> SMA, EMA, MACD, RSI, Stoch, BB, ATR, OBV
         │
         ▼
[Phase 4: Features] ───────> Crossover, Positions, Returns, Labeling T+5
         │
         ├───► [Phase 5: ML Models] ──────────> RF + XGBoost ──► ML Score (40%)
         │                                                            │
         └───► [Phase 6A: Rule-based Score] ──> 5 nhóm chấm ──► Rule Score (60%)
                                                                      │
                                                                      ▼
                                                         [Phase 6B: Total Score & Signal]
                                                                      │
                                                         ┌────────────┴────────────┐
                                                         ▼                         ▼
                                                [Bảng Khuyến Nghị]     [Phase 7: Backtest]
                                                [Terminal 30 mã VN30]  [So sánh vs Buy&Hold]
```

---

## 2. Cấu trúc thư mục

```
DSS/
├── DSS_FULL_CODE_GUIDE.md   # File hướng dẫn này
├── requirements.txt         # Thư viện
├── config.py                # Cấu hình chung
├── main.py                  # Chạy khuyến nghị hôm nay
├── backtest_runner.py       # Chạy kiểm chứng lịch sử
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py      # Phase 1: Lấy VN30 tự động
│   ├── data_cleaner.py      # Phase 2: Cleaning
│   ├── indicators.py        # Phase 3: Technical Indicators
│   ├── features.py          # Phase 4: Features & Labels
│   ├── ml_models.py         # Phase 5: RF + XGBoost
│   ├── scoring.py           # Phase 6A: Rule-based
│   ├── decision.py          # Phase 6B: Tổng hợp & Terminal
│   └── backtester.py        # Phase 7: Giả lập giao dịch
└── models/                  # Lưu model .pkl (tự động tạo)
```

---

## 3. Cài đặt môi trường (`requirements.txt`)

```txt
vnstock
pandas
numpy
ta
scikit-learn
xgboost
joblib
tabulate
colorama
```

```bash
pip install -r requirements.txt
```

---

## 4. File Cấu hình (`config.py`)

```python
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
ML_FORWARD_DAYS = 5         # Dự đoán T+5
ML_PROFIT_THRESHOLD = 0.03  # Ngưỡng +3% MUA, -3% BÁN
TEST_SIZE_RATIO = 0.2       # Walk-forward: 20% cuối làm Test

# ── 3. Trọng số quyết định ──
WEIGHT_RULE_BASED = 0.60
WEIGHT_ML_MODEL = 0.40

# ── 4. Ngưỡng tín hiệu (0-100) ──
SCORE_STRONG_BUY = 75
SCORE_BUY = 60
SCORE_HOLD = 40
SCORE_SELL = 25

# ── 5. Backtest ──
BACKTEST_MONTHS = 6         # Kiểm chứng 6 tháng gần nhất

# ── 6. Đường dẫn ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
```

---

## 5. Phase 1: Thu thập dữ liệu VN30 tự động

### File: `src/data_fetcher.py`
```python
"""
Phase 1: Tự động quét danh sách VN30 từ vnstock và kéo dữ liệu OHLCV
"""
import pandas as pd
from vnstock import Vnstock, stock_historical_data, get_index_series
import config


def fetch_vn30_symbols() -> list[str]:
    """
    Tự động lấy danh sách 30 mã thuộc rổ VN30 tại thời điểm hiện tại.
    """
    try:
        print("📡 [Phase 1] Đang quét danh sách rổ VN30 mới nhất...")
        stock = Vnstock().stock(symbol='ACB', source='VCI')
        listing = stock.listing.symbols_by_group('VN30')
        symbols = listing.tolist() if hasattr(listing, 'tolist') else list(listing)
        print(f"✅ Tìm thấy {len(symbols)} mã VN30: {', '.join(symbols[:5])}...")
        return symbols
    except Exception as e:
        print(f"⚠️ Không thể tự động quét VN30: {e}")
        print("ℹ️ Sử dụng danh sách VN30 mặc định.")
        return [
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR",
            "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB",
            "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
            "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]


def fetch_stock_ohlcv(symbol: str, start_date: str = config.START_DATE, end_date: str = config.END_DATE) -> pd.DataFrame:
    """
    Lấy dữ liệu giá lịch sử OHLCV cho 1 mã cổ phiếu.
    """
    symbol = symbol.strip().upper()
    try:
        print(f"📡 Đang tải dữ liệu {symbol} ({start_date} → {end_date})...")
        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        if df is None or df.empty:
            print(f"❌ Không có dữ liệu cho mã {symbol}")
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"❌ Lỗi tải {symbol}: {e}")
        return pd.DataFrame()


def fetch_market_index(index_code: str = "VNINDEX", time_range: str = "ThreeYears") -> pd.DataFrame:
    """
    Lấy dữ liệu chỉ số thị trường VNINDEX.
    """
    try:
        print(f"📡 Đang tải chỉ số {index_code}...")
        df = get_index_series(index_code=index_code, time_range=time_range)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Không thể tải {index_code}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    symbols = fetch_vn30_symbols()
    print(f"\nDanh sách VN30 ({len(symbols)} mã): {symbols}")
    df = fetch_stock_ohlcv("TCB")
    print(df.tail())
```

---

## 6. Phase 2: Làm sạch dữ liệu

### File: `src/data_cleaner.py`
```python
"""
Phase 2: Làm sạch, chuẩn hóa dữ liệu OHLCV và VNINDEX
"""
import pandas as pd
import numpy as np


def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Chuẩn hóa types (float, int, datetime)
    - Sắp xếp theo ngày tăng dần
    - Forward fill + Backward fill missing values
    - Flag outlier biến động > 6.8% (Trần/sàn HOSE)
    """
    if df.empty:
        return df

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Chuẩn hóa cột ngày
    date_col = 'tradingDate' if 'tradingDate' in df.columns else 'time'
    df['tradingDate'] = pd.to_datetime(df[date_col])
    df = df.sort_values('tradingDate').reset_index(drop=True)

    # Chuẩn hóa kiểu số
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

    # Fill missing values
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill().bfill()

    # Flag biến động bất thường
    df['daily_return'] = df['close'].pct_change()
    df['is_extreme'] = df['daily_return'].abs() >= 0.068

    df = df.dropna(subset=['close']).reset_index(drop=True)
    return df


def clean_index_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu VNINDEX
    """
    if df.empty:
        return df

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    date_col = 'tradingDate' if 'tradingDate' in df.columns else 'time'
    df['tradingDate'] = pd.to_datetime(df[date_col])
    df = df.sort_values('tradingDate').reset_index(drop=True)

    val_col = 'indexValue' if 'indexValue' in df.columns else 'close'
    df['indexValue'] = pd.to_numeric(df[val_col], errors='coerce').ffill().bfill()

    return df[['tradingDate', 'indexValue']]


if __name__ == "__main__":
    from data_fetcher import fetch_stock_ohlcv
    raw = fetch_stock_ohlcv("FPT")
    clean = clean_ohlcv_data(raw)
    print("\n--- TEST PHASE 2 ---")
    print(clean.info())
    print(clean.tail())
```

---

## 7. Phase 3: Chỉ báo kỹ thuật

### File: `src/indicators.py`
```python
"""
Phase 3: Tính toán chỉ báo kỹ thuật (4 nhóm: Trend, Momentum, Volatility, Volume)
"""
import pandas as pd
import ta


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        print("⚠️ Dữ liệu quá ngắn để tính chỉ báo.")
        return df

    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # ── 1. Xu hướng (Trend) ──
    df['sma_10'] = ta.trend.sma_indicator(close, window=10)
    df['sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['sma_50'] = ta.trend.sma_indicator(close, window=50)
    df['sma_200'] = ta.trend.sma_indicator(close, window=200)
    df['ema_12'] = ta.trend.ema_indicator(close, window=12)
    df['ema_26'] = ta.trend.ema_indicator(close, window=26)

    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # ── 2. Động lượng (Momentum) ──
    df['rsi'] = ta.momentum.rsi(close, window=14)

    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['williams_r'] = ta.momentum.williams_r(high=high, low=low, close=close, lbp=14)

    # ── 3. Biến động (Volatility) ──
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-9)

    df['atr'] = ta.volatility.average_true_range(high=high, low=low, close=close, window=14)
    df['atr_pct'] = (df['atr'] / (close + 1e-9)) * 100

    # ── 4. Khối lượng (Volume) ──
    df['obv'] = ta.volume.on_balance_volume(close, volume)
    df['vol_sma_20'] = ta.trend.sma_indicator(volume.astype(float), window=20)
    df['vol_ratio'] = volume / (df['vol_sma_20'] + 1e-9)

    return df


if __name__ == "__main__":
    from data_fetcher import fetch_stock_ohlcv
    from data_cleaner import clean_ohlcv_data
    df = clean_ohlcv_data(fetch_stock_ohlcv("HPG"))
    df = calculate_technical_indicators(df)
    print("\n--- TEST PHASE 3 ---")
    print(df[['tradingDate', 'close', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'vol_ratio']].tail())
```

---

## 8. Phase 4: Feature Engineering & Labeling

### File: `src/features.py`
```python
"""
Phase 4: Tạo Features phái sinh cho ML và gán nhãn T+5 (MUA/GIỮ/BÁN)
"""
import pandas as pd
import numpy as np
import ta
import config

FEATURE_COLUMNS = [
    'price_vs_sma50', 'price_vs_sma200', 'sma50_vs_sma200',
    'macd_hist', 'macd_hist_slope',
    'rsi', 'stoch_k', 'stoch_d', 'williams_r',
    'bb_position', 'bb_width', 'atr_pct',
    'vol_ratio', 'obv_slope',
    'return_1d', 'return_5d', 'return_20d',
    'vnindex_vs_sma50', 'vnindex_return_5d'
]


def build_features_and_labels(stock_df: pd.DataFrame, index_df: pd.DataFrame = None,
                              forward_days: int = config.ML_FORWARD_DAYS,
                              threshold: float = config.ML_PROFIT_THRESHOLD) -> pd.DataFrame:
    df = stock_df.copy()

    # ── 1. Features vị trí giá ──
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-9) * 100
    df['price_vs_sma200'] = (df['close'] - df['sma_200']) / (df['sma_200'] + 1e-9) * 100
    df['sma50_vs_sma200'] = (df['sma_50'] - df['sma_200']) / (df['sma_200'] + 1e-9) * 100

    # Độ dốc MACD Histogram
    df['macd_hist_slope'] = df['macd_hist'] - df['macd_hist'].shift(3)

    # Vị trí trong Bollinger Bands (0=Lower, 1=Upper)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-9)

    # Độ dốc OBV
    df['obv_slope'] = (df['obv'] - df['obv'].shift(5)) / (df['obv'].shift(5).abs() + 1e-9)

    # Returns quá khứ
    df['return_1d'] = df['close'].pct_change(1) * 100
    df['return_5d'] = df['close'].pct_change(5) * 100
    df['return_20d'] = df['close'].pct_change(20) * 100

    # Tín hiệu giao cắt (dùng cho Rule-based scoring)
    df['golden_cross'] = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
    df['death_cross'] = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

    # ── 2. Tích hợp VNINDEX ──
    if index_df is not None and not index_df.empty:
        idx = index_df.copy()
        idx['vnindex_sma50'] = ta.trend.sma_indicator(idx['indexValue'], window=50)
        idx['vnindex_vs_sma50'] = (idx['indexValue'] - idx['vnindex_sma50']) / (idx['vnindex_sma50'] + 1e-9) * 100
        idx['vnindex_return_5d'] = idx['indexValue'].pct_change(5) * 100

        df = pd.merge(df, idx[['tradingDate', 'vnindex_vs_sma50', 'vnindex_return_5d']], on='tradingDate', how='left')
        df['vnindex_vs_sma50'] = df['vnindex_vs_sma50'].ffill().bfill().fillna(0)
        df['vnindex_return_5d'] = df['vnindex_return_5d'].ffill().bfill().fillna(0)
    else:
        df['vnindex_vs_sma50'] = 0.0
        df['vnindex_return_5d'] = 0.0

    # ── 3. Gán nhãn T+5 ──
    df['future_return'] = df['close'].shift(-forward_days) / df['close'] - 1
    df['label'] = 0
    df.loc[df['future_return'] >= threshold, 'label'] = 1    # MUA
    df.loc[df['future_return'] <= -threshold, 'label'] = -1  # BÁN

    return df


if __name__ == "__main__":
    from data_fetcher import fetch_stock_ohlcv, fetch_market_index
    from data_cleaner import clean_ohlcv_data, clean_index_data
    from indicators import calculate_technical_indicators

    stock = clean_ohlcv_data(fetch_stock_ohlcv("VNM"))
    index = clean_index_data(fetch_market_index("VNINDEX"))
    ind = calculate_technical_indicators(stock)
    feat = build_features_and_labels(ind, index)
    print("\n--- TEST PHASE 4 ---")
    print("Phân bố nhãn:")
    print(feat['label'].value_counts(dropna=False))
```

---

## 9. Phase 5: Machine Learning Models

### File: `src/ml_models.py`
```python
"""
Phase 5: Train Random Forest + XGBoost (Walk-forward), Ensemble Predict
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import config
from features import FEATURE_COLUMNS


def train_ml_models(df: pd.DataFrame, symbol: str):
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['label']).copy()
    df_pool = df_clean.iloc[:-config.ML_FORWARD_DAYS]

    X = df_pool[FEATURE_COLUMNS]
    y = df_pool['label'].map({-1: 0, 0: 1, 1: 2})

    if len(X) < 100:
        print(f"⚠️ [Phase 5] Dữ liệu {symbol} quá ít ({len(X)} mẫu).")
        return None, None

    # Walk-forward split
    split_idx = int(len(X) * (1 - config.TEST_SIZE_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"🎓 [Phase 5] Train {symbol} (Train: {len(X_train)}, Test: {len(X_test)})...")

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight='balanced', random_state=42
    )
    rf_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42
    )
    xgb_model.fit(X_train, y_train)

    acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"✅ Accuracy -> RF: {acc_rf:.1%}, XGBoost: {acc_xgb:.1%}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(rf_model, os.path.join(config.MODEL_DIR, f"{symbol}_rf.pkl"))
    joblib.dump(xgb_model, os.path.join(config.MODEL_DIR, f"{symbol}_xgb.pkl"))

    return rf_model, xgb_model


def predict_latest_signal(df: pd.DataFrame, rf_model, xgb_model) -> tuple[float, tuple]:
    latest_row = df[FEATURE_COLUMNS].iloc[[-1]]
    if latest_row.isnull().values.any():
        latest_row = latest_row.ffill().bfill().fillna(0)

    p_rf = rf_model.predict_proba(latest_row)[0]
    p_xgb = xgb_model.predict_proba(latest_row)[0]

    avg_proba = (p_rf + p_xgb) / 2.0
    ml_score = avg_proba[2] * 100  # P(MUA) * 100

    return ml_score, tuple(avg_proba)
```

---

## 10. Phase 6: Scoring & Decision Engine

### File: `src/scoring.py`
```python
"""
Phase 6A: Chấm điểm Rule-based (5 nhóm: Xu hướng, Động lượng, Khối lượng, Biến động, VNINDEX)
"""
import pandas as pd


def calculate_rule_based_score(df: pd.DataFrame) -> tuple[float, list[str]]:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    score = 50.0
    reasons = []

    # ── 1. Xu hướng (30 điểm) ──
    if latest['close'] > latest['sma_50'] > latest['sma_200']:
        score += 10
        reasons.append("Uptrend mạnh (Giá > SMA50 > SMA200)")
    elif latest['close'] < latest['sma_50'] < latest['sma_200']:
        score -= 8
        reasons.append("Downtrend (Giá < SMA50 < SMA200)")

    if df['golden_cross'].iloc[-20:].any():
        score += 8
        reasons.append("Golden Cross gần đây")
    elif df['death_cross'].iloc[-20:].any():
        score -= 10
        reasons.append("Death Cross gần đây")

    if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > prev['macd_hist']:
        score += 7
        reasons.append("MACD tích cực & Histogram mở rộng")
    elif latest['macd'] < latest['macd_signal']:
        score -= 10
        reasons.append("MACD dưới Signal")

    if latest['close'] > latest['sma_50']:
        score += 5

    # ── 2. Động lượng (25 điểm) ──
    rsi = latest['rsi']
    if 30 <= rsi <= 50 and rsi > prev['rsi']:
        score += 10
        reasons.append(f"RSI phục hồi ({rsi:.1f})")
    elif 50 < rsi <= 60:
        score += 7
    elif rsi > 80:
        score -= 12
        reasons.append(f"RSI quá mua mạnh ({rsi:.1f})")
    elif rsi > 70:
        score -= 8
        reasons.append(f"RSI quá mua ({rsi:.1f})")

    if latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']:
        score += 8
        reasons.append("Stochastic cắt mua vùng Oversold")
    elif latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
        score -= 8

    # ── 3. Khối lượng (20 điểm) ──
    vol_ratio = latest['vol_ratio']
    price_change = (latest['close'] - prev['close']) / (prev['close'] + 1e-9)

    if vol_ratio >= 1.5 and price_change > 0:
        score += 10
        reasons.append(f"Volume bùng nổ ({vol_ratio:.1f}x) + Giá tăng")
    elif vol_ratio >= 1.5 and price_change < -0.02:
        score -= 10
        reasons.append(f"Áp lực bán tháo ({vol_ratio:.1f}x)")

    if len(df) >= 5 and df['obv'].iloc[-1] > df['obv'].iloc[-5]:
        score += 5
    elif len(df) >= 5 and df['obv'].iloc[-1] < df['obv'].iloc[-5]:
        score -= 5

    # ── 4. Biến động (15 điểm) ──
    bb_pos = latest['bb_position']
    if bb_pos < 0.2 and latest['close'] > prev['close']:
        score += 8
        reasons.append("Bật tăng từ dải dưới BB")
    elif 0.2 <= bb_pos <= 0.5:
        score += 4
    if bb_pos >= 0.95:
        score -= 5
        reasons.append("Chạm dải trên BB")

    if latest['atr_pct'] < 3.0:
        score += 3
    elif latest['atr_pct'] > 5.0:
        score -= 5

    # ── 5. Thị trường VNINDEX (10 điểm) ──
    vnindex_vs_sma50 = latest.get('vnindex_vs_sma50', 0)
    vnindex_return_5d = latest.get('vnindex_return_5d', 0)

    if vnindex_vs_sma50 > 0:
        score += 5
    else:
        score -= 5

    if vnindex_return_5d > 1.0:
        score += 3
    elif vnindex_return_5d < -3.0:
        score -= 5
        reasons.append("VNINDEX giảm mạnh tuần qua")

    return max(0.0, min(100.0, score)), reasons
```

---

### File: `src/decision.py`
```python
"""
Phase 6B: Tổng hợp điểm & xuất bảng Terminal
"""
from colorama import Fore, Style, init
from tabulate import tabulate
import config

init(autoreset=True)


def generate_decision(symbol: str, current_price: float, rule_score: float, ml_score: float, reasons: list[str]) -> dict:
    total_score = (rule_score * config.WEIGHT_RULE_BASED) + (ml_score * config.WEIGHT_ML_MODEL)

    if total_score >= config.SCORE_STRONG_BUY:
        signal = f"{Fore.GREEN}{Style.BRIGHT}🟢 MUA MẠNH{Style.RESET_ALL}"
    elif total_score >= config.SCORE_BUY:
        signal = f"{Fore.GREEN}🟡 MUA{Style.RESET_ALL}"
    elif total_score >= config.SCORE_HOLD:
        signal = f"{Fore.YELLOW}⚪ GIỮ{Style.RESET_ALL}"
    elif total_score >= config.SCORE_SELL:
        signal = f"{Fore.MAGENTA}🟠 BÁN{Style.RESET_ALL}"
    else:
        signal = f"{Fore.RED}{Style.BRIGHT}🔴 BÁN MẠNH{Style.RESET_ALL}"

    return {
        "symbol": symbol,
        "price": f"{current_price:,.0f} đ",
        "total_score": f"{total_score:.1f}",
        "rule_score": f"{rule_score:.1f}",
        "ml_score": f"{ml_score:.1f}",
        "signal": signal,
        "primary_reason": "; ".join(reasons[:2]) if reasons else "Trung tính"
    }


def print_terminal_report(results: list[dict]):
    headers = ["MÃ", "GIÁ", "TỔNG", "RULES (60%)", "ML (40%)", "KHUYẾN NGHỊ", "LÝ DO"]
    table_data = []
    for r in results:
        table_data.append([
            f"{Fore.CYAN}{Style.BRIGHT}{r['symbol']}{Style.RESET_ALL}",
            r['price'], f"{Style.BRIGHT}{r['total_score']}{Style.RESET_ALL}",
            r['rule_score'], r['ml_score'], r['signal'], r['primary_reason']
        ])

    print("\n" + "=" * 95)
    print(f"{Fore.YELLOW}{Style.BRIGHT}       DSS KHUYẾN NGHỊ CỔ PHIẾU VN30{Style.RESET_ALL}")
    print("=" * 95)
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print(f"\n💡 Tổng điểm = 60% Rules + 40% ML Ensemble (RF + XGBoost).")
```

---

## 11. Phase 7: Backtesting

### File: `src/backtester.py`
```python
"""
Phase 7: Giả lập giao dịch lịch sử — So sánh DSS vs Buy & Hold
"""
import pandas as pd
import numpy as np
import config
from features import FEATURE_COLUMNS
from scoring import calculate_rule_based_score
from ml_models import predict_latest_signal


def run_backtest(feat_df: pd.DataFrame, rf_model, xgb_model,
                 backtest_months: int = config.BACKTEST_MONTHS) -> dict:
    """
    Giả lập giao dịch trên dữ liệu lịch sử:
    - Quét từng phiên trong khoảng backtest
    - Mua khi Total Score >= SCORE_BUY (60)
    - Bán (chốt) sau T+5 phiên hoặc khi Score < SCORE_SELL (25)
    - So sánh tổng lợi nhuận với Buy & Hold
    """
    # Lấy khoảng backtest (N tháng cuối)
    backtest_days = backtest_months * 21  # ~21 phiên/tháng
    if len(feat_df) < backtest_days + 50:
        backtest_days = len(feat_df) - 50  # Đảm bảo đủ data cho indicators

    df_backtest = feat_df.iloc[-backtest_days:].copy()

    trades = []
    holding = False
    buy_price = 0
    buy_date = None
    hold_days = 0

    for i in range(len(df_backtest)):
        row = df_backtest.iloc[i]
        # Lấy slice dữ liệu từ đầu đến phiên hiện tại để tính score
        current_idx = df_backtest.index[i]
        df_up_to_now = feat_df.loc[:current_idx]

        if len(df_up_to_now) < 50:
            continue

        # Tính Rule Score
        rule_score, _ = calculate_rule_based_score(df_up_to_now)

        # Tính ML Score
        try:
            latest_features = df_up_to_now[FEATURE_COLUMNS].iloc[[-1]]
            if latest_features.isnull().values.any():
                latest_features = latest_features.ffill().bfill().fillna(0)
            p_rf = rf_model.predict_proba(latest_features)[0]
            p_xgb = xgb_model.predict_proba(latest_features)[0]
            ml_score = ((p_rf + p_xgb) / 2.0)[2] * 100
        except:
            ml_score = 50.0

        total_score = rule_score * config.WEIGHT_RULE_BASED + ml_score * config.WEIGHT_ML_MODEL

        current_price = row['close']
        current_date = row['tradingDate']

        if not holding:
            # Tìm điểm MUA
            if total_score >= config.SCORE_BUY:
                holding = True
                buy_price = current_price
                buy_date = current_date
                hold_days = 0
        else:
            hold_days += 1
            # Chốt lời/lỗ sau T+5 hoặc khi có tín hiệu BÁN
            if hold_days >= config.ML_FORWARD_DAYS or total_score < config.SCORE_SELL:
                pnl_pct = (current_price - buy_price) / buy_price * 100
                trades.append({
                    'buy_date': buy_date,
                    'sell_date': current_date,
                    'buy_price': buy_price,
                    'sell_price': current_price,
                    'pnl_pct': pnl_pct,
                    'hold_days': hold_days
                })
                holding = False

    # Tính Buy & Hold
    start_price = df_backtest['close'].iloc[0]
    end_price = df_backtest['close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price * 100

    # Tính tổng lợi nhuận DSS
    if trades:
        trades_df = pd.DataFrame(trades)
        dss_total_return = trades_df['pnl_pct'].sum()
        win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
        avg_pnl = trades_df['pnl_pct'].mean()
    else:
        dss_total_return = 0.0
        win_rate = 0.0
        avg_pnl = 0.0
        trades_df = pd.DataFrame()

    return {
        'trades': trades_df,
        'num_trades': len(trades),
        'dss_return': dss_total_return,
        'buy_hold_return': buy_hold_return,
        'win_rate': win_rate,
        'avg_pnl_per_trade': avg_pnl,
        'backtest_start': df_backtest['tradingDate'].iloc[0],
        'backtest_end': df_backtest['tradingDate'].iloc[-1],
    }


def print_backtest_report(symbol: str, result: dict):
    """In kết quả backtest cho 1 mã"""
    print(f"\n{'─' * 60}")
    print(f"📊 BACKTEST: {symbol}")
    print(f"   Giai đoạn: {result['backtest_start'].strftime('%Y-%m-%d')} → {result['backtest_end'].strftime('%Y-%m-%d')}")
    print(f"   Số lệnh: {result['num_trades']}")
    print(f"   Tỷ lệ thắng: {result['win_rate']:.1f}%")
    print(f"   Lãi TB/lệnh: {result['avg_pnl_per_trade']:.2f}%")
    print(f"   ──────────────────────────────────")
    print(f"   📈 Lợi nhuận DSS:       {result['dss_return']:+.2f}%")
    print(f"   📉 Lợi nhuận Buy&Hold:  {result['buy_hold_return']:+.2f}%")

    diff = result['dss_return'] - result['buy_hold_return']
    if diff > 0:
        print(f"   ✅ DSS vượt trội: +{diff:.2f}%")
    else:
        print(f"   ⚠️ Buy&Hold tốt hơn: {diff:.2f}%")
```

---

## 12. File thực thi chính

### File: `main.py` (Chạy khuyến nghị hôm nay)
```python
"""
main.py — Chạy DSS cho toàn bộ rổ VN30, xuất bảng khuyến nghị Terminal
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_fetcher import fetch_vn30_symbols, fetch_stock_ohlcv, fetch_market_index
from data_cleaner import clean_ohlcv_data, clean_index_data
from indicators import calculate_technical_indicators
from features import build_features_and_labels
from ml_models import train_ml_models, predict_latest_signal
from scoring import calculate_rule_based_score
from decision import generate_decision, print_terminal_report


def run_dss():
    print("=" * 65)
    print(" 📈 HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH MUA BÁN CỔ PHIẾU (DSS — VN30)")
    print("=" * 65)

    # 1. Lấy danh sách VN30 tự động
    symbols = fetch_vn30_symbols()

    # 2. Lấy VNINDEX dùng chung
    raw_index = fetch_market_index("VNINDEX")
    clean_index = clean_index_data(raw_index)

    all_results = []
    for symbol in symbols:
        print(f"\n{'='*20} 🔍 {symbol} {'='*20}")

        raw_df = fetch_stock_ohlcv(symbol)
        if raw_df.empty:
            continue

        clean_df = clean_ohlcv_data(raw_df)
        ind_df = calculate_technical_indicators(clean_df)
        feat_df = build_features_and_labels(ind_df, clean_index)

        rf_model, xgb_model = train_ml_models(feat_df, symbol)
        if rf_model is None:
            ml_score = 50.0
        else:
            ml_score, _ = predict_latest_signal(feat_df, rf_model, xgb_model)

        rule_score, reasons = calculate_rule_based_score(feat_df)
        current_price = clean_df['close'].iloc[-1]
        decision = generate_decision(symbol, current_price, rule_score, ml_score, reasons)
        all_results.append(decision)

    if all_results:
        print_terminal_report(all_results)
    else:
        print("❌ Không có kết quả.")


if __name__ == "__main__":
    run_dss()
```

---

### File: `backtest_runner.py` (Chạy kiểm chứng lịch sử)
```python
"""
backtest_runner.py — Chạy Backtest cho VN30, so sánh DSS vs Buy & Hold
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_fetcher import fetch_vn30_symbols, fetch_stock_ohlcv, fetch_market_index
from data_cleaner import clean_ohlcv_data, clean_index_data
from indicators import calculate_technical_indicators
from features import build_features_and_labels
from ml_models import train_ml_models
from backtester import run_backtest, print_backtest_report


def run_all_backtests():
    print("=" * 65)
    print(" 📊 KIỂM CHỨNG HIỆU SUẤT DSS (BACKTEST — VN30)")
    print("=" * 65)

    symbols = fetch_vn30_symbols()
    raw_index = fetch_market_index("VNINDEX")
    clean_index = clean_index_data(raw_index)

    total_dss = 0.0
    total_bh = 0.0
    count = 0

    for symbol in symbols:
        print(f"\n{'='*20} 🔍 {symbol} {'='*20}")

        raw_df = fetch_stock_ohlcv(symbol)
        if raw_df.empty:
            continue

        clean_df = clean_ohlcv_data(raw_df)
        ind_df = calculate_technical_indicators(clean_df)
        feat_df = build_features_and_labels(ind_df, clean_index)

        rf_model, xgb_model = train_ml_models(feat_df, symbol)
        if rf_model is None:
            continue

        result = run_backtest(feat_df, rf_model, xgb_model)
        print_backtest_report(symbol, result)

        total_dss += result['dss_return']
        total_bh += result['buy_hold_return']
        count += 1

    if count > 0:
        print(f"\n{'='*65}")
        print(f"📈 TỔNG KẾT ({count} mã VN30):")
        print(f"   Lợi nhuận TB DSS:       {total_dss/count:+.2f}%")
        print(f"   Lợi nhuận TB Buy&Hold:  {total_bh/count:+.2f}%")
        print(f"{'='*65}")


if __name__ == "__main__":
    run_all_backtests()
```

---

## 13. Hướng dẫn chạy & Kiểm thử

### Bước 1: Cài đặt thư viện
```bash
cd /Users/phanducduy/Desktop/ml+dl/DSS
pip install -r requirements.txt
```

### Bước 2: Test từng Phase
```bash
python src/data_fetcher.py    # Test quét VN30 + fetch data
python src/data_cleaner.py    # Test cleaning
python src/indicators.py      # Test chỉ báo kỹ thuật
python src/features.py        # Test features & labeling
```

### Bước 3: Chạy khuyến nghị hôm nay
```bash
python main.py
```
→ Tự động quét 30 mã VN30, train ML, tính điểm và in bảng khuyến nghị Terminal.

### Bước 4: Chạy kiểm chứng lịch sử (Backtest)
```bash
python backtest_runner.py
```
→ Giả lập giao dịch 6 tháng qua cho 30 mã VN30, so sánh lợi nhuận DSS vs Buy & Hold.
