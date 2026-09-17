"""Phase 2: Làm sạch và chuẩn hóa dữ liệu OHLCV + VNINDEX."""

import pandas as pd

OHLC_COLUMNS = ["open", "high", "low", "close"]
REQUIRED_OHLCV_COLUMNS = {"time", *OHLC_COLUMNS, "volume"}

# Ngưỡng cảnh báo biến động mạnh — CHỈ đúng với nến NGÀY (daily, ~trần/sàn HOSE 7%).
# Nến phút/tick phải định nghĩa ngưỡng riêng, không tái dùng hằng này.
EXTREME_DAILY_RETURN_THRESHOLD = 0.068


def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa types, sort theo time, fill missing, gắn cờ cảnh báo."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if not REQUIRED_OHLCV_COLUMNS.issubset(df.columns):
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    for col in OHLC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        # NaN = "không lấy được dữ liệu", khác với 0 = "không có giao dịch" -> giữ cờ
        df["volume_missing"] = df["volume"].isna()
        df["volume"] = df["volume"].fillna(0).astype(int)

    # CHỈ ffill (lấy giá quá khứ đắp vào lỗ hổng). CẤM bfill vì đó là
    # lấy dữ liệu TƯƠNG LAI điền về quá khứ -> leakage cho ML.
    # NaN đầu file (không có quá khứ để fill) sẽ bị drop ở cuối.
    present_ohlc = [c for c in OHLC_COLUMNS if c in df.columns]
    df[present_ohlc] = df[present_ohlc].ffill()

    # Return của 1 kỳ (không gắn timeframe để dùng được cho cả nến phút)
    df["period_return"] = df["close"].pct_change()
    df["is_extreme"] = df["period_return"].abs() >= EXTREME_DAILY_RETURN_THRESHOLD

    # Nến sai logic (VD: high < close) -> flag để phase sau biết, không tự xóa
    df["is_ohlc_invalid"] = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
    )

    return df.dropna(subset=["time", "close"]).reset_index(drop=True)


def clean_index_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch VNINDEX, luôn trả về 2 cột time + indexValue."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "indexValue" not in df.columns:
        if "close" not in df.columns:
            return pd.DataFrame()
        df["indexValue"] = pd.to_numeric(df["close"], errors="coerce")

    if "time" in df.columns:
        time_col = df["time"]
    elif "tradingDate" in df.columns:
        time_col = df["tradingDate"]
    else:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(time_col, errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    df["indexValue"] = pd.to_numeric(df["indexValue"], errors="coerce").ffill()
    return df[["time", "indexValue"]].dropna().reset_index(drop=True)
