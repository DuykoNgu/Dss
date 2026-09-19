"""
Phase 1: Thu thập OHLCV từ vnstock và cập nhật SQLite.

- Quét rổ VN30 (có danh sách dự phòng khi API lỗi).
- Tải song song, retry lỗi mạng, chờ khi gặp rate limit.
- Mã chưa có dữ liệu hoặc lịch sử ngắn hơn LOOKBACK_YEARS: tải full.
- Mã đã có dữ liệu: tải chồng REFRESH_DAYS ngày cuối rồi upsert.
- Giá điều chỉnh bị đổi (cổ tức, chia tách): tải full lại để cả chuỗi cùng
  một cơ sở giá.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Thêm thư mục gốc dự án vào sys.path để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

import config
from src.data import store

# Retry khi gọi API
MAX_RETRIES = 5         # Số lần thử tối đa mỗi request
RETRY_BASE_DELAY = 5    # Chờ RETRY_BASE_DELAY * (attempt + 1) giây giữa các lần thử
RATE_LIMIT_WAIT = 70    # Chờ khi bị chặn tần suất

MAX_WORKERS = 3
# Giãn cách mọi lần gọi API (chung cho mọi luồng). Gói Guest của vnstock cho 20 đơn vị
# quota/phút và mỗi lần tải OHLCV tốn 2 đơn vị -> tối đa ~10 lần/phút. Có API key
# (community, 60 đơn vị/phút) thì có thể giảm xuống ~2,5 giây.
SECONDS_PER_CALL = 7

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
MARKET_CLOSE_HOUR = 15          # HOSE đóng cửa 14:45; từ 15:00 coi nến ngày đã chốt
REFRESH_DAYS = 10               # Số ngày lịch tải chồng lên cuối dữ liệu mỗi lần cập nhật
BACKFILL_TOLERANCE_DAYS = 30    # Lịch sử bắt đầu muộn hơn START_DATE quá mức này -> tải full
ADJUSTMENT_TOLERANCE = 0.005    # Giá đóng cửa cũ/mới cùng ngày lệch > 0,5% -> giá đã điều chỉnh

# Rổ VN30 (cập nhật 09/2026), chỉ dùng khi API lấy danh sách gặp sự cố
DEFAULT_VN30_SYMBOLS = [
    "ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR", "HDB",
    "HPG", "LPB", "MBB", "MCH", "MSN", "MWG", "SAB", "SHB",
    "SSB", "SSI", "STB", "TCB", "TCX", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
]


_throttle_lock = threading.Lock()
_next_call_at = 0.0


def _throttle() -> None:
    global _next_call_at
    with _throttle_lock:
        wait = _next_call_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _next_call_at = time.monotonic() + SECONDS_PER_CALL


def _vnstock():
    """Import vnstock khi thực sự gọi API.

    Import vnstock tự ghi file chỉ dẫn AI (AGENTS.md...) vào thư mục đang chạy,
    nên tắt tính năng đó trước khi import. Import muộn cũng giúp code offline
    (clean, train, backtest, test) không phụ thuộc vnstock.
    """
    os.environ.setdefault("VNSTOCK_DISABLE_AGENT_SETUP", "1")
    import vnstock
    return vnstock


def drop_unclosed_candle(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Bỏ nến của hôm nay khi phiên chưa đóng cửa để không cache nến dở dang."""
    now = now or datetime.now(VN_TZ)
    if df.empty or now.hour >= MARKET_CLOSE_HOUR:
        return df
    return df[df["time"].dt.date < now.date()].reset_index(drop=True)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột, ép time về datetime, sort tăng dần, bỏ nến chưa chốt."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "time" not in df.columns:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"]).astype("datetime64[ns]")
    return drop_unclosed_candle(df.sort_values("time").reset_index(drop=True))


def needs_full_refetch(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Cùng ngày mà giá đóng cửa cũ/mới lệch nhau -> lịch sử đã được điều chỉnh lại."""
    old = old[["time", "close"]].assign(time=lambda frame: frame["time"].dt.normalize())
    new = new[["time", "close"]].assign(time=lambda frame: frame["time"].dt.normalize())
    overlap = old.merge(
        new, on="time", suffixes=("_old", "_new")
    )
    drift = (overlap["close_old"] / overlap["close_new"] - 1).abs()
    return bool((drift > ADJUSTMENT_TOLERANCE).any())


def fetch_vn30_symbols() -> list[str]:
    """Lấy rổ VN30 từ API; lỗi hoặc trả về < 20 mã thì dùng danh sách dự phòng."""
    try:
        print("📡 [Phase 1] Quét rổ VN30 mới nhất...")
        _throttle()
        symbols = _vnstock().Reference().equity.list_by_group("VN30").tolist()
        symbols = [s.strip().upper() for s in symbols if s]
        if len(symbols) >= 20:
            print(f"✅ Tìm thấy {len(symbols)} mã VN30.")
            return symbols
        print(f"⚠️ VN30 trả về thiếu ({len(symbols)} mã), dùng fallback.")
    except Exception as e:
        print(f"⚠️ Không quét được VN30: {e}")
    print("ℹ️ Dùng danh sách VN30 mặc định (30 mã).")
    return list(DEFAULT_VN30_SYMBOLS)


def _fetch_ohlcv(name: str, source, start_date: str, end_date: str) -> pd.DataFrame:
    """Gọi source().ohlcv có retry; trả DataFrame rỗng nếu thất bại."""
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            df = source().ohlcv(start=start_date, end=end_date, count=config.DATA_COUNT)
            return pd.DataFrame() if df is None or df.empty else normalize_ohlcv(df)
        except (Exception, SystemExit) as e:
            # vnai gọi sys.exit khi vượt quota -> bắt lại để chờ và thử tiếp, không dừng cả tiến trình
            message = str(e).lower()
            if "rate limit" in message or "giới hạn" in message:
                print(f"  ⏳ {name}: rate limit, chờ {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
            elif isinstance(e, SystemExit):
                raise
            elif attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (attempt + 1))
    return pd.DataFrame()


def fetch_stock_ohlcv(symbol: str,
                      start_date: str = config.START_DATE,
                      end_date: str | None = None) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    return _fetch_ohlcv(symbol, lambda: _vnstock().Market().equity(symbol=symbol),
                        start_date, end_date or datetime.now(VN_TZ).strftime("%Y-%m-%d"))


def fetch_market_index(index_code: str = "VNINDEX",
                       start_date: str = config.START_DATE,
                       end_date: str | None = None) -> pd.DataFrame:
    return _fetch_ohlcv(index_code, lambda: _vnstock().Market().index(symbol=index_code),
                        start_date, end_date or datetime.now(VN_TZ).strftime("%Y-%m-%d"))


def _fetch_symbol_frame(symbol: str) -> tuple[str, pd.DataFrame, bool]:
    latest = store.latest_date(symbol)
    first = store.first_date(symbol)
    backfill_from = pd.Timestamp(config.START_DATE) + timedelta(days=BACKFILL_TOLERANCE_DAYS)
    replace = not (latest and first and pd.Timestamp(first) <= backfill_from)
    if not replace:
        start = (pd.Timestamp(latest) - timedelta(days=REFRESH_DAYS)).strftime("%Y-%m-%d")
        frame = fetch_stock_ohlcv(symbol, start_date=start)
        if frame.empty:
            raise RuntimeError(f"{symbol}: API không trả nến")
        if needs_full_refetch(store.read_bars(symbol), frame):
            frame = fetch_stock_ohlcv(symbol)
            replace = True
    else:
        frame = fetch_stock_ohlcv(symbol)
    if frame.empty:
        raise RuntimeError(f"{symbol}: API không trả nến")
    return symbol, frame, replace


def fetch_latest_quotes(symbols: list[str]) -> dict[str, dict]:
    """Lấy bảng giá cả rổ bằng một request, không ghi nến trong phiên vào CSV."""
    if not symbols:
        return {}
    try:
        _throttle()
        board = _vnstock().Market().quote(symbol=symbols)
        if board is None or board.empty:
            return {}
        return {str(row["symbol"]).upper(): row.to_dict() for _, row in board.iterrows()
                if str(row.get("symbol", "")).upper() in symbols}
    except Exception as error:
        print(f"Không lấy được bảng giá: {error}", flush=True)
        return {}


def save_data(symbols: list[str], df_index: pd.DataFrame, full_sync: bool = False) -> dict:
    """Fetch before writing, then publish the index and stocks in one transaction."""
    if df_index is None or df_index.empty:
        raise RuntimeError("VNINDEX: API không trả nến")
    index_rows = store.candle_rows(store.INDEX_SYMBOL, df_index)
    valid_dates = [row[1] for row in index_rows if row[7]]
    if not valid_dates or max(valid_dates) != max(row[1] for row in index_rows):
        raise RuntimeError("VNINDEX: nến mới nhất không hợp lệ; giữ dữ liệu cũ")
    target = max(valid_dates)
    frames, replacements = _fetch_symbol_frames(symbols)
    for symbol, frame in frames.items():
        valid_dates = [row[1] for row in store.candle_rows(symbol, frame) if row[7]]
        if not valid_dates or max(valid_dates) != target:
            raise RuntimeError(f"{symbol}: thiếu nến hợp lệ ngày {target}; giữ dữ liệu cũ")
    manifest = symbols if full_sync else sorted(set(store.current_symbols()) | set(symbols))
    rejected = store.write_batch({store.INDEX_SYMBOL: df_index, **frames}, manifest, replacements)
    print(f"Đồng bộ {target}: {len(symbols)} mã; {sum(rejected.values())} nến không hợp lệ đã đánh dấu")
    return {"VN30_count": len(symbols), "symbols": symbols, "rejected": rejected}


def sync_symbols(symbols: list[str]) -> dict:
    frames, replacements = _fetch_symbol_frames(symbols)
    rejected = store.write_batch(frames, replace_symbols=replacements)
    return {"symbols": symbols, "rejected": rejected}


def _fetch_symbol_frames(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], set[str]]:
    frames = {}
    replacements = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_fetch_symbol_frame, symbol) for symbol in symbols]
        for future in as_completed(futures):
            symbol, frame, replace = future.result()
            frames[symbol] = frame
            if replace:
                replacements.add(symbol)
    return frames, replacements
