"""
Phase 1: Thu thập OHLCV từ vnstock và cache CSV.

- Quét rổ VN30 (có danh sách dự phòng khi API lỗi).
- Tải song song, retry lỗi mạng, chờ khi gặp rate limit.
- Mã chưa có cache hoặc lịch sử ngắn hơn LOOKBACK_YEARS: tải full rồi ghi đè.
- Mã đã có cache: tải chồng REFRESH_DAYS ngày cuối rồi gộp, để nến bị lưu dở
  được thay bằng nến đã chốt.
- Giá điều chỉnh bị đổi (cổ tức, chia tách): tải full lại để cả chuỗi cùng
  một cơ sở giá.
"""

import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Thêm thư mục gốc dự án vào sys.path để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

import config

DATA_DIR = config.DATA_DIR

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
REFRESH_DAYS = 10               # Số ngày lịch tải chồng lên cuối cache mỗi lần cập nhật
BACKFILL_TOLERANCE_DAYS = 30    # Cache bắt đầu muộn hơn START_DATE quá mức này -> tải full
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
    overlap = old[["time", "close"]].merge(
        new[["time", "close"]], on="time", suffixes=("_old", "_new")
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


def merge_and_save(csv_path: str, new_df: pd.DataFrame) -> tuple[int, int]:
    """Gộp nến mới vào CSV, trùng time thì giữ bản mới. Trả về (tổng dòng, dòng mới thêm)."""
    old_len = 0
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        old_len = len(old)
        # Align schema theo file cũ để API đổi cột giữa 2 lần fetch không tạo cột rác
        new_df = new_df.reindex(columns=old.columns)
        combined = pd.concat([old, new_df], ignore_index=True) if not new_df.empty else old
    else:
        combined = new_df
    if combined.empty:
        return 0, 0
    combined["time"] = pd.to_datetime(combined["time"])
    combined = combined.drop_duplicates(subset=["time"], keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)
    save_csv_atomic(csv_path, combined)
    return len(combined), len(combined) - old_len


def save_csv_atomic(csv_path: str, frame: pd.DataFrame) -> None:
    """Replace a CSV only after the complete new file is on disk."""
    directory = os.path.dirname(csv_path)
    os.makedirs(directory, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, suffix=".tmp",
                                         delete=False, encoding="utf-8") as output:
            temporary = output.name
            frame.to_csv(output, index=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, csv_path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _stock_path(sym: str) -> str:
    return os.path.join(DATA_DIR, "stocks", f"{sym}.csv")


def fetch_one_symbol(sym: str) -> tuple[str, dict]:
    """Tải full lịch sử và ghi đè cache. Tải lỗi thì giữ nguyên cache cũ."""
    df = fetch_stock_ohlcv(sym)
    if df.empty:
        return sym, {"rows": 0, "error": "no data"}
    save_csv_atomic(_stock_path(sym), df)
    return sym, {"rows": len(df), "cached": False, "full": True}


def update_one_symbol(sym: str) -> tuple[str, dict]:
    """Tải chồng REFRESH_DAYS ngày cuối rồi gộp; tự chuyển sang tải full khi cần."""
    csv_path = _stock_path(sym)
    try:
        old = pd.read_csv(csv_path)
        old["time"] = pd.to_datetime(old["time"])
    except Exception:
        return fetch_one_symbol(sym)

    # Cache ngắn hơn LOOKBACK_YEARS (VD: vừa tăng LOOKBACK) -> tải full để bổ sung lịch sử.
    # Mã mới niêm yết luôn rơi vào nhánh này; vẫn chỉ tốn 1 request như cập nhật thường.
    backfill_from = pd.Timestamp(config.START_DATE) + timedelta(days=BACKFILL_TOLERANCE_DAYS)
    if old["time"].min() > backfill_from:
        return fetch_one_symbol(sym)

    # Tải chồng lên đoạn cuối: nến từng bị lưu khi phiên chưa đóng cửa sẽ được ghi đè
    start = (old["time"].max() - timedelta(days=REFRESH_DAYS)).strftime("%Y-%m-%d")
    new_df = fetch_stock_ohlcv(sym, start_date=start)
    if new_df.empty:
        return sym, {"rows": len(old), "cached": True}
    if needs_full_refetch(old, new_df):
        return fetch_one_symbol(sym)

    total, added = merge_and_save(csv_path, new_df)
    return sym, {"rows": total, "cached": False, "added": added}


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
    """Lưu VNINDEX + đồng bộ song song các mã.

    symbols.json: full_sync=True (chạy full rổ) thì ghi đè; ngược lại hợp nhất
    với manifest cũ để lần chạy giới hạn (--limit/--symbols) không làm mất list.
    """
    os.makedirs(os.path.join(DATA_DIR, "stocks"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "index"), exist_ok=True)

    manifest = os.path.join(DATA_DIR, "symbols.json")
    if full_sync or not os.path.exists(manifest):
        manifest_symbols = list(symbols)
    else:
        try:
            with open(manifest) as f:
                old_symbols = json.load(f)
        except Exception:
            old_symbols = []
        manifest_symbols = list(dict.fromkeys(list(old_symbols) + list(symbols)))
    with open(manifest, "w") as f:
        json.dump(manifest_symbols, f, indent=2, ensure_ascii=False)

    if df_index is None or df_index.empty:
        print("⚠️ VNINDEX rỗng, vẫn lưu symbols + stocks.")
    else:
        total, added = merge_and_save(config.INDEX_PATH, df_index)
        print(f"📈 VNINDEX: {total} dòng (+{added} mới)")

    return sync_symbols(symbols)


def sync_symbols(symbols: list[str]) -> dict:
    """Đồng bộ song song CSV của các mã (không đụng symbols.json, VNINDEX)."""
    os.makedirs(os.path.join(DATA_DIR, "stocks"), exist_ok=True)
    summary = {"VN30_count": len(symbols), "symbols": symbols}
    missing = [s for s in symbols if not os.path.exists(_stock_path(s))]
    existing = [s for s in symbols if s not in missing]
    print(f"📦 {len(missing)} mã tải mới, 🔄 {len(existing)} mã cập nhật.")
    tasks = [(s, fetch_one_symbol) for s in missing] + [(s, update_one_symbol) for s in existing]

    if tasks:
        print(f"🚀 Đồng bộ song song {len(tasks)} mã với {MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(fn, sym) for sym, fn in tasks]
            for future in as_completed(futures):
                sym, info = future.result()
                summary[sym] = info
                if info.get("rows", 0) == 0:
                    print(f"  ❌ {sym}: no data")
                elif info.get("full"):
                    print(f"  ✅ {sym}: {info['rows']} rows (tải full)")
                elif info.get("cached"):
                    print(f"  ⏭️ {sym}: {info['rows']} rows (không đổi)")
                else:
                    print(f"  ✅ {sym}: {info['rows']} rows (+{info['added']} mới)")

    return summary
