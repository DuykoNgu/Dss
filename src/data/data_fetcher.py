"""
src/data_fetcher.py
-------------------
Module đảm nhiệm việc thu thập dữ liệu lịch sử giá cổ phiếu (OHLCV) từ thư viện vnstock.
Các tính năng chính:
- Tự động quét danh sách rổ cổ phiếu VN30 (kèm danh sách dự phòng - fallback).
- Thu thập dữ liệu lịch sử giá của từng cổ phiếu trong VN30 và chỉ số VNINDEX.
- Hỗ trợ tải song song (đa luồng - Multithreading) để tối ưu thời gian tải dữ liệu.
- Cơ chế retry thông minh: xử lý lỗi mạng và tự động chờ dãn cách khi gặp Rate Limit của API.
- Tự động kiểm tra cache dữ liệu: không tải lại các mã đã có file CSV.
"""

import sys
import os

# Thêm thư mục gốc của dự án (DSS/) vào sys.path để có thể import config từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import pandas as pd
from vnstock import Market, Reference
import config

# ==============================================================================
# CẤU HÌNH & HẰNG SỐ (CONSTANTS)
# ==============================================================================

# Đường dẫn đến thư mục chứa dữ liệu đầu ra: DSS/data/ (lấy từ config làm chuẩn duy nhất)
DATA_DIR = os.path.join(config.BASE_DIR, "data")
assert os.path.basename(DATA_DIR) == "data" and DATA_DIR.endswith("DSS/data"), DATA_DIR

# Cấu hình cơ chế thử lại (Retry) khi gọi API
MAX_RETRIES = 5         # Số lần thử lại tối đa khi gọi API thất bại
RETRY_BASE_DELAY = 5    # Thời gian chờ cơ bản (giây) giữa các lần thử: delay = RETRY_BASE_DELAY * (attempt + 1)
RATE_LIMIT_WAIT = 70    # Thời gian chờ (giây) khi bị chặn tần suất gọi API (Rate Limit)

# Cấu hình retry khi lưu từng mã cổ phiếu
SAVE_MAX_RETRIES = 3    # Số lần thử lại tối đa cho mỗi mã cổ phiếu
SAVE_RETRY_DELAY = 8    # Thời gian nghỉ (giây) giữa các lần thử lại lưu mã

# Cấu hình đa luồng (Multi-threading) để tải dữ liệu song song
MAX_WORKERS = 3         # Số luồng chạy đồng thời (giữ mức vừa phải để tránh bị API chặn vì spam request)
WORKER_DELAY = 4        # Thời gian nghỉ (giây) sau mỗi tác vụ worker nhằm giảm áp lực lên server API

# Danh sách 30 mã cổ phiếu rổ VN30 mặc định (dùng làm fallback khi API lấy danh sách gặp sự cố)
DEFAULT_VN30_SYMBOLS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR",
    "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB",
    "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
    "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]


# ==============================================================================
# CÁC HÀM XỬ LÝ & THU THẬP DỮ LIỆU
# ==============================================================================

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa định dạng DataFrame OHLCV sau khi lấy về từ API:
    - Loại bỏ khoảng trắng thừa ở tên các cột.
    - Kiểm tra sự tồn tại của cột mốc thời gian ('time').
    - Chuyển đổi cột 'time' sang kiểu datetime chuẩn.
    - Sắp xếp dữ liệu theo thứ tự thời gian tăng dần và reset lại index.

    Args:
        df (pd.DataFrame): DataFrame dữ liệu thô lấy từ API.

    Returns:
        pd.DataFrame: DataFrame đã được làm sạch và chuẩn hóa, hoặc DataFrame rỗng nếu thiếu cột 'time'.
    """
    df = df.copy()
    # Chuẩn hóa tên cột: xóa khoảng trắng đầu/cuối tên cột
    df.columns = [c.strip() for c in df.columns]
    
    # Kiểm tra cột bắt buộc 'time'
    if "time" not in df.columns:
        return pd.DataFrame()
        
    # Ép kiểu dữ liệu thời gian sang datetime để tiện phân tích chuỗi thời gian
    df["time"] = pd.to_datetime(df["time"])
    
    # Sắp xếp từ ngày cũ đến ngày mới nhất và đánh lại chỉ số dòng
    return df.sort_values("time").reset_index(drop=True)


def fetch_vn30_symbols() -> list[str]:
    """
    Lấy danh sách mã chứng khoán thuộc rổ VN30 mới nhất từ vnstock Reference API:
    - Thử quét danh sách VN30 qua API Reference().equity.list_by_group("VN30").
    - Kiểm tra tính hợp lệ: nếu danh sách trả về đủ tin cậy (>= 20 mã), sử dụng danh sách này.
    - Nếu có lỗi hoặc dữ liệu trả về không đầy đủ (< 20 mã), tự động kích hoạt cơ chế dự phòng
      và sử dụng danh sách mặc định `DEFAULT_VN30_SYMBOLS`.

    Returns:
        list[str]: Danh sách các mã cổ phiếu trong rổ VN30 (định dạng chữ hoa).
    """
    try:
        print("📡 [Phase 1] Quét rổ VN30 mới nhất...")
        ref = Reference()
        # Gọi API lấy danh sách cổ phiếu theo nhóm VN30
        symbols = ref.equity.list_by_group("VN30").tolist()
        # Chuẩn hóa mã cổ phiếu (viết hoa, bỏ khoảng trắng)
        symbols = [s.strip().upper() for s in symbols if s]
        
        # Nếu lấy được từ 20 mã trở lên thì coi là hợp lệ
        if len(symbols) >= 20:
            print(f"✅ Tìm thấy {len(symbols)} mã VN30.")
            return symbols
        print(f"⚠️ VN30 trả về thiếu ({len(symbols)} mã), dùng fallback.")
    except Exception as e:
        print(f"⚠️ Không quét được VN30: {e}")
        
    # Trường hợp xảy ra lỗi hoặc thiếu mã -> Sử dụng danh sách tĩnh mặc định
    print("ℹ️ Dùng danh sách VN30 mặc định (30 mã).")
    return list(DEFAULT_VN30_SYMBOLS)


def fetch_stock_ohlcv(symbol: str,
                      start_date: str = config.START_DATE,
                      end_date: str = config.END_DATE) -> pd.DataFrame:
    """
    Lấy dữ liệu lịch sử giá nến (OHLCV) của một mã cổ phiếu cụ thể từ vnstock Market API:
    - Có cơ chế thử lại (retry) tối đa MAX_RETRIES lần nếu gặp lỗi kết nối.
    - Tự động nhận diện lỗi Rate Limit (giới hạn tần suất) và dừng chờ RATE_LIMIT_WAIT giây trước khi thử lại.
    - Áp dụng thời gian chờ tăng dần theo cấp số tuyến tính cho các lỗi mạng thông thường.

    Args:
        symbol (str): Mã cổ phiếu cần lấy dữ liệu (VD: 'FPT', 'HPG',...).
        start_date (str): Ngày bắt đầu lấy dữ liệu (mặc định lấy từ config).
        end_date (str): Ngày kết thúc lấy dữ liệu (mặc định lấy từ config).

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu OHLCV đã chuẩn hóa, hoặc rỗng nếu không lấy được.
    """
    symbol = symbol.strip().upper()
    for attempt in range(MAX_RETRIES):
        try:
            # Gọi API vnstock lấy dữ liệu giá nến cổ phiếu
            df = Market().equity(symbol=symbol).ohlcv(
                start=start_date, end=end_date, count=config.DATA_COUNT
            )
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Chuẩn hóa dữ liệu đầu ra
            normalized = normalize_ohlcv(df)
            if normalized.empty:
                return pd.DataFrame()
            return normalized
        except Exception as e:
            # Kiểm tra nếu lỗi do bị rate limit từ máy chủ dữ liệu
            is_rate_limit = "rate limit" in str(e).lower() or "giới hạn" in str(e).lower()
            if is_rate_limit:
                print(f"  ⏳ {symbol}: rate limit, chờ {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            
            # Nếu là lỗi khác và còn số lần thử lại, tăng thời gian chờ rồi thử lại
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (attempt + 1))
            else:
                return pd.DataFrame()
    return pd.DataFrame()


def fetch_market_index(index_code: str = "VNINDEX",
                       start_date: str = config.START_DATE,
                       end_date: str = config.END_DATE) -> pd.DataFrame:
    """
    Lấy dữ liệu lịch sử giá nến (OHLCV) của chỉ số thị trường (mặc định là VNINDEX):
    - Tương tự fetch_stock_ohlcv, nhưng sử dụng hàm Market().index() chuyên dụng cho chỉ số.
    - Có cơ chế retry và xử lý Rate Limit tương tự nhằm đảm bảo độ tin cậy.

    Args:
        index_code (str): Mã chỉ số thị trường (mặc định: 'VNINDEX').
        start_date (str): Ngày bắt đầu (từ config).
        end_date (str): Ngày kết thúc (từ config).

    Returns:
        pd.DataFrame: DataFrame dữ liệu nến của chỉ số thị trường đã được chuẩn hóa.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Gọi API vnstock lấy dữ liệu chỉ số thị trường
            df = Market().index(symbol=index_code).ohlcv(
                start=start_date, end=end_date, count=config.DATA_COUNT
            )
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Chuẩn hóa dữ liệu
            normalized = normalize_ohlcv(df)
            if normalized.empty:
                return pd.DataFrame()
            return normalized
        except Exception as e:
            # Bắt lỗi rate limit
            is_rate_limit = "rate limit" in str(e).lower() or "giới hạn" in str(e).lower()
            if is_rate_limit:
                print(f"  ⏳ {index_code}: rate limit, chờ {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            
            # Thử lại với delay tăng dần
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (attempt + 1))
            else:
                return pd.DataFrame()
    return pd.DataFrame()


def merge_and_save(csv_path: str, new_df: pd.DataFrame) -> tuple[int, int]:
    """Gộp nến mới vào CSV, khử trùng theo time. Trả về (tổng dòng, dòng mới thêm)."""
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
    combined.to_csv(csv_path, index=False)
    return len(combined), len(combined) - old_len


def fetch_one_symbol(sym: str) -> tuple[str, dict]:
    csv_path = os.path.join(DATA_DIR, "stocks", f"{sym}.csv")
    df = pd.DataFrame()

    # Vòng lặp thử tải dữ liệu cho mã cổ phiếu
    for _ in range(SAVE_MAX_RETRIES):
        try:
            df = fetch_stock_ohlcv(sym)
            if not df.empty:
                break  # Tải thành công thì thoát vòng lặp retry
        except Exception:
            pass
        time.sleep(SAVE_RETRY_DELAY)

    # Nếu có dữ liệu hợp lệ -> Lưu vào file CSV
    if not df.empty:
        total, _ = merge_and_save(csv_path, df)
        info = {"rows": total, "columns": list(df.columns), "cached": False}
    else:
        info = {"rows": 0, "error": "no data"}

    # Thời gian nghỉ của worker để dãn cách các request
    time.sleep(WORKER_DELAY)
    return sym, info


def update_one_symbol(sym: str) -> tuple[str, dict]:
    """Chỉ tải nến mới sau ngày cuối trong CSV rồi append (không tải lại lịch sử)."""
    csv_path = os.path.join(DATA_DIR, "stocks", f"{sym}.csv")
    try:
        old = pd.read_csv(csv_path, usecols=["time"])
        last_date = pd.to_datetime(old["time"]).max().date()
    except Exception:
        return fetch_one_symbol(sym)

    today = pd.to_datetime(config.END_DATE).date()
    # Lưu ý: nến hôm nay (chưa đóng cửa) chỉ được lấy 1 lần trong ngày;
    # các lần chạy sau cùng ngày skip để khỏi tốn request. DSS chạy theo nến ngày.
    if last_date >= today:
        df = pd.read_csv(csv_path)
        return sym, {"rows": len(df), "columns": list(df.columns), "cached": True}

    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    new_df = pd.DataFrame()
    for _ in range(SAVE_MAX_RETRIES):
        try:
            new_df = fetch_stock_ohlcv(sym, start_date=start, end_date=config.END_DATE)
            if not new_df.empty:
                break
        except Exception:
            pass
        time.sleep(SAVE_RETRY_DELAY)

    if new_df.empty:
        df = pd.read_csv(csv_path)
        info = {"rows": len(df), "columns": list(df.columns), "cached": True, "note": "no new candles"}
    else:
        total, added = merge_and_save(csv_path, new_df)
        info = {"rows": total, "columns": list(new_df.columns), "cached": False, "added": added}
    time.sleep(WORKER_DELAY)
    return sym, info


def fetch_latest_quote(symbol: str) -> dict:
    """Polling nhẹ giá/bảng lệnh hiện tại (1 call), không tải lại lịch sử OHLCV."""
    symbol = symbol.strip().upper()
    try:
        q = Market().equity(symbol=symbol).quote()
        if q is None or (hasattr(q, "empty") and q.empty):
            return {}
        row = q.iloc[0].to_dict() if hasattr(q, "iloc") else dict(q)
        return {
            "symbol": symbol,
            "price": row.get("close_price", row.get("price")),
            "open": row.get("open_price"),
            "high": row.get("high_price"),
            "low": row.get("low_price"),
            "volume": row.get("volume_accumulated", row.get("volume")),
            "change": row.get("price_change"),
            "pct_change": row.get("percent_change"),
        }
    except Exception:
        return {}


def save_data(symbols: list[str], df_index: pd.DataFrame, full_sync: bool = False) -> dict:
    """Lưu trữ toàn bộ dữ liệu và điều phối tải song song.

    symbols.json: full_sync=True (chạy full rổ) thì ghi đè; ngược lại hợp nhất
    với manifest cũ để lần chạy giới hạn (--limit/--symbols) không làm mất list.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "stocks"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "index"), exist_ok=True)

    # Manifest symbols.json: full rổ thì ghi đè, chạy giới hạn thì hợp nhất
    manifest = os.path.join(DATA_DIR, "symbols.json")
    if full_sync or not os.path.exists(manifest):
        manifest_symbols = list(symbols)
    else:
        try:
            old_symbols = json.load(open(manifest))
        except Exception:
            old_symbols = []
        manifest_symbols = list(dict.fromkeys(list(old_symbols) + list(symbols)))
    with open(manifest, "w") as f:
        json.dump(manifest_symbols, f, indent=2, ensure_ascii=False)

    # Lưu dữ liệu chỉ số VNINDEX vào file CSV (gộp + khử trùng, không ghi đè)
    index_path = os.path.join(DATA_DIR, "index", "VNINDEX.csv")
    if df_index is None or df_index.empty:
        print("⚠️ VNINDEX rỗng, vẫn lưu symbols + stocks.")
    else:
        total, added = merge_and_save(index_path, df_index)
        print(f"📈 VNINDEX: {total} dòng (+{added} mới)")

    summary = {"VN30_count": len(symbols), "symbols": symbols}
    missing, stale = [], []

    # Phân loại: mã chưa có file -> tải full; mã đã có -> chỉ cập nhật nến mới
    for sym in symbols:
        csv_path = os.path.join(DATA_DIR, "stocks", f"{sym}.csv")
        if os.path.exists(csv_path):
            stale.append(sym)
        else:
            missing.append(sym)

    print(f"📦 {len(missing)} mã tải mới, 🔄 {len(stale)} mã kiểm tra nến mới.")
    tasks = [(s, fetch_one_symbol) for s in missing] + [(s, update_one_symbol) for s in stale]

    # Nếu có các mã cần tải/cập nhật -> Dùng ThreadPoolExecutor tải đa luồng
    if tasks:
        print(f"🚀 Đồng bộ song song {len(tasks)} mã với {MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Đưa từng task vào pool luồng
            futures = {executor.submit(fn, sym): sym for sym, fn in tasks}
            # Lắng nghe kết quả khi từng luồng hoàn thành
            for future in as_completed(futures):
                sym, info = future.result()
                summary[sym] = info
                if info.get("rows", 0) > 0:
                    extra = f" (+{info['added']} mới)" if "added" in info else ""
                    mark = "⏭️" if info.get("cached") else "✅"
                    print(f"  {mark} {sym}: {info['rows']} rows{extra}")
                else:
                    print(f"  ❌ {sym}: no data")

    return summary


# ==============================================================================
# ĐIỂM CHẠY CHÍNH (ENTRYPOINT)
# ==============================================================================
