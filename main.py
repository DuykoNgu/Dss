"""Điểm chạy chính của DSS: nạp dữ liệu VN30 -> làm sạch -> chỉ báo -> features -> khuyến nghị.

Ví dụ:
    python main.py                          # full VN30, đồng bộ incremental
    python main.py --no-fetch --limit 2     # chạy offline 2 mã từ cache
    python main.py --symbols FPT,HPG        # chỉ chạy mã chỉ định
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pandas as pd

from src.data import (
    clean_index_data,
    clean_ohlcv_data,
    fetch_market_index,
    fetch_vn30_symbols,
    save_data,
)
from src.features import build_features_and_labels, calculate_technical_indicators
from src.scoring import generate_decisions

DATA_DIR = os.path.join(BASE_DIR, "data")
STOCKS_DIR = os.path.join(DATA_DIR, "stocks")
INDEX_PATH = os.path.join(DATA_DIR, "index", "VNINDEX.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hệ thống hỗ trợ quyết định mua/bán VN30.")
    parser.add_argument("--symbols", default="", help="VD: FPT,HPG (mặc định: quét rổ VN30)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số mã (test nhanh)")
    parser.add_argument("--no-fetch", action="store_true", help="Không gọi API, chỉ dùng CSV cache")
    parser.add_argument("--fetch-only", action="store_true", help="Chỉ đồng bộ dữ liệu rồi dừng")
    return parser.parse_args()


def get_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.no_fetch:
        cache = os.path.join(DATA_DIR, "symbols.json")
        if os.path.exists(cache):
            with open(cache) as f:
                symbols = json.load(f)
        else:
            symbols = sorted(f[:-4] for f in os.listdir(STOCKS_DIR) if f.endswith(".csv"))
    else:
        symbols = fetch_vn30_symbols()
    return symbols[: args.limit] if args.limit > 0 else symbols


def sync_data(symbols: list[str], fetch: bool, full_sync: bool) -> None:
    if fetch:
        save_data(symbols, fetch_market_index(), full_sync=full_sync)


def load_clean_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    cleaned = {}
    for sym in symbols:
        path = os.path.join(STOCKS_DIR, f"{sym}.csv")
        if not os.path.exists(path):
            print(f"Cảnh báo: thiếu cache {sym}, bỏ qua.", file=sys.stderr)
            continue
        df = clean_ohlcv_data(pd.read_csv(path))
        if not df.empty:
            cleaned[sym] = df
    return cleaned


def load_market_index() -> pd.DataFrame:
    if not os.path.exists(INDEX_PATH):
        return pd.DataFrame()
    return clean_index_data(pd.read_csv(INDEX_PATH))


def build_recommendations(featured: dict[str, pd.DataFrame]) -> list[dict]:
    return generate_decisions(featured)


def print_report(rows: list[dict]) -> None:
    print(f"{'Mã':<6}{'Giá':>10}{'Tổng':>7}{'Rules':>7}{'ML':>7}  Khuyến nghị  Lý do")
    for r in rows:
        print(f"{r['symbol']:<6}{r['price']:>10.0f}{r['total_score']:>7.1f}"
              f"{r['rule_score']:>7.1f}{r['ml_score']:>7.1f}  {r['signal']:<10} {r['primary_reason']}")


def main() -> int:
    args = parse_args()
    symbols = get_symbols(args)
    if not symbols:
        print("Không có mã nào để chạy.", file=sys.stderr)
        return 1

    # [1/5] Đồng bộ dữ liệu thô: mã thiếu tải full, mã cũ chỉ lấy nến mới
    print(f"[1/5] Đồng bộ dữ liệu {len(symbols)} mã...")
    full_sync = not args.symbols and not args.limit
    sync_data(symbols, fetch=not args.no_fetch, full_sync=full_sync)
    if args.fetch_only:
        print("Xong đồng bộ dữ liệu.")
        return 0

    # [2/5] Làm sạch: chuẩn hóa kiểu, sort time, fill thiếu, gắn cờ cảnh báo
    print("[2/5] Làm sạch dữ liệu...")
    cleaned = load_clean_data(symbols)
    if not cleaned:
        print("Không có dữ liệu sạch để xử lý.", file=sys.stderr)
        return 1
    print(f"      Xong {len(cleaned)}/{len(symbols)} mã.")

    # [3/5] Chỉ báo kỹ thuật: SMA/EMA/MACD/RSI/Stoch/BB/ATR/OBV (~25 cột)
    print("[3/5] Tính chỉ báo kỹ thuật...")
    indicated = {sym: calculate_technical_indicators(df) for sym, df in cleaned.items()}

    # [4/5] Features + nhãn: 19 features tương đối, merge VNINDEX, nhãn T+5
    print("[4/5] Dựng features...")
    market = load_market_index()
    featured = {sym: build_features_and_labels(df, market) for sym, df in indicated.items()}
    trainable = sum(len(f.dropna()) for f in featured.values())
    print(f"      Xong, {trainable} dòng trainable trên {len(featured)} mã.")

    # [5/5] Chấm điểm Rule + ML, tổng hợp 60/40 thành tín hiệu
    print("[5/5] Train ML + chấm điểm + khuyến nghị...")
    rows = build_recommendations(featured)

    print_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
