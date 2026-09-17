"""Điểm chạy chính của DSS: nạp dữ liệu VN30 -> làm sạch -> chỉ báo -> features -> khuyến nghị.

Ví dụ:
    python main.py                          # full VN30, đồng bộ incremental
    python main.py --no-fetch --limit 2     # chạy offline 2 mã từ cache
    python main.py --symbols FPT,HPG        # chỉ chạy mã chỉ định
"""

import argparse
import sys

import config
from src.pipeline import cached_symbols, load_featured, load_market_index, parse_symbols
from src.scoring import generate_decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hệ thống hỗ trợ quyết định mua/bán VN30.")
    parser.add_argument("--symbols", default="", help="VD: FPT,HPG (mặc định: quét rổ VN30)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số mã (test nhanh)")
    parser.add_argument("--no-fetch", action="store_true", help="Không gọi API, chỉ dùng CSV cache")
    parser.add_argument("--fetch-only", action="store_true", help="Chỉ đồng bộ dữ liệu rồi dừng")
    parser.add_argument("--with-history", action="store_true",
                        help="Đồng bộ thêm các mã từng thuộc VN30 (cho backtest --universe history)")
    parser.add_argument("--retrain", action="store_true", help="Train lại model thay vì ưu tiên model cache")
    parser.add_argument("--pooled", action="store_true", default=config.ML_POOLED,
                        help="Dùng 1 cặp model học chung dữ liệu mọi mã")
    return parser.parse_args()


def get_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        symbols = parse_symbols(args.symbols)
    elif args.no_fetch:
        symbols = cached_symbols()
    else:
        from src.data import fetch_vn30_symbols
        symbols = fetch_vn30_symbols()
    return symbols[: args.limit] if args.limit > 0 else symbols


def print_report(rows: list[dict]) -> None:
    latest = max(r["time"] for r in rows)
    print(f"\nKhuyến nghị theo nến ngày {latest:%Y-%m-%d} (ML '–' = thiếu model/feature, dùng 50)")
    print(f"{'Mã':<6}{'Giá':>10}{'Tổng':>7}{'Rules':>7}{'ML':>7}  Khuyến nghị  Lý do")
    for r in rows:
        ml = f"{r['ml_score']:>7.1f}" if r["ml_available"] else f"{'–':>7}"
        stale = f" (dữ liệu đến {r['time']:%Y-%m-%d})" if r["time"] < latest else ""
        print(f"{r['symbol']:<6}{r['price']:>10.2f}{r['total_score']:>7.1f}"
              f"{r['rule_score']:>7.1f}{ml}  {r['signal']:<10} {r['primary_reason']}{stale}")


def main() -> int:
    args = parse_args()
    symbols = get_symbols(args)
    if not symbols:
        print("Không có mã nào để chạy.", file=sys.stderr)
        return 1

    # [1/4] Đồng bộ dữ liệu: mã thiếu tải full, mã cũ tải chồng đoạn cuối
    if not args.no_fetch:
        from src.data import fetch_market_index, save_data
        print(f"[1/4] Đồng bộ dữ liệu {len(symbols)} mã...")
        full_sync = not args.symbols and not args.limit
        save_data(symbols, fetch_market_index(), full_sync=full_sync)
        if args.with_history:
            from src.data import sync_symbols
            from src.data.universe import load_vn30_changes, vn30_symbols_ever
            former = sorted(set(vn30_symbols_ever(load_vn30_changes())) - set(symbols))
            print(f"      Đồng bộ thêm {len(former)} mã từng thuộc VN30: {', '.join(former)}")
            sync_symbols(former)
    if args.fetch_only:
        print("Xong đồng bộ dữ liệu.")
        return 0

    # [2/4] Làm sạch + chỉ báo + features (nhãn T+5 chỉ dùng để train)
    print("[2/4] Làm sạch, tính chỉ báo, dựng features...")
    market = load_market_index()
    featured = {symbol: load_featured(symbol, market) for symbol in symbols}
    featured = {symbol: df for symbol, df in featured.items() if not df.empty}
    skipped = sorted(set(symbols) - set(featured))
    if skipped:
        print(f"      Bỏ qua (thiếu cache hoặc quá ít dữ liệu): {', '.join(skipped)}", file=sys.stderr)
    if not featured:
        print("Không có dữ liệu sạch để xử lý.", file=sys.stderr)
        return 1
    print(f"      Xong {len(featured)}/{len(symbols)} mã.")

    # [3/4] Model: dùng cache nếu còn mới, cũ hơn dữ liệu thì tự train lại
    print(f"[3/4] Nạp/train model ({'pooled' if args.pooled else 'mỗi mã'})...")
    # [4/4] Chấm điểm Rule + ML, tổng hợp 60/40 thành tín hiệu
    rows = generate_decisions(featured, force_retrain=args.retrain, pooled=args.pooled)
    print("[4/4] Chấm điểm + khuyến nghị.")

    print_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
