"""Chạy walk-forward backtest cho VN30 và in báo cáo tổng hợp.

Ví dụ:
    python backtest_runner.py                        # full VN30, 6 tháng
    python backtest_runner.py --symbols FPT,HPG
    python backtest_runner.py --limit 5 --months 3
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pandas as pd

import config
from src.backtest import print_backtest_report, run_symbol_backtest
from src.data import clean_index_data, clean_ohlcv_data
from src.features import build_features_and_labels, calculate_technical_indicators

DATA_DIR = os.path.join(BASE_DIR, "data")
STOCKS_DIR = os.path.join(DATA_DIR, "stocks")
INDEX_PATH = os.path.join(DATA_DIR, "index", "VNINDEX.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest DSS VN30.")
    parser.add_argument("--symbols", default="", help="VD: FPT,HPG (mặc định: toàn rổ)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số mã")
    parser.add_argument("--months", type=int, default=config.BACKTEST_MONTHS,
                        help="Số tháng backtest")
    parser.add_argument("--retrain-every", type=int, default=config.BACKTEST_RETRAIN_DAYS,
                        help="Retrain model mỗi N phiên")
    return parser.parse_args()


def get_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    else:
        cache = os.path.join(DATA_DIR, "symbols.json")
        if os.path.exists(cache):
            with open(cache) as file:
                symbols = json.load(file)
        else:
            symbols = sorted(name[:-4] for name in os.listdir(STOCKS_DIR) if name.endswith(".csv"))
    return symbols[: args.limit] if args.limit > 0 else symbols


def load_market_index() -> pd.DataFrame:
    if not os.path.exists(INDEX_PATH):
        return pd.DataFrame()
    return clean_index_data(pd.read_csv(INDEX_PATH))


def build_featured(symbol: str, index_df: pd.DataFrame) -> pd.DataFrame:
    path = os.path.join(STOCKS_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    clean = clean_ohlcv_data(pd.read_csv(path))
    return build_features_and_labels(calculate_technical_indicators(clean), index_df)


def summarize(results: list[dict], months: int) -> None:
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("equity", "trades")}
                          for r in results])
    total_trades = int(frame["num_trades"].sum())
    beat = int((frame["dss_return"] > frame["buy_hold_return"]).sum())
    vnindex_mean = frame["vnindex_return"].dropna().mean() if frame["vnindex_return"].notna().any() else None

    print(f"\n{'═' * 72}")
    print(f"📈 TỔNG KẾT ({len(frame)} mã, {months} tháng)")
    print(f"   Lợi nhuận TB DSS:       {frame['dss_return'].mean():+.2%}")
    print(f"   Lợi nhuận TB Buy&Hold:  {frame['buy_hold_return'].mean():+.2%}")
    if vnindex_mean is not None:
        print(f"   Lợi nhuận TB VNINDEX:   {vnindex_mean:+.2%}")
    print(f"   Sharpe TB: {frame['sharpe'].mean():.2f} | MDD TB: {frame['max_drawdown'].mean():.2%}")
    print(f"   Tổng lệnh: {total_trades} | DSS thắng B&H: {beat}/{len(frame)} mã")
    print(f"{'═' * 72}")


def main() -> int:
    args = parse_args()
    symbols = get_symbols(args)
    index_df = load_market_index()
    if index_df.empty:
        print("Thiếu data/index/VNINDEX.csv — chạy ./run.sh fetch trước.", file=sys.stderr)
        return 1

    print(f"Backtest {len(symbols)} mã | {args.months} tháng | "
          f"retrain mỗi {args.retrain_every} phiên | phí {config.BACKTEST_FEE_RATE:.2%}/chiều")

    results = []
    for symbol in symbols:
        featured = build_featured(symbol, index_df)
        result = run_symbol_backtest(
            featured,
            symbol,
            months=args.months,
            retrain_every=args.retrain_every,
            index_df=index_df,
        )
        if result is None:
            print(f"⏭️  {symbol}: không đủ dữ liệu để backtest.")
            continue
        print_backtest_report(result)
        results.append(result)

    if not results:
        print("Không có mã nào backtest được.", file=sys.stderr)
        return 1

    summarize(results, args.months)

    os.makedirs(REPORT_DIR, exist_ok=True)
    summary_frame = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("equity", "trades")} for r in results
    ])
    summary_frame.to_csv(os.path.join(REPORT_DIR, "backtest_symbols.csv"), index=False)
    trades_frame = pd.concat(
        [r["trades"].assign(symbol=r["symbol"]) for r in results if not r["trades"].empty],
        ignore_index=True,
    ) if any(not r["trades"].empty for r in results) else pd.DataFrame()
    trades_frame.to_csv(os.path.join(REPORT_DIR, "backtest_trades.csv"), index=False)
    print(f"\n💾 Báo cáo: {REPORT_DIR}/backtest_symbols.csv, backtest_trades.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
