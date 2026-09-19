"""Walk-forward backtest DSS: so sánh nguồn điểm blend / rule / ml.

Ví dụ:
    python backtest_runner.py                                   # rổ hiện tại, 12 tháng, T+5
    python backtest_runner.py --pooled --universe history --months 72 --retrain-every 60
    python backtest_runner.py --label-strategy excess --horizon 20
    python backtest_runner.py --label-strategy triple_barrier --exit barrier
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pandas as pd

import config
from src.backtest import (
    DEFAULT_BARRIER,
    SCORE_MODES,
    daily_rank_ic,
    equal_weight_return,
    index_return,
    score_history,
    simulate_portfolio,
    simulate_symbol,
)
from src.data.universe import load_vn30_changes, membership_mask, vn30_symbols_ever
from src.pipeline import cached_symbols, load_featured, load_market_index, parse_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest DSS VN30.")
    parser.add_argument("--symbols", default="", help="VD: FPT,HPG (mặc định: toàn rổ)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số mã")
    parser.add_argument("--months", type=int, default=config.BACKTEST_MONTHS,
                        help="Số tháng backtest")
    parser.add_argument("--retrain-every", type=int, default=config.BACKTEST_RETRAIN_DAYS,
                        help="Retrain model mỗi N phiên")
    parser.add_argument("--slots", type=int, default=config.BACKTEST_PORTFOLIO_SLOTS,
                        help="Số phần vốn của danh mục")
    parser.add_argument("--horizon", type=int, default=config.ML_FORWARD_DAYS,
                        help="Tầm nhìn nhãn và số phiên nắm giữ tối đa")
    parser.add_argument("--label-strategy", default="fixed",
                        choices=["fixed", "volatility", "excess", "triple_barrier"],
                        help="Nhãn dùng để train model trong backtest")
    parser.add_argument("--exit", default="horizon", choices=["horizon", "barrier"],
                        help="horizon: bán khi hết tầm nhìn; barrier: thêm chốt lời/cắt lỗ theo barrier")
    parser.add_argument("--universe", default="current", choices=["current", "history"],
                        help="current: rổ VN30 hiện tại cho mọi ngày; history: thành phần VN30 theo từng kỳ")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--pooled", dest="pooled", action="store_true", default=config.ML_POOLED,
                       help="1 cặp model học chung dữ liệu mọi mã")
    scope.add_argument("--per-symbol", dest="pooled", action="store_false",
                       help="Mỗi mã 1 cặp model riêng")
    return parser.parse_args()


def load_universe(args: argparse.Namespace, index_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Nạp features; với --universe history thêm cột in_universe theo thành phần VN30 từng kỳ."""
    changes = load_vn30_changes() if args.universe == "history" else None
    if args.symbols:
        symbols = parse_symbols(args.symbols)
    elif changes is not None:
        symbols = sorted(set(vn30_symbols_ever(changes)) | set(cached_symbols()))
    else:
        symbols = cached_symbols()
    symbols = symbols[: args.limit] if args.limit > 0 else symbols

    featured = {}
    for symbol in symbols:
        df = load_featured(symbol, index_df, args.label_strategy, forward_days=args.horizon)
        if df.empty:
            print(f"⚠️  {symbol}: thiếu dữ liệu, bỏ qua (chạy python main.py --fetch-only --with-history)")
            continue
        if changes is not None:
            df["in_universe"] = membership_mask(changes, symbol, df["time"])
        featured[symbol] = df
    return featured


def main() -> int:
    args = parse_args()
    index_df = load_market_index()
    if index_df.empty:
        print("Thiếu VNINDEX trong SQLite — chạy ./run.sh fetch trước.", file=sys.stderr)
        return 1

    model_scope = "pooled" if args.pooled else "per_symbol"
    barrier = DEFAULT_BARRIER if args.exit == "barrier" else None
    run_name = (f"backtest_{model_scope}_{args.label_strategy}_h{args.horizon}"
                f"_{args.exit}_{args.universe}_{args.months}m")
    print(f"{run_name} | retrain mỗi {args.retrain_every} phiên | danh mục {args.slots} slot")

    featured = load_universe(args, index_df)
    scores = score_history(featured, args.months, args.retrain_every, args.pooled, args.horizon)
    if scores.empty:
        print("Không có mã nào backtest được.", file=sys.stderr)
        return 1

    benchmark = {
        "equal_weight_return": equal_weight_return(scores),
        "vnindex_return": index_return(scores["time"].drop_duplicates(), index_df),
    }
    summary_rows, symbol_rows, trade_frames, ic_rows = [], [], [], []
    for mode, column in SCORE_MODES.items():
        per_symbol = [
            result for _, group in scores.groupby("symbol")
            if (result := simulate_symbol(group, column, index_df, args.horizon, barrier)) is not None
        ]
        portfolio = simulate_portfolio(scores, column, args.slots, args.horizon, barrier)
        frame = pd.DataFrame([{k: v for k, v in r.items() if k != "trades"} for r in per_symbol])
        closed = pd.concat([r["trades"] for r in per_symbol] + [pd.DataFrame({"net_return": []})])
        symbol_rows.append(frame.assign(mode=mode))
        trade_frames += [r["trades"].assign(mode=mode, scope="symbol") for r in per_symbol]
        trade_frames.append(portfolio["trades"].assign(mode=mode, scope="portfolio"))

        daily_ic = daily_rank_ic(scores, column)
        yearly_ic = daily_ic.groupby(daily_ic.index.year).mean()
        year_end = portfolio["equity"].groupby(portfolio["equity"].index.year).last()
        yearly_return = year_end / year_end.shift(1, fill_value=1.0) - 1
        ic_rows += [{"mode": mode, "year": year, "rank_ic": ic, "portfolio_return": yearly_return.get(year)}
                    for year, ic in yearly_ic.items()]
        summary_rows.append({
            "mode": mode,
            "run": run_name,
            "rank_ic": daily_ic.mean(),
            "rank_ic_positive_years": f"{int((yearly_ic > 0).sum())}/{len(yearly_ic)}",
            "symbol_avg_return": frame["dss_return"].mean(),
            "symbol_beat_buy_hold": int((frame["dss_return"] > frame["buy_hold_return"]).sum()),
            "symbols": len(frame),
            "symbol_trades": int(frame["num_trades"].sum()),
            "symbol_avg_trade_return": closed["net_return"].mean() if len(closed) else 0.0,
            "portfolio_return": portfolio["portfolio_return"],
            "portfolio_trades": portfolio["num_trades"],
            "portfolio_win_rate": portfolio["win_rate"],
            "portfolio_avg_trade_return": portfolio["avg_return"],
            "portfolio_sharpe": portfolio["sharpe"],
            "portfolio_max_drawdown": portfolio["max_drawdown"],
            "portfolio_exposure": portfolio["exposure"],
            "buy_hold_avg_return": frame["buy_hold_return"].mean(),
            **benchmark,
        })

    summary = pd.DataFrame(summary_rows)
    ic_table = pd.DataFrame(ic_rows)
    start, end = scores["time"].min(), scores["time"].max()
    vnindex = benchmark["vnindex_return"]
    vnindex_text = f"{vnindex:+.2%}" if vnindex is not None else "N/A"
    print(f"\n{'═' * 100}")
    print(f"📈 {start:%Y-%m-%d} → {end:%Y-%m-%d} | Nắm đều các mã trong rổ: "
          f"{benchmark['equal_weight_return']:+.2%} | VNINDEX: {vnindex_text}")
    print(f"{'Nguồn':<7}{'Rank IC':>9}{'IC>0 năm':>10}{'Lãi/lệnh':>10}{'Danh mục':>11}"
          f"{'Lệnh DM':>9}{'Win DM':>8}{'Lãi/lệnh DM':>13}{'Sharpe DM':>11}{'MDD DM':>9}")
    for row in summary.itertuples():
        print(f"{row.mode:<7}{row.rank_ic:>+9.3f}{row.rank_ic_positive_years:>10}"
              f"{row.symbol_avg_trade_return:>+10.2%}{row.portfolio_return:>+11.2%}"
              f"{row.portfolio_trades:>9}{row.portfolio_win_rate:>8.1%}{row.portfolio_avg_trade_return:>+13.2%}"
              f"{row.portfolio_sharpe:>11.2f}{row.portfolio_max_drawdown:>9.2%}")
    if not ic_table.empty:
        print("\nRank IC theo năm:")
        print(ic_table.pivot(index="year", columns="mode", values="rank_ic").round(3).to_string())
    print(f"{'═' * 100}")

    output_dir = os.path.join(config.REPORT_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, "backtest_summary.csv"), index=False)
    ic_table.to_csv(os.path.join(output_dir, "backtest_ic_by_year.csv"), index=False)
    pd.concat(symbol_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, "backtest_symbols.csv"), index=False)
    pd.concat([t for t in trade_frames if not t.empty] or [pd.DataFrame()], ignore_index=True).to_csv(
        os.path.join(output_dir, "backtest_trades.csv"), index=False)
    scores.to_csv(os.path.join(output_dir, "backtest_scores.csv"), index=False)
    print(f"\n💾 Báo cáo: {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
