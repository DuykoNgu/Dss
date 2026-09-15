"""Phase 7: Walk-forward backtest cho tín hiệu DSS, có phí/thuế/slippage."""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

import config
from src.models import predict_ml_score, train_ml_models
from src.scoring import calculate_rule_based_score

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252
MIN_WINDOW_DAYS = 20

BUY_COST_MULTIPLIER = 1 + config.BACKTEST_FEE_RATE + config.BACKTEST_SLIPPAGE_RATE
SELL_PROCEEDS_MULTIPLIER = (
    1 - config.BACKTEST_FEE_RATE - config.BACKTEST_TAX_RATE - config.BACKTEST_SLIPPAGE_RATE
)


def _score_at(df: pd.DataFrame, idx: int, rf, xgb) -> float:
    rule_score, _ = calculate_rule_based_score(df.iloc[: idx + 1])
    ml_score = predict_ml_score(df.iloc[[idx]], rf, xgb)
    return rule_score * config.WEIGHT_RULE_BASED + ml_score * config.WEIGHT_ML_MODEL


def _train_purged(df: pd.DataFrame, idx: int, symbol: str) -> tuple:
    # Bỏ ML_FORWARD_DAYS dòng cuối: label của chúng nhìn vào tương lai
    train_df = df.iloc[: max(0, idx - config.ML_FORWARD_DAYS)]
    return train_ml_models(train_df, symbol, save=False, verbose=False)


def _build_trade(position: dict, exit_idx: int, df: pd.DataFrame,
                 exit_price: float, reason: str) -> dict:
    net_return = (exit_price * SELL_PROCEEDS_MULTIPLIER) / position["buy_cost"] - 1
    return {
        "entry_date": df["time"].iloc[position["entry_idx"]],
        "exit_date": df["time"].iloc[exit_idx],
        "entry_price": position["entry_price"],
        "exit_price": float(exit_price),
        "hold_days": exit_idx - position["entry_idx"],
        "net_return": float(net_return),
        "exit_reason": reason,
    }


def _sharpe(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def _profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gains = trades.loc[trades["net_return"] > 0, "net_return"].sum()
    losses = trades.loc[trades["net_return"] <= 0, "net_return"].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))


def _index_return(df: pd.DataFrame, start: int, index_df: pd.DataFrame | None) -> float | None:
    if index_df is None or index_df.empty:
        return None
    window = index_df[index_df["time"].isin(df["time"].iloc[start:])]
    if len(window) < 2:
        return None
    return float(window["indexValue"].iloc[-1] / window["indexValue"].iloc[0] - 1)


def run_symbol_backtest(
    featured: pd.DataFrame,
    symbol: str,
    months: int = config.BACKTEST_MONTHS,
    retrain_every: int = config.BACKTEST_RETRAIN_DAYS,
    index_df: pd.DataFrame | None = None,
) -> dict | None:
    """Mô phỏng giao dịch trên N tháng cuối, model chỉ học từ quá khứ.

    - Tín hiệu tại close ngày i, khớp lệnh tại open ngày i+1.
    - Vào lệnh khi Total >= SCORE_BUY, thoát khi đủ T+5 hoặc Total < SCORE_SELL.
    - Retrain định kỳ trên dữ liệu đã purge (không dùng label chạm tương lai).
    """
    if featured is None or featured.empty:
        return None

    df = featured.dropna(subset=["time", "open", "close"]).reset_index(drop=True)
    if df.empty:
        return None
    total = len(df)
    window = min(months * TRADING_DAYS_PER_MONTH, total - config.MIN_TRAIN_ROWS)
    if window < MIN_WINDOW_DAYS:
        return None
    start = total - window

    rf = xgb = None
    last_train_idx = None
    position = None
    entry_flag = stop_flag = False
    trades = []
    equity_values = []
    exposure_days = 0
    current_equity = 1.0

    for i in range(start, total):
        # 1. Khớp lệnh chờ tại giá mở cửa hôm nay
        if position is not None and stop_flag:
            exit_price = float(df["open"].iloc[i])
            current_equity = position["units"] * exit_price * SELL_PROCEEDS_MULTIPLIER
            trades.append(_build_trade(position, i, df, exit_price, "stop-loss"))
            position = None
        if position is None and entry_flag:
            open_price = float(df["open"].iloc[i])
            buy_cost = open_price * BUY_COST_MULTIPLIER
            position = {
                "entry_idx": i,
                "entry_price": open_price,
                "buy_cost": buy_cost,
                "units": current_equity / buy_cost,
            }

        # 2. Chốt đủ T+5 tại giá đóng cửa
        if position is not None and (i - position["entry_idx"]) >= config.ML_FORWARD_DAYS:
            exit_price = float(df["close"].iloc[i])
            current_equity = position["units"] * exit_price * SELL_PROCEEDS_MULTIPLIER
            trades.append(_build_trade(position, i, df, exit_price, "T+5"))
            position = None

        # 3. Đánh dấu equity theo giá đóng cửa, chưa trừ phí thanh lý lần nữa.
        if position is not None:
            exposure_days += 1
            current_equity = position["units"] * float(df["close"].iloc[i])
        equity_values.append(current_equity)

        # 4. Retrain định kỳ trên dữ liệu đã purge
        if rf is None or (i - last_train_idx) >= retrain_every:
            rf, xgb = _train_purged(df, i, symbol)
            last_train_idx = i

        # 5. Tính tín hiệu tại close hôm nay, khớp lệnh ngày mai
        total_score = _score_at(df, i, rf, xgb)
        entry_flag = (
            i < total - 1
            and position is None
            and total_score >= config.SCORE_BUY
        )
        stop_flag = position is not None and total_score < config.SCORE_SELL

    equity = pd.Series(equity_values, index=df.index[start:])
    daily_returns = equity.pct_change().dropna()
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        win_rate = avg_win = avg_loss = 0.0
    else:
        wins = trades_df.loc[trades_df["net_return"] > 0, "net_return"]
        losses = trades_df.loc[trades_df["net_return"] <= 0, "net_return"]
        win_rate = float((trades_df["net_return"] > 0).mean())
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0

    benchmark_start = min(start + 1, total - 1)
    benchmark_buy_cost = float(df["open"].iloc[benchmark_start]) * BUY_COST_MULTIPLIER
    benchmark_sell_proceeds = float(df["close"].iloc[-1]) * SELL_PROCEEDS_MULTIPLIER
    buy_hold_return = benchmark_sell_proceeds / benchmark_buy_cost - 1

    return {
        "symbol": symbol,
        "start": df["time"].iloc[start],
        "end": df["time"].iloc[-1],
        "num_trades": len(trades_df),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_return": float(trades_df["net_return"].mean()) if not trades_df.empty else 0.0,
        "profit_factor": _profit_factor(trades_df),
        "dss_return": float(equity.iloc[-1] - 1),
        "buy_hold_return": buy_hold_return,
        "vnindex_return": _index_return(df, start, index_df),
        "sharpe": _sharpe(daily_returns),
        "max_drawdown": _max_drawdown(equity),
        "exposure": exposure_days / len(equity_values) if equity_values else 0.0,
        "equity": equity,
        "trades": trades_df,
    }


def print_backtest_report(result: dict) -> None:
    if result is None:
        return
    vnindex = result["vnindex_return"]
    vnindex_text = f"{vnindex:+.2%}" if vnindex is not None else "N/A"
    profit_factor = result["profit_factor"]
    pf_text = "∞" if profit_factor == float("inf") else f"{profit_factor:.2f}"
    print(f"\n{'─' * 72}")
    print(f"📊 {result['symbol']} | {result['start']:%Y-%m-%d} → {result['end']:%Y-%m-%d}")
    print(f"   Lệnh: {result['num_trades']} | Win rate: {result['win_rate']:.1%} | "
          f"Profit factor: {pf_text} | Exposure: {result['exposure']:.0%}")
    print(f"   Lãi TB/lệnh: {result['avg_return']:+.2%} "
          f"(win {result['avg_win']:+.2%} / loss {result['avg_loss']:+.2%})")
    print(f"   DSS: {result['dss_return']:+.2%} | Buy&Hold: {result['buy_hold_return']:+.2%} | "
          f"VNINDEX: {vnindex_text}")
    print(f"   Sharpe: {result['sharpe']:.2f} | Max drawdown: {result['max_drawdown']:.2%}")
