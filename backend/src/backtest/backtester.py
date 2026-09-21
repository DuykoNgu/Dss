"""Phase 7: Walk-forward backtest cho tín hiệu DSS, có phí/thuế/slippage.

1. `score_history`: chấm Rule/ML/Total từng phiên trong cửa sổ backtest. Model
   chỉ học từ quá khứ: retrain định kỳ, bỏ các dòng có label chạm tương lai.
2. Từ bảng điểm đó, với từng nguồn điểm (blend / rule / ml):
   - `simulate_symbol`: mỗi mã giao dịch riêng với toàn bộ vốn.
   - `simulate_portfolio`: một danh mục chung vốn, chọn mã điểm cao nhất.
   - `daily_rank_ic`: điểm có xếp hạng đúng lợi nhuận T+N giữa các mã hay không.

Nếu dữ liệu có cột `in_universe` (thành phần VN30 theo từng kỳ), chỉ train, vào
lệnh và xếp hạng trên các dòng thuộc rổ tại ngày đó.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.models import train_ml_models
from src.models.ml_models import TrainingWeights, predict_ml_probabilities
from src.scoring import calculate_rule_based_score

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252
MIN_WINDOW_DAYS = 20
SCORE_MODES = {"blend": "total_score", "rule": "rule_score", "ml": "ml_score"}
# Barrier của backtest khớp với nhãn triple_barrier: ngưỡng lợi nhuận + chi phí vòng mua-bán
DEFAULT_BARRIER = config.ML_PROFIT_THRESHOLD + config.ML_LABEL_COST_RATE

BUY_COST_MULTIPLIER = 1 + config.BACKTEST_FEE_RATE + config.BACKTEST_SLIPPAGE_RATE
SELL_PROCEEDS_MULTIPLIER = (
    1 - config.BACKTEST_FEE_RATE - config.BACKTEST_TAX_RATE - config.BACKTEST_SLIPPAGE_RATE
)


def _purged_history(df: pd.DataFrame, day: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Dữ liệu train tại close `day`: bỏ các dòng cuối có label còn nhìn vào tương lai,
    và chỉ giữ dòng thuộc rổ tại ngày của dòng đó (nếu có cột in_universe)."""
    past = df[df["time"] <= day]
    if "label_end" in past:
        past = past[past["label_end"] < day]
    else:
        past = past.iloc[: max(0, len(past) - horizon - 1)]
    return past[past["in_universe"]] if "in_universe" in past.columns else past


def score_history(
    featured: dict[str, pd.DataFrame],
    months: int = config.BACKTEST_MONTHS,
    retrain_every: int = config.BACKTEST_RETRAIN_DAYS,
    pooled: bool = config.ML_POOLED,
    horizon: int = config.ML_FORWARD_DAYS,
    weighting: TrainingWeights = TrainingWeights(),
) -> pd.DataFrame:
    """Điểm Rule/ML/Total của mọi mã cho từng phiên trong `months` tháng cuối."""
    frames = {
        symbol: df.dropna(subset=["time", "open", "close"]).sort_values("time").reset_index(drop=True)
        for symbol, df in featured.items()
        if df is not None and not df.empty
    }
    if not frames:
        return pd.DataFrame()
    calendar = np.sort(pd.concat([df["time"] for df in frames.values()]).unique())
    window = calendar[-months * TRADING_DAYS_PER_MONTH:]

    rows = []
    for block_start in range(0, len(window), retrain_every):
        block = window[block_start:block_start + retrain_every]
        print(f"[backtest] {pd.Timestamp(block[0]):%Y-%m-%d} "
              f"({block_start + 1}/{len(window)} phiên)", flush=True)
        # Retrain tại close ngày đầu block, dùng cho cả block
        if pooled:
            history = pd.concat([_purged_history(df, block[0], horizon) for df in frames.values()],
                                ignore_index=True)
            shared_models = train_ml_models(history, "POOLED", save=False, verbose=False,
                                            weighting=weighting)
        for symbol, df in frames.items():
            positions = np.flatnonzero(df["time"].isin(block).to_numpy())
            if len(positions) == 0:
                continue
            rf, xgb = shared_models if pooled else train_ml_models(
                _purged_history(df, block[0], horizon), symbol, save=False, verbose=False,
                weighting=weighting
            )
            signals = predict_ml_probabilities(df.iloc[positions], rf, xgb)
            ml_scores = np.nan_to_num(50 + 50 * (signals[:, 2] - signals[:, 0]), nan=50)
            predictions = np.where(np.isfinite(signals).all(axis=1), signals.argmax(axis=1), np.nan)
            for position, ml_score, prediction in zip(positions, ml_scores, predictions):
                rule_score, _ = calculate_rule_based_score(df.iloc[: position + 1])
                row = df.iloc[position]
                rows.append({
                    "time": row["time"],
                    "symbol": symbol,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "in_universe": bool(row.get("in_universe", True)),
                    "rule_score": rule_score,
                    "ml_score": float(ml_score),
                    "total_score": rule_score * config.WEIGHT_RULE_BASED
                    + ml_score * config.WEIGHT_ML_MODEL,
                    "future_return": row.get("future_return"),
                    "label": row.get("label"),
                    "label_end": row.get("label_end"),
                    "ml_prediction": prediction,
                })
    return pd.DataFrame(rows)


def _trade(symbol: str, entry: dict, exit_date, exit_price: float, hold_days: int,
           reason: str) -> dict:
    return {
        "symbol": symbol,
        "entry_date": entry["entry_date"],
        "exit_date": exit_date,
        "entry_price": entry["entry_price"],
        "exit_price": float(exit_price),
        "hold_days": hold_days,
        "net_return": float(exit_price * SELL_PROCEEDS_MULTIPLIER / entry["buy_cost"] - 1),
        "exit_reason": reason,
    }


def _open_position(day, price: float, budget: float, entry_idx: int) -> dict:
    buy_cost = price * BUY_COST_MULTIPLIER
    return {"entry_idx": entry_idx, "entry_date": day, "entry_price": float(price),
            "buy_cost": buy_cost, "units": budget / buy_cost}


def _intraday_exit(position: dict, days_held: int, bar: pd.Series, horizon: int,
                   barrier: float | None) -> tuple[float, str] | None:
    """Giá và lý do thoát trong phiên hôm nay (sau khi đã khớp lệnh chờ tại open).

    - barrier: chạm ngưỡng +/- quanh giá vào thì bán tại ngưỡng (mở cửa vượt ngưỡng
      thì bán tại giá mở cửa). Chạm cả hai trong cùng phiên thì giả định chạm cắt lỗ
      trước (bảo thủ). Chỉ xét từ phiên T+3 vì T+2 cổ phiếu mới về tài khoản buổi chiều.
    - Hết `horizon` phiên thì bán tại giá đóng cửa.
    """
    if barrier is not None and days_held >= config.MIN_DAYS_BEFORE_OPEN_SELL:
        lower = position["entry_price"] * (1 - barrier)
        upper = position["entry_price"] * (1 + barrier)
        if bar["low"] <= lower:
            return min(bar["open"], lower), "barrier-stop"
        if bar["high"] >= upper:
            return max(bar["open"], upper), "barrier-profit"
    if days_held >= horizon and not pd.isna(bar["close"]):
        return bar["close"], f"T+{horizon}"
    return None


def _sharpe(equity: pd.Series) -> float:
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def _trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"num_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "avg_return": 0.0, "profit_factor": 0.0}
    returns = trades["net_return"]
    wins, losses = returns[returns > 0], returns[returns <= 0]
    if losses.sum() == 0:
        profit_factor = float("inf") if wins.sum() > 0 else 0.0
    else:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    return {
        "num_trades": len(trades),
        "win_rate": float((returns > 0).mean()),
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "avg_return": float(returns.mean()),
        "profit_factor": profit_factor,
    }


def index_return(times: pd.Series, index_df: pd.DataFrame | None) -> float | None:
    if index_df is None or index_df.empty:
        return None
    window = index_df[index_df["time"].isin(times)]
    if len(window) < 2:
        return None
    return float(window["indexValue"].iloc[-1] / window["indexValue"].iloc[0] - 1)


def simulate_symbol(scores: pd.DataFrame, score_col: str,
                    index_df: pd.DataFrame | None = None,
                    horizon: int = config.ML_FORWARD_DAYS,
                    barrier: float | None = None) -> dict | None:
    """Một mã, toàn bộ vốn.

    - Điểm tại close ngày i >= SCORE_BUY (và mã thuộc rổ) -> mua tại open ngày i+1.
    - Thoát theo `_intraday_exit`, hoặc điểm < SCORE_SELL -> bán tại open hôm sau
      (chỉ khi cổ phiếu đã về tài khoản theo T+2).
    """
    df = scores.sort_values("time").reset_index(drop=True)
    if len(df) < MIN_WINDOW_DAYS:
        return None
    symbol = df["symbol"].iloc[0]
    tradable = df["in_universe"] if "in_universe" in df.columns else pd.Series(True, index=df.index)

    position = None
    entry_flag = stop_flag = False
    trades, equity_values = [], []
    exposure_days = 0
    current_equity = 1.0
    for i in range(len(df)):
        bar = df.iloc[i]
        # 1. Khớp lệnh chờ tại giá mở cửa hôm nay
        if position is not None and stop_flag:
            current_equity = position["units"] * bar["open"] * SELL_PROCEEDS_MULTIPLIER
            trades.append(_trade(symbol, position, bar["time"], bar["open"],
                                 i - position["entry_idx"], "stop-loss"))
            position = None
        if position is None and entry_flag:
            position = _open_position(bar["time"], bar["open"], current_equity, i)

        # 2. Thoát trong phiên: barrier hoặc hết tầm nhìn
        if position is not None:
            exit_ = _intraday_exit(position, i - position["entry_idx"], bar, horizon, barrier)
            if exit_ is not None:
                current_equity = position["units"] * exit_[0] * SELL_PROCEEDS_MULTIPLIER
                trades.append(_trade(symbol, position, bar["time"], exit_[0],
                                     i - position["entry_idx"], exit_[1]))
                position = None

        # 3. Đánh dấu equity theo giá đóng cửa (chưa trừ phí thanh lý)
        if position is not None:
            exposure_days += 1
            current_equity = position["units"] * bar["close"]
        equity_values.append(current_equity)

        # 4. Tín hiệu tại close hôm nay, khớp lệnh ngày mai
        score = bar[score_col]
        entry_flag = (i < len(df) - 1 and position is None and bool(tradable.iloc[i])
                      and score >= config.SCORE_BUY)
        stop_flag = (
            position is not None
            and score < config.SCORE_SELL
            and i + 1 - position["entry_idx"] >= config.MIN_DAYS_BEFORE_OPEN_SELL
        )

    equity = pd.Series(equity_values, index=df["time"])
    trades_df = pd.DataFrame(trades)
    buy_hold = (df["close"].iloc[-1] * SELL_PROCEEDS_MULTIPLIER
                / (df["open"].iloc[1] * BUY_COST_MULTIPLIER) - 1)
    return {
        "symbol": symbol,
        "start": df["time"].iloc[0],
        "end": df["time"].iloc[-1],
        **_trade_stats(trades_df),
        "dss_return": float(equity.iloc[-1] - 1),
        "buy_hold_return": float(buy_hold),
        "vnindex_return": index_return(df["time"], index_df),
        "sharpe": _sharpe(equity),
        "max_drawdown": _max_drawdown(equity),
        "exposure": exposure_days / len(equity_values),
        "trades": trades_df,
    }


def simulate_portfolio(scores: pd.DataFrame, score_col: str,
                       slots: int = config.BACKTEST_PORTFOLIO_SLOTS,
                       horizon: int = config.ML_FORWARD_DAYS,
                       barrier: float | None = None) -> dict:
    """Danh mục chung vốn chia `slots` phần bằng nhau.

    Mỗi phiên, slot trống được lấp bằng các mã (thuộc rổ) có điểm >= SCORE_BUY cao
    nhất tại close hôm trước, khớp open hôm nay. Quy tắc thoát giống `simulate_symbol`.
    """
    bars = {column: scores.pivot(index="time", columns="symbol", values=column)
            for column in ("open", "high", "low", "close")}
    marks = bars["close"].ffill()
    points = scores.pivot(index="time", columns="symbol", values=score_col)
    if "in_universe" in scores.columns:
        points = points.where(scores.pivot(index="time", columns="symbol", values="in_universe").eq(True))
    dates = points.index

    cash = 1.0
    positions: dict[str, dict] = {}
    trades, equity_values, invested = [], [], []
    to_buy: list[str] = []
    to_sell: list[str] = []
    for k, day in enumerate(dates):
        # 1. Khớp lệnh chờ tại open: bán trước để có tiền mua
        for symbol in to_sell:
            price = bars["open"].at[day, symbol]
            if pd.isna(price):
                continue
            position = positions.pop(symbol)
            cash += position["units"] * price * SELL_PROCEEDS_MULTIPLIER
            trades.append(_trade(symbol, position, day, price, k - position["entry_idx"], "stop-loss"))
        slot_budget = (equity_values[-1] if equity_values else 1.0) / slots
        for symbol in to_buy:
            price = bars["open"].at[day, symbol]
            if pd.isna(price) or symbol in positions or cash <= 1e-9:
                continue
            budget = min(cash, slot_budget)
            positions[symbol] = _open_position(day, price, budget, k)
            cash -= budget

        # 2. Thoát trong phiên: barrier hoặc hết tầm nhìn
        for symbol, position in list(positions.items()):
            bar = pd.Series({column: frame.at[day, symbol] for column, frame in bars.items()})
            if bar.isna().any():
                continue  # mã không giao dịch phiên này
            exit_ = _intraday_exit(position, k - position["entry_idx"], bar, horizon, barrier)
            if exit_ is not None:
                cash += position["units"] * exit_[0] * SELL_PROCEEDS_MULTIPLIER
                trades.append(_trade(symbol, position, day, exit_[0], k - position["entry_idx"], exit_[1]))
                del positions[symbol]

        # 3. Equity theo giá đóng cửa gần nhất
        holdings = sum(p["units"] * marks.at[day, s] for s, p in positions.items())
        equity_values.append(cash + holdings)
        invested.append(holdings / (cash + holdings))

        # 4. Tín hiệu tại close hôm nay
        to_sell = [
            s for s, p in positions.items()
            if points.at[day, s] < config.SCORE_SELL
            and k + 1 - p["entry_idx"] >= config.MIN_DAYS_BEFORE_OPEN_SELL
        ]
        free_slots = slots - len(positions) + len(to_sell)
        to_buy = []
        if k < len(dates) - 1 and free_slots > 0:
            candidates = points.loc[day].drop(labels=list(positions)).dropna()
            candidates = candidates[candidates >= config.SCORE_BUY]
            to_buy = candidates.sort_values(ascending=False).index[:free_slots].tolist()

    equity = pd.Series(equity_values, index=dates)
    trades_df = pd.DataFrame(trades)
    return {
        "start": dates[0],
        "end": dates[-1],
        **_trade_stats(trades_df),
        "portfolio_return": float(equity.iloc[-1] - 1),
        "sharpe": _sharpe(equity),
        "max_drawdown": _max_drawdown(equity),
        "exposure": float(np.mean(invested)),
        "equity": equity,
        "trades": trades_df,
    }


def equal_weight_return(scores: pd.DataFrame) -> float:
    """Benchmark: nắm giữ đều mọi mã thuộc rổ, tái cân bằng mỗi ngày, không phí."""
    closes = scores.pivot(index="time", columns="symbol", values="close")
    daily = closes.ffill().pct_change(fill_method=None)  # phiên tạm ngừng giao dịch: lợi nhuận 0
    if "in_universe" in scores.columns:
        member = scores.pivot(index="time", columns="symbol", values="in_universe")
        daily = daily.where(member.eq(True).shift(1, fill_value=False))
    return float((1 + daily.mean(axis=1).fillna(0)).prod() - 1)


def daily_rank_ic(scores: pd.DataFrame, score_col: str) -> pd.Series:
    """Tương quan hạng (Spearman) theo ngày giữa điểm và lợi nhuận T+N thực tế của các mã thuộc rổ.

    > 0: mã điểm cao thực sự tăng tốt hơn mã điểm thấp. ~0: điểm không xếp hạng được.
    Các ngày mọi mã cùng điểm (VD: ML trung tính) cho NaN.
    """
    valid = scores.dropna(subset=[score_col, "future_return"])
    if "in_universe" in valid.columns:
        valid = valid[valid["in_universe"]]
    if valid.empty:
        return pd.Series(dtype=float)
    ranks = valid.groupby("time")[[score_col, "future_return"]].rank()
    return ranks.groupby(valid["time"]).corr().xs(score_col, level=1)["future_return"]


def rank_ic(scores: pd.DataFrame, score_col: str) -> float:
    daily = daily_rank_ic(scores, score_col)
    return float(daily.mean()) if not daily.empty else float("nan")
