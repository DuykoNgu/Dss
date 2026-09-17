"""Estimate position profit or loss from closed VN30 sessions."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

import config
from api.features.market import MarketState

VND = Decimal("1")
PRICE_UNIT = Decimal("1000")
MAX_QUANTITY = 10_000_000
MAX_PRICE = 1_000
FEE_RATE = Decimal(str(config.BACKTEST_FEE_RATE))
TAX_RATE = Decimal(str(config.BACKTEST_TAX_RATE))
SLIPPAGE_RATE = Decimal(str(config.BACKTEST_SLIPPAGE_RATE))
BUY_COST_RATE = FEE_RATE + SLIPPAGE_RATE
SELL_COST_RATE = FEE_RATE + TAX_RATE + SLIPPAGE_RATE


def money(value: Decimal) -> int:
    return int(value.quantize(VND, rounding=ROUND_HALF_UP))


def estimate_profit_loss(state: MarketState, body: dict) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("symbol"), str):
        raise ValueError("Mã chứng khoán không hợp lệ.")
    symbol = body["symbol"].strip().upper()
    candles = state.history(symbol)
    if not candles:
        raise ValueError("Mã này không thuộc rổ VN30 hiện tại hoặc chưa có lịch sử giá.")

    entry_date = body.get("entry_date")
    if not isinstance(entry_date, str) or entry_date not in {row["date"] for row in candles}:
        raise ValueError("Ngày mua phải là một phiên có trong lịch sử giá hiển thị.")
    quantity = body.get("quantity")
    if type(quantity) is not int or not 0 < quantity <= MAX_QUANTITY:
        raise ValueError("Số cổ phiếu phải là số nguyên dương hợp lệ.")
    buy_price = body.get("buy_price")
    if (isinstance(buy_price, bool) or not isinstance(buy_price, (int, float))
            or not math.isfinite(buy_price) or not 0 < buy_price <= MAX_PRICE):
        raise ValueError("Giá mua phải là số dương hợp lệ (nghìn đồng/cổ phiếu).")

    purchase_value = Decimal(str(buy_price)) * quantity * PRICE_UNIT
    buy_costs = money(purchase_value * BUY_COST_RATE)
    buy_total = money(purchase_value) + buy_costs
    sessions = []
    previous_pnl = None
    for candle in candles:
        if candle["date"] < entry_date:
            continue
        close_value = Decimal(str(candle["close"])) * quantity * PRICE_UNIT
        sell_costs = money(close_value * SELL_COST_RATE)
        pnl = money(close_value) - sell_costs - buy_total
        sessions.append({
            "date": candle["date"],
            "close": candle["close"],
            "closing_value_vnd": money(close_value),
            "sell_costs_vnd": sell_costs,
            "pnl_vnd": pnl,
            "pnl_pct": float((Decimal(pnl) / buy_total * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )),
            "daily_change_vnd": None if previous_pnl is None else pnl - previous_pnl,
        })
        previous_pnl = pnl

    return {
        "symbol": symbol,
        "entry_date": entry_date,
        "quantity": quantity,
        "buy_price": buy_price,
        "purchase_value_vnd": money(purchase_value),
        "buy_costs_vnd": buy_costs,
        "buy_total_vnd": buy_total,
        "buy_cost_rate_pct": float(BUY_COST_RATE * 100),
        "sell_cost_rate_pct": float(SELL_COST_RATE * 100),
        "sessions": sessions,
    }


POST_ROUTES = {"/api/profit-loss": estimate_profit_loss}
