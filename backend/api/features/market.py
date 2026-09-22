"""Market snapshot, quote polling, and market API routes."""

from __future__ import annotations

import math
import logging
import hashlib
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config
from src.data import clean_ohlcv_data, fetch_latest_quotes, fetch_market_index, save_data
from src.data import store
from src.data.universe import load_vn30_changes, membership_mask, vn30_members, vn30_symbols_ever
from src.models import explain_ml_score, load_ml_models, model_is_stale, train_ml_models
from src.pipeline import load_featured, load_market_index
from api.snapshot_cache import CACHE_PATH, read_snapshot, write_snapshot

MODEL_NAME = "WEB_POOLED_EXCESS_H20_HISTORY"
HORIZON = 20
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
POLL_SECONDS = 20
SYNC_RETRY_SECONDS = 900
NO_SESSION_RETRY_SECONDS = 3600
QUOTE_MAX_AGE_SECONDS = 60
QUOTE_SOURCE_MAX_AGE_SECONDS = 300
QUOTE_PRICE_UNIT = 1000
WEB_MODEL_SPEC = {"label_strategy": "excess", "horizon": HORIZON,
                  "feature_set": "baseline"}
LOGGER = logging.getLogger(__name__)


def recommendation(rows: list[dict], symbol: str) -> dict:
    by_symbol = {row["symbol"]: row for row in rows}
    if symbol not in by_symbol:
        raise ValueError("Mã này không thuộc rổ VN30 hiện tại.")
    selected = by_symbol[symbol]
    if selected["ml_score"] is None:
        raise ValueError(f"{symbol} chưa đủ dữ liệu để mô hình ML xếp hạng.")
    ranked = sorted((row for row in rows if row["ml_score"] is not None),
                    key=lambda row: (-row["ml_score"], row["symbol"]))
    better = [row for row in ranked if row["symbol"] != symbol and row["ml_score"] > selected["ml_score"]]
    options = better[:3] or [row for row in ranked if row["symbol"] != symbol][:3]
    return {
        "selected": selected,
        "rank": next(index for index, row in enumerate(ranked, 1) if row["symbol"] == symbol),
        "ranked_count": len(ranked),
        "better_available": bool(better),
        "alternatives": [{**row, "score_gap": round(row["ml_score"] - selected["ml_score"], 1)} for row in options],
    }


def load_snapshot() -> dict:
    index = load_market_index()
    if index.empty:
        raise RuntimeError("Thiếu VNINDEX trong SQLite. Chạy ./run.sh fetch trước.")
    date = index["time"].max()
    changes = load_vn30_changes()
    current = sorted(vn30_members(changes, date))
    featured = {}
    for symbol in sorted(set(vn30_symbols_ever(changes)) | set(current)):
        frame = load_featured(symbol, index, label_strategy="excess", forward_days=HORIZON)
        if not frame.empty:
            featured[symbol] = frame

    training = []
    for symbol, frame in featured.items():
        historic = frame.loc[membership_mask(changes, symbol, frame["time"])]
        training.append(historic)
    rf, xgb = load_ml_models(MODEL_NAME)
    if (rf is None or xgb is None or model_is_stale(rf, date, WEB_MODEL_SPEC)
            or model_is_stale(xgb, date, WEB_MODEL_SPEC)
            or rf.data_end_ != date or xgb.data_end_ != date):
        print("Huấn luyện mô hình ML excess T+20 trên lịch sử VN30...", flush=True)
        training_frame = pd.concat(training, ignore_index=True)
        training_frame.attrs["model_spec"] = WEB_MODEL_SPEC
        rf, xgb = train_ml_models(training_frame, MODEL_NAME, verbose=False)
        if rf is None:
            raise RuntimeError("Không đủ dữ liệu để huấn luyện mô hình ML.")
    model_version = hashlib.sha256(json.dumps({"config": rf.config_, "spec": rf.spec_,
                                               "data_end": str(rf.data_end_)},
                                              sort_keys=True).encode()).hexdigest()[:12]

    explanations = {}
    latest = []
    history = {}
    for symbol in current:
        frame = featured.get(symbol)
        if frame is None or frame.empty or frame["time"].max() != date:
            continue
        last = frame.iloc[[-1]]
        explanations[symbol] = explain_ml_score(last, rf, xgb)
        if explanations[symbol]:
            explanations[symbol]["label_threshold_pct"] = round(
                (config.ML_PROFIT_THRESHOLD + config.ML_LABEL_COST_RATE) * 100, 2
            )
    for symbol in current:
        candles = clean_ohlcv_data(store.read_bars(symbol))
        if candles.empty:
            continue
        last = candles.iloc[-1]
        prior = candles.iloc[-2]["close"] if len(candles) > 1 else None
        change = (last["close"] / prior - 1) * 100 if prior else None
        history[symbol] = [
            {"date": candle.time.date().isoformat(), "close": round(float(candle.close), 2),
             "volume": int(candle.volume)}
            for candle in candles.tail(120).itertuples()
        ]
        latest.append({
            "symbol": symbol,
            "date": last["time"].date().isoformat(),
            "previous_close": round(float(prior), 2) if prior else None,
            "open": round(float(last["open"]), 2),
            "high": round(float(last["high"]), 2),
            "low": round(float(last["low"]), 2),
            "close": round(float(last["close"]), 2),
            "change_pct": round(float(change), 2) if change is not None else None,
            "volume": int(last["volume"]),
            "ml_score": explanations[symbol]["score"] if explanations.get(symbol) else None,
            "explanation": explanations.get(symbol),
        })
    latest.sort(key=lambda row: row["symbol"])
    report_path = Path(config.REPORT_DIR) / "backtest_pooled_excess_h20_horizon_history_72m" / "backtest_summary.csv"
    data_files = [Path(config.MARKET_DB_PATH), Path(f"{config.MARKET_DB_PATH}-wal")]
    data_updated_at = max((path.stat().st_mtime_ns for path in data_files if path.exists()), default=0)
    research = None
    if report_path.exists() and report_path.stat().st_mtime_ns >= data_updated_at:
        summary = pd.read_csv(report_path).set_index("mode").loc["ml"]
        research = {
            "rank_ic": round(float(summary["rank_ic"]), 3),
            "positive_years": summary["rank_ic_positive_years"],
            "portfolio_return": round(float(summary["portfolio_return"]) * 100, 1),
            "equal_weight_return": round(float(summary["equal_weight_return"]) * 100, 1),
            "max_drawdown": round(float(summary["portfolio_max_drawdown"]) * 100, 1),
        }
    if len(latest) != len(current) or any(row["date"] != date.date().isoformat() for row in latest):
        raise RuntimeError("Thiếu nến đã đóng của một hoặc nhiều mã VN30; giữ snapshot trước đó.")
    return {
        "date": date.date().isoformat(),
        "data_as_of": date.date().isoformat(),
        "snapshot_generated_at": datetime.now(VN_TZ).isoformat(timespec="seconds"),
        "index": round(float(index.iloc[-1]["indexValue"]), 2),
        "index_change_pct": round(float((index.iloc[-1]["indexValue"] / index.iloc[-2]["indexValue"] - 1) * 100), 2),
        "model": "ML pooled · vượt VNINDEX · T+20",
        "model_id": MODEL_NAME,
        "model_version": model_version,
        "model_spec": WEB_MODEL_SPEC,
        "contract_version": 1,
        "stocks": latest,
        "research": research,
        "history": history,
    }


def quote_value(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def quote_price(value) -> float | None:
    number = quote_value(value)
    return round(number / QUOTE_PRICE_UNIT, 2) if number is not None and number > 0 else None


def live_quote(row: dict, now: datetime) -> dict | None:
    source_time = row.get("time")
    if source_time is None:
        return None
    try:
        if isinstance(source_time, (int, float)):
            observed_at = datetime.fromtimestamp(source_time / 1000, VN_TZ)
        else:
            observed_at = pd.Timestamp(source_time).to_pydatetime()
            observed_at = (observed_at.replace(tzinfo=VN_TZ) if observed_at.tzinfo is None
                           else observed_at.astimezone(VN_TZ))
        age = (now - observed_at).total_seconds()
        if observed_at.date() != now.date() or not 0 <= age <= QUOTE_SOURCE_MAX_AGE_SECONDS:
            return None
    except (ValueError, TypeError, OverflowError):
        return None
    price = quote_price(row.get("close_price"))
    if price is None:
        return None
    reference = quote_price(row.get("reference_price"))
    volume = quote_value(row.get("volume_accumulated"))
    return {
        "price": price,
        "reference": reference,
        "open": quote_price(row.get("open_price")),
        "high": quote_price(row.get("high_price")),
        "low": quote_price(row.get("low_price")),
        "change_pct": round((price / reference - 1) * 100, 2) if reference else None,
        "volume": int(volume) if volume is not None and volume >= 0 else None,
        "source_at": observed_at.isoformat(timespec="seconds") if observed_at else None,
        "updated_at": now.isoformat(timespec="seconds"),
    }


class MarketState:
    def __init__(self, snapshot: dict):
        self.snapshot = snapshot
        self.quotes: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.sync_lock = threading.Lock()
        self.cache_mtime = 0
        self.quote_status = "closed"
        self.quote_as_of = None
        self.sync_error = snapshot.get("_sync_error")
        self.requested_session_date = None

    def market(self) -> dict:
        with self.lock:
            return {**{key: value for key, value in self.snapshot.items()
                       if key != "history" and not key.startswith("_")},
                    "quote_status": self.quote_status,
                    "quote_as_of": self.quote_as_of,
                    "sync_status": "error" if self.sync_error else "ok",
                    "stocks": [
                {**stock, "quote": self.quotes.get(stock["symbol"])}
                for stock in self.snapshot["stocks"]
            ]}

    def history(self, symbol: str) -> list[dict] | None:
        with self.lock:
            return self.snapshot["history"].get(symbol)

    def refresh_from_cache(self) -> None:
        modified = CACHE_PATH.stat().st_mtime_ns
        published = read_snapshot() if modified != self.cache_mtime else None
        with self.lock:
            if published is not None:
                self.snapshot = published
                self.quotes = published.get("_quotes", {})
                self.quote_status = published.get("_quote_status", "closed")
                self.quote_as_of = published.get("_quote_as_of")
                self.sync_error = published.get("_sync_error")
                self.cache_mtime = modified
            now = datetime.now(VN_TZ)
            if now.weekday() >= 5 or not 9 <= now.hour < 15:
                self.quotes = {}
                self.quote_status = "closed"
            else:
                self.quotes = {
                    symbol: quote for symbol, quote in self.quotes.items()
                    if quote.get("updated_at") and
                    0 <= (now - datetime.fromisoformat(quote["updated_at"])).total_seconds() <= QUOTE_MAX_AGE_SECONDS
                }
                if not self.quotes:
                    self.quote_status = "unavailable"

    def publish(self) -> None:
        with self.lock:
            published = {**self.snapshot, "_quotes": self.quotes,
                         "_quote_status": self.quote_status, "_quote_as_of": self.quote_as_of,
                         "_sync_error": self.sync_error}
        write_snapshot(published)

    def poll(self, now: datetime) -> None:
        if now.weekday() >= 5 or not 9 <= now.hour < 15:
            with self.lock:
                self.quotes = {}
                self.quote_status = "closed"
            return
        with self.lock:
            symbols = [row["symbol"] for row in self.snapshot["stocks"]]
        board = fetch_latest_quotes(symbols)
        quotes = {symbol: quote for symbol, row in board.items()
                  if symbol in symbols and (quote := live_quote(row, now))}
        with self.lock:
            previous_status = self.quote_status
            self.quotes = quotes
            self.quote_status = ("live" if symbols and len(quotes) == len(symbols) else
                                 "degraded" if quotes else "unavailable")
            self.quote_as_of = now.isoformat(timespec="seconds") if quotes else self.quote_as_of
            status = self.quote_status
        if status != previous_status:
            LOGGER.info("Quote status=%s coverage=%d/%d", status, len(quotes), len(symbols))

    def sync_daily(self) -> bool:
        if not self.sync_lock.acquire(blocking=False):
            return False
        try:
            return self._sync_daily()
        finally:
            self.sync_lock.release()

    def _sync_daily(self) -> bool:
        index = fetch_market_index()
        if index.empty or index["time"].max().date().isoformat() <= self.snapshot["date"]:
            LOGGER.info("VNINDEX chưa có nến đóng mới hơn %s", self.snapshot["date"])
            return False
        session_date = index["time"].max().date()
        symbols = sorted(vn30_members(load_vn30_changes(), pd.Timestamp(session_date)))
        if not symbols:
            raise RuntimeError("Chưa có danh sách VN30 hợp lệ cho phiên mới")
        save_data(symbols, index, full_sync=True)
        snapshot = load_snapshot()
        with self.lock:
            self.snapshot = snapshot
            self.quotes = {}
            self.sync_error = None
        LOGGER.info("Đồng bộ dữ liệu ngày %s hoàn tất với %d mã", session_date, len(symbols))
        return True

    def request_session_sync(self, session_date: str) -> None:
        try:
            requested_date = datetime.strptime(session_date, "%Y-%m-%d").date()
        except ValueError as error:
            raise ValueError("Ngày phiên giao dịch không hợp lệ.") from error
        if requested_date > datetime.now(VN_TZ).date():
            raise ValueError("Không thể đồng bộ phiên trong tương lai.")
        with self.lock:
            if (session_date <= self.snapshot["date"]
                    or session_date == self.requested_session_date):
                return
            self.requested_session_date = session_date
        threading.Thread(target=self._sync_requested_session, daemon=True).start()

    def _sync_requested_session(self) -> None:
        try:
            if self.sync_daily():
                self.publish()
        except Exception as error:
            with self.lock:
                self.sync_error = str(error)
            LOGGER.exception("Không cập nhật được phiên người dùng yêu cầu")

    def run(self, stop: threading.Event) -> None:
        next_sync_at = None
        first_poll = True
        while not stop.is_set():
            now = datetime.now(VN_TZ)
            try:
                if now.hour >= 15:
                    self.poll(now)
                with self.lock:
                    snapshot_date = self.snapshot["date"]
                if ((first_poll or (now.weekday() < 5 and now.hour >= 15))
                        and snapshot_date < now.date().isoformat()
                        and (next_sync_at is None or now >= next_sync_at)):
                    next_sync_at = now + timedelta(seconds=SYNC_RETRY_SECONDS)
                    if not self.sync_daily():
                        next_sync_at = now + timedelta(seconds=NO_SESSION_RETRY_SECONDS)
                if now.hour < 15:
                    self.poll(now)
                self.publish()
            except Exception as error:
                with self.lock:
                    self.sync_error = str(error)
                LOGGER.exception("Không cập nhật được thị trường")
            first_poll = False
            stop.wait(POLL_SECONDS)


def get_market(state: MarketState, query: dict) -> dict:
    requested_date = query.get("session_date", [""])[0]
    if requested_date:
        state.request_session_sync(requested_date)
    return state.market()


def get_history(state: MarketState, query: dict) -> dict:
    symbol = query.get("symbol", [""])[0].strip().upper()
    candles = state.history(symbol)
    if candles is None:
        raise ValueError("Mã này không thuộc rổ VN30 hiện tại.")
    return {"symbol": symbol, "candles": candles}


def get_recommendation(state: MarketState, query: dict) -> dict:
    symbol = query.get("symbol", [""])[0].strip().upper()
    return recommendation(state.market()["stocks"], symbol)


GET_ROUTES = {
    "/api/market": get_market,
    "/api/history": get_history,
    "/api/recommend": get_recommendation,
}
