"""SQLite store for validated daily candles."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config

INDEX_SYMBOL = "VNINDEX"
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
COLUMNS = ("open", "high", "low", "close", "volume")


@contextmanager
def connection():
    path = Path(config.MARKET_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT NOT NULL, time TEXT NOT NULL, open REAL, high REAL,
            low REAL, close REAL, volume INTEGER, valid INTEGER NOT NULL,
            updated_at TEXT NOT NULL, PRIMARY KEY (symbol, time))""")
        db.execute("CREATE TABLE IF NOT EXISTS current_symbols (symbol TEXT PRIMARY KEY)")
        yield db
    finally:
        db.close()


def candle_rows(symbol: str, frame: pd.DataFrame) -> list[tuple]:
    if frame is None or frame.empty:
        return []
    if "time" not in frame or not set(COLUMNS).issubset(frame.columns):
        raise ValueError(f"{symbol}: thiếu cột nến bắt buộc")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    if frame["time"].isna().any():
        raise ValueError(f"{symbol}: thời gian nến không hợp lệ")
    frame = frame.drop_duplicates("time", keep="last")
    fetched_at = datetime.now(VN_TZ).isoformat(timespec="seconds")
    today = datetime.now(VN_TZ).date()
    rows = []
    for row in frame.itertuples(index=False):
        values = []
        for column in COLUMNS:
            try:
                value = float(getattr(row, column))
                values.append(value if math.isfinite(value) else None)
            except (TypeError, ValueError):
                values.append(None)
        opening, high, low, close, volume = values
        session_date = row.time.date()
        if session_date > today:
            raise ValueError(f"{symbol}: nến tương lai {session_date}")
        valid = int(all(value is not None and value > 0 for value in values[:4])
                    and volume is not None and volume >= 0
                    and high >= max(opening, low, close)
                    and low <= min(opening, high, close))
        rows.append((symbol, session_date.isoformat(), opening, high, low, close,
                     int(volume) if volume is not None else None, valid, fetched_at))
    return rows


def _upsert(db: sqlite3.Connection, rows: list[tuple]) -> None:
    db.executemany("""INSERT INTO market_data
        (symbol, time, open, high, low, close, volume, valid, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, time) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume, valid=excluded.valid,
            updated_at=excluded.updated_at
        WHERE excluded.valid = 1 AND (
            market_data.open IS NOT excluded.open OR market_data.high IS NOT excluded.high
            OR market_data.low IS NOT excluded.low OR market_data.close IS NOT excluded.close
            OR market_data.volume IS NOT excluded.volume OR market_data.valid = 0)""", rows)


def ensure_store() -> None:
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM market_data LIMIT 1").fetchone():
            db.commit()
            return
        paths = sorted(Path(config.STOCKS_DIR).glob("*.csv"))
        if Path(config.INDEX_PATH).exists():
            paths.insert(0, Path(config.INDEX_PATH))
        try:
            rejected = 0
            for path in paths:
                rows = candle_rows(path.stem, pd.read_csv(path))
                _upsert(db, rows)
                rejected += sum(not row[7] for row in rows)
            manifest = Path(config.DATA_DIR) / "symbols.json"
            if manifest.exists():
                symbols = json.loads(manifest.read_text())
            else:
                symbols = [path.stem for path in paths if path.stem != INDEX_SYMBOL]
            db.executemany("INSERT OR IGNORE INTO current_symbols VALUES (?)",
                           [(symbol,) for symbol in symbols])
            db.commit()
            if paths:
                print(f"Đã nhập {len(paths)} CSV vào SQLite; {rejected} nến sai được đánh dấu")
        except Exception:
            db.rollback()
            raise


def latest_date(symbol: str) -> str | None:
    ensure_store()
    with connection() as db:
        row = db.execute("SELECT MAX(time) FROM market_data WHERE symbol=? AND valid=1", (symbol,)).fetchone()
    return row[0]


def first_date(symbol: str) -> str | None:
    ensure_store()
    with connection() as db:
        row = db.execute("SELECT MIN(time) FROM market_data WHERE symbol=? AND valid=1", (symbol,)).fetchone()
    return row[0]


def read_bars(symbol: str) -> pd.DataFrame:
    ensure_store()
    with connection() as db:
        frame = pd.read_sql_query("""SELECT time, open, high, low, close, volume FROM market_data
            WHERE symbol=? AND valid=1 ORDER BY time""", db, params=(symbol,), parse_dates=["time"])
    return frame


def current_symbols() -> list[str]:
    ensure_store()
    with connection() as db:
        symbols = [row[0] for row in db.execute("SELECT symbol FROM current_symbols ORDER BY symbol")]
    return symbols


def write_batch(frames: dict[str, pd.DataFrame], symbols: list[str] | None = None,
                replace_symbols: set[str] | None = None) -> dict[str, int]:
    ensure_store()
    prepared = {symbol: candle_rows(symbol, frame) for symbol, frame in frames.items()}
    if any(not rows for rows in prepared.values()):
        raise ValueError("API trả nến rỗng; giữ dữ liệu đã công bố")
    with connection() as db:
        try:
            db.execute("BEGIN IMMEDIATE")
            for symbol, rows in prepared.items():
                if replace_symbols and symbol in replace_symbols:
                    db.execute("DELETE FROM market_data WHERE symbol=?", (symbol,))
                _upsert(db, rows)
            if symbols is not None:
                db.execute("DELETE FROM current_symbols")
                db.executemany("INSERT INTO current_symbols VALUES (?)", [(symbol,) for symbol in symbols])
            db.commit()
        except Exception:
            db.rollback()
            raise
    return {symbol: sum(not row[7] for row in rows) for symbol, rows in prepared.items()}
