"""Atomic snapshot cache shared by the producer and read-only web workers."""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import config

CACHE_PATH = Path(config.DATA_DIR) / "web_snapshot_sqlite.json"
LOCK_PATH = Path(config.DATA_DIR) / "web_snapshot.lock"
CACHE_VERSION = 6


def source_stamp() -> list[list[int | str]]:
    paths = [Path(config.MARKET_DB_PATH), Path(f"{config.MARKET_DB_PATH}-wal"),
             Path(config.BASE_DIR) / "reference/vn30_changes.csv"]
    return [[str(path.relative_to(config.BASE_DIR)), path.stat().st_size, path.stat().st_mtime_ns]
            for path in paths if path.exists()]


@contextmanager
def build_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_snapshot() -> dict:
    with CACHE_PATH.open(encoding="utf-8") as source:
        return json.load(source)


def load_or_build_snapshot(builder: Callable[[], dict], rebuild: bool = False) -> dict:
    with build_lock():
        stamp = source_stamp()
        cached = None
        if not rebuild:
            try:
                cached = read_snapshot()
                if (cached.get("_cache_version") == CACHE_VERSION
                        and cached.get("_source_stamp") == stamp and not cached.get("_sync_error")):
                    cached.pop("_quotes", None)
                    return cached
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        try:
            snapshot = builder()
        except Exception as error:
            if cached and cached.get("date") and cached.get("stocks"):
                cached.pop("_quotes", None)
                cached["_sync_error"] = str(error)
                write_snapshot(cached)
                return cached
            raise
        write_snapshot(snapshot)
        return snapshot


def write_snapshot(snapshot: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_name(f"{CACHE_PATH.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump({**snapshot, "_cache_version": CACHE_VERSION, "_source_stamp": source_stamp()},
                      output, ensure_ascii=False, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, CACHE_PATH)
    finally:
        temporary.unlink(missing_ok=True)
