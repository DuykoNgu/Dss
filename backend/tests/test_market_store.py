"""Daily candle storage boundaries."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data import store


def candle(day: str, close: float = 100) -> pd.DataFrame:
    return pd.DataFrame({"time": [day], "open": [100], "high": [101], "low": [99],
                         "close": [close], "volume": [1000]})


class MarketStoreTests(unittest.TestCase):
    def test_upsert_replaces_corrected_bar_but_keeps_good_bar_on_invalid_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("config.MARKET_DB_PATH", str(Path(directory) / "market.sqlite3")), \
                 patch("config.STOCKS_DIR", str(Path(directory) / "stocks")), \
                 patch("config.INDEX_PATH", str(Path(directory) / "VNINDEX.csv")), \
                 patch("config.DATA_DIR", directory):
                store.write_batch({"FPT": candle("2026-09-17")})
                store.write_batch({"FPT": candle("2026-09-17", 101)})
                store.write_batch({"FPT": candle("2026-09-17", 102)})
                self.assertEqual(store.read_bars("FPT")["close"].tolist(), [101])

    def test_import_flags_invalid_csv_bars_without_serving_them(self):
        with tempfile.TemporaryDirectory() as directory:
            stocks = Path(directory) / "stocks"
            stocks.mkdir()
            pd.concat([candle("2026-09-16"), candle("2026-09-17", 102)]).to_csv(
                stocks / "FPT.csv", index=False)
            (Path(directory) / "symbols.json").write_text('["FPT"]')
            with patch("config.MARKET_DB_PATH", str(Path(directory) / "market.sqlite3")), \
                 patch("config.DATA_DIR", directory), patch("config.STOCKS_DIR", str(stocks)), \
                 patch("config.INDEX_PATH", str(Path(directory) / "VNINDEX.csv")):
                self.assertEqual(store.current_symbols(), ["FPT"])
                self.assertEqual(store.read_bars("FPT")["time"].dt.strftime("%Y-%m-%d").tolist(),
                                 ["2026-09-16"])
                with sqlite3.connect(Path(directory) / "market.sqlite3") as db:
                    self.assertEqual(db.execute("SELECT valid FROM market_data WHERE time='2026-09-17'").fetchone(),
                                     (0,))

    def test_batch_rolls_back_index_if_stock_write_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "market.sqlite3"
            with patch("config.MARKET_DB_PATH", str(db_path)), \
                 patch("config.STOCKS_DIR", str(Path(directory) / "stocks")), \
                 patch("config.INDEX_PATH", str(Path(directory) / "VNINDEX.csv")), \
                 patch("config.DATA_DIR", directory):
                store.write_batch({"VNINDEX": candle("2026-09-16")}, ["FPT"])
                with sqlite3.connect(db_path) as db:
                    db.execute("""CREATE TRIGGER reject_fpt BEFORE INSERT ON market_data
                        WHEN NEW.symbol='FPT' BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
                with self.assertRaises(sqlite3.IntegrityError):
                    store.write_batch({"VNINDEX": candle("2026-09-17"),
                                       "FPT": candle("2026-09-17")}, ["FPT"])
                self.assertEqual(store.latest_date("VNINDEX"), "2026-09-16")

    def test_full_refetch_replaces_old_price_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("config.MARKET_DB_PATH", str(Path(directory) / "market.sqlite3")), \
                 patch("config.STOCKS_DIR", str(Path(directory) / "stocks")), \
                 patch("config.INDEX_PATH", str(Path(directory) / "VNINDEX.csv")), \
                 patch("config.DATA_DIR", directory):
                store.write_batch({"FPT": pd.concat([candle("2026-09-15"), candle("2026-09-16")])})
                store.write_batch({"FPT": candle("2026-09-16", 101)},
                                  replace_symbols={"FPT"})
                self.assertEqual(store.read_bars("FPT")["time"].dt.strftime("%Y-%m-%d").tolist(),
                                 ["2026-09-16"])


if __name__ == "__main__":
    unittest.main()
