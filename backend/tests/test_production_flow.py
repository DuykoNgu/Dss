"""Checks for safe publication and the production HTTP boundary."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from api.features.market import MarketState
from api.server import create_app
from src.data.data_fetcher import save_csv_atomic
from src.features import FEATURE_COLUMNS
from src.models.ml_models import load_ml_models, model_is_stale, train_ml_models


class PersistedModel:
    pass


class ProductionFlowTests(unittest.TestCase):
    def test_csv_replace_keeps_previous_file_when_write_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "FPT.csv"
            target.write_text("complete\n")
            frame = Mock()
            def fail_write(output, index):
                output.write("partial")
                raise OSError("disk error")
            frame.to_csv.side_effect = fail_write
            with self.assertRaises(OSError):
                save_csv_atomic(str(target), frame)
            self.assertEqual(target.read_text(), "complete\n")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_model_pair_is_one_bundle_and_validates_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame({name: [0.0] * 3 for name in FEATURE_COLUMNS})
            frame["label"] = [-1, 0, 1]
            frame["time"] = pd.date_range("2026-09-15", periods=3)
            with patch("config.MODEL_DIR", directory), patch("config.MIN_TRAIN_ROWS", 3), \
                 patch("src.models.ml_models.fit_models", return_value=(PersistedModel(), PersistedModel())):
                train_ml_models(frame, "TEST", verbose=False)
                rf, xgb = load_ml_models("TEST")
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["TEST_bundle.pkl"])
            self.assertFalse(model_is_stale(rf, frame["time"].max()))
            self.assertFalse(model_is_stale(xgb, frame["time"].max()))
            self.assertTrue(model_is_stale(rf, frame["time"].max(), {"horizon": 20}))

    def test_daily_sync_keeps_snapshot_when_one_stock_is_old(self):
        now = datetime(2026, 9, 17, 15, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
        snapshot = {"date": "2026-09-16", "stocks": [{"symbol": "FPT"}], "history": {}}
        state = MarketState(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "FPT.csv").write_text("time\n2026-09-16\n")
            index = pd.DataFrame({"time": pd.to_datetime(["2026-09-17"])})
            with patch("api.features.market.fetch_market_index", return_value=index), \
                 patch("api.features.market.save_data"), \
                 patch("api.features.market.vn30_members", return_value={"FPT"}), \
                 patch("api.features.market.config.STOCKS_DIR", directory), \
                 patch("api.features.market.load_snapshot") as build:
                with self.assertRaises(RuntimeError):
                    state.sync_daily(now)
                build.assert_not_called()
        self.assertIs(state.snapshot, snapshot)

    def test_failed_daily_sync_is_retried_after_backoff(self):
        state = MarketState({"date": "2026-09-16", "stocks": [], "history": {}})
        moments = [datetime(2026, 9, 17, 15, minute, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
                   for minute in (0, 5, 16)]
        stop = Mock()
        stop.is_set.side_effect = [False, False, False, True]
        with patch("api.features.market.datetime") as clock, \
             patch.object(state, "poll"), patch.object(state, "publish"), \
             patch("api.features.market.LOGGER"), \
             patch.object(state, "sync_daily", side_effect=[RuntimeError("upstream"), True]) as sync:
            clock.now.side_effect = moments
            state.run(stop)
        self.assertEqual(sync.call_count, 2)

    def test_http_routes_and_readiness_use_one_snapshot(self):
        snapshot = {"date": "2026-09-17", "data_as_of": "2026-09-17",
                    "stocks": [{"symbol": "FPT", "ml_score": 51}], "history": {"FPT": []}}
        with patch("api.server.read_snapshot", return_value=snapshot), \
             patch("api.server.configure_logging"), \
             patch.object(MarketState, "refresh_from_cache"):
            with TestClient(create_app(worker=True)) as client:
                self.assertEqual(client.get("/health/ready").status_code, 200)
                market = client.get("/api/market")
                self.assertEqual(market.json()["stocks"][0]["symbol"], "FPT")
                self.assertEqual(client.get("/api/market", headers={"If-None-Match": market.headers["etag"]}).status_code, 304)
                self.assertEqual(client.get("/api/history?symbol=XYZ").status_code, 400)
                self.assertEqual(client.post("/api/ask", content="bad", headers={"Content-Type": "text/plain"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
