"""Snapshot sharing and score explanation checks."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.models import explain_ml_score, predict_ml_scores
from api.features.market import MarketState
from api.snapshot_cache import load_or_build_snapshot, write_snapshot


class FixedModel:
    def predict_proba(self, rows):
        return np.tile([0.2, 0.3, 0.5], (len(rows), 1))


class WebSnapshotTests(unittest.TestCase):
    def test_explanation_reconstructs_the_served_score(self):
        row = pd.DataFrame([{name: 0.0 for name in FEATURE_COLUMNS}])
        model = FixedModel()
        explanation = explain_ml_score(row, model, model)
        self.assertEqual(explanation["score"], round(predict_ml_scores(row, model, model)[0], 1))
        self.assertEqual([explanation[key] for key in ("below", "neutral", "above")], [20, 30, 50])

    def test_second_worker_reads_published_snapshot_without_building(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "snapshot.json"
            lock = Path(directory) / "snapshot.lock"
            with patch("api.snapshot_cache.CACHE_PATH", cache), patch("api.snapshot_cache.LOCK_PATH", lock), \
                 patch("api.snapshot_cache.source_stamp", return_value=[["data", 1, 2]]), \
                 patch("api.features.market.CACHE_PATH", cache):
                first = load_or_build_snapshot(lambda: {"date": "2026-09-17", "stocks": [], "history": {}})
                second = load_or_build_snapshot(lambda: self.fail("Unexpected model training"))
                self.assertEqual(first["date"], second["date"])
                state = MarketState(first)
                write_snapshot({**first, "_quotes": {"FPT": {
                    "price": 100,
                    "updated_at": "2026-09-17T10:30:00+07:00",
                }}})
                with patch("api.features.market.datetime") as clock:
                    clock.now.return_value = datetime(2026, 9, 17, 10, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
                    clock.fromisoformat = datetime.fromisoformat
                    state.refresh_from_cache()
                    self.assertEqual(state.quotes["FPT"]["price"], 100)
                    clock.now.return_value = datetime(2026, 9, 17, 10, 31, 1, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
                    state.refresh_from_cache()
                    self.assertEqual(state.quotes, {})


if __name__ == "__main__":
    unittest.main()
