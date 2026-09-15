import unittest

import pandas as pd

from src.data.data_cleaner import clean_index_data, clean_ohlcv_data
from src.features import FEATURE_COLUMNS
from src.models.ml_models import train_ml_models


class DataCleanerRegressionTests(unittest.TestCase):
    def test_ohlcv_with_missing_required_column_returns_empty(self):
        frame = pd.DataFrame({"time": ["2026-01-01"], "close": [100]})

        cleaned = clean_ohlcv_data(frame)

        self.assertTrue(cleaned.empty)

    def test_index_is_sorted_before_forward_fill(self):
        frame = pd.DataFrame({
            "time": ["2026-01-02", "2026-01-01"],
            "indexValue": [102.0, None],
        })

        cleaned = clean_index_data(frame)

        self.assertEqual(cleaned["time"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-02"])


class ModelRegressionTests(unittest.TestCase):
    def test_training_skips_when_training_window_lacks_class(self):
        frame = pd.DataFrame({column: 0.0 for column in FEATURE_COLUMNS}, index=range(120))
        frame["label"] = 0

        rf, xgb = train_ml_models(frame, "TEST", save=False, verbose=False)

        self.assertIsNone(rf)
        self.assertIsNone(xgb)


if __name__ == "__main__":
    unittest.main()
