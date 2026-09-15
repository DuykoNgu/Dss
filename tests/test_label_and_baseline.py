import unittest

import numpy as np
import pandas as pd

from src.features.features import _build_labels
from src.models.validation import _baseline_predictions


class LabelTests(unittest.TestCase):
    def test_fixed_label_includes_round_trip_cost(self):
        frame = pd.DataFrame({"future_return": [0.032, 0.037, -0.032, -0.037]})

        labels = _build_labels(frame, forward_days=5, threshold=0.03, strategy="fixed")

        self.assertEqual(labels.tolist(), [0.0, 1.0, 0.0, -1.0])

    def test_invalid_forward_days_is_rejected(self):
        frame = pd.DataFrame({"future_return": [0.0]})

        with self.assertRaises(ValueError):
            _build_labels(frame, forward_days=0, threshold=0.03, strategy="fixed")


class BaselineTests(unittest.TestCase):
    def test_baselines_are_deterministic_and_three_class(self):
        frame = pd.DataFrame({"return_5d": [2.0, 0.0, -2.0]})

        predictions = _baseline_predictions(frame)

        np.testing.assert_array_equal(predictions["BaselineHold"], [1, 1, 1])
        np.testing.assert_array_equal(predictions["BaselineMomentum"], [2, 1, 0])


if __name__ == "__main__":
    unittest.main()
