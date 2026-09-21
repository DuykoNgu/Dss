import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.backtest.backtester import _purged_history
from src.features import build_features_and_labels, calculate_technical_indicators
from src.models.ml_models import TrainingWeights, _class_signals, fit_models
from src.models.validation import walk_forward_validate


class LabelAlignmentTests(unittest.TestCase):
    def test_excess_uses_stock_end_date_when_calendars_differ(self):
        dates = pd.bdate_range("2025-01-01", periods=65)
        stock = pd.DataFrame({"time": dates.delete(60), "open": 100.0, "high": 101.0,
                              "low": 99.0, "close": 100.0, "volume": 1000})
        index = pd.DataFrame({"time": dates.delete(59), "indexValue": np.arange(64) + 1000.0})
        featured = build_features_and_labels(calculate_technical_indicators(stock), index,
                                             forward_days=2, label_strategy="excess")
        prices = index.set_index("time")["indexValue"]
        row = featured.loc[featured.time == dates[58]].iloc[0]
        self.assertAlmostEqual(row.vnindex_future_return, prices[dates[61]] / prices[dates[58]] - 1)

    def test_missing_index_endpoint_is_not_filled_into_a_label(self):
        dates = pd.bdate_range("2025-01-01", periods=65)
        stock = pd.DataFrame({"time": dates, "open": 100.0, "high": 101.0,
                              "low": 99.0, "close": 100.0, "volume": 1000})
        index = pd.DataFrame({"time": dates.delete(60), "indexValue": 1000.0})
        featured = build_features_and_labels(calculate_technical_indicators(stock), index,
                                             forward_days=2, label_strategy="excess")
        self.assertTrue(pd.isna(featured.loc[58, "label"]))

    def test_purge_uses_actual_label_end_before_filtering_membership(self):
        frame = pd.DataFrame({"time": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                              "label_end": pd.to_datetime(["2025-01-10", "2025-01-05", "2025-01-06"]),
                              "in_universe": [True, False, True]})
        history = _purged_history(frame, pd.Timestamp("2025-01-10"), 2)
        self.assertEqual(history.index.tolist(), [2])


class WeightingTests(unittest.TestCase):
    def test_temporal_weight_halves_at_each_half_life(self):
        dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-11", "2025-01-21"]))
        weights = TrainingWeights(half_life_days=10).temporal_weights(dates)
        np.testing.assert_allclose(weights / weights[-1], [0.25, 0.5, 1])

    def test_both_models_share_weights_and_unweighted_predictions_stay_unadjusted(self):
        features = pd.DataFrame({"x": range(6)})
        labels = pd.Series([0, 1, 1, 1, 1, 2])
        temporal = np.array([1, 2, 2, 3, 3, 4], dtype=float)
        for balance in (None, "balanced"):
            with self.subTest(balance=balance), \
                 patch("src.models.ml_models.RandomForestClassifier") as forest, \
                 patch("src.models.ml_models.XGBClassifier") as boosted:
                forest.return_value.fit.return_value = forest.return_value
                forest.return_value.predict_proba.return_value = np.array([[0.2, 0.3, 0.5]])
                boosted.return_value.predict_proba.return_value = np.array([[0.2, 0.3, 0.5]])
                rf, xgb = fit_models(features, labels, rf_params={"class_weight": balance},
                                     sample_weight=temporal)
                actual = forest.return_value.fit.call_args.kwargs["sample_weight"]
                np.testing.assert_allclose(actual, boosted.return_value.fit.call_args.kwargs["sample_weight"])
                if balance is None:
                    np.testing.assert_allclose(_class_signals(features.iloc[:1], rf, xgb), [[0.2, 0.3, 0.5]])
                else:
                    np.testing.assert_allclose([actual[labels == label].sum() for label in range(3)], [2, 2, 2])

    def test_partial_rf_parameters_keep_default_balancing(self):
        with patch("src.models.ml_models.RandomForestClassifier") as forest, \
             patch("src.models.ml_models.XGBClassifier"):
            fit_models(pd.DataFrame({"x": range(4)}), pd.Series([0, 1, 1, 2]),
                       rf_params={"max_depth": 3})
            np.testing.assert_allclose(forest.return_value.fit.call_args.kwargs["sample_weight"],
                                       [4 / 3, 2 / 3, 2 / 3, 4 / 3])


class WalkForwardCoverageTests(unittest.TestCase):
    def test_pooled_folds_keep_dates_together_purge_labels_and_cover_tail(self):
        dates = pd.bdate_range("2024-01-01", periods=215)
        frame = pd.DataFrame({"time": np.repeat(dates, 2), "day": np.repeat(np.arange(215), 2),
                              "label": np.tile([-1, 0, 1], 144)[:430], "return_5d": 0.0})
        frame["label_end"] = frame.time + pd.Timedelta(days=30)
        checks = []

        def fit(features, labels, *args):
            latest_label = frame.loc[features.index, "label_end"].max()
            model = Mock()

            def predict(validation):
                checks.append(latest_label < frame.loc[validation.index, "time"].min())
                return np.ones(len(validation), dtype=int)

            model.predict.side_effect = predict
            model.predict_proba.side_effect = lambda values: np.tile([0.2, 0.6, 0.2], (len(values), 1))
            return model, model

        with patch("src.models.validation.fit_models", side_effect=fit):
            result = walk_forward_validate(frame.sample(frac=1, random_state=42), gap=5,
                                           feature_columns=["day"])
        predicted = pd.DataFrame(result["predictions"]).query("model == 'RF'")
        expected = set(dates[int(len(dates) * 0.5) + 5:])
        self.assertEqual((set(predicted.time), predicted.groupby("time").size().unique().tolist(), all(checks)),
                         (expected, [2], True))


if __name__ == "__main__":
    unittest.main()
