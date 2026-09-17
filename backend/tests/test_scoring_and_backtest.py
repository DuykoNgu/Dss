import unittest
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.backtest.backtester import rank_ic, simulate_portfolio, simulate_symbol
from src.data.data_fetcher import DEFAULT_VN30_SYMBOLS, drop_unclosed_candle, needs_full_refetch
from src.data.universe import load_vn30_changes, membership_mask, vn30_members
from src.features.features import _build_labels
from src.features import FEATURE_COLUMNS
from src.models.ml_models import DEFAULT_MODEL_SPEC, model_config, model_is_stale, predict_ml_scores
from src.scoring import calculate_rule_based_score


class FakeModel:
    def __init__(self, proba, priors):
        self.proba = np.array(proba)
        self.class_priors_ = priors

    def predict_proba(self, x):
        return np.tile(self.proba, (len(x), 1))


def feature_rows(count=1, value=0.0):
    return pd.DataFrame({column: [value] * count for column in FEATURE_COLUMNS})


class FetcherTests(unittest.TestCase):
    def test_unclosed_candle_is_dropped_only_during_session(self):
        frame = pd.DataFrame({"time": pd.to_datetime(["2026-09-16", "2026-09-17"])})

        during = drop_unclosed_candle(frame, now=datetime(2026, 9, 17, 10, 0))
        after = drop_unclosed_candle(frame, now=datetime(2026, 9, 17, 15, 5))

        self.assertEqual(len(during), 1)
        self.assertEqual(len(after), 2)

    def test_price_adjustment_triggers_full_refetch(self):
        times = pd.to_datetime(["2026-09-15", "2026-09-16"])
        old = pd.DataFrame({"time": times, "close": [100.0, 101.0]})

        self.assertFalse(needs_full_refetch(old, old.copy()))
        self.assertTrue(needs_full_refetch(old, old.assign(close=[98.0, 99.0])))


class MlScoreTests(unittest.TestCase):
    def test_neutral_model_scores_fifty_on_rule_scale(self):
        priors = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
        neutral = FakeModel([1 / 3, 1 / 3, 1 / 3], priors)
        bullish = FakeModel([0.1, 0.3, 0.6], priors)

        self.assertAlmostEqual(predict_ml_scores(feature_rows(), neutral, neutral)[0], 50.0)
        self.assertAlmostEqual(predict_ml_scores(feature_rows(), bullish, bullish)[0], 75.0)

    def test_missing_model_or_nan_features_fall_back_to_neutral(self):
        priors = {0: 0.2, 1: 0.6, 2: 0.2}
        bullish = FakeModel([0.1, 0.3, 0.6], priors)

        self.assertEqual(predict_ml_scores(feature_rows(), None, None)[0], 50.0)
        self.assertEqual(predict_ml_scores(feature_rows(value=np.nan), bullish, bullish)[0], 50.0)

    def test_model_staleness(self):
        model = FakeModel([1 / 3] * 3, {})
        self.assertTrue(model_is_stale(model, pd.Timestamp("2026-09-17")))

        model.config_ = model_config()
        model.spec_ = DEFAULT_MODEL_SPEC
        model.data_end_ = pd.Timestamp("2026-09-10")
        self.assertFalse(model_is_stale(model, pd.Timestamp("2026-09-17")))
        self.assertTrue(model_is_stale(model, pd.Timestamp("2026-09-30")))


class RuleScoreTests(unittest.TestCase):
    def test_uptrend_is_not_double_counted_and_missing_index_is_neutral(self):
        row = {"close": 110.0, "sma_50": 105.0, "sma_200": 100.0, "vnindex_vs_sma50": 0.0}

        score, reasons = calculate_rule_based_score(pd.DataFrame([row, row]))

        self.assertIn("Uptrend mạnh (Giá > SMA50 > SMA200)", reasons)
        self.assertNotIn("Giá trên SMA50", reasons)
        self.assertNotIn("VNINDEX dưới SMA50", reasons)


def score_frame(symbol, scores, price=10.0):
    return pd.DataFrame({
        "time": pd.bdate_range("2026-01-01", periods=len(scores)),
        "symbol": symbol,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "score": scores,
    })


class BacktestTests(unittest.TestCase):
    def test_stop_loss_waits_for_t_plus_2_settlement(self):
        # Tín hiệu mua phiên 0 -> khớp open phiên 1; điểm sập ngay từ phiên 1
        frame = score_frame("AAA", [70] + [10] * 24)

        trades = simulate_symbol(frame, "score")["trades"]

        self.assertEqual(trades["exit_reason"].iloc[0], "stop-loss")
        self.assertGreaterEqual(trades["hold_days"].iloc[0], config.MIN_DAYS_BEFORE_OPEN_SELL)

    def test_portfolio_never_exceeds_slots(self):
        frames = [score_frame(symbol, [70] * 25) for symbol in ("AAA", "BBB", "CCC")]

        trades = simulate_portfolio(pd.concat(frames), "score", slots=2)["trades"]

        self.assertEqual(trades.groupby("entry_date")["symbol"].count().max(), 2)

    def test_barrier_exit_takes_profit_at_barrier_price(self):
        frame = score_frame("AAA", [70] + [50] * 24)
        frame.loc[4, "high"] = 11.0  # entry open 10 (phiên 1), phiên 4 chạm +5%

        trades = simulate_symbol(frame, "score", horizon=10, barrier=0.05)["trades"]

        self.assertEqual(trades["exit_reason"].iloc[0], "barrier-profit")
        self.assertAlmostEqual(trades["exit_price"].iloc[0], 10.5)

    def test_barrier_touching_both_sides_assumes_stop_first(self):
        frame = score_frame("AAA", [70] + [50] * 24)
        frame.loc[4, ["high", "low"]] = [11.0, 9.0]

        trades = simulate_symbol(frame, "score", horizon=10, barrier=0.05)["trades"]

        self.assertEqual(trades["exit_reason"].iloc[0], "barrier-stop")
        self.assertAlmostEqual(trades["exit_price"].iloc[0], 9.5)

    def test_portfolio_skips_symbols_outside_universe(self):
        inside = score_frame("AAA", [70] * 25).assign(in_universe=True)
        outside = score_frame("BBB", [90] * 25).assign(in_universe=False)

        trades = simulate_portfolio(pd.concat([inside, outside]), "score", slots=2)["trades"]

        self.assertEqual(set(trades["symbol"]), {"AAA"})

    def test_rank_ic_is_one_for_perfect_ranking(self):
        day = pd.Timestamp("2026-01-02")
        frame = pd.DataFrame({
            "time": day,
            "symbol": ["A", "B", "C", "D"],
            "score": [10, 20, 30, 40],
            "future_return": [-0.02, 0.0, 0.01, 0.05],
        })

        self.assertAlmostEqual(rank_ic(frame, "score"), 1.0)


class ExcessLabelTests(unittest.TestCase):
    def test_excess_label_compares_with_vnindex(self):
        frame = pd.DataFrame({
            "future_return": [0.05, 0.05, -0.05],
            "vnindex_future_return": [0.01, 0.03, -0.01],
        })

        labels = _build_labels(frame, forward_days=5, threshold=0.03, strategy="excess")

        self.assertEqual(labels.tolist(), [1.0, 0.0, -1.0])


class Vn30HistoryTests(unittest.TestCase):
    def test_change_log_is_consistent_and_ends_at_current_basket(self):
        changes = load_vn30_changes()
        members = set(changes.loc[changes["action"] == "base", "symbol"])
        self.assertEqual(len(members), 30)
        for date, group in changes[changes["action"] != "base"].groupby("effective_date"):
            for row in group.itertuples():
                if row.action == "add":
                    self.assertNotIn(row.symbol, members, row)
                    members.add(row.symbol)
                else:
                    self.assertIn(row.symbol, members, row)
                    members.discard(row.symbol)
            self.assertEqual(len(members), 30, date)

        self.assertEqual(members, set(DEFAULT_VN30_SYMBOLS))
        self.assertEqual(vn30_members(changes, "2026-09-17"), set(DEFAULT_VN30_SYMBOLS))

    def test_membership_mask_follows_exit_and_reentry(self):
        # SAB: có trong rổ gốc, bị loại 01/02/2021, vào lại 02/08/2021
        times = pd.Series(pd.to_datetime(["2020-01-02", "2020-09-01", "2021-03-01", "2021-09-01"]))

        mask = membership_mask(load_vn30_changes(), "SAB", times)

        self.assertEqual(mask.tolist(), [True, True, False, True])


if __name__ == "__main__":
    unittest.main()
