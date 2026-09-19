"""Boundary checks for the VN30-only research flows."""

import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.data_fetcher import fetch_latest_quotes
from api.features.assistant import POST_ROUTES, answer_question
from api.features.market import GET_ROUTES, MarketState, live_quote, recommendation


ROWS = [
    {"symbol": "FPT", "ml_score": 51.0},
    {"symbol": "HPG", "ml_score": 55.0},
    {"symbol": "ACB", "ml_score": 49.0},
]


class WebFlowTests(unittest.TestCase):
    def test_quote_batch_keeps_daily_ml_snapshot_separate(self):
        now = datetime(2026, 9, 17, 10, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
        board = pd.DataFrame([
            {"symbol": "FPT", "close_price": 120000, "reference_price": 100000,
             "volume_accumulated": 3000, "time": 1789615740000},
            {"symbol": "HPG", "close_price": float("nan"), "reference_price": 25},
        ])
        market = type("Market", (), {"quote": lambda self, symbol: board})()
        provider = type("Provider", (), {"Market": lambda self: market})()
        with patch("src.data.data_fetcher._vnstock", return_value=provider), patch("src.data.data_fetcher._throttle"):
            quotes = fetch_latest_quotes(["FPT", "HPG"])
        self.assertEqual(set(quotes), {"FPT", "HPG"})
        state = MarketState({"date": "2026-09-16", "stocks": [
            {"symbol": "FPT", "close": 98, "ml_score": 55},
            {"symbol": "HPG", "close": 25, "ml_score": 51},
        ], "history": {"FPT": [{"date": "2026-09-16", "close": 98}]}})
        with patch("api.features.market.fetch_latest_quotes", return_value=quotes):
            state.poll(now)
        self.assertEqual(state.market()["stocks"][0]["quote"]["change_pct"], 20)
        self.assertEqual(state.market()["stocks"][0]["quote"]["price"], 120)
        self.assertEqual(state.market()["stocks"][0]["quote"]["source_at"][11:16], "10:29")
        self.assertNotIn("history", state.market())
        self.assertIsNone(state.market()["stocks"][1]["quote"])
        self.assertEqual(state.market()["stocks"][0]["close"], 98)
        self.assertEqual(state.history("FPT")[0]["close"], 98)
        self.assertIsNone(live_quote({"time": 1789529340000, "close_price": 120000}, now))
        with patch("api.features.market.fetch_latest_quotes", return_value={}):
            state.poll(now)
        self.assertIsNone(state.market()["stocks"][0]["quote"])
        state.poll(now.replace(hour=16))
        self.assertIsNone(state.market()["stocks"][0]["quote"])

    def test_live_quote_requires_recent_source_time_in_vietnam(self):
        now = datetime(2026, 9, 17, 10, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
        row = {"close_price": 120000}
        self.assertIsNone(live_quote(row, now))
        self.assertIsNone(live_quote({**row, "time": "2026-09-17 10:24:59"}, now))
        self.assertIsNone(live_quote({**row, "time": "2026-09-17 10:30:01"}, now))
        self.assertEqual(live_quote({**row, "time": "2026-09-17 10:29:00"}, now)["source_at"],
                         "2026-09-17T10:29:00+07:00")

    def test_feature_routes_share_one_snapshot(self):
        state = MarketState({"date": "2026-09-17", "stocks": ROWS,
                             "history": {"FPT": [{"date": "2026-09-17", "close": 74.3}]}})
        self.assertEqual(GET_ROUTES["/api/market"](state, {})["date"], "2026-09-17")
        self.assertEqual(GET_ROUTES["/api/history"](state, {"symbol": ["fpt"]})["candles"][0]["close"], 74.3)
        self.assertEqual(GET_ROUTES["/api/recommend"](state, {"symbol": ["FPT"]})["selected"]["symbol"], "FPT")
        self.assertEqual(POST_ROUTES["/api/ask"](state, {"question": "Mã nào được ML xếp cao nhất?"})["kind"], "top")
        with self.assertRaises(ValueError):
            GET_ROUTES["/api/history"](state, {"symbol": ["XYZ"]})

    def test_recommend_only_higher_scored_alternatives(self):
        result = recommendation(ROWS, "FPT")
        self.assertEqual([row["symbol"] for row in result["alternatives"]], ["HPG"])

    def test_top_ranked_does_not_claim_better_alternative(self):
        result = recommendation(ROWS, "HPG")
        self.assertFalse(result["better_available"])

    def test_reject_non_vn30_ticker_and_off_topic_question(self):
        self.assertEqual(answer_question("AAPL có tốt không?", ROWS)["kind"], "blocked")
        self.assertEqual(answer_question("Thời tiết hôm nay?", ROWS)["kind"], "blocked")
        self.assertEqual(answer_question("Mã nào được ML xếp cao nhất ở Mỹ?", ROWS)["kind"], "blocked")

    def test_guide_unknown_investor_to_top_ml_rank(self):
        result = answer_question("Mã nào được ML xếp cao nhất?", ROWS)
        self.assertEqual([row["symbol"] for row in result["items"]], ["HPG", "FPT", "ACB"])
        self.assertEqual(answer_question("Tôi không biết đầu tư gì", ROWS)["kind"], "top")


if __name__ == "__main__":
    unittest.main()
