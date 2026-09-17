"""Per-session position estimates and request validation."""

import json
import unittest

from api.features.profit_loss import estimate_profit_loss


class ExampleMarket:
    def history(self, symbol):
        if symbol != "ACB":
            return None
        return [
            {"date": "2026-09-15", "close": 20},
            {"date": "2026-09-16", "close": 22},
            {"date": "2026-09-17", "close": 19},
        ]


class ProfitLossTests(unittest.TestCase):
    def test_profit_and_loss_after_each_closed_session(self):
        result = estimate_profit_loss(ExampleMarket(), {
            "symbol": "ACB", "entry_date": "2026-09-15",
            "buy_price": 20, "quantity": 100,
        })
        self.assertEqual(result["buy_total_vnd"], 2_005_000)
        self.assertEqual([row["pnl_vnd"] for row in result["sessions"]],
                         [-12_000, 187_300, -111_650])
        self.assertEqual([row["daily_change_vnd"] for row in result["sessions"]],
                         [None, 199_300, -298_950])
        json.dumps(result, allow_nan=False)

    def test_rejects_invalid_inputs(self):
        request = {"symbol": "ACB", "entry_date": "2026-09-15",
                   "buy_price": 20, "quantity": 100}
        for bad in ({"quantity": 0}, {"quantity": 1.5}, {"buy_price": float("nan")},
                    {"entry_date": "2026-09-14"}, {"symbol": "XYZ"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimate_profit_loss(ExampleMarket(), request | bad)


if __name__ == "__main__":
    unittest.main()
