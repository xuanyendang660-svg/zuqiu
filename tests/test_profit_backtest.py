import unittest

from profit_model.backtest import SettledBet, summarize_records


class ProfitBacktestTests(unittest.TestCase):
    def test_reports_singles_and_parlays_separately(self):
        records = [
            SettledBet("s1", "single", 1.0, 2.0, "win", 1.90),
            SettledBet("s2", "single", 1.0, 1.90, "loss", 1.95),
            SettledBet("p1", "parlay", 0.5, 4.0, "win", 3.60),
        ]
        report = summarize_records(records)
        self.assertEqual(report["overall"]["bets"], 3)
        self.assertEqual(report["singles"]["bets"], 2)
        self.assertEqual(report["parlays"]["bets"], 1)
        self.assertAlmostEqual(report["singles"]["total_profit"], 0.0)
        self.assertGreater(report["parlays"]["roi"], 0.0)
        self.assertGreater(report["overall"]["average_clv"], 0.0)

    def test_drawdown_and_losing_streak(self):
        records = [
            SettledBet("a", "single", 1.0, 2.0, "loss"),
            SettledBet("b", "single", 1.0, 2.0, "loss"),
            SettledBet("c", "single", 1.0, 2.0, "win"),
        ]
        report = summarize_records(records)["overall"]
        self.assertEqual(report["longest_losing_streak"], 2)
        self.assertEqual(report["max_drawdown_units"], 2.0)


if __name__ == "__main__":
    unittest.main()
