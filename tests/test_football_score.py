"""Mathematical and temporal gates for the experimental score workflow."""

import math
import unittest
from datetime import datetime, timedelta, timezone

from football_score.model import (fair_cover, fuse, joint_distribution,
                                  market_distribution, settlement, top_score)
from football_score.review import review
from football_score.workflow import _rebuild_prior, market_asof


class ScoreWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.row = {"match_id": "m", "date": "2025-01-05", "league_avg_goals_pre": 2.6,
                    "league_home_goals_pre": 1.4, "league_away_goals_pre": 1.2,
                    "home_recent8_gf": 1.6, "home_recent8_ga": 1.1,
                    "away_recent8_gf": 1.0, "away_recent8_ga": 1.7,
                    "home_elo_pre": 1550, "away_elo_pre": 1470}
        self.theta = [0, 0, .2, .2, 0, 0, .2, .2, math.log(10), math.log(12)]

    def test_joint_adaptive_mass_and_exact_argmax(self):
        scores = joint_distribution(self.row, self.theta)
        self.assertAlmostEqual(sum(scores.values()), 1, places=10)
        self.assertGreater(max(h + a for h, a in scores), 7)
        self.assertEqual(top_score(scores), max(scores, key=scores.get))
        self.assertLess(scores.get((9, 9), 0), scores.get((2, 1), 0))

    def test_asian_quarter_split_and_push(self):
        self.assertEqual(settlement(0, -.25), (0., .5))
        self.assertEqual(settlement(1, -.75), (.5, 0.))
        self.assertEqual(settlement(3, -3), (0., 0.))
        self.assertEqual(settlement(2, -2.25), (0., .5))
        self.assertAlmostEqual(fair_cover({(2, 0): .5, (3, 0): .5}, 2.25), 2 / 3)

    def test_market_fit_and_fusion_stays_normalized(self):
        odds = {"one_x_two_odds": {"home": 2.0, "draw": 3.5, "away": 4.2},
                "over_under_odds": {"over": 1.9, "under": 1.95}, "total_line": 2.5,
                "asian_handicap": -.5, "asian_home_odds": 1.9, "asian_away_odds": 1.95}
        qm, diagnostics = market_distribution(odds)
        self.assertAlmostEqual(sum(qm.values()), 1, places=8)
        self.assertLess(max(map(abs, diagnostics["fit_residuals"])), .08)
        self.assertIsNotNone(diagnostics["asian_check_residual"])
        mixed = fuse(joint_distribution(self.row, self.theta), qm, .25)
        self.assertAlmostEqual(sum(mixed.values()), 1, places=8)

    def test_temporal_feature_group_does_not_see_same_day_result(self):
        def row(mid, day, h, a):
            return {"match_id": mid, "date": day, "league_code": "L",
                    "home_team": "Home", "away_team": "Away",
                    "actual_home_goals": str(h), "actual_away_goals": str(a)}
        same = [row("a", "2025-01-01", 9, 0), row("b", "2025-01-01", 0, 0),
                row("c", "2025-01-02", 0, 0)]
        _rebuild_prior(same)
        self.assertEqual(same[0]["home_recent8_gf"], same[1]["home_recent8_gf"])
        self.assertEqual(same[0]["home_elo_pre"], same[1]["home_elo_pre"])
        self.assertNotEqual(same[1]["home_elo_pre"], same[2]["home_elo_pre"])

    def test_after_kickoff_quote_is_blocked(self):
        quote = {"match_id": "m", "kickoff_utc": "2025-01-05T15:00:00Z",
                 "observed_at": "2025-01-05T15:05:00Z",
                 "published_at": "2025-01-05T14:00:00Z", "source": "test"}
        self.assertTrue(market_asof(self.row, quote)[1].startswith("MARKET_BLOCKED"))

    def test_weekly_audit_does_not_invent_stop_state_or_cause(self):
        kick = datetime.now(timezone.utc) - timedelta(days=2)
        old = []
        results = []
        for i in range(4):
            when = kick + timedelta(days=i // 2)
            match = {"match_id": str(i), "score": [1, 0],
                     "home_team": "H", "away_team": "A", "kickoff_utc": when.isoformat(),
                     "frozen_at": (when - timedelta(hours=2)).isoformat(),
                     "stop_state": "UNKNOWN_PREMATCH"}
            old.append(match)
            results.append({"match_id": str(i), "score": [0, 2], "status": "FT_90",
                            "home_team": "H", "away_team": "A"})
        out = review(old, results)
        self.assertEqual(out["reviewed_count"], 4)
        self.assertIsNone(out["matches"][0]["axes_hit"]["stop_state"])
        self.assertEqual(out["next_week_validation"][0]["module"], "direction")
        self.assertEqual(len(out["next_week_validation"]), 2)
        self.assertIsNone(out["matches"][0]["axes_hit"]["margin"])
        self.assertIn("UNDETERMINED", out["matches"][0]["attribution"])


if __name__ == "__main__":
    unittest.main()
