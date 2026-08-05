import unittest

from profit_model import (
    BetOffer,
    RiskConfig,
    build_best_two_leg_parlay,
    evaluate_offer,
    select_singles,
)


class ProfitModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RiskConfig(
            min_single_ev=0.02,
            min_probability_edge=0.02,
            uncertainty_haircut=0.02,
            min_parlay_ev=0.05,
        )

    def test_positive_ev_offer_is_selected(self):
        offer = BetOffer(
            match_id="A",
            market="asian_handicap",
            selection="home -0.25",
            decimal_odds=2.0,
            model_probability=0.58,
            model_probability_lower=0.55,
        )
        decision = evaluate_offer(offer, self.config)
        self.assertEqual(decision.action, "BET")
        self.assertGreater(decision.expected_value, 0.0)
        self.assertLessEqual(
            decision.stake_fraction,
            self.config.max_single_stake_fraction,
        )

    def test_negative_ev_offer_returns_no_bet(self):
        offer = BetOffer(
            match_id="A",
            market="totals",
            selection="under 2.5",
            decimal_odds=1.80,
            model_probability=0.54,
            model_probability_lower=0.52,
        )
        decision = evaluate_offer(offer, self.config)
        self.assertEqual(decision.action, "NO_BET")
        self.assertEqual(decision.stake_fraction, 0.0)

    def test_only_one_single_is_kept_per_match(self):
        offers = [
            BetOffer("A", "totals", "over 2.5", 2.0, 0.58, 0.55),
            BetOffer("A", "asian_handicap", "home -0.25", 2.1, 0.57, 0.54),
        ]
        selected, rejected = select_singles(offers, self.config)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].reason, "ONE_BET_PER_MATCH_RULE")

    def test_parlay_requires_two_different_matches(self):
        offers = [
            BetOffer("A", "totals", "over 2.5", 2.0, 0.58, 0.55),
            BetOffer("B", "asian_handicap", "away +0.25", 2.0, 0.58, 0.55),
        ]
        selected, _ = select_singles(offers, self.config)
        parlay = build_best_two_leg_parlay(selected, self.config)
        self.assertEqual(parlay.action, "PARLAY")
        self.assertEqual(len(parlay.legs), 2)
        self.assertGreater(parlay.expected_value, 0.0)

    def test_same_match_parlay_is_blocked(self):
        first = evaluate_offer(
            BetOffer("A", "totals", "over 2.5", 2.0, 0.58, 0.55),
            self.config,
        )
        second = evaluate_offer(
            BetOffer("A", "asian_handicap", "home -0.25", 2.0, 0.58, 0.55),
            self.config,
        )
        parlay = build_best_two_leg_parlay([first, second], self.config)
        self.assertEqual(parlay.action, "NO_PARLAY")


if __name__ == "__main__":
    unittest.main()
