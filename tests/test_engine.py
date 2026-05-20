import unittest

from score_model import ExactScoreModel, MatchInput
from score_model.engine import BETTING_BOARD_SCORES, SCOREBOOK_SCORES


def _base_match(**overrides):
    data = {
        "match_id": "test-001",
        "league": "Test League",
        "home_team": "Home",
        "away_team": "Away",
        "league_avg_goals": 2.65,
        "home_attack": 1.0,
        "away_attack": 1.0,
        "home_defense": 1.0,
        "away_defense": 1.0,
        "market": {
            "confidence": 0.5,
            "expected_home_goals": 1.4,
            "expected_away_goals": 1.1,
            "exact_score_odds": {
                "0-0": 10.0,
                "1-0": 7.5,
                "0-1": 9.5,
                "1-1": 6.5,
                "2-0": 10.5,
                "2-1": 8.5,
                "1-2": 12.5,
                "2-2": 14.0,
                "3-1": 22.0,
            },
        },
    }
    data.update(overrides)
    return MatchInput.from_dict(data)


def _public_home_favorite_match(match_id="public-home"):
    return _base_match(
        match_id=match_id,
        league_avg_goals=3.0,
        league_home_advantage=0.08,
        league_draw_bias=0.02,
        league_volatility=0.72,
        home_attack=1.35,
        away_attack=1.08,
        home_defense=1.22,
        away_defense=1.28,
        recent_home_xg=1.9,
        recent_away_xg=1.35,
        recent_home_xga=1.4,
        recent_away_xga=1.55,
        home_form=0.18,
        away_form=0.04,
        home_motivation=0.20,
        away_motivation=0.18,
        market={
            "confidence": 0.72,
            "expected_home_goals": 1.9,
            "expected_away_goals": 1.05,
            "one_x_two_odds": {"home": 1.55, "draw": 4.1, "away": 5.4},
            "exact_score_odds": {
                "1-0": 6.0,
                "2-0": 6.5,
                "2-1": 7.0,
                "1-1": 7.5,
                "3-1": 10.0,
                "3-0": 11.0,
                "2-2": 13.0,
                "1-2": 18.0,
                "3-2": 20.0,
                "4-1": 21.0,
            },
        },
    )


class ExactScoreModelTests(unittest.TestCase):
    def test_predict_returns_single_score_inside_expanded_goal_range(self):
        pick = ExactScoreModel(score_profile="full").predict(_base_match())

        home, away = [int(part) for part in pick.score.split("-")]
        self.assertLessEqual(home + away, 10)
        self.assertEqual(len(pick.layer_winners), 11)
        self.assertTrue(all(candidate.total_goals <= 10 for candidate in pick.layer_winners))

    def test_layer_winners_cover_every_total_goal_layer(self):
        pick = ExactScoreModel(score_profile="full").predict(_base_match())

        self.assertEqual([candidate.total_goals for candidate in pick.layer_winners], list(range(11)))

    def test_default_scorebook_uses_user_score_list(self):
        model = ExactScoreModel()

        self.assertEqual(set(model.score_grid), set(SCOREBOOK_SCORES))
        self.assertEqual(len(model.score_grid), 41)
        self.assertIn((6, 0), model.score_grid)
        self.assertIn((4, 3), model.score_grid)

    def test_default_scorebook_evaluates_every_betting_board_score(self):
        model = ExactScoreModel()
        match = _base_match()
        _home_mu, _away_mu, candidates, _layer_winners = model._score_match(match)
        candidate_scores = {(candidate.home_goals, candidate.away_goals) for candidate in candidates}

        self.assertEqual(len(BETTING_BOARD_SCORES), 28)
        self.assertTrue(set(BETTING_BOARD_SCORES).issubset(set(model.score_grid)))
        self.assertTrue(set(BETTING_BOARD_SCORES).issubset(candidate_scores))

    def test_model_can_promote_non_comfort_away_score_when_structure_supports_it(self):
        match = _base_match(
            home_attack=0.88,
            away_attack=1.32,
            home_defense=1.22,
            away_defense=0.82,
            home_absence_impact=0.55,
            away_motivation=0.45,
            league_volatility=0.72,
            market={
                "confidence": 0.65,
                "expected_home_goals": 1.2,
                "expected_away_goals": 1.25,
                "one_x_two_odds": {"home": 2.35, "draw": 3.4, "away": 2.95},
                "exact_score_odds": {
                    "0-0": 12.0,
                    "1-0": 8.0,
                    "0-1": 8.5,
                    "1-1": 6.8,
                    "2-1": 9.0,
                    "1-2": 13.0,
                    "0-2": 18.0,
                    "1-3": 38.0,
                    "2-3": 55.0,
                },
            },
        )

        pick = ExactScoreModel().predict(match)
        layer_scores = {candidate.score for candidate in pick.layer_winners}
        self.assertTrue(layer_scores.intersection({"0-2", "1-2", "1-3", "2-3"}))

    def test_final_pick_can_be_aggressive_when_edge_and_structure_align(self):
        match = _base_match(
            league_avg_goals=3.05,
            league_home_advantage=0.04,
            league_draw_bias=-0.2,
            league_volatility=0.85,
            home_attack=0.78,
            away_attack=1.48,
            home_defense=1.42,
            away_defense=0.78,
            recent_home_xg=0.9,
            recent_away_xg=2.0,
            recent_home_xga=1.9,
            recent_away_xga=0.95,
            home_form=-0.45,
            away_form=0.5,
            home_absence_impact=0.72,
            away_absence_impact=0.05,
            home_rotation_risk=0.3,
            away_rotation_risk=0.02,
            away_motivation=0.6,
            market={
                "confidence": 0.72,
                "expected_home_goals": 1.25,
                "expected_away_goals": 1.95,
                "one_x_two_odds": {"home": 2.5, "draw": 3.45, "away": 2.65},
                "exact_score_odds": {
                    "0-0": 13.0,
                    "1-0": 9.0,
                    "0-1": 8.0,
                    "1-1": 7.0,
                    "0-2": 15.0,
                    "1-2": 12.0,
                    "0-3": 34.0,
                    "1-3": 42.0,
                    "2-3": 60.0,
                    "1-4": 90.0,
                },
            },
        )

        pick = ExactScoreModel().predict(match)
        self.assertIn(pick.score, {"1-2", "1-3", "1-4", "2-3"})
        self.assertIn(pick.confidence_band, {"B", "C+", "C"})

    def test_aggressive_mode_can_move_off_crowded_draw(self):
        match = _base_match(
            league_avg_goals=2.78,
            league_home_advantage=0.08,
            league_draw_bias=0.08,
            league_volatility=0.62,
            home_attack=1.40 / 1.39,
            away_attack=1.13 / 1.39,
            home_defense=1.53 / 1.39,
            away_defense=0.87 / 1.39,
            recent_home_xg=1.40,
            recent_away_xg=1.13,
            recent_home_xga=1.53,
            recent_away_xga=0.87,
            home_form=-0.10,
            away_form=0.12,
            home_motivation=0.18,
            away_motivation=0.05,
            market={
                "confidence": 0.64,
                "expected_home_goals": 1.65,
                "expected_away_goals": 1.25,
                "one_x_two_odds": {"home": 1.98, "draw": 3.60, "away": 3.60},
                "exact_score_odds": {
                    "1-0": 7.0,
                    "1-1": 6.2,
                    "2-1": 8.0,
                    "2-2": 10.0,
                    "2-0": 9.5,
                    "3-1": 15.0,
                    "3-2": 18.0,
                    "1-2": 12.0,
                    "0-0": 12.0,
                    "0-1": 11.0,
                    "0-2": 20.0,
                },
            },
        )

        balanced_pick = ExactScoreModel(mode="balanced").predict(match)
        aggressive_pick = ExactScoreModel(mode="aggressive").predict(match)

        self.assertEqual(balanced_pick.score, "1-1")
        self.assertNotEqual(aggressive_pick.score, "1-1")

    def test_contrarian_mode_can_fade_public_home_favorite_score(self):
        match = _public_home_favorite_match()

        balanced_pick = ExactScoreModel(mode="balanced").predict(match)
        contrarian_pick = ExactScoreModel(mode="contrarian").predict(match)

        self.assertEqual(balanced_pick.score, "2-1")
        self.assertEqual(contrarian_pick.score, "2-2")

    def test_predict_many_diversifies_repeated_two_two_scores(self):
        model = ExactScoreModel(mode="contrarian")
        matches = [_public_home_favorite_match(f"slate-{index}") for index in range(6)]

        isolated_scores = [model.predict(match).score for match in matches]
        slate_scores = [pick.score for pick in model.predict_many(matches)]

        self.assertEqual(isolated_scores, ["2-2"] * 6)
        self.assertLessEqual(slate_scores.count("2-2"), 1)
        self.assertGreater(len(set(slate_scores)), 1)

    def test_predict_many_reaudits_overcrowded_comfort_structure(self):
        model = ExactScoreModel(mode="adaptive")
        matches = [_public_home_favorite_match(f"comfort-audit-{index}") for index in range(12)]

        isolated_picks = [model.predict(match) for match in matches]
        slate_picks = model.predict_many(matches)
        isolated_public = sum(
            model._is_public_or_comfort_pick(pick.candidate, matches[index], pick.expected_home_goals, pick.expected_away_goals)
            for index, pick in enumerate(isolated_picks)
        )
        slate_public = sum(
            model._is_public_or_comfort_pick(pick.candidate, matches[index], pick.expected_home_goals, pick.expected_away_goals)
            for index, pick in enumerate(slate_picks)
        )

        self.assertEqual(isolated_public, 12)
        self.assertLessEqual(slate_public, 6)
        self.assertTrue(
            any(
                "slate structure audit" in " ".join(pick.candidate.reasons)
                or "slate goal balance audit" in " ".join(pick.candidate.reasons)
                for pick in slate_picks
            )
        )

    def test_predict_many_moderates_overcrowded_high_total_structure(self):
        model = ExactScoreModel(mode="adaptive")
        matches = [_public_home_favorite_match(f"goal-audit-{index}") for index in range(12)]

        slate_picks = model.predict_many(matches)
        high_total_count = sum(1 for pick in slate_picks if pick.candidate.total_goals >= 4)

        self.assertLessEqual(high_total_count, 5)
        self.assertTrue(
            any("slate goal balance audit" in " ".join(pick.candidate.reasons) for pick in slate_picks)
        )

    def test_goal_balance_uses_dynamic_target_for_high_pressure_slate(self):
        def high_pressure_match(index: int) -> MatchInput:
            return _base_match(
                match_id=f"high-pressure-slate-{index}",
                league_avg_goals=3.18,
                league_volatility=0.86,
                home_attack=1.45,
                away_attack=1.22,
                home_defense=1.28,
                away_defense=1.44,
                recent_home_xg=2.05,
                recent_away_xg=1.55,
                recent_home_xga=1.45,
                recent_away_xga=1.95,
                home_motivation=0.75,
                away_motivation=0.60,
                home_table_pressure=0.85,
                away_table_pressure=0.70,
                home_survival_pressure=0.75 if index % 2 == 0 else 0.0,
                away_survival_pressure=0.75 if index % 2 == 1 else 0.0,
                is_final_round=True,
                endgame_chaos=0.78,
                market={
                    "confidence": 0.62,
                    "expected_home_goals": 1.95,
                    "expected_away_goals": 1.45,
                    "one_x_two_odds": {"home": 2.05, "draw": 3.65, "away": 3.25},
                    "total_goals_line": 3.25,
                    "over_money_heat": 0.76,
                    "exact_score_odds": {
                        "2-1": 8.0,
                        "1-2": 10.0,
                        "2-2": 11.0,
                        "3-1": 15.0,
                        "1-3": 22.0,
                        "3-2": 28.0,
                        "2-3": 32.0,
                        "4-2": 58.0,
                        "3-3": 65.0,
                        "3-0": 26.0,
                        "0-3": 38.0,
                    },
                },
            )

        model = ExactScoreModel(mode="adaptive")
        matches = [high_pressure_match(index) for index in range(12)]
        contexts = []
        for match in matches:
            home_mu, away_mu, candidates, layer_winners = model._score_match(match)
            selected = model._select_final(match, home_mu, away_mu, candidates, layer_winners)
            contexts.append((match, home_mu, away_mu, candidates, layer_winners, selected))

        picks = model.predict_many(matches)

        self.assertGreaterEqual(model._slate_high_total_target(contexts), 7)
        self.assertGreaterEqual(sum(1 for pick in picks if pick.candidate.total_goals >= 4), 7)

    def test_slate_audit_does_not_force_true_hot_favorite_batch(self):
        def true_hot_match(index: int) -> MatchInput:
            return _base_match(
                match_id=f"true-hot-batch-{index}",
                league_avg_goals=2.85,
                league_volatility=0.58,
                home_attack=1.62,
                away_attack=0.84,
                home_defense=0.72,
                away_defense=1.36,
                recent_home_xg=2.05,
                recent_away_xg=0.88,
                recent_home_xga=0.82,
                recent_away_xga=1.75,
                home_form=0.35,
                away_form=-0.25,
                home_motivation=0.32,
                away_motivation=0.04,
                market={
                    "confidence": 0.76,
                    "expected_home_goals": 2.05,
                    "expected_away_goals": 0.82,
                    "one_x_two_odds": {"home": 1.36, "draw": 5.0, "away": 8.5},
                    "exact_score_odds": {
                        "1-0": 6.8,
                        "2-0": 6.4,
                        "2-1": 7.2,
                        "3-0": 8.5,
                        "3-1": 9.0,
                        "1-1": 11.0,
                        "2-2": 21.0,
                    },
                },
            )

        picks = ExactScoreModel(mode="adaptive").predict_many([true_hot_match(index) for index in range(8)])

        self.assertFalse(any("slate structure audit" in " ".join(pick.candidate.reasons) for pick in picks))
        self.assertTrue(all(pick.score in {"2-0", "2-1", "3-0", "3-1", "3-2", "4-0", "4-1"} for pick in picks))

    def test_adaptive_mode_keeps_true_hot_favorite_on_normal_path(self):
        match = _base_match(
            league_avg_goals=2.85,
            league_volatility=0.58,
            home_attack=1.62,
            away_attack=0.84,
            home_defense=0.72,
            away_defense=1.36,
            recent_home_xg=2.05,
            recent_away_xg=0.88,
            recent_home_xga=0.82,
            recent_away_xga=1.75,
            home_form=0.35,
            away_form=-0.25,
            home_motivation=0.32,
            away_motivation=0.04,
            market={
                "confidence": 0.76,
                "expected_home_goals": 2.05,
                "expected_away_goals": 0.82,
                "one_x_two_odds": {"home": 1.36, "draw": 5.0, "away": 8.5},
                "exact_score_odds": {
                    "1-0": 6.8,
                    "2-0": 6.4,
                    "2-1": 7.2,
                    "3-0": 8.5,
                    "3-1": 9.0,
                    "1-1": 11.0,
                    "2-2": 21.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"2-0", "2-1", "3-0", "3-1"})
        self.assertIn("scenario gate: normal", " ".join(pick.candidate.reasons))

    def test_adaptive_mode_can_mark_fake_hot_favorite_as_draw_or_upset(self):
        match = _base_match(
            league_avg_goals=2.62,
            league_draw_bias=0.10,
            league_volatility=0.50,
            home_attack=1.02,
            away_attack=1.06,
            home_defense=1.14,
            away_defense=0.94,
            recent_home_xg=1.12,
            recent_away_xg=1.22,
            recent_home_xga=1.42,
            recent_away_xga=1.02,
            home_form=-0.28,
            away_form=0.22,
            home_absence_impact=0.35,
            away_motivation=0.38,
            market={
                "confidence": 0.70,
                "expected_home_goals": 1.30,
                "expected_away_goals": 1.05,
                "one_x_two_odds": {"home": 1.72, "draw": 3.55, "away": 4.80},
                "exact_score_odds": {
                    "1-0": 6.4,
                    "2-0": 8.2,
                    "2-1": 7.6,
                    "1-1": 6.8,
                    "1-2": 15.0,
                    "0-1": 13.0,
                    "0-2": 26.0,
                    "2-2": 18.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"1-1", "0-1", "1-2", "0-2"})
        self.assertRegex(" ".join(pick.candidate.reasons), r"scenario gate: (draw|upset)|market independence layer")

    def test_market_independence_layer_treats_line_as_public_flow_not_result_cause(self):
        match = _base_match(
            match_id="market-line-is-not-cause",
            league_avg_goals=2.62,
            league_volatility=0.54,
            home_attack=0.82,
            away_attack=1.42,
            home_defense=1.48,
            away_defense=0.82,
            recent_home_xg=0.82,
            recent_away_xg=1.72,
            recent_home_xga=1.85,
            recent_away_xga=0.88,
            home_form=-0.34,
            away_form=0.36,
            home_absence_impact=0.38,
            away_motivation=0.62,
            away_table_pressure=0.78,
            market={
                "confidence": 0.78,
                "expected_home_goals": 1.32,
                "expected_away_goals": 1.02,
                "one_x_two_odds": {"home": 1.52, "draw": 3.90, "away": 6.20},
                "asian_handicap": -1.25,
                "home_money_heat": 0.88,
                "favorite_money_heat": 0.84,
                "away_money_heat": 0.22,
                "bookmaker_trap_risk": 0.72,
                "exact_score_odds": {
                    "1-0": 6.0,
                    "2-0": 7.6,
                    "2-1": 7.2,
                    "1-1": 7.0,
                    "0-1": 17.0,
                    "0-2": 36.0,
                    "1-2": 19.0,
                    "1-3": 55.0,
                    "2-2": 18.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"0-1", "0-2", "1-2", "1-3"})
        self.assertNotIn(pick.score, {"1-0", "2-0", "2-1", "3-1"})
        self.assertIn("market independence", " ".join(pick.candidate.reasons))

    def test_draw_sufficient_table_script_promotes_draw_layer(self):
        match = _base_match(
            match_id="draw-enough",
            league_avg_goals=2.55,
            league_draw_bias=0.08,
            league_volatility=0.42,
            home_attack=1.02,
            away_attack=1.00,
            home_defense=0.95,
            away_defense=0.98,
            home_draw_sufficient=True,
            away_draw_sufficient=True,
            market={
                "confidence": 0.60,
                "expected_home_goals": 1.15,
                "expected_away_goals": 1.05,
                "one_x_two_odds": {"home": 2.45, "draw": 3.05, "away": 3.10},
                "draw_money_heat": 0.35,
                "exact_score_odds": {
                    "0-0": 8.5,
                    "1-1": 5.6,
                    "2-2": 11.0,
                    "1-0": 8.0,
                    "0-1": 8.5,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"0-0", "1-1", "2-2"})
        self.assertIn("table script allows both sides to accept draw", " ".join(pick.candidate.reasons))

    def test_under_draw_script_can_select_zero_zero_not_only_one_one(self):
        match = _base_match(
            match_id="low-lock",
            league_avg_goals=2.05,
            league_home_advantage=0.02,
            league_draw_bias=0.22,
            league_volatility=0.28,
            home_attack=0.78,
            away_attack=0.75,
            home_defense=0.70,
            away_defense=0.72,
            recent_home_xg=0.78,
            recent_away_xg=0.70,
            recent_home_xga=0.62,
            recent_away_xga=0.66,
            home_draw_sufficient=True,
            away_draw_sufficient=True,
            weather_goal_drag=0.45,
            market={
                "confidence": 0.68,
                "expected_home_goals": 0.82,
                "expected_away_goals": 0.72,
                "one_x_two_odds": {"home": 2.75, "draw": 2.82, "away": 3.05},
                "total_goals_line": 2.0,
                "under_money_heat": 0.78,
                "draw_money_heat": 0.52,
                "exact_score_odds": {
                    "0-0": 6.8,
                    "1-1": 5.8,
                    "1-0": 7.2,
                    "0-1": 8.0,
                    "2-0": 15.0,
                    "0-2": 18.0,
                    "2-1": 13.0,
                    "1-2": 15.0,
                    "2-2": 24.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertEqual(pick.score, "0-0")
        self.assertIn("under/draw script keeps 0-0 route live", " ".join(pick.candidate.reasons))

    def test_strong_match_under_trap_can_reverse_false_cover_to_small_cold(self):
        match = _base_match(
            match_id="strong-match-under-upset",
            league="CSL",
            home_team="Hot Home",
            away_team="Elite Away",
            league_avg_goals=3.05,
            league_volatility=0.54,
            home_attack=1.52,
            away_attack=1.45,
            home_defense=0.92,
            away_defense=1.05,
            recent_home_xg=2.05,
            recent_away_xg=1.72,
            recent_home_xga=0.82,
            recent_away_xga=1.22,
            home_form=0.55,
            away_form=0.25,
            home_motivation=0.50,
            away_motivation=0.45,
            market={
                "confidence": 0.76,
                "expected_home_goals": 1.85,
                "expected_away_goals": 1.35,
                "one_x_two_odds": {"home": 1.78, "draw": 3.75, "away": 4.0},
                "asian_handicap": -0.75,
                "total_goals_line": 3.25,
                "home_money_heat": 0.74,
                "favorite_money_heat": 0.72,
                "over_money_heat": 0.42,
                "under_money_heat": 0.58,
                "bookmaker_trap_risk": 0.28,
                "exact_score_odds": {
                    "1-0": 9.0,
                    "2-0": 10.0,
                    "2-1": 7.5,
                    "3-1": 12.0,
                    "3-2": 18.0,
                    "4-1": 30.0,
                    "4-2": 70.0,
                    "1-1": 7.5,
                    "0-1": 15.0,
                    "0-2": 30.0,
                    "1-2": 13.0,
                    "2-2": 10.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertEqual(pick.score, "0-1")
        self.assertIn("strong-match under trap arbitration", " ".join(pick.candidate.reasons))

    def test_balanced_under_draw_compresses_two_two_to_one_one(self):
        match = _base_match(
            match_id="balanced-under-draw-compression",
            league="CSL",
            home_team="Sticky Home",
            away_team="Public Away",
            league_avg_goals=2.90,
            league_draw_bias=0.05,
            league_volatility=0.56,
            home_attack=1.12,
            away_attack=1.38,
            home_defense=1.10,
            away_defense=1.02,
            recent_home_xg=1.32,
            recent_away_xg=1.60,
            recent_home_xga=1.25,
            recent_away_xga=1.16,
            home_form=0.15,
            away_form=0.30,
            home_motivation=0.36,
            away_motivation=0.32,
            market={
                "confidence": 0.74,
                "expected_home_goals": 1.20,
                "expected_away_goals": 1.55,
                "one_x_two_odds": {"home": 3.45, "draw": 3.70, "away": 2.0},
                "asian_handicap": 0.75,
                "total_goals_line": 2.75,
                "away_money_heat": 0.70,
                "favorite_money_heat": 0.68,
                "over_money_heat": 0.38,
                "under_money_heat": 0.62,
                "bookmaker_trap_risk": 0.34,
                "exact_score_odds": {
                    "0-1": 8.0,
                    "0-2": 11.0,
                    "1-2": 8.0,
                    "1-1": 7.0,
                    "2-2": 10.0,
                    "2-1": 14.0,
                    "1-0": 13.0,
                    "2-3": 25.0,
                    "3-3": 60.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertEqual(pick.score, "1-1")
        self.assertIn("strong-match under trap arbitration", " ".join(pick.candidate.reasons))

    def test_deep_handicap_trap_can_stop_blind_cover_score(self):
        match = _base_match(
            match_id="deep-trap",
            league_avg_goals=2.75,
            league_volatility=0.55,
            home_attack=1.55,
            away_attack=0.95,
            home_defense=0.82,
            away_defense=1.25,
            recent_home_xg=2.0,
            recent_away_xg=1.0,
            recent_home_xga=0.85,
            recent_away_xga=1.6,
            home_form=0.25,
            away_motivation=0.45,
            home_settled=True,
            away_table_pressure=0.80,
            market={
                "confidence": 0.75,
                "expected_home_goals": 2.10,
                "expected_away_goals": 0.75,
                "one_x_two_odds": {"home": 1.35, "draw": 5.0, "away": 8.0},
                "asian_handicap": -1.75,
                "home_money_heat": 0.82,
                "favorite_money_heat": 0.80,
                "bookmaker_trap_risk": 0.55,
                "exact_score_odds": {
                    "2-0": 6.4,
                    "3-0": 7.0,
                    "3-1": 8.0,
                    "2-1": 7.2,
                    "1-1": 14.0,
                    "1-2": 30.0,
                    "2-2": 22.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertNotIn(pick.score, {"3-0", "4-0", "4-1", "5-0"})
        self.assertRegex(" ".join(pick.candidate.reasons), r"deep-(draw|noncover|upset)")

    def test_true_deep_with_big_win_need_keeps_cover_route(self):
        match = _base_match(
            match_id="true-deep",
            league_avg_goals=2.95,
            league_volatility=0.68,
            home_attack=1.65,
            away_attack=0.72,
            home_defense=0.72,
            away_defense=1.55,
            recent_home_xg=2.25,
            recent_away_xg=0.72,
            recent_home_xga=0.75,
            recent_away_xga=1.95,
            home_form=0.35,
            away_form=-0.25,
            home_motivation=0.65,
            away_motivation=-0.20,
            home_table_pressure=0.85,
            home_big_win_need=0.65,
            away_settled=True,
            market={
                "confidence": 0.74,
                "expected_home_goals": 2.35,
                "expected_away_goals": 0.62,
                "one_x_two_odds": {"home": 1.28, "draw": 5.7, "away": 10.0},
                "asian_handicap": -1.75,
                "home_money_heat": 0.65,
                "favorite_money_heat": 0.62,
                "bookmaker_trap_risk": 0.05,
                "exact_score_odds": {
                    "2-0": 6.8,
                    "3-0": 6.5,
                    "3-1": 8.0,
                    "4-0": 11.0,
                    "4-1": 12.0,
                    "5-0": 18.0,
                    "1-0": 9.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"3-0", "3-1", "4-0", "4-1", "5-0"})
        self.assertIn("deep-cover", " ".join(pick.candidate.reasons))

    def test_final_round_parade_can_stretch_past_public_three_one(self):
        match = _base_match(
            match_id="final-parade",
            league_avg_goals=3.05,
            league_volatility=0.72,
            league_home_advantage=0.16,
            home_attack=1.85,
            away_attack=0.95,
            home_defense=0.82,
            away_defense=1.95,
            recent_home_xg=2.55,
            recent_away_xg=0.95,
            recent_home_xga=0.85,
            recent_away_xga=2.35,
            home_form=0.45,
            away_form=-0.50,
            home_motivation=0.35,
            away_motivation=-0.20,
            home_settled=True,
            away_settled=True,
            is_final_round=True,
            endgame_chaos=0.75,
            home_celebration_risk=0.80,
            away_collapse_risk=0.85,
            market={
                "confidence": 0.70,
                "expected_home_goals": 2.45,
                "expected_away_goals": 0.82,
                "one_x_two_odds": {"home": 1.32, "draw": 5.8, "away": 9.0},
                "asian_handicap": -1.75,
                "total_goals_line": 3.25,
                "home_money_heat": 0.62,
                "favorite_money_heat": 0.58,
                "over_money_heat": 0.64,
                "bookmaker_trap_risk": 0.05,
                "exact_score_odds": {
                    "2-0": 8.0,
                    "3-0": 7.0,
                    "3-1": 8.0,
                    "4-0": 11.0,
                    "4-1": 12.0,
                    "5-0": 19.0,
                    "5-1": 24.0,
                    "2-1": 7.5,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"4-1", "5-0", "5-1"})
        self.assertIn("celebration script can stretch winning margin", " ".join(pick.candidate.reasons))

    def test_final_round_name_team_collapse_can_pick_direct_reverse_blowout(self):
        match = _base_match(
            match_id="final-collapse",
            league_avg_goals=2.78,
            league_volatility=0.66,
            league_home_advantage=0.10,
            home_attack=1.05,
            away_attack=1.38,
            home_defense=1.72,
            away_defense=0.92,
            recent_home_xg=1.05,
            recent_away_xg=1.70,
            recent_home_xga=2.10,
            recent_away_xga=0.88,
            home_form=-0.50,
            away_form=0.40,
            home_motivation=0.02,
            away_motivation=0.62,
            home_settled=True,
            is_final_round=True,
            endgame_chaos=0.72,
            home_collapse_risk=0.82,
            away_table_pressure=0.78,
            market={
                "confidence": 0.68,
                "expected_home_goals": 1.05,
                "expected_away_goals": 1.55,
                "one_x_two_odds": {"home": 2.15, "draw": 3.5, "away": 3.1},
                "asian_handicap": -0.25,
                "total_goals_line": 2.75,
                "home_money_heat": 0.62,
                "away_money_heat": 0.36,
                "favorite_money_heat": 0.62,
                "over_money_heat": 0.56,
                "bookmaker_trap_risk": 0.62,
                "exact_score_odds": {
                    "1-1": 6.8,
                    "2-1": 8.0,
                    "1-2": 10.0,
                    "0-2": 18.0,
                    "1-3": 32.0,
                    "0-3": 55.0,
                    "2-3": 45.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"0-3", "1-3", "1-4"})
        self.assertIn("final-round home collapse risk supports blowout branch", " ".join(pick.candidate.reasons))

    def test_final_round_survival_team_can_attack_past_underdog_market(self):
        match = _base_match(
            match_id="survival-home-burst",
            league_avg_goals=3.25,
            league_volatility=0.88,
            league_home_advantage=0.16,
            home_attack=1.85,
            away_attack=0.95,
            home_defense=1.20,
            away_defense=2.45,
            recent_home_xg=2.85,
            recent_away_xg=1.05,
            recent_home_xga=1.20,
            recent_away_xga=2.90,
            home_form=0.55,
            away_form=-0.75,
            home_motivation=0.95,
            away_motivation=-0.20,
            home_table_pressure=1.0,
            home_survival_pressure=1.0,
            away_settled=True,
            away_collapse_risk=0.95,
            is_final_round=True,
            endgame_chaos=0.95,
            market={
                "confidence": 0.62,
                "expected_home_goals": 2.55,
                "expected_away_goals": 1.05,
                "one_x_two_odds": {"home": 2.7, "draw": 3.9, "away": 2.35},
                "asian_handicap": 0.25,
                "total_goals_line": 3.25,
                "home_money_heat": 0.32,
                "away_money_heat": 0.54,
                "favorite_money_heat": 0.58,
                "over_money_heat": 0.78,
                "bookmaker_trap_risk": 0.60,
                "exact_score_odds": {
                    "2-1": 9.0,
                    "3-1": 18.0,
                    "3-0": 26.0,
                    "4-1": 50.0,
                    "5-1": 120.0,
                    "6-1": 260.0,
                    "3-2": 25.0,
                    "4-2": 55.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"3-0", "4-0", "4-1", "5-1", "6-1"})
        self.assertIn("home survival pressure supports proactive attack route", " ".join(pick.candidate.reasons))
        self.assertNotIn(pick.score, {"1-1", "2-1", "1-2"})

    def test_forced_exit_gate_overrides_fake_hot_public_favorite(self):
        match = _base_match(
            match_id="forced-fake-hot",
            league_avg_goals=2.75,
            league_volatility=0.60,
            league_draw_bias=0.05,
            home_attack=1.22,
            away_attack=1.12,
            home_defense=1.28,
            away_defense=1.05,
            recent_home_xg=1.35,
            recent_away_xg=1.45,
            recent_home_xga=1.70,
            recent_away_xga=1.08,
            home_form=-0.20,
            away_form=0.18,
            home_absence_impact=0.35,
            home_motivation=0.08,
            away_motivation=0.58,
            away_table_pressure=0.78,
            market={
                "confidence": 0.72,
                "expected_home_goals": 1.45,
                "expected_away_goals": 1.15,
                "one_x_two_odds": {"home": 1.68, "draw": 3.75, "away": 5.0},
                "asian_handicap": -1.0,
                "home_money_heat": 0.82,
                "favorite_money_heat": 0.78,
                "bookmaker_trap_risk": 0.62,
                "exact_score_odds": {
                    "1-0": 6.4,
                    "2-0": 7.8,
                    "2-1": 7.2,
                    "3-1": 11.0,
                    "1-1": 7.2,
                    "2-2": 16.0,
                    "1-2": 18.0,
                    "0-1": 16.0,
                    "0-2": 34.0,
                    "1-3": 48.0,
                    "2-3": 60.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"1-2", "0-2", "1-3", "2-3"})
        self.assertNotIn(pick.score, {"1-0", "2-0", "2-1", "3-1", "1-1"})
        self.assertIn("forced exit gate", " ".join(pick.candidate.reasons))

    def test_shape_arbitration_can_promote_explicit_extreme_clean_sheet(self):
        match = _base_match(
            match_id="shape-extreme-clean",
            league_avg_goals=3.02,
            league_volatility=0.72,
            home_attack=1.82,
            away_attack=0.62,
            home_defense=0.60,
            away_defense=1.82,
            recent_home_xg=2.65,
            recent_away_xg=0.55,
            recent_home_xga=0.58,
            recent_away_xga=2.35,
            home_form=0.48,
            away_form=-0.42,
            home_motivation=0.82,
            away_motivation=-0.18,
            home_table_pressure=0.88,
            home_big_win_need=0.78,
            away_settled=True,
            market={
                "confidence": 0.74,
                "expected_home_goals": 2.48,
                "expected_away_goals": 0.52,
                "one_x_two_odds": {"home": 1.25, "draw": 6.1, "away": 11.0},
                "asian_handicap": -2.0,
                "total_goals_line": 3.0,
                "home_money_heat": 0.58,
                "favorite_money_heat": 0.58,
                "over_money_heat": 0.52,
                "bookmaker_trap_risk": 0.04,
                "exact_score_odds": {
                    "2-0": 6.6,
                    "3-0": 6.8,
                    "3-1": 8.8,
                    "4-0": 13.0,
                    "4-1": 15.0,
                    "5-0": 26.0,
                    "1-0": 9.5,
                },
            },
        )
        model = ExactScoreModel(mode="adaptive")
        home_mu, away_mu, candidates, _layers = model._score_match(match)
        selected = next(candidate for candidate in candidates if candidate.score == "3-0")

        shaped = model._shape_arbitration_challenger(match, home_mu, away_mu, selected, candidates)

        self.assertIn(shaped.score, {"4-0", "5-0"})
        self.assertIn("shape arbitration: extreme_clean", " ".join(shaped.reasons))

    def test_shape_arbitration_can_promote_high_draw_when_draw_and_chaos_align(self):
        match = _base_match(
            match_id="shape-high-draw",
            league_avg_goals=3.15,
            league_draw_bias=0.12,
            league_volatility=0.88,
            home_attack=1.38,
            away_attack=1.34,
            home_defense=1.34,
            away_defense=1.36,
            recent_home_xg=1.92,
            recent_away_xg=1.86,
            recent_home_xga=1.72,
            recent_away_xga=1.78,
            home_table_pressure=0.82,
            away_table_pressure=0.78,
            home_survival_pressure=0.55,
            away_survival_pressure=0.52,
            is_final_round=True,
            endgame_chaos=0.86,
            market={
                "confidence": 0.62,
                "expected_home_goals": 1.72,
                "expected_away_goals": 1.68,
                "one_x_two_odds": {"home": 2.42, "draw": 3.45, "away": 2.72},
                "total_goals_line": 3.0,
                "draw_money_heat": 0.48,
                "over_money_heat": 0.68,
                "exact_score_odds": {
                    "1-1": 6.4,
                    "2-2": 10.0,
                    "2-1": 8.6,
                    "1-2": 9.4,
                    "3-2": 22.0,
                    "2-3": 24.0,
                    "3-3": 58.0,
                    "4-2": 60.0,
                },
            },
        )
        model = ExactScoreModel(mode="adaptive")
        home_mu, away_mu, candidates, _layers = model._score_match(match)
        selected = next(candidate for candidate in candidates if candidate.score == "2-2")

        shaped = model._shape_arbitration_challenger(match, home_mu, away_mu, selected, candidates)

        self.assertEqual(shaped.score, "3-3")
        self.assertIn("shape arbitration: high_draw", " ".join(shaped.reasons))

    def test_shape_arbitration_can_promote_small_ball_cold(self):
        match = _base_match(
            match_id="shape-small-cold",
            league_avg_goals=2.18,
            league_draw_bias=0.10,
            league_volatility=0.36,
            home_attack=0.92,
            away_attack=1.04,
            home_defense=1.12,
            away_defense=0.78,
            recent_home_xg=0.92,
            recent_away_xg=1.08,
            recent_home_xga=1.32,
            recent_away_xga=0.78,
            home_form=-0.30,
            away_form=0.22,
            home_absence_impact=0.34,
            away_motivation=0.66,
            away_table_pressure=0.76,
            away_survival_pressure=0.58,
            market={
                "confidence": 0.70,
                "expected_home_goals": 1.02,
                "expected_away_goals": 0.86,
                "one_x_two_odds": {"home": 1.74, "draw": 3.35, "away": 5.25},
                "asian_handicap": -1.0,
                "total_goals_line": 2.0,
                "home_money_heat": 0.82,
                "favorite_money_heat": 0.80,
                "away_money_heat": 0.32,
                "under_money_heat": 0.72,
                "bookmaker_trap_risk": 0.64,
                "exact_score_odds": {
                    "1-0": 5.8,
                    "2-0": 7.8,
                    "2-1": 8.0,
                    "1-1": 6.6,
                    "0-1": 16.0,
                    "0-2": 34.0,
                    "1-2": 22.0,
                    "0-0": 8.6,
                },
            },
        )
        model = ExactScoreModel(mode="adaptive")
        home_mu, away_mu, candidates, _layers = model._score_match(match)
        selected = next(candidate for candidate in candidates if candidate.score == "1-0")

        shaped = model._shape_arbitration_challenger(match, home_mu, away_mu, selected, candidates)

        self.assertIn(shaped.score, {"0-1", "0-2"})
        self.assertIn("shape arbitration: small_cold", " ".join(shaped.reasons))

    def test_result_script_key_distinguishes_user_score_scripts(self):
        match = _base_match(
            match_id="script-key-map",
            league_avg_goals=2.75,
            home_attack=1.20,
            away_attack=1.00,
            home_defense=0.95,
            away_defense=1.10,
            market={
                "confidence": 0.60,
                "expected_home_goals": 1.55,
                "expected_away_goals": 1.10,
                "one_x_two_odds": {"home": 1.85, "draw": 3.50, "away": 4.20},
                "total_goals_line": 2.5,
                "exact_score_odds": {},
            },
        )
        model = ExactScoreModel(mode="adaptive")
        home_mu, away_mu, candidates, _layers = model._score_match(match)
        by_score = {candidate.score: candidate for candidate in candidates}

        expected_scripts = {
            "1-0": "small_normal",
            "2-1": "big_normal",
            "3-2": "big_exchange_normal",
            "3-0": "big_cover_normal",
            "1-1": "small_draw",
            "2-2": "big_draw",
            "0-1": "small_cold",
            "2-3": "big_exchange_cold",
            "1-3": "big_cover_cold",
        }

        for score, script in expected_scripts.items():
            self.assertEqual(model._result_script_key(by_score[score], match, home_mu, away_mu), script)

    def test_result_script_arbitration_can_promote_small_normal(self):
        match = _base_match(
            match_id="script-small-normal",
            league_avg_goals=2.12,
            league_volatility=0.32,
            home_attack=1.15,
            away_attack=0.72,
            home_defense=0.72,
            away_defense=1.08,
            recent_home_xg=1.20,
            recent_away_xg=0.62,
            recent_home_xga=0.60,
            recent_away_xga=1.18,
            home_form=0.20,
            away_form=-0.18,
            weather_goal_drag=0.30,
            market={
                "confidence": 0.66,
                "expected_home_goals": 1.16,
                "expected_away_goals": 0.58,
                "one_x_two_odds": {"home": 1.70, "draw": 3.35, "away": 5.40},
                "total_goals_line": 2.0,
                "home_money_heat": 0.42,
                "under_money_heat": 0.76,
                "over_money_heat": 0.18,
                "exact_score_odds": {
                    "1-0": 5.8,
                    "2-0": 7.2,
                    "2-1": 9.2,
                    "3-1": 20.0,
                    "1-1": 7.4,
                },
            },
        )
        model = ExactScoreModel(mode="adaptive")
        home_mu, away_mu, candidates, _layers = model._score_match(match)
        selected = next(candidate for candidate in candidates if candidate.score == "2-1")

        scripted = model._result_script_challenger(match, home_mu, away_mu, selected, candidates)

        self.assertIn(scripted.score, {"1-0", "2-0"})
        self.assertIn("result script arbitration: small_normal", " ".join(scripted.reasons))

    def test_result_script_arbitration_can_promote_big_cold_script(self):
        match = _base_match(
            match_id="script-big-exchange-cold",
            league_avg_goals=3.08,
            league_volatility=0.84,
            home_attack=1.22,
            away_attack=1.42,
            home_defense=1.44,
            away_defense=1.08,
            recent_home_xg=1.55,
            recent_away_xg=1.82,
            recent_home_xga=1.88,
            recent_away_xga=1.10,
            home_form=-0.22,
            away_form=0.35,
            home_absence_impact=0.32,
            away_motivation=0.72,
            away_table_pressure=0.82,
            market={
                "confidence": 0.68,
                "expected_home_goals": 1.58,
                "expected_away_goals": 1.55,
                "one_x_two_odds": {"home": 1.80, "draw": 3.75, "away": 4.40},
                "asian_handicap": -0.75,
                "total_goals_line": 2.75,
                "home_money_heat": 0.78,
                "favorite_money_heat": 0.76,
                "away_money_heat": 0.34,
                "over_money_heat": 0.66,
                "bookmaker_trap_risk": 0.58,
                "exact_score_odds": {
                    "2-1": 7.2,
                    "3-1": 12.0,
                    "1-2": 14.0,
                    "2-3": 42.0,
                    "3-2": 20.0,
                    "1-3": 36.0,
                    "2-2": 12.0,
                },
            },
        )
        model = ExactScoreModel(mode="adaptive")
        home_mu, away_mu, candidates, _layers = model._score_match(match)
        selected = next(candidate for candidate in candidates if candidate.score == "2-1")

        scripted = model._result_script_challenger(match, home_mu, away_mu, selected, candidates)

        self.assertIn(scripted.score, {"1-3", "2-3", "1-2"})
        self.assertRegex(" ".join(scripted.reasons), r"result script arbitration: (big_exchange_cold|big_cover_cold|small_cold)")

    def test_result_script_arbitration_protects_big_cover_from_exchange(self):
        match = _base_match(
            match_id="script-cover-protection",
            league_avg_goals=3.25,
            league_volatility=0.88,
            league_home_advantage=0.16,
            home_attack=1.85,
            away_attack=0.95,
            home_defense=1.20,
            away_defense=2.45,
            recent_home_xg=2.85,
            recent_away_xg=1.05,
            recent_home_xga=1.20,
            recent_away_xga=2.90,
            home_form=0.55,
            away_form=-0.75,
            home_motivation=0.95,
            away_motivation=-0.20,
            home_table_pressure=1.0,
            home_survival_pressure=1.0,
            away_settled=True,
            away_collapse_risk=0.95,
            is_final_round=True,
            endgame_chaos=0.95,
            market={
                "confidence": 0.62,
                "expected_home_goals": 2.55,
                "expected_away_goals": 1.05,
                "one_x_two_odds": {"home": 2.7, "draw": 3.9, "away": 2.35},
                "asian_handicap": 0.25,
                "total_goals_line": 3.25,
                "home_money_heat": 0.32,
                "away_money_heat": 0.54,
                "favorite_money_heat": 0.58,
                "over_money_heat": 0.78,
                "bookmaker_trap_risk": 0.60,
                "exact_score_odds": {
                    "2-1": 9.0,
                    "3-1": 18.0,
                    "3-0": 26.0,
                    "4-1": 50.0,
                    "5-1": 120.0,
                    "6-1": 260.0,
                    "3-2": 25.0,
                    "4-2": 55.0,
                },
            },
        )
        model = ExactScoreModel(mode="adaptive")
        home_mu, away_mu, candidates, _layers = model._score_match(match)
        selected = next(candidate for candidate in candidates if candidate.score == "4-1")

        scripted = model._result_script_challenger(match, home_mu, away_mu, selected, candidates)

        self.assertEqual(scripted.score, "4-1")

    def test_forced_exit_gate_does_not_fade_true_hot_favorite(self):
        match = _base_match(
            match_id="true-hot-no-force",
            league_avg_goals=2.85,
            league_volatility=0.58,
            home_attack=1.62,
            away_attack=0.84,
            home_defense=0.72,
            away_defense=1.36,
            recent_home_xg=2.05,
            recent_away_xg=0.88,
            recent_home_xga=0.82,
            recent_away_xga=1.75,
            home_form=0.35,
            away_form=-0.25,
            home_motivation=0.32,
            away_motivation=0.04,
            market={
                "confidence": 0.76,
                "expected_home_goals": 2.05,
                "expected_away_goals": 0.82,
                "one_x_two_odds": {"home": 1.36, "draw": 5.0, "away": 8.5},
                "exact_score_odds": {
                    "1-0": 6.8,
                    "2-0": 6.4,
                    "2-1": 7.2,
                    "3-0": 8.5,
                    "3-1": 9.0,
                    "1-1": 11.0,
                    "2-2": 21.0,
                },
            },
        )

        pick = ExactScoreModel(mode="adaptive").predict(match)

        self.assertIn(pick.score, {"2-0", "2-1", "3-0", "3-1"})
        self.assertNotIn("forced exit gate", " ".join(pick.candidate.reasons))


if __name__ == "__main__":
    unittest.main()
