from __future__ import annotations

from football_v2.halftime_core_predictor import (
    CORE_RULES,
    predict_core_halftime_exact_score,
    predict_core_halftime_exact_score_from_odds,
)


def test_core_rules_are_only_the_two_high_sample_nil_nil_rules() -> None:
    assert [rule.rule_id for rule in CORE_RULES] == ["HTX-03", "HTX-04"]
    assert all(rule.halftime_underdog == 0 for rule in CORE_RULES)
    assert all(rule.halftime_favourite == 0 for rule in CORE_RULES)
    assert all(rule.oriented_prediction == "0-1" for rule in CORE_RULES)


def test_core_predictor_returns_favourite_1_0_from_probabilities() -> None:
    result = predict_core_halftime_exact_score(
        home_probability=0.70,
        draw_probability=0.18,
        away_probability=0.12,
        halftime_home_score=0,
        halftime_away_score=0,
    )

    assert result.accepted
    assert result.rule_id == "HTX-03"
    assert result.oriented_score == "0-1"
    assert (result.home_score, result.away_score) == (1, 0)


def test_core_predictor_abstains_after_a_halftime_goal() -> None:
    result = predict_core_halftime_exact_score(
        home_probability=0.70,
        draw_probability=0.18,
        away_probability=0.12,
        halftime_home_score=1,
        halftime_away_score=0,
    )

    assert not result.accepted
    assert result.reason == "core method abstained"


def test_core_odds_entrypoint() -> None:
    result = predict_core_halftime_exact_score_from_odds(
        home_odds=1.35,
        draw_odds=5.00,
        away_odds=10.00,
        halftime_home_score=0,
        halftime_away_score=0,
    )

    assert result.accepted
    assert (result.home_score, result.away_score) == (1, 0)
