from __future__ import annotations

import pandas as pd

from football_v2.halftime_final_audit import paired_comparison
from football_v2.halftime_selective_predictor import (
    predict_halftime_exact_score,
    predict_halftime_exact_score_from_odds,
)


def test_home_underdog_rule_returns_home_1_0() -> None:
    prediction = predict_halftime_exact_score(
        home_probability=0.30,
        draw_probability=0.27,
        away_probability=0.43,
        halftime_home_score=1,
        halftime_away_score=0,
    )

    assert prediction.accepted
    assert prediction.rule_id == "HTX-01"
    assert prediction.underdog_side == "home"
    assert (prediction.home_score, prediction.away_score) == (1, 0)
    assert prediction.oriented_score == "1-0"


def test_away_underdog_nil_nil_strong_favourite_returns_home_1_0() -> None:
    prediction = predict_halftime_exact_score(
        home_probability=0.72,
        draw_probability=0.18,
        away_probability=0.10,
        halftime_home_score=0,
        halftime_away_score=0,
    )

    assert prediction.accepted
    assert prediction.rule_id == "HTX-03"
    assert prediction.underdog_side == "away"
    assert prediction.oriented_score == "0-1"
    assert (prediction.home_score, prediction.away_score) == (1, 0)


def test_uncovered_state_abstains() -> None:
    prediction = predict_halftime_exact_score(
        home_probability=0.42,
        draw_probability=0.28,
        away_probability=0.30,
        halftime_home_score=1,
        halftime_away_score=1,
    )

    assert not prediction.accepted
    assert prediction.home_score is None
    assert prediction.rule_id is None


def test_odds_entrypoint_devigs_and_matches_rule() -> None:
    prediction = predict_halftime_exact_score_from_odds(
        home_odds=1.40,
        draw_odds=4.50,
        away_odds=9.00,
        halftime_home_score=0,
        halftime_away_score=0,
    )

    assert prediction.accepted
    assert prediction.oriented_score == "0-1"
    assert (prediction.home_score, prediction.away_score) == (1, 0)


def test_paired_comparison_detects_model_advantage() -> None:
    actual = pd.Series(["1-0", "0-1", "2-0", "1-1"] * 50)
    model = pd.Series(["1-0", "0-1", "2-0", "0-0"] * 50)
    baseline = pd.Series(["1-0", "1-0", "0-0", "0-0"] * 50)

    result = paired_comparison(
        actual,
        model,
        baseline,
        bootstrap_samples=5_000,
        random_state=7,
    )

    assert result.model_accuracy == 0.75
    assert result.baseline_accuracy == 0.25
    assert result.improvement == 0.50
    assert result.model_only_correct == 100
    assert result.baseline_only_correct == 0
    assert result.improvement_bootstrap_95_low > 0
    assert result.mcnemar_exact_p_value < 0.001
