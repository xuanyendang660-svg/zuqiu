from __future__ import annotations

from football_v2.halftime_topflight_predictor import (
    predict_topflight_halftime_exact_score,
)


def test_topflight_domain_accepts_valid_core_state() -> None:
    result = predict_topflight_halftime_exact_score(
        is_top_flight_domestic_league=True,
        home_probability=0.70,
        draw_probability=0.18,
        away_probability=0.12,
        halftime_home_score=0,
        halftime_away_score=0,
    )

    assert result.accepted
    assert (result.home_score, result.away_score) == (1, 0)


def test_non_topflight_domain_is_forced_to_abstain() -> None:
    result = predict_topflight_halftime_exact_score(
        is_top_flight_domestic_league=False,
        home_probability=0.70,
        draw_probability=0.18,
        away_probability=0.12,
        halftime_home_score=0,
        halftime_away_score=0,
    )

    assert not result.accepted
    assert result.reason == "method is validated only for top-flight domestic leagues"
