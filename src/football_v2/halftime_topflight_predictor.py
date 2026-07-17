from __future__ import annotations

from .halftime_core_predictor import (
    predict_core_halftime_exact_score,
    predict_core_halftime_exact_score_from_odds,
)
from .halftime_selective_predictor import HalftimeExactPrediction, _abstention


def predict_topflight_halftime_exact_score(
    *,
    is_top_flight_domestic_league: bool,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    halftime_home_score: int,
    halftime_away_score: int,
) -> HalftimeExactPrediction:
    if not is_top_flight_domestic_league:
        return _abstention("method is validated only for top-flight domestic leagues")
    return predict_core_halftime_exact_score(
        home_probability=home_probability,
        draw_probability=draw_probability,
        away_probability=away_probability,
        halftime_home_score=halftime_home_score,
        halftime_away_score=halftime_away_score,
    )


def predict_topflight_halftime_exact_score_from_odds(
    *,
    is_top_flight_domestic_league: bool,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    halftime_home_score: int,
    halftime_away_score: int,
) -> HalftimeExactPrediction:
    if not is_top_flight_domestic_league:
        return _abstention("method is validated only for top-flight domestic leagues")
    return predict_core_halftime_exact_score_from_odds(
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        halftime_home_score=halftime_home_score,
        halftime_away_score=halftime_away_score,
    )
