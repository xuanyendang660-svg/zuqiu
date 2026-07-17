from __future__ import annotations

from math import isfinite

from .halftime_selective_predictor import (
    FROZEN_RULES,
    FrozenHalftimeRule,
    HalftimeExactPrediction,
    _abstention,
    _market_gap_bin,
    _normalise_probabilities,
    _probability_bin,
)


CORE_RULE_IDS = ("HTX-03", "HTX-04")
CORE_RULES: tuple[FrozenHalftimeRule, ...] = tuple(
    rule for rule in FROZEN_RULES if rule.rule_id in CORE_RULE_IDS
)


def predict_core_halftime_exact_score(
    *,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    halftime_home_score: int,
    halftime_away_score: int,
) -> HalftimeExactPrediction:
    """Predict only the frozen high-sample 0-0 strong-favourite state.

    Core selection uses only the pre-shadow second-division transfer set:
    at least 500 matches, at least 27% exact accuracy, at least +7 percentage
    points over the score-state baseline, and positive improvement in all five
    transfer leagues. No shadow result is used to choose these rules.
    """

    home_probability, draw_probability, away_probability = _normalise_probabilities(
        home_probability,
        draw_probability,
        away_probability,
    )
    if halftime_home_score < 0 or halftime_away_score < 0:
        raise ValueError("halftime scores cannot be negative")
    if abs(home_probability - away_probability) <= 1e-12:
        return _abstention("home and away win probabilities are tied")

    home_underdog = home_probability < away_probability
    underdog_probability = home_probability if home_underdog else away_probability
    favourite_probability = away_probability if home_underdog else home_probability
    halftime_underdog = halftime_home_score if home_underdog else halftime_away_score
    halftime_favourite = halftime_away_score if home_underdog else halftime_home_score
    probability_bin = _probability_bin(underdog_probability)
    gap_bin = _market_gap_bin(favourite_probability - underdog_probability)

    for rule in CORE_RULES:
        if rule.halftime_underdog != min(int(halftime_underdog), 3):
            continue
        if rule.halftime_favourite != min(int(halftime_favourite), 3):
            continue
        if rule.probability_bin is not None and rule.probability_bin != probability_bin:
            continue
        if rule.market_gap_bin is not None and rule.market_gap_bin != gap_bin:
            continue

        if home_underdog:
            home_score = rule.prediction_underdog
            away_score = rule.prediction_favourite
            side = "home"
        else:
            home_score = rule.prediction_favourite
            away_score = rule.prediction_underdog
            side = "away"
        return HalftimeExactPrediction(
            accepted=True,
            home_score=home_score,
            away_score=away_score,
            oriented_score=rule.oriented_prediction,
            rule_id=rule.rule_id,
            rule_family=rule.family,
            underdog_side=side,
            transfer_matches=rule.transfer_matches,
            transfer_accuracy=rule.transfer_accuracy,
            transfer_baseline_accuracy=rule.transfer_baseline_accuracy,
            reason="frozen high-sample core rule matched",
        )

    return _abstention("core method abstained")


def predict_core_halftime_exact_score_from_odds(
    *,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    halftime_home_score: int,
    halftime_away_score: int,
) -> HalftimeExactPrediction:
    odds = (home_odds, draw_odds, away_odds)
    if not all(isfinite(float(value)) and float(value) > 1.0 for value in odds):
        raise ValueError("decimal odds must be finite and greater than 1")
    inverse = tuple(1.0 / float(value) for value in odds)
    return predict_core_halftime_exact_score(
        home_probability=inverse[0],
        draw_probability=inverse[1],
        away_probability=inverse[2],
        halftime_home_score=halftime_home_score,
        halftime_away_score=halftime_away_score,
    )
