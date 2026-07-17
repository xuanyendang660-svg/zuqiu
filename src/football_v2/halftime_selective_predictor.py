from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class FrozenHalftimeRule:
    rule_id: str
    family: str
    halftime_underdog: int
    halftime_favourite: int
    prediction_underdog: int
    prediction_favourite: int
    probability_bin: str | None = None
    market_gap_bin: str | None = None
    home_underdog: bool | None = None
    transfer_matches: int = 0
    transfer_accuracy: float = 0.0
    transfer_baseline_accuracy: float = 0.0

    @property
    def oriented_prediction(self) -> str:
        return f"{self.prediction_underdog}-{self.prediction_favourite}"

    @property
    def transfer_improvement(self) -> float:
        return self.transfer_accuracy - self.transfer_baseline_accuracy


@dataclass(frozen=True)
class HalftimeExactPrediction:
    accepted: bool
    home_score: int | None
    away_score: int | None
    oriented_score: str | None
    rule_id: str | None
    rule_family: str | None
    underdog_side: str | None
    transfer_matches: int
    transfer_accuracy: float | None
    transfer_baseline_accuracy: float | None
    reason: str


FROZEN_RULES: tuple[FrozenHalftimeRule, ...] = (
    FrozenHalftimeRule(
        "HTX-01",
        "score_state_gap_home",
        1,
        0,
        1,
        0,
        market_gap_bin="g15",
        home_underdog=True,
        transfer_matches=52,
        transfer_accuracy=0.25,
        transfer_baseline_accuracy=0.19230769230769232,
    ),
    FrozenHalftimeRule(
        "HTX-02",
        "score_state_gap",
        0,
        2,
        0,
        2,
        market_gap_bin="g05",
        transfer_matches=54,
        transfer_accuracy=0.24074074074074073,
        transfer_baseline_accuracy=0.1111111111111111,
    ),
    FrozenHalftimeRule(
        "HTX-03",
        "score_state_prob",
        0,
        0,
        0,
        1,
        probability_bin="p20",
        transfer_matches=511,
        transfer_accuracy=0.2837573385518591,
        transfer_baseline_accuracy=0.18395303326810175,
    ),
    FrozenHalftimeRule(
        "HTX-04",
        "score_state_gap",
        0,
        0,
        0,
        1,
        market_gap_bin="g50",
        transfer_matches=583,
        transfer_accuracy=0.27101200686106347,
        transfer_baseline_accuracy=0.1955403087478559,
    ),
    FrozenHalftimeRule(
        "HTX-05",
        "score_state_prob",
        0,
        1,
        0,
        2,
        probability_bin="p20",
        transfer_matches=417,
        transfer_accuracy=0.2422062350119904,
        transfer_baseline_accuracy=0.2014388489208633,
    ),
    FrozenHalftimeRule(
        "HTX-06",
        "score_state_gap",
        0,
        1,
        0,
        2,
        market_gap_bin="g50",
        transfer_matches=497,
        transfer_accuracy=0.23742454728370221,
        transfer_baseline_accuracy=0.19919517102615694,
    ),
    FrozenHalftimeRule(
        "HTX-07",
        "score_state_gap",
        1,
        0,
        1,
        0,
        market_gap_bin="g30",
        transfer_matches=217,
        transfer_accuracy=0.2534562211981567,
        transfer_baseline_accuracy=0.22580645161290322,
    ),
    FrozenHalftimeRule(
        "HTX-08",
        "score_state_gap",
        2,
        0,
        2,
        0,
        market_gap_bin="g05",
        transfer_matches=61,
        transfer_accuracy=0.4098360655737705,
        transfer_baseline_accuracy=0.16393442622950818,
    ),
    FrozenHalftimeRule(
        "HTX-09",
        "score_state_prob_home",
        0,
        2,
        0,
        2,
        probability_bin="p35",
        home_underdog=False,
        transfer_matches=72,
        transfer_accuracy=0.25,
        transfer_baseline_accuracy=0.125,
    ),
)


def _probability_bin(value: float) -> str:
    if value <= 0.20:
        return "p20"
    if value <= 0.25:
        return "p25"
    if value <= 0.30:
        return "p30"
    if value <= 0.35:
        return "p35"
    if value <= 0.40:
        return "p40"
    return "p50"


def _market_gap_bin(value: float) -> str:
    if value <= 0.05:
        return "g05"
    if value <= 0.10:
        return "g10"
    if value <= 0.15:
        return "g15"
    if value <= 0.20:
        return "g20"
    if value <= 0.30:
        return "g30"
    return "g50"


def _normalise_probabilities(
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> tuple[float, float, float]:
    values = (home_probability, draw_probability, away_probability)
    if not all(isfinite(float(value)) and float(value) > 0.0 for value in values):
        raise ValueError("all three probabilities must be finite and positive")
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("probability total must be positive")
    return tuple(float(value) / total for value in values)  # type: ignore[return-value]


def _abstention(reason: str) -> HalftimeExactPrediction:
    return HalftimeExactPrediction(
        accepted=False,
        home_score=None,
        away_score=None,
        oriented_score=None,
        rule_id=None,
        rule_family=None,
        underdog_side=None,
        transfer_matches=0,
        transfer_accuracy=None,
        transfer_baseline_accuracy=None,
        reason=reason,
    )


def predict_halftime_exact_score(
    *,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    halftime_home_score: int,
    halftime_away_score: int,
) -> HalftimeExactPrediction:
    """Return one frozen exact score only when a validated state rule fires.

    Probabilities must be pre-match probabilities. The function normalises them,
    determines the underdog orientation, and refuses to predict on all uncovered
    states. Rules and priority are frozen before the final shadow evaluation.
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
    state_underdog = min(int(halftime_underdog), 3)
    state_favourite = min(int(halftime_favourite), 3)
    probability_bin = _probability_bin(underdog_probability)
    gap_bin = _market_gap_bin(favourite_probability - underdog_probability)

    for rule in FROZEN_RULES:
        if rule.halftime_underdog != state_underdog:
            continue
        if rule.halftime_favourite != state_favourite:
            continue
        if rule.probability_bin is not None and rule.probability_bin != probability_bin:
            continue
        if rule.market_gap_bin is not None and rule.market_gap_bin != gap_bin:
            continue
        if rule.home_underdog is not None and rule.home_underdog != home_underdog:
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
            reason="frozen validated state rule matched",
        )

    return _abstention("no frozen validated state rule matched")


def predict_halftime_exact_score_from_odds(
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
    return predict_halftime_exact_score(
        home_probability=inverse[0],
        draw_probability=inverse[1],
        away_probability=inverse[2],
        halftime_home_score=halftime_home_score,
        halftime_away_score=halftime_away_score,
    )
