"""Pricing helpers for decimal-odds markets."""

from __future__ import annotations


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be above 1.0")
    return 1.0 / decimal_odds


def expected_value(probability: float, decimal_odds: float) -> float:
    _validate_probability(probability)
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be above 1.0")
    return probability * decimal_odds - 1.0


def fair_odds(probability: float) -> float:
    _validate_probability(probability)
    return 1.0 / probability


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Return full Kelly fraction, floored at zero."""
    _validate_probability(probability)
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be above 1.0")
    net_odds = decimal_odds - 1.0
    loss_probability = 1.0 - probability
    return max(0.0, (net_odds * probability - loss_probability) / net_odds)


def de_vig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    raw_a = implied_probability(odds_a)
    raw_b = implied_probability(odds_b)
    total = raw_a + raw_b
    if total <= 0.0:
        raise ValueError("invalid two-way market")
    return raw_a / total, raw_b / total


def _validate_probability(probability: float) -> None:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")
