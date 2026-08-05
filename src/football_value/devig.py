from __future__ import annotations

from collections.abc import Sequence


def _validate_odds(odds: Sequence[float]) -> None:
    if len(odds) < 2:
        raise ValueError("at least two odds are required")
    if any(price <= 1.0 for price in odds):
        raise ValueError("decimal odds must be greater than 1.0")


def proportional_devig(odds: Sequence[float]) -> tuple[float, ...]:
    """Remove the overround by normalizing implied probabilities."""
    _validate_odds(odds)
    implied = [1.0 / price for price in odds]
    total = sum(implied)
    return tuple(value / total for value in implied)


def power_devig(odds: Sequence[float], tolerance: float = 1e-12) -> tuple[float, ...]:
    """Power-method de-vig: find k such that sum(implied**k) == 1."""
    _validate_odds(odds)
    implied = [1.0 / price for price in odds]

    low, high = 0.01, 20.0
    for _ in range(200):
        midpoint = (low + high) / 2.0
        total = sum(value**midpoint for value in implied)
        if abs(total - 1.0) <= tolerance:
            break
        if total > 1.0:
            low = midpoint
        else:
            high = midpoint

    probabilities = [value**midpoint for value in implied]
    normalizer = sum(probabilities)
    return tuple(value / normalizer for value in probabilities)


def devig_probabilities(
    odds: Sequence[float], method: str = "proportional"
) -> tuple[float, ...]:
    normalized_method = method.strip().lower()
    if normalized_method == "proportional":
        return proportional_devig(odds)
    if normalized_method == "power":
        return power_devig(odds)
    raise ValueError(f"unsupported de-vig method: {method}")
