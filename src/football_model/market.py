from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DevigResult:
    probabilities: dict[str, float]
    overround: float


def multiplicative_devig(decimal_odds: Mapping[str, float]) -> DevigResult:
    """Remove bookmaker margin by normalising inverse decimal odds."""
    if not decimal_odds:
        raise ValueError("decimal_odds cannot be empty")
    if any(price <= 1.0 for price in decimal_odds.values()):
        raise ValueError("decimal odds must be greater than 1.0")

    keys = list(decimal_odds)
    raw = np.array([1.0 / decimal_odds[key] for key in keys], dtype=float)
    overround = float(raw.sum() - 1.0)
    fair = raw / raw.sum()
    return DevigResult(dict(zip(keys, fair, strict=True)), overround)


def infer_correct_score_cluster(
    correct_score_odds: Mapping[tuple[int, int], float],
    *,
    cumulative_mass: float = 0.45,
) -> tuple[tuple[int, int], ...]:
    """Infer a market score cluster from correct-score prices.

    This identifies the low-price cluster; it does not label that cluster as wrong.
    """
    if not correct_score_odds:
        return ()
    raw = {score: 1.0 / price for score, price in correct_score_odds.items() if price > 1.0}
    total = sum(raw.values())
    if total <= 0:
        return ()
    ranked = sorted(raw.items(), key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, int]] = []
    running = 0.0
    for score, mass in ranked:
        selected.append(score)
        running += mass / total
        if running >= cumulative_mass:
            break
    return tuple(selected)
