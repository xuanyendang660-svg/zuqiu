from __future__ import annotations

from math import exp, lgamma, log
from typing import Mapping

import numpy as np


def _poisson_pmf(k: int, rate: float) -> float:
    if rate <= 0:
        raise ValueError("Poisson rate must be positive")
    return exp(k * log(rate) - rate - lgamma(k + 1))


def independent_poisson_matrix(
    home_rate: float,
    away_rate: float,
    *,
    max_goals: int = 7,
) -> np.ndarray:
    """Return a complete 0..max_goals independent Poisson score grid."""
    if max_goals < 1:
        raise ValueError("max_goals must be positive")
    home = np.array([_poisson_pmf(k, home_rate) for k in range(max_goals + 1)])
    away = np.array([_poisson_pmf(k, away_rate) for k in range(max_goals + 1)])
    matrix = np.outer(home, away)
    return matrix / matrix.sum()


def dixon_coles_matrix(
    home_rate: float,
    away_rate: float,
    *,
    rho: float = -0.08,
    max_goals: int = 7,
) -> np.ndarray:
    """Apply the Dixon-Coles low-score correction to a Poisson grid."""
    matrix = independent_poisson_matrix(home_rate, away_rate, max_goals=max_goals)
    tau = {
        (0, 0): 1 - home_rate * away_rate * rho,
        (0, 1): 1 + home_rate * rho,
        (1, 0): 1 + away_rate * rho,
        (1, 1): 1 - rho,
    }
    for score, multiplier in tau.items():
        if multiplier <= 0:
            raise ValueError("rho produces a non-positive Dixon-Coles correction")
        matrix[score] *= multiplier
    return matrix / matrix.sum()


def multiplicative_devig(decimal_odds: Mapping[str, float]) -> dict[str, float]:
    """Remove bookmaker overround by normalising inverse decimal odds."""
    if not decimal_odds:
        raise ValueError("odds cannot be empty")
    if any(price <= 1 for price in decimal_odds.values()):
        raise ValueError("decimal odds must be greater than 1")
    raw = {key: 1 / price for key, price in decimal_odds.items()}
    total = sum(raw.values())
    return {key: probability / total for key, probability in raw.items()}


def matrix_top_scores(matrix: np.ndarray, top_k: int = 10) -> list[tuple[tuple[int, int], float]]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    order = np.argsort(matrix.ravel())[::-1][:top_k]
    return [
        (tuple(int(value) for value in np.unravel_index(index, matrix.shape)), float(matrix.ravel()[index]))
        for index in order
    ]
