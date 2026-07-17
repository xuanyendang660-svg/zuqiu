from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GoalLicenses:
    home_first: float = 1.0
    home_second: float = 1.0
    home_third_plus: float = 1.0
    away_first: float = 1.0
    away_second: float = 1.0
    away_third_plus: float = 1.0
    draw_stop: float = 1.0
    home_lead_stop: float = 1.0
    away_lead_stop: float = 1.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SelectionResult:
    score: tuple[int, int]
    probability: float
    adjusted_score: float
    ranked_scores: tuple[tuple[tuple[int, int], float], ...]
    comfort_cluster_warning: bool


def _goal_multiplier(goals: int, first: float, second: float, third_plus: float) -> float:
    result = 1.0
    if goals >= 1:
        result *= first
    if goals >= 2:
        result *= second
    if goals >= 3:
        result *= third_plus ** (goals - 2)
    return result


def mechanism_matrix(matrix: np.ndarray, licenses: GoalLicenses) -> np.ndarray:
    licenses.validate()
    adjusted = np.zeros_like(matrix, dtype=float)
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            multiplier = _goal_multiplier(
                home_goals,
                licenses.home_first,
                licenses.home_second,
                licenses.home_third_plus,
            )
            multiplier *= _goal_multiplier(
                away_goals,
                licenses.away_first,
                licenses.away_second,
                licenses.away_third_plus,
            )
            if home_goals == away_goals:
                multiplier *= licenses.draw_stop
            elif home_goals > away_goals:
                multiplier *= licenses.home_lead_stop
            else:
                multiplier *= licenses.away_lead_stop
            adjusted[home_goals, away_goals] = matrix[home_goals, away_goals] * multiplier
    return adjusted / adjusted.sum()


def select_score(
    matrix: np.ndarray,
    *,
    licenses: GoalLicenses | None = None,
    market_cluster: tuple[tuple[int, int], ...] = (),
    top_n: int = 12,
) -> SelectionResult:
    """Select a score without adding an anti-market or diversity bonus."""
    adjusted = mechanism_matrix(matrix, licenses or GoalLicenses())
    flat_order = np.argsort(adjusted.ravel())[::-1]
    ranked: list[tuple[tuple[int, int], float]] = []
    for index in flat_order[:top_n]:
        score = tuple(int(v) for v in np.unravel_index(index, adjusted.shape))
        ranked.append((score, float(adjusted[score])))
    best_score, best_adjusted = ranked[0]

    common_scores = {(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)}
    nearest = ranked[1][1] if len(ranked) > 1 else 0.0
    weak_separation = best_adjusted < nearest * 1.12 if nearest > 0 else False
    warning = best_score in common_scores and best_score in set(market_cluster) and weak_separation

    return SelectionResult(
        score=best_score,
        probability=float(matrix[best_score]),
        adjusted_score=best_adjusted,
        ranked_scores=tuple(ranked),
        comfort_cluster_warning=warning,
    )
