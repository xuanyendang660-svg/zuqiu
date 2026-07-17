from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass(frozen=True)
class ScenarioWeights:
    low_event: float = 0.10
    normal: float = 0.55
    home_control: float = 0.10
    away_control: float = 0.05
    shootout: float = 0.10
    home_blowout: float = 0.07
    away_blowout: float = 0.03

    def normalized(self) -> "ScenarioWeights":
        values = np.array(list(self.__dict__.values()), dtype=float)
        if np.any(values < 0):
            raise ValueError("scenario weights cannot be negative")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("at least one scenario weight must be positive")
        scaled = values / total
        return ScenarioWeights(**dict(zip(self.__dict__.keys(), scaled, strict=True)))


_SCENARIO_MODIFIERS: dict[str, tuple[float, float, float, float]] = {
    "low_event": (0.68, 0.68, 0.92, 0.92),
    "normal": (1.00, 1.00, 1.00, 1.00),
    "home_control": (1.15, 0.62, 1.04, 0.96),
    "away_control": (0.62, 1.15, 0.96, 1.04),
    "shootout": (1.35, 1.35, 1.08, 1.08),
    "home_blowout": (1.70, 0.78, 1.22, 0.96),
    "away_blowout": (0.78, 1.70, 0.96, 1.22),
}


def _poisson_outer(home_lambda: float, away_lambda: float, max_goals: int) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    home = poisson.pmf(goals, home_lambda)
    away = poisson.pmf(goals, away_lambda)
    matrix = np.outer(home, away)
    return matrix / matrix.sum()


def _apply_tail_tilt(matrix: np.ndarray, home_tilt: float, away_tilt: float) -> np.ndarray:
    max_goals = matrix.shape[0] - 1
    home_goals = np.arange(max_goals + 1)[:, None]
    away_goals = np.arange(max_goals + 1)[None, :]
    home_factor = np.where(home_goals >= 3, home_tilt ** (home_goals - 2), 1.0)
    away_factor = np.where(away_goals >= 3, away_tilt ** (away_goals - 2), 1.0)
    tilted = matrix * home_factor * away_factor
    return tilted / tilted.sum()


def build_score_matrix(
    home_lambda: float,
    away_lambda: float,
    *,
    scenario_weights: ScenarioWeights | None = None,
    max_goals: int = 7,
) -> np.ndarray:
    """Build a score matrix as a mixture of match-state distributions."""
    if home_lambda <= 0 or away_lambda <= 0:
        raise ValueError("expected-goal rates must be positive")
    if max_goals < 3:
        raise ValueError("max_goals must be at least 3")

    weights = (scenario_weights or ScenarioWeights()).normalized()
    result = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for scenario, weight in weights.__dict__.items():
        h_mod, a_mod, h_tail, a_tail = _SCENARIO_MODIFIERS[scenario]
        component = _poisson_outer(home_lambda * h_mod, away_lambda * a_mod, max_goals)
        component = _apply_tail_tilt(component, h_tail, a_tail)
        result += weight * component
    return result / result.sum()


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    home = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, k=1).sum())
    return {"home": home, "draw": draw, "away": away}


def totals_probability(matrix: np.ndarray, line: float = 2.5) -> float:
    indices = np.indices(matrix.shape)
    return float(matrix[(indices[0] + indices[1]) > line].sum())


def btts_probability(matrix: np.ndarray) -> float:
    return float(matrix[1:, 1:].sum())
