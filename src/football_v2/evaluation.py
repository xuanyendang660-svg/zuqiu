from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from .labels import ScoreArchetype, classify_score
from .model import TailAwareExactScoreModel


@dataclass(frozen=True)
class EvaluationReport:
    matches: int
    exact_accuracy: float
    top_k_accuracy: float
    direction_accuracy: float
    archetype_accuracy: float
    negative_log_likelihood: float
    extreme_tail_recall: float | None


def _direction(score: tuple[int, int]) -> int:
    return (score[0] > score[1]) - (score[0] < score[1])


def _distribution_archetype(matrix: np.ndarray) -> ScoreArchetype:
    masses = {archetype: 0.0 for archetype in ScoreArchetype}
    for home in range(matrix.shape[0]):
        for away in range(matrix.shape[1]):
            masses[classify_score(home, away)] += float(matrix[home, away])
    return max(masses, key=masses.get)


def evaluate_predictions(
    model: TailAwareExactScoreModel,
    features: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    *,
    top_k: int = 5,
) -> EvaluationReport:
    home = np.asarray(home_goals, dtype=int)
    away = np.asarray(away_goals, dtype=int)
    if home.shape != away.shape or home.ndim != 1:
        raise ValueError("goal arrays must be matching one-dimensional arrays")

    distributions = model.predict_distribution(features)
    if len(distributions) != len(home):
        raise ValueError("features and goal arrays must have matching rows")

    exact_hits = 0
    top_k_hits = 0
    direction_hits = 0
    archetype_hits = 0
    nll = 0.0
    extreme_actual = 0
    extreme_detected = 0
    extreme_types = {
        ScoreArchetype.HOME_BLOWOUT,
        ScoreArchetype.AWAY_BLOWOUT,
        ScoreArchetype.SHOOTOUT,
    }

    for matrix, actual_home, actual_away in zip(distributions, home, away, strict=True):
        actual = (int(actual_home), int(actual_away))
        flat_order = np.argsort(matrix.ravel())[::-1]
        predicted = tuple(
            int(value) for value in np.unravel_index(int(flat_order[0]), matrix.shape)
        )
        top_scores = {
            tuple(int(value) for value in np.unravel_index(int(index), matrix.shape))
            for index in flat_order[:top_k]
        }
        exact_hits += predicted == actual
        top_k_hits += actual in top_scores
        direction_hits += _direction(predicted) == _direction(actual)

        actual_archetype = classify_score(*actual)
        predicted_archetype = _distribution_archetype(matrix)
        archetype_hits += predicted_archetype is actual_archetype
        if actual_archetype in extreme_types:
            extreme_actual += 1
            extreme_detected += predicted_archetype is actual_archetype

        probability = (
            float(matrix[actual])
            if actual[0] < matrix.shape[0] and actual[1] < matrix.shape[1]
            else 0.0
        )
        nll -= log(max(probability, 1e-15))

    count = len(home)
    if count == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return EvaluationReport(
        matches=count,
        exact_accuracy=exact_hits / count,
        top_k_accuracy=top_k_hits / count,
        direction_accuracy=direction_hits / count,
        archetype_accuracy=archetype_hits / count,
        negative_log_likelihood=nll / count,
        extreme_tail_recall=(extreme_detected / extreme_actual) if extreme_actual else None,
    )
