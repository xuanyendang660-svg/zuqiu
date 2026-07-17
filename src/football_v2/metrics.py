from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from .labels import ScoreArchetype, classify_score


@dataclass(frozen=True)
class DistributionMetrics:
    matches: int
    exact_accuracy: float
    top_k_accuracy: float
    direction_accuracy: float
    archetype_accuracy: float
    negative_log_likelihood: float
    extreme_tail_recall: float | None
    mean_actual_score_rank: float


def _direction(score: tuple[int, int]) -> int:
    return (score[0] > score[1]) - (score[0] < score[1])


def dominant_archetype(matrix: np.ndarray) -> ScoreArchetype:
    masses = {archetype: 0.0 for archetype in ScoreArchetype}
    for home in range(matrix.shape[0]):
        for away in range(matrix.shape[1]):
            masses[classify_score(home, away)] += float(matrix[home, away])
    return max(masses, key=masses.get)


def evaluate_matrices(
    matrices: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    *,
    top_k: int = 5,
) -> DistributionMetrics:
    if len(matrices) == 0:
        raise ValueError("cannot evaluate empty predictions")
    home = np.asarray(home_goals, dtype=int)
    away = np.asarray(away_goals, dtype=int)
    if home.shape != away.shape or len(home) != len(matrices):
        raise ValueError("prediction and target lengths differ")

    exact_hits = 0
    top_k_hits = 0
    direction_hits = 0
    archetype_hits = 0
    negative_log_likelihood = 0.0
    ranks: list[int] = []
    extreme_total = 0
    extreme_hits = 0
    extreme_types = {
        ScoreArchetype.HOME_BLOWOUT,
        ScoreArchetype.AWAY_BLOWOUT,
        ScoreArchetype.SHOOTOUT,
    }

    for matrix, actual_home, actual_away in zip(matrices, home, away, strict=True):
        actual = (int(actual_home), int(actual_away))
        order = np.argsort(matrix.ravel())[::-1]
        predicted = tuple(int(v) for v in np.unravel_index(int(order[0]), matrix.shape))
        top_scores = {
            tuple(int(v) for v in np.unravel_index(int(index), matrix.shape))
            for index in order[:top_k]
        }
        actual_index = np.ravel_multi_index(actual, matrix.shape)
        rank = int(np.where(order == actual_index)[0][0]) + 1
        ranks.append(rank)
        exact_hits += predicted == actual
        top_k_hits += actual in top_scores
        direction_hits += _direction(predicted) == _direction(actual)

        actual_archetype = classify_score(*actual)
        predicted_archetype = dominant_archetype(matrix)
        archetype_hits += predicted_archetype is actual_archetype
        if actual_archetype in extreme_types:
            extreme_total += 1
            extreme_hits += predicted_archetype is actual_archetype

        negative_log_likelihood -= log(max(float(matrix[actual]), 1e-15))

    count = len(matrices)
    return DistributionMetrics(
        matches=count,
        exact_accuracy=exact_hits / count,
        top_k_accuracy=top_k_hits / count,
        direction_accuracy=direction_hits / count,
        archetype_accuracy=archetype_hits / count,
        negative_log_likelihood=negative_log_likelihood / count,
        extreme_tail_recall=(extreme_hits / extreme_total) if extreme_total else None,
        mean_actual_score_rank=float(np.mean(ranks)),
    )
