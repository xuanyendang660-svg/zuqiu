from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np


@dataclass(frozen=True)
class MatchEvaluation:
    exact_hit: bool
    actual_score_rank: int
    negative_log_likelihood: float
    direction_hit: bool
    total_bucket_hit: bool
    btts_hit: bool


def _direction(score: tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "home"
    if score[0] < score[1]:
        return "away"
    return "draw"


def _total_bucket(total: int) -> str:
    if total <= 2:
        return "low"
    if total == 3:
        return "normal"
    if total <= 5:
        return "open"
    return "extreme"


def evaluate_match(
    matrix: np.ndarray,
    predicted_score: tuple[int, int],
    actual_score: tuple[int, int],
) -> MatchEvaluation:
    if actual_score[0] >= matrix.shape[0] or actual_score[1] >= matrix.shape[1]:
        actual_probability = 1e-15
        rank = matrix.size + 1
    else:
        actual_probability = float(matrix[actual_score])
        order = np.argsort(matrix.ravel())[::-1]
        actual_index = np.ravel_multi_index(actual_score, matrix.shape)
        rank = int(np.where(order == actual_index)[0][0]) + 1

    return MatchEvaluation(
        exact_hit=predicted_score == actual_score,
        actual_score_rank=rank,
        negative_log_likelihood=-log(max(actual_probability, 1e-15)),
        direction_hit=_direction(predicted_score) == _direction(actual_score),
        total_bucket_hit=_total_bucket(sum(predicted_score)) == _total_bucket(sum(actual_score)),
        btts_hit=(predicted_score[0] > 0 and predicted_score[1] > 0)
        == (actual_score[0] > 0 and actual_score[1] > 0),
    )
