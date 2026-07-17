import numpy as np

from football_model.score_matrix import build_score_matrix
from football_model.selection import GoalLicenses, mechanism_matrix, select_score


def test_low_second_goal_license_demotes_two_goal_scores() -> None:
    matrix = build_score_matrix(1.6, 0.7)
    neutral = mechanism_matrix(matrix, GoalLicenses())
    restricted = mechanism_matrix(
        matrix,
        GoalLicenses(home_first=1.05, home_second=0.55, home_third_plus=0.45),
    )
    assert restricted[1, 0] / restricted[2, 0] > neutral[1, 0] / neutral[2, 0]


def test_market_cluster_never_mechanically_changes_best_score() -> None:
    matrix = np.zeros((4, 4))
    matrix[2, 1] = 0.4
    matrix[1, 1] = 0.35
    matrix[3, 1] = 0.25
    result = select_score(matrix, market_cluster=((2, 1),))
    assert result.score == (2, 1)
