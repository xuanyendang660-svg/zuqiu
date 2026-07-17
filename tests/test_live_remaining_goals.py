import numpy as np

from football_v2.live_remaining_goals import (
    _score_matrices,
    infer_market_goal_rates,
)


def test_market_rate_solver_respects_home_favorite_and_over_market() -> None:
    home_rate, away_rate = infer_market_goal_rates(0.62, 0.22, 0.16, 0.58)
    assert home_rate > away_rate
    assert home_rate + away_rate > 2.3


def test_conditional_score_matrix_never_goes_below_current_score() -> None:
    matrices = _score_matrices(
        np.array([1]),
        np.array([3]),
        np.array([0.8]),
        np.array([1.1]),
        family="negative_binomial",
        home_dispersion=0.25,
        away_dispersion=0.35,
        maximum_final_goals=8,
    )
    matrix = matrices[0]
    assert np.isclose(matrix.sum(), 1.0)
    assert np.all(matrix[:1, :] == 0)
    assert np.all(matrix[:, :3] == 0)
    assert np.isclose(matrix[1:, 3:].sum(), 1.0)
