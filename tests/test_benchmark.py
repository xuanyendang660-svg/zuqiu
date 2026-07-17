from types import SimpleNamespace

import numpy as np

from football_v2.baselines import dixon_coles_matrix
from football_v2.benchmark import evaluate_matrices, market_conditioned_matrix


def test_market_conditioning_is_normalized_and_moves_home_mass() -> None:
    base = dixon_coles_matrix(1.25, 1.10, max_goals=7)
    home_goals, away_goals = np.indices(base.shape)
    base_home = float(base[home_goals > away_goals].sum())
    row = SimpleNamespace(
        market_home_prob=0.62,
        market_draw_prob=0.23,
        market_away_prob=0.15,
        market_over25_prob=0.56,
    )
    conditioned = market_conditioned_matrix(base, row)
    conditioned_home = float(conditioned[home_goals > away_goals].sum())
    assert np.isclose(conditioned.sum(), 1.0)
    assert conditioned_home > base_home


def test_distribution_metrics_rank_actual_score() -> None:
    matrix = np.full((8, 8), 1e-6)
    matrix[1, 5] = 0.70
    matrix[1, 4] = 0.20
    matrix[0, 4] = 0.10
    matrix /= matrix.sum()
    report = evaluate_matrices(
        np.stack([matrix]),
        np.array([1]),
        np.array([5]),
        top_k=3,
    )
    assert report.exact_accuracy == 1.0
    assert report.top_k_accuracy == 1.0
    assert report.extreme_tail_recall == 1.0
    assert report.mean_actual_score_rank == 1.0
