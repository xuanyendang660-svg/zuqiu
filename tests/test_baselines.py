import numpy as np

from football_v2.baselines import (
    dixon_coles_matrix,
    independent_poisson_matrix,
    multiplicative_devig,
)


def test_poisson_matrix_is_complete_distribution() -> None:
    matrix = independent_poisson_matrix(1.6, 1.1, max_goals=7)
    assert matrix.shape == (8, 8)
    assert np.isclose(matrix.sum(), 1.0)
    assert np.all(matrix >= 0)


def test_dixon_coles_changes_low_score_cells() -> None:
    poisson = independent_poisson_matrix(1.4, 1.0)
    corrected = dixon_coles_matrix(1.4, 1.0, rho=-0.08)
    assert np.isclose(corrected.sum(), 1.0)
    assert not np.isclose(poisson[1, 1], corrected[1, 1])


def test_devig_returns_unit_mass() -> None:
    probabilities = multiplicative_devig({"home": 1.80, "draw": 3.60, "away": 4.80})
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert probabilities["home"] > probabilities["draw"] > probabilities["away"]
