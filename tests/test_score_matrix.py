import numpy as np

from football_model.score_matrix import ScenarioWeights, build_score_matrix


def test_matrix_is_probability_distribution() -> None:
    matrix = build_score_matrix(1.5, 1.1)
    assert matrix.shape == (8, 8)
    assert np.isclose(matrix.sum(), 1.0)
    assert np.all(matrix >= 0)


def test_home_blowout_weight_increases_home_tail() -> None:
    normal = build_score_matrix(
        1.45,
        0.9,
        scenario_weights=ScenarioWeights(
            low_event=0,
            normal=1,
            home_control=0,
            away_control=0,
            shootout=0,
            home_blowout=0,
            away_blowout=0,
        ),
    )
    blowout = build_score_matrix(
        1.45,
        0.9,
        scenario_weights=ScenarioWeights(
            low_event=0,
            normal=0.4,
            home_control=0,
            away_control=0,
            shootout=0,
            home_blowout=0.6,
            away_blowout=0,
        ),
    )
    assert blowout[3:, :].sum() > normal[3:, :].sum()
