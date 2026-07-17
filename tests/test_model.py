import numpy as np

from football_v2.evaluation import evaluate_predictions
from football_v2.model import ModelConfig, TailAwareExactScoreModel


def _synthetic_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    regimes = [
        (np.array([0, 0, 0, 0, 0, 0], dtype=float), (0, 0)),
        (np.array([1, 0, 0, 0, 0, 0], dtype=float), (2, 1)),
        (np.array([0, 1, 0, 0, 0, 0], dtype=float), (6, 1)),
        (np.array([0, 0, 1, 0, 0, 0], dtype=float), (1, 5)),
        (np.array([0, 0, 0, 1, 0, 0], dtype=float), (3, 3)),
        (np.array([0, 0, 0, 0, 1, 0], dtype=float), (3, 0)),
        (np.array([0, 0, 0, 0, 0, 1], dtype=float), (0, 3)),
    ]
    features: list[np.ndarray] = []
    home_goals: list[int] = []
    away_goals: list[int] = []
    for center, score in regimes:
        for _ in range(70):
            features.append(center + rng.normal(0, 0.025, size=center.shape))
            home_goals.append(score[0])
            away_goals.append(score[1])
    return np.vstack(features), np.array(home_goals), np.array(away_goals)


def test_model_can_select_one_five_as_final_score() -> None:
    x, home, away = _synthetic_training_data()
    model = TailAwareExactScoreModel(
        ModelConfig(n_estimators=140, random_state=11, max_goals=7)
    ).fit(x, home, away)

    away_blowout_features = np.array([[0, 0, 1, 0, 0, 0]], dtype=float)
    distribution = model.predict_distribution(away_blowout_features)[0]

    assert distribution.shape == (8, 8)
    assert np.isclose(distribution.sum(), 1.0)
    assert model.predict_score(away_blowout_features) == [(1, 5)]
    assert distribution[1, 5] == distribution.max()


def test_model_reports_tail_recall_on_separable_holdout() -> None:
    x, home, away = _synthetic_training_data()
    model = TailAwareExactScoreModel(
        ModelConfig(n_estimators=120, random_state=13, max_goals=7)
    ).fit(x, home, away)
    holdout = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    report = evaluate_predictions(
        model,
        holdout,
        np.array([6, 1, 3, 2]),
        np.array([1, 5, 3, 1]),
        top_k=3,
    )
    assert report.exact_accuracy == 1.0
    assert report.extreme_tail_recall == 1.0
