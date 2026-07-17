import numpy as np

from football_v2.residual import MarketResidualScoreModel, ResidualConfig


def _market_base(rows: int) -> np.ndarray:
    matrix = np.full((8, 8), 0.002, dtype=float)
    matrix[2, 1] = 0.22
    matrix[1, 1] = 0.18
    matrix[1, 0] = 0.14
    matrix[2, 0] = 0.12
    matrix[1, 2] = 0.10
    matrix[3, 1] = 0.07
    matrix[1, 5] = 0.025
    matrix[6, 1] = 0.025
    matrix[3, 3] = 0.025
    matrix /= matrix.sum()
    return np.repeat(matrix[None, :, :], rows, axis=0)


def _cyclic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    scores = [
        (2, 1),
        (6, 1),
        (1, 5),
        (3, 3),
        (0, 0),
        (3, 0),
        (0, 3),
        (1, 1),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 2),
    ]
    centers = np.eye(len(scores), dtype=float)
    features: list[np.ndarray] = []
    home: list[int] = []
    away: list[int] = []
    for index in range(720):
        regime = index % len(scores)
        features.append(centers[regime] + rng.normal(0, 0.02, size=len(scores)))
        home.append(scores[regime][0])
        away.append(scores[regime][1])
    return np.vstack(features), np.array(home), np.array(away)


def _config(seed: int, trees: int) -> ResidualConfig:
    return ResidualConfig(
        n_estimators=trees,
        min_samples_leaf=1,
        random_state=seed,
        calibration_fraction=0.20,
        maximum_override_rate=0.25,
        minimum_lift=1.10,
    )


def test_residual_can_override_market_two_one_with_one_five() -> None:
    features, home, away = _cyclic_data()
    base = _market_base(len(features))
    model = MarketResidualScoreModel(_config(23, 120)).fit(features, home, away, base)

    query = np.zeros((1, 12), dtype=float)
    query[0, 2] = 1.0
    distribution = model.predict_distribution(query, _market_base(1))[0]
    predicted = tuple(
        int(value) for value in np.unravel_index(np.argmax(distribution), distribution.shape)
    )

    assert predicted == (1, 5)
    assert distribution[1, 5] > distribution[2, 1]
    assert model.selection_.alpha > 0 or model.selection_.beta > 0
    assert model.selection_.calibration_alert_metrics.lift is not None
    assert model.selection_.calibration_alert_metrics.lift > 1.0


def test_residual_calibration_keeps_market_guardrails() -> None:
    features, home, away = _cyclic_data()
    base = _market_base(len(features))
    model = MarketResidualScoreModel(_config(29, 100)).fit(features, home, away, base)
    selected = model.selection_.calibration_metrics
    market = model.selection_.calibration_market_metrics

    assert selected.exact_accuracy >= market.exact_accuracy - 0.001
    assert selected.top_k_accuracy >= market.top_k_accuracy - 0.003
    assert selected.negative_log_likelihood <= market.negative_log_likelihood + 0.005
