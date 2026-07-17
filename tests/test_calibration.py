import numpy as np

from football_model.calibration import MarketTargets, calibrate_distribution, distribution_summary
from football_model.score_matrix import build_score_matrix


def test_calibration_matches_market_moments() -> None:
    prior = build_score_matrix(1.4, 1.0)
    targets = MarketTargets(home=0.51, draw=0.27, away=0.22, over_2_5=0.52, btts=0.50)
    calibrated = calibrate_distribution(prior, targets)
    summary = distribution_summary(calibrated)
    for key, target in targets.__dict__.items():
        assert np.isclose(summary[key], target, atol=3e-4)
