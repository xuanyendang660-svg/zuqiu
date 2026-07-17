from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .score_matrix import btts_probability, outcome_probabilities, totals_probability


@dataclass(frozen=True)
class MarketTargets:
    home: float | None = None
    draw: float | None = None
    away: float | None = None
    over_2_5: float | None = None
    btts: float | None = None

    def validate(self) -> None:
        values = [v for v in self.__dict__.values() if v is not None]
        if any(v <= 0 or v >= 1 for v in values):
            raise ValueError("market targets must be strictly between 0 and 1")
        result_values = [self.home, self.draw, self.away]
        if all(v is not None for v in result_values):
            if not np.isclose(sum(result_values), 1.0, atol=1e-4):
                raise ValueError("home/draw/away targets must sum to 1")


def _feature_stack(shape: tuple[int, int], targets: MarketTargets) -> tuple[list[str], np.ndarray]:
    h, a = np.indices(shape)
    names: list[str] = []
    features: list[np.ndarray] = []
    for name, target, feature in [
        ("home", targets.home, h > a),
        ("draw", targets.draw, h == a),
        ("away", targets.away, h < a),
        ("over_2_5", targets.over_2_5, (h + a) > 2.5),
        ("btts", targets.btts, (h > 0) & (a > 0)),
    ]:
        if target is not None:
            names.append(name)
            features.append(feature.astype(float))
    return names, np.stack(features) if features else np.empty((0, *shape))


def _target_vector(targets: MarketTargets, names: list[str]) -> np.ndarray:
    return np.array([getattr(targets, name) for name in names], dtype=float)


def calibrate_distribution(
    prior: np.ndarray,
    targets: MarketTargets,
    *,
    tolerance: float = 2e-4,
) -> np.ndarray:
    """KL-project a prior score matrix onto observable market moments."""
    targets.validate()
    if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
        raise ValueError("prior must be a square score matrix")
    if np.any(prior < 0) or prior.sum() <= 0:
        raise ValueError("prior must be a non-negative distribution")
    prior = prior / prior.sum()

    names, features = _feature_stack(prior.shape, targets)
    if not names:
        return prior.copy()
    target = _target_vector(targets, names)
    log_prior = np.log(np.clip(prior, 1e-15, None))

    def distribution(theta: np.ndarray) -> np.ndarray:
        log_q = log_prior + np.tensordot(theta, features, axes=(0, 0))
        log_q -= np.max(log_q)
        q = np.exp(log_q)
        return q / q.sum()

    def residual(theta: np.ndarray) -> np.ndarray:
        q = distribution(theta)
        moments = np.array([(q * f).sum() for f in features])
        return moments - target

    solution = least_squares(residual, x0=np.zeros(len(names)), max_nfev=4000)
    q = distribution(solution.x)
    max_error = float(np.max(np.abs(residual(solution.x))))
    if max_error > tolerance:
        raise RuntimeError(
            f"market calibration did not converge within tolerance; max error={max_error:.6f}"
        )
    return q


def distribution_summary(matrix: np.ndarray) -> dict[str, float]:
    return {
        **outcome_probabilities(matrix),
        "over_2_5": totals_probability(matrix, 2.5),
        "btts": btts_probability(matrix),
    }
