from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import MarketTargets, calibrate_distribution, distribution_summary
from .score_matrix import ScenarioWeights, build_score_matrix
from .selection import GoalLicenses, SelectionResult, select_score


@dataclass(frozen=True)
class MatchModelInput:
    home_lambda: float
    away_lambda: float
    scenario_weights: ScenarioWeights = ScenarioWeights()
    market_targets: MarketTargets | None = None
    licenses: GoalLicenses = GoalLicenses()
    market_cluster: tuple[tuple[int, int], ...] = ()
    max_goals: int = 7


@dataclass(frozen=True)
class ModelResult:
    prior_matrix: np.ndarray
    calibrated_matrix: np.ndarray
    selection: SelectionResult
    summary: dict[str, float]


def run_model(model_input: MatchModelInput) -> ModelResult:
    prior = build_score_matrix(
        model_input.home_lambda,
        model_input.away_lambda,
        scenario_weights=model_input.scenario_weights,
        max_goals=model_input.max_goals,
    )
    calibrated = (
        calibrate_distribution(prior, model_input.market_targets)
        if model_input.market_targets is not None
        else prior
    )
    selection = select_score(
        calibrated,
        licenses=model_input.licenses,
        market_cluster=model_input.market_cluster,
    )
    return ModelResult(
        prior_matrix=prior,
        calibrated_matrix=calibrated,
        selection=selection,
        summary=distribution_summary(calibrated),
    )
