"""Football market and exact-score modelling package."""

from .calibration import MarketTargets, calibrate_distribution
from .pipeline import MatchModelInput, ModelResult, run_model
from .score_matrix import ScenarioWeights, build_score_matrix
from .selection import GoalLicenses, SelectionResult, select_score

__all__ = [
    "GoalLicenses",
    "MarketTargets",
    "MatchModelInput",
    "ModelResult",
    "ScenarioWeights",
    "SelectionResult",
    "build_score_matrix",
    "calibrate_distribution",
    "run_model",
    "select_score",
]
