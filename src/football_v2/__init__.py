"""Data-driven football exact-score modelling."""

from .baselines import dixon_coles_matrix, independent_poisson_matrix
from .evaluation import EvaluationReport, evaluate_predictions
from .freeze import Evidence, EvidenceKind, FrozenPrediction
from .labels import ScoreArchetype, classify_score
from .model import TailAwareExactScoreModel

__all__ = [
    "EvaluationReport",
    "Evidence",
    "EvidenceKind",
    "FrozenPrediction",
    "ScoreArchetype",
    "TailAwareExactScoreModel",
    "classify_score",
    "dixon_coles_matrix",
    "evaluate_predictions",
    "independent_poisson_matrix",
]
