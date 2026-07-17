"""Data-driven football exact-score modelling."""

from .baselines import dixon_coles_matrix, independent_poisson_matrix
from .benchmark import BenchmarkReport, DistributionMetrics, walk_forward_benchmark
from .dataset import HistoricalDataset, build_historical_dataset, load_premier_league_frame
from .evaluation import EvaluationReport, evaluate_predictions
from .freeze import Evidence, EvidenceKind, FrozenPrediction
from .labels import ScoreArchetype, classify_score
from .model import ModelConfig, TailAwareExactScoreModel
from .persistence import ModelBundle, load_bundle, save_bundle

__all__ = [
    "BenchmarkReport",
    "DistributionMetrics",
    "EvaluationReport",
    "Evidence",
    "EvidenceKind",
    "FrozenPrediction",
    "HistoricalDataset",
    "ModelBundle",
    "ModelConfig",
    "ScoreArchetype",
    "TailAwareExactScoreModel",
    "build_historical_dataset",
    "classify_score",
    "dixon_coles_matrix",
    "evaluate_predictions",
    "independent_poisson_matrix",
    "load_bundle",
    "load_premier_league_frame",
    "save_bundle",
    "walk_forward_benchmark",
]
