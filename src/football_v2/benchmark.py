from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Iterable

import numpy as np

from .baselines import dixon_coles_matrix, independent_poisson_matrix
from .dataset import HistoricalDataset
from .labels import ScoreArchetype, classify_score
from .model import ModelConfig, TailAwareExactScoreModel


@dataclass(frozen=True)
class DistributionMetrics:
    matches: int
    exact_accuracy: float
    top_k_accuracy: float
    direction_accuracy: float
    archetype_accuracy: float
    negative_log_likelihood: float
    extreme_tail_recall: float | None
    mean_actual_score_rank: float


@dataclass(frozen=True)
class BenchmarkReport:
    matches: int
    folds: int
    first_test_date: str
    last_test_date: str
    metrics: dict[str, DistributionMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "folds": self.folds,
            "first_test_date": self.first_test_date,
            "last_test_date": self.last_test_date,
            "metrics": {name: asdict(value) for name, value in self.metrics.items()},
        }


def _direction(score: tuple[int, int]) -> int:
    return (score[0] > score[1]) - (score[0] < score[1])


def _dominant_archetype(matrix: np.ndarray) -> ScoreArchetype:
    masses = {archetype: 0.0 for archetype in ScoreArchetype}
    for home in range(matrix.shape[0]):
        for away in range(matrix.shape[1]):
            masses[classify_score(home, away)] += float(matrix[home, away])
    return max(masses, key=masses.get)


def evaluate_matrices(
    matrices: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    *,
    top_k: int = 5,
) -> DistributionMetrics:
    if len(matrices) == 0:
        raise ValueError("cannot evaluate empty predictions")
    home = np.asarray(home_goals, dtype=int)
    away = np.asarray(away_goals, dtype=int)
    if home.shape != away.shape or len(home) != len(matrices):
        raise ValueError("prediction and target lengths differ")

    exact_hits = 0
    top_k_hits = 0
    direction_hits = 0
    archetype_hits = 0
    negative_log_likelihood = 0.0
    ranks: list[int] = []
    extreme_total = 0
    extreme_hits = 0
    extreme_types = {
        ScoreArchetype.HOME_BLOWOUT,
        ScoreArchetype.AWAY_BLOWOUT,
        ScoreArchetype.SHOOTOUT,
    }

    for matrix, actual_home, actual_away in zip(matrices, home, away, strict=True):
        actual = (int(actual_home), int(actual_away))
        order = np.argsort(matrix.ravel())[::-1]
        predicted = tuple(int(v) for v in np.unravel_index(int(order[0]), matrix.shape))
        top_scores = {
            tuple(int(v) for v in np.unravel_index(int(index), matrix.shape))
            for index in order[:top_k]
        }
        actual_index = np.ravel_multi_index(actual, matrix.shape)
        rank = int(np.where(order == actual_index)[0][0]) + 1
        ranks.append(rank)
        exact_hits += predicted == actual
        top_k_hits += actual in top_scores
        direction_hits += _direction(predicted) == _direction(actual)

        actual_archetype = classify_score(*actual)
        predicted_archetype = _dominant_archetype(matrix)
        archetype_hits += predicted_archetype is actual_archetype
        if actual_archetype in extreme_types:
            extreme_total += 1
            extreme_hits += predicted_archetype is actual_archetype

        negative_log_likelihood -= log(max(float(matrix[actual]), 1e-15))

    count = len(matrices)
    return DistributionMetrics(
        matches=count,
        exact_accuracy=exact_hits / count,
        top_k_accuracy=top_k_hits / count,
        direction_accuracy=direction_hits / count,
        archetype_accuracy=archetype_hits / count,
        negative_log_likelihood=negative_log_likelihood / count,
        extreme_tail_recall=(extreme_hits / extreme_total) if extreme_total else None,
        mean_actual_score_rank=float(np.mean(ranks)),
    )


def _nanmean(values: Iterable[float], fallback: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else fallback


def _rates_from_row(row: object) -> tuple[float, float]:
    league_home = float(row.league_home_goals) if np.isfinite(row.league_home_goals) else 1.50
    league_away = float(row.league_away_goals) if np.isfinite(row.league_away_goals) else 1.15
    home_rate = _nanmean(
        [row.home_gf10, row.home_venue_gf10, row.away_ga10, row.away_venue_ga10],
        league_home,
    )
    away_rate = _nanmean(
        [row.away_gf10, row.away_venue_gf10, row.home_ga10, row.home_venue_ga10],
        league_away,
    )
    return float(np.clip(home_rate, 0.20, 4.50)), float(np.clip(away_rate, 0.20, 4.50))


def _scale_binary_event(matrix: np.ndarray, mask: np.ndarray, target: float | None) -> np.ndarray:
    if target is None or not np.isfinite(target) or not 0 < target < 1:
        return matrix
    current = float(matrix[mask].sum())
    complement = 1.0 - current
    if current <= 0 or complement <= 0:
        return matrix
    adjusted = matrix.copy()
    adjusted[mask] *= target / current
    adjusted[~mask] *= (1.0 - target) / complement
    return adjusted / adjusted.sum()


def market_conditioned_matrix(base: np.ndarray, row: object, *, iterations: int = 25) -> np.ndarray:
    """Rake a score grid toward de-vigged 1X2, total and BTTS market moments."""
    home_goals, away_goals = np.indices(base.shape)
    masks = {
        "home": home_goals > away_goals,
        "draw": home_goals == away_goals,
        "away": home_goals < away_goals,
        "over": home_goals + away_goals > 2,
        "btts": (home_goals > 0) & (away_goals > 0),
    }
    targets = {
        "home": row.market_home_prob,
        "draw": row.market_draw_prob,
        "away": row.market_away_prob,
        "over": row.market_over25_prob,
        "btts": np.nan,
    }
    matrix = base.copy() / base.sum()
    for _ in range(iterations):
        for name in ("home", "draw", "away", "over", "btts"):
            matrix = _scale_binary_event(matrix, masks[name], targets[name])
    return matrix / matrix.sum()


def baseline_distributions(dataset: HistoricalDataset, *, kind: str) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for row in dataset.frame.itertuples(index=False):
        home_rate, away_rate = _rates_from_row(row)
        if kind == "poisson":
            matrix = independent_poisson_matrix(home_rate, away_rate, max_goals=7)
        elif kind == "dixon_coles":
            matrix = dixon_coles_matrix(home_rate, away_rate, rho=-0.08, max_goals=7)
        elif kind == "market":
            base = dixon_coles_matrix(home_rate, away_rate, rho=-0.08, max_goals=7)
            matrix = market_conditioned_matrix(base, row)
        else:
            raise ValueError(f"unknown baseline kind: {kind!r}")
        matrices.append(matrix)
    return np.stack(matrices)


def walk_forward_benchmark(
    dataset: HistoricalDataset,
    *,
    folds: int = 3,
    initial_train_fraction: float = 0.65,
    model_config: ModelConfig | None = None,
    top_k: int = 5,
) -> tuple[BenchmarkReport, TailAwareExactScoreModel]:
    """Run expanding-window, strictly chronological sample-out testing."""
    if folds < 1:
        raise ValueError("folds must be positive")
    if not 0.50 <= initial_train_fraction <= 0.90:
        raise ValueError("initial_train_fraction must be between 0.50 and 0.90")
    count = len(dataset.frame)
    initial = int(count * initial_train_fraction)
    if initial < 1000 or initial >= count:
        raise ValueError("insufficient data for walk-forward backtest")
    remaining = count - initial
    window = max(1, remaining // folds)

    learned_parts: list[np.ndarray] = []
    poisson_parts: list[np.ndarray] = []
    dixon_parts: list[np.ndarray] = []
    market_parts: list[np.ndarray] = []
    home_parts: list[np.ndarray] = []
    away_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    last_model: TailAwareExactScoreModel | None = None

    for fold in range(folds):
        train_end = initial + fold * window
        test_end = count if fold == folds - 1 else min(count, train_end + window)
        if test_end <= train_end:
            continue
        train = HistoricalDataset(
            dataset.frame.iloc[:train_end].reset_index(drop=True), dataset.feature_columns
        )
        test = HistoricalDataset(
            dataset.frame.iloc[train_end:test_end].reset_index(drop=True), dataset.feature_columns
        )
        model = TailAwareExactScoreModel(model_config).fit(
            train.features, train.home_goals, train.away_goals
        )
        learned_parts.append(model.predict_distribution(test.features))
        poisson_parts.append(baseline_distributions(test, kind="poisson"))
        dixon_parts.append(baseline_distributions(test, kind="dixon_coles"))
        market_parts.append(baseline_distributions(test, kind="market"))
        home_parts.append(test.home_goals)
        away_parts.append(test.away_goals)
        date_parts.append(test.frame["date"].to_numpy())
        last_model = model

    if last_model is None:
        raise RuntimeError("no walk-forward folds were produced")
    home = np.concatenate(home_parts)
    away = np.concatenate(away_parts)
    dates = np.concatenate(date_parts)
    predictions = {
        "v2": np.concatenate(learned_parts),
        "market": np.concatenate(market_parts),
        "dixon_coles": np.concatenate(dixon_parts),
        "poisson": np.concatenate(poisson_parts),
    }
    metrics = {
        name: evaluate_matrices(matrices, home, away, top_k=top_k)
        for name, matrices in predictions.items()
    }
    report = BenchmarkReport(
        matches=len(home),
        folds=folds,
        first_test_date=str(np.min(dates).astype("datetime64[D]")),
        last_test_date=str(np.max(dates).astype("datetime64[D]")),
        metrics=metrics,
    )
    final_model = TailAwareExactScoreModel(model_config).fit(
        dataset.features, dataset.home_goals, dataset.away_goals
    )
    return report, final_model
