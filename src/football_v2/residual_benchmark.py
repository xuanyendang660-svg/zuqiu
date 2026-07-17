from __future__ import annotations

import numpy as np

from .benchmark import BenchmarkReport, baseline_distributions, evaluate_matrices
from .dataset import HistoricalDataset
from .residual import MarketResidualScoreModel, ResidualConfig


def walk_forward_residual_benchmark(
    dataset: HistoricalDataset,
    *,
    folds: int = 3,
    initial_train_fraction: float = 0.65,
    residual_config: ResidualConfig | None = None,
    top_k: int = 5,
) -> tuple[BenchmarkReport, MarketResidualScoreModel]:
    """Expanding-window benchmark for the market-preserving residual model."""
    if folds < 1:
        raise ValueError("folds must be positive")
    if not 0.50 <= initial_train_fraction <= 0.90:
        raise ValueError("initial_train_fraction must be between 0.50 and 0.90")
    count = len(dataset.frame)
    initial = int(count * initial_train_fraction)
    if initial < 1000 or initial >= count:
        raise ValueError("insufficient data for walk-forward backtest")
    window = max(1, (count - initial) // folds)

    residual_parts: list[np.ndarray] = []
    market_parts: list[np.ndarray] = []
    poisson_parts: list[np.ndarray] = []
    dixon_parts: list[np.ndarray] = []
    home_parts: list[np.ndarray] = []
    away_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    selections: list[dict[str, float]] = []

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
        train_market = baseline_distributions(train, kind="market")
        test_market = baseline_distributions(test, kind="market")
        model = MarketResidualScoreModel(residual_config).fit(
            train.features,
            train.home_goals,
            train.away_goals,
            train_market,
        )
        residual_parts.append(model.predict_distribution(test.features, test_market))
        market_parts.append(test_market)
        poisson_parts.append(baseline_distributions(test, kind="poisson"))
        dixon_parts.append(baseline_distributions(test, kind="dixon_coles"))
        home_parts.append(test.home_goals)
        away_parts.append(test.away_goals)
        date_parts.append(test.frame["date"].to_numpy())
        selections.append(
            {
                "alpha": model.selection_.alpha,
                "beta": model.selection_.beta,
                "tail_boost": model.selection_.tail_boost,
            }
        )

    if not residual_parts:
        raise RuntimeError("no walk-forward folds were produced")
    home = np.concatenate(home_parts)
    away = np.concatenate(away_parts)
    dates = np.concatenate(date_parts)
    predictions = {
        "v2_residual": np.concatenate(residual_parts),
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
    all_market = baseline_distributions(dataset, kind="market")
    final_model = MarketResidualScoreModel(residual_config).fit(
        dataset.features,
        dataset.home_goals,
        dataset.away_goals,
        all_market,
    )
    final_model.walk_forward_selections_ = selections
    return report, final_model
