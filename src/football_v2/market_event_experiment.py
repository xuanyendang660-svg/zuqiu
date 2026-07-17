from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .event_tail_model import (
    AlertMetrics,
    EventTailConfig,
    _fit_model,
    _predict_fold,
    _score_metrics,
    alert_metrics,
)
from .statsbomb_events import EventDataset


@dataclass(frozen=True)
class FeatureSetResult:
    alerts: AlertMetrics
    exact_accuracy: float | None
    top5_accuracy: float | None
    mean_rank: float | None
    thresholds: tuple[float, ...]


@dataclass(frozen=True)
class MarketEventReport:
    matches: int
    target_name: str
    first_test_date: str
    last_test_date: str
    full: FeatureSetResult
    market_only: FeatureSetResult
    event_only: FeatureSetResult

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "target_name": self.target_name,
            "first_test_date": self.first_test_date,
            "last_test_date": self.last_test_date,
            "full": {
                "alerts": asdict(self.full.alerts),
                "exact_accuracy": self.full.exact_accuracy,
                "top5_accuracy": self.full.top5_accuracy,
                "mean_rank": self.full.mean_rank,
                "thresholds": list(self.full.thresholds),
            },
            "market_only": {
                "alerts": asdict(self.market_only.alerts),
                "exact_accuracy": self.market_only.exact_accuracy,
                "top5_accuracy": self.market_only.top5_accuracy,
                "mean_rank": self.market_only.mean_rank,
                "thresholds": list(self.market_only.thresholds),
            },
            "event_only": {
                "alerts": asdict(self.event_only.alerts),
                "exact_accuracy": self.event_only.exact_accuracy,
                "top5_accuracy": self.event_only.top5_accuracy,
                "mean_rank": self.event_only.mean_rank,
                "thresholds": list(self.event_only.thresholds),
            },
        }


def _feature_masks(columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    market = np.array(
        [
            column.startswith("market_")
            or column in {"ah_line", "division_id", "month_sin", "month_cos"}
            for column in columns
        ],
        dtype=bool,
    )
    event = np.logical_not(
        np.array(
            [
                column.startswith("market_")
                or column in {"ah_line", "division_id"}
                for column in columns
            ],
            dtype=bool,
        )
    )
    return {
        "full": np.ones(len(columns), dtype=bool),
        "market_only": market,
        "event_only": event,
    }


def _run_feature_set(
    dataset: EventDataset,
    target: np.ndarray,
    feature_mask: np.ndarray,
    *,
    folds: int,
    initial_train_fraction: float,
    config: EventTailConfig,
) -> tuple[FeatureSetResult, np.ndarray]:
    count = len(dataset.frame)
    initial = int(count * initial_train_fraction)
    window = max(1, (count - initial) // folds)
    outputs = []
    target_parts: list[np.ndarray] = []
    home_parts: list[np.ndarray] = []
    away_parts: list[np.ndarray] = []
    thresholds: list[float] = []
    for fold in range(folds):
        train_end = initial + fold * window
        test_end = count if fold == folds - 1 else min(count, train_end + window)
        if test_end <= train_end:
            continue
        train_x = dataset.features[:train_end, feature_mask]
        test_x = dataset.features[train_end:test_end, feature_mask]
        model = _fit_model(
            train_x,
            target[:train_end],
            dataset.tail_type[:train_end],
            dataset.home_goals[:train_end],
            dataset.away_goals[:train_end],
            config,
        )
        output = _predict_fold(model, test_x, config)
        outputs.append(output)
        target_parts.append(target[train_end:test_end])
        home_parts.append(dataset.home_goals[train_end:test_end])
        away_parts.append(dataset.away_goals[train_end:test_end])
        thresholds.append(output.threshold)
    alerts = np.concatenate([output.alerts for output in outputs])
    distributions = np.concatenate([output.distributions for output in outputs])
    combined_target = np.concatenate(target_parts)
    home = np.concatenate(home_parts)
    away = np.concatenate(away_parts)
    exact, top5, rank = _score_metrics(distributions, alerts, home, away)
    return (
        FeatureSetResult(
            alerts=alert_metrics(alerts, combined_target),
            exact_accuracy=exact,
            top5_accuracy=top5,
            mean_rank=rank,
            thresholds=tuple(thresholds),
        ),
        combined_target,
    )


def walk_forward_market_event_test(
    dataset: EventDataset,
    *,
    target_column: str = "tail_target",
    folds: int = 3,
    initial_train_fraction: float = 0.55,
    config: EventTailConfig | None = None,
) -> MarketEventReport:
    config = config or EventTailConfig()
    if target_column not in dataset.frame.columns:
        raise ValueError(f"missing target column: {target_column}")
    target = dataset.frame[target_column].to_numpy(dtype=int)
    masks = _feature_masks(dataset.feature_columns)
    results: dict[str, FeatureSetResult] = {}
    combined_target: np.ndarray | None = None
    for name, mask in masks.items():
        result, current_target = _run_feature_set(
            dataset,
            target,
            mask,
            folds=folds,
            initial_train_fraction=initial_train_fraction,
            config=config,
        )
        results[name] = result
        if combined_target is None:
            combined_target = current_target
    initial = int(len(dataset.frame) * initial_train_fraction)
    dates = dataset.frame.iloc[initial:]["date"].to_numpy()
    return MarketEventReport(
        matches=len(combined_target) if combined_target is not None else 0,
        target_name=target_column,
        first_test_date=str(np.min(dates).astype("datetime64[D]")),
        last_test_date=str(np.max(dates).astype("datetime64[D]")),
        full=results["full"],
        market_only=results["market_only"],
        event_only=results["event_only"],
    )
