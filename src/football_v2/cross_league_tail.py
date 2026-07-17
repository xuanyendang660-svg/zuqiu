from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .event_tail_model import (
    AlertMetrics,
    EventTailConfig,
    _fit_model,
    _predict_fold,
    alert_metrics,
)
from .statsbomb_events import EventDataset
from .tail_score_classifier import TailExactScoreModel, score_ranking_metrics


@dataclass(frozen=True)
class CrossLeagueReport:
    matches: int
    target_name: str
    full_alerts: AlertMetrics
    market_alerts: AlertMetrics
    full_exact: dict[str, float | int | None]
    market_exact: dict[str, float | int | None]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "target_name": self.target_name,
            "full_alerts": asdict(self.full_alerts),
            "market_alerts": asdict(self.market_alerts),
            "full_exact": self.full_exact,
            "market_exact": self.market_exact,
            "fold_details": list(self.fold_details),
        }


def _market_mask(columns: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [
            column.startswith("market_")
            or column in {"ah_line", "division_id", "month_sin", "month_cos"}
            for column in columns
        ],
        dtype=bool,
    )


def leave_one_league_out_test(
    dataset: EventDataset,
    *,
    target_column: str = "tail_target",
    config: EventTailConfig | None = None,
) -> CrossLeagueReport:
    config = config or EventTailConfig()
    if target_column not in dataset.frame.columns:
        raise ValueError(f"missing target column: {target_column}")
    target = dataset.frame[target_column].to_numpy(dtype=int)
    league = dataset.frame["division_id"].to_numpy(dtype=int)
    market_mask = _market_mask(dataset.feature_columns)

    full_alert_parts: list[np.ndarray] = []
    market_alert_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    full_distribution_parts: list[np.ndarray] = []
    market_distribution_parts: list[np.ndarray] = []
    home_parts: list[np.ndarray] = []
    away_parts: list[np.ndarray] = []
    fold_details: list[dict[str, object]] = []

    for held_out in sorted(np.unique(league)):
        train_mask = league != held_out
        test_mask = league == held_out
        if int(train_mask.sum()) < 1000 or int(test_mask.sum()) < 250:
            continue
        train_order = np.argsort(
            dataset.frame.loc[train_mask, "date"].to_numpy(dtype="datetime64[ns]")
        )
        test_order = np.argsort(
            dataset.frame.loc[test_mask, "date"].to_numpy(dtype="datetime64[ns]")
        )
        train_x = dataset.features[train_mask][train_order]
        test_x = dataset.features[test_mask][test_order]
        train_target = target[train_mask][train_order]
        test_target = target[test_mask][test_order]
        train_types = dataset.tail_type[train_mask][train_order]
        train_home = dataset.home_goals[train_mask][train_order]
        train_away = dataset.away_goals[train_mask][train_order]
        test_home = dataset.home_goals[test_mask][test_order]
        test_away = dataset.away_goals[test_mask][test_order]

        full_model = _fit_model(
            train_x,
            train_target,
            train_types,
            train_home,
            train_away,
            config,
        )
        market_model = _fit_model(
            train_x[:, market_mask],
            train_target,
            train_types,
            train_home,
            train_away,
            config,
        )
        full_output = _predict_fold(full_model, test_x, config)
        market_output = _predict_fold(market_model, test_x[:, market_mask], config)

        full_exact_model = TailExactScoreModel(random_state=config.random_state).fit(
            train_x,
            train_home,
            train_away,
            train_target,
        )
        market_exact_model = TailExactScoreModel(random_state=config.random_state + 1).fit(
            train_x[:, market_mask],
            train_home,
            train_away,
            train_target,
        )
        full_exact_distribution = full_exact_model.predict_distribution(test_x)
        market_exact_distribution = market_exact_model.predict_distribution(
            test_x[:, market_mask]
        )

        full_alert_parts.append(full_output.alerts)
        market_alert_parts.append(market_output.alerts)
        target_parts.append(test_target)
        full_distribution_parts.append(full_exact_distribution)
        market_distribution_parts.append(market_exact_distribution)
        home_parts.append(test_home)
        away_parts.append(test_away)
        fold_details.append(
            {
                "held_out_league": int(held_out),
                "test_matches": int(test_mask.sum()),
                "full_threshold": full_output.threshold,
                "market_threshold": market_output.threshold,
                "full_alerts": asdict(alert_metrics(full_output.alerts, test_target)),
                "market_alerts": asdict(
                    alert_metrics(market_output.alerts, test_target)
                ),
            }
        )

    if not target_parts:
        raise RuntimeError("no league holdout folds were produced")
    combined_target = np.concatenate(target_parts)
    full_alerts = np.concatenate(full_alert_parts)
    market_alerts = np.concatenate(market_alert_parts)
    home = np.concatenate(home_parts)
    away = np.concatenate(away_parts)
    full_distributions = np.concatenate(full_distribution_parts)
    market_distributions = np.concatenate(market_distribution_parts)
    return CrossLeagueReport(
        matches=len(combined_target),
        target_name=target_column,
        full_alerts=alert_metrics(full_alerts, combined_target),
        market_alerts=alert_metrics(market_alerts, combined_target),
        full_exact=score_ranking_metrics(
            full_distributions, full_alerts, home, away
        ),
        market_exact=score_ranking_metrics(
            market_distributions, market_alerts, home, away
        ),
        fold_details=tuple(fold_details),
    )
