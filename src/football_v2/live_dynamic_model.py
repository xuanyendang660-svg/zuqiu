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
from .live_snapshots import LiveSnapshotDataset
from .tail_score_classifier import TailExactScoreModel, score_ranking_metrics


@dataclass(frozen=True)
class LiveCutoffResult:
    cutoff: int
    matches: int
    full_alerts: AlertMetrics
    state_alerts: AlertMetrics
    full_exact: dict[str, float | int | None]
    state_exact: dict[str, float | int | None]


@dataclass(frozen=True)
class LiveDynamicReport:
    target_name: str
    cutoffs: tuple[LiveCutoffResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "cutoffs": [
                {
                    "cutoff": result.cutoff,
                    "matches": result.matches,
                    "full_alerts": asdict(result.full_alerts),
                    "state_alerts": asdict(result.state_alerts),
                    "full_exact": result.full_exact,
                    "state_exact": result.state_exact,
                }
                for result in self.cutoffs
            ],
            "fold_details": list(self.fold_details),
        }


def _state_mask(columns: tuple[str, ...]) -> np.ndarray:
    state_columns = {
        "ah_line",
        "division_id",
        "month_sin",
        "month_cos",
        "snapshot_minute",
        "remaining_minutes",
        "live_home_score",
        "live_away_score",
        "live_score_diff",
        "live_total_goals",
        "minutes_since_goal",
        "goals_last10",
    }
    return np.array(
        [column.startswith("market_") or column in state_columns for column in columns],
        dtype=bool,
    )


def _safe_exact_model(
    features: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
    score_target: np.ndarray,
    *,
    random_state: int,
) -> TailExactScoreModel | None:
    if int(np.asarray(score_target, dtype=bool).sum()) < 40:
        return None
    return TailExactScoreModel(
        random_state=random_state,
        n_estimators=120,
    ).fit(features, home, away, score_target)


def leave_one_league_out_live_test(
    dataset: LiveSnapshotDataset,
    *,
    target_column: str = "final_jackpot_target",
    config: EventTailConfig | None = None,
) -> LiveDynamicReport:
    config = config or EventTailConfig(
        max_alert_coverage=0.08,
        minimum_alerts=6,
        minimum_lift=1.25,
        max_iter=180,
        min_samples_leaf=24,
    )
    if target_column not in dataset.frame.columns:
        raise ValueError(f"missing target column: {target_column}")
    target = dataset.frame[target_column].to_numpy(dtype=int)
    score_target = dataset.frame["final_jackpot_target"].to_numpy(dtype=int)
    league = dataset.frame["division_id"].to_numpy(dtype=int)
    cutoffs = dataset.frame["snapshot_minute"].to_numpy(dtype=int)
    home = dataset.frame["home_score"].to_numpy(dtype=int)
    away = dataset.frame["away_score"].to_numpy(dtype=int)
    state_mask = _state_mask(dataset.feature_columns)

    fold_outputs: list[dict[str, object]] = []
    collected: dict[int, dict[str, list[np.ndarray]]] = {
        cutoff: {
            "target": [],
            "home": [],
            "away": [],
            "full_alerts": [],
            "state_alerts": [],
            "full_distributions": [],
            "state_distributions": [],
        }
        for cutoff in sorted(np.unique(cutoffs))
    }

    for held_out in sorted(np.unique(league)):
        train_mask = league != held_out
        test_mask = league == held_out
        if int(train_mask.sum()) < 2500 or int(test_mask.sum()) < 700:
            continue
        train_order = np.lexsort(
            (
                dataset.frame.loc[train_mask, "snapshot_minute"].to_numpy(),
                dataset.frame.loc[train_mask, "match_id"].to_numpy(),
                dataset.frame.loc[train_mask, "date"].to_numpy(dtype="datetime64[ns]"),
            )
        )
        test_order = np.lexsort(
            (
                dataset.frame.loc[test_mask, "snapshot_minute"].to_numpy(),
                dataset.frame.loc[test_mask, "match_id"].to_numpy(),
                dataset.frame.loc[test_mask, "date"].to_numpy(dtype="datetime64[ns]"),
            )
        )
        train_x = dataset.features[train_mask][train_order]
        test_x = dataset.features[test_mask][test_order]
        train_target = target[train_mask][train_order]
        test_target = target[test_mask][test_order]
        train_score_target = score_target[train_mask][train_order]
        train_home = home[train_mask][train_order]
        train_away = away[train_mask][train_order]
        test_home = home[test_mask][test_order]
        test_away = away[test_mask][test_order]
        test_cutoffs = cutoffs[test_mask][test_order]
        train_types = dataset.frame.loc[train_mask, "final_jackpot_type"].to_numpy(dtype=str)[
            train_order
        ]

        full_model = _fit_model(
            train_x,
            train_target,
            train_types,
            train_home,
            train_away,
            config,
        )
        state_model = _fit_model(
            train_x[:, state_mask],
            train_target,
            train_types,
            train_home,
            train_away,
            config,
        )
        full_output = _predict_fold(full_model, test_x, config)
        state_output = _predict_fold(state_model, test_x[:, state_mask], config)
        full_exact_model = _safe_exact_model(
            train_x,
            train_home,
            train_away,
            train_score_target,
            random_state=config.random_state,
        )
        state_exact_model = _safe_exact_model(
            train_x[:, state_mask],
            train_home,
            train_away,
            train_score_target,
            random_state=config.random_state + 1,
        )
        full_distributions = (
            full_exact_model.predict_distribution(test_x)
            if full_exact_model is not None
            else full_output.distributions
        )
        state_distributions = (
            state_exact_model.predict_distribution(test_x[:, state_mask])
            if state_exact_model is not None
            else state_output.distributions
        )

        fold_outputs.append(
            {
                "held_out_league": int(held_out),
                "test_snapshots": int(test_mask.sum()),
                "full_threshold": full_output.threshold,
                "state_threshold": state_output.threshold,
                "full_alerts": asdict(alert_metrics(full_output.alerts, test_target)),
                "state_alerts": asdict(
                    alert_metrics(state_output.alerts, test_target)
                ),
            }
        )
        for cutoff in collected:
            mask = test_cutoffs == cutoff
            collected[cutoff]["target"].append(test_target[mask])
            collected[cutoff]["home"].append(test_home[mask])
            collected[cutoff]["away"].append(test_away[mask])
            collected[cutoff]["full_alerts"].append(full_output.alerts[mask])
            collected[cutoff]["state_alerts"].append(state_output.alerts[mask])
            collected[cutoff]["full_distributions"].append(full_distributions[mask])
            collected[cutoff]["state_distributions"].append(state_distributions[mask])

    if not fold_outputs:
        raise RuntimeError("no live league holdout folds were produced")
    results: list[LiveCutoffResult] = []
    for cutoff, parts in collected.items():
        actual = np.concatenate(parts["target"])
        actual_home = np.concatenate(parts["home"])
        actual_away = np.concatenate(parts["away"])
        full_alerts = np.concatenate(parts["full_alerts"])
        state_alerts = np.concatenate(parts["state_alerts"])
        full_distributions = np.concatenate(parts["full_distributions"])
        state_distributions = np.concatenate(parts["state_distributions"])
        results.append(
            LiveCutoffResult(
                cutoff=cutoff,
                matches=len(actual),
                full_alerts=alert_metrics(full_alerts, actual),
                state_alerts=alert_metrics(state_alerts, actual),
                full_exact=score_ranking_metrics(
                    full_distributions, full_alerts, actual_home, actual_away
                ),
                state_exact=score_ranking_metrics(
                    state_distributions, state_alerts, actual_home, actual_away
                ),
            )
        )
    return LiveDynamicReport(
        target_name=target_column,
        cutoffs=tuple(results),
        fold_details=tuple(fold_outputs),
    )
