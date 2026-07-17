from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .event_tail_model import EventTailConfig, _fit_model, _predict_fold, alert_metrics
from .live_dynamic_model import _state_mask
from .live_mechanism_rules import _rules
from .live_remaining_goals import (
    ScoreDistributionMetrics,
    _choose_family,
    _distribution_metrics,
    _dynamic_feature_mask,
    _fit_regressor,
    _market_rates_for_frame,
    _score_matrices,
)
from .live_snapshots import LiveSnapshotDataset


@dataclass(frozen=True)
class ColdTailRuleResult:
    alerts: int
    precision: float | None
    recall: float
    lift: float | None
    exact_scores: ScoreDistributionMetrics | None


@dataclass(frozen=True)
class ColdTailCutoffResult:
    cutoff: int
    rules: dict[str, ColdTailRuleResult]


@dataclass(frozen=True)
class ColdTailIntersectionReport:
    cutoffs: tuple[ColdTailCutoffResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoffs": [
                {
                    "cutoff": result.cutoff,
                    "rules": {
                        name: {
                            "alerts": rule.alerts,
                            "precision": rule.precision,
                            "recall": rule.recall,
                            "lift": rule.lift,
                            "exact_scores": (
                                asdict(rule.exact_scores)
                                if rule.exact_scores
                                else None
                            ),
                        }
                        for name, rule in result.rules.items()
                    },
                }
                for result in self.cutoffs
            ],
            "fold_details": list(self.fold_details),
        }


def _intersection_rules(frame: object, tail_alerts: np.ndarray) -> dict[str, np.ndarray]:
    mechanism = _rules(frame)
    open_or_resistant = np.logical_or(
        mechanism["underdog_lead_open_game"],
        mechanism["underdog_two_goals_resistance"],
    )
    return {
        "tail_alert_only": tail_alerts,
        "tail_and_lead_two": np.logical_and(
            tail_alerts, mechanism["underdog_lead_two"]
        ),
        "tail_and_two_goals_ahead": np.logical_and(
            tail_alerts, mechanism["underdog_two_goals_ahead"]
        ),
        "tail_and_open_game": np.logical_and(
            tail_alerts, mechanism["underdog_lead_open_game"]
        ),
        "tail_and_resistance": np.logical_and(
            tail_alerts, mechanism["underdog_two_goals_resistance"]
        ),
        "tail_and_open_or_resistant": np.logical_and(
            tail_alerts, open_or_resistant
        ),
        "tail_and_favorite_red": np.logical_and(
            tail_alerts, mechanism["underdog_lead_favorite_red"]
        ),
    }


def leave_one_league_out_cold_tail_test(
    dataset: LiveSnapshotDataset,
    *,
    cutoffs_to_test: tuple[int, ...] = (30, 45),
) -> ColdTailIntersectionReport:
    frame = dataset.frame
    league = frame["division_id"].to_numpy(dtype=int)
    cutoffs = frame["snapshot_minute"].to_numpy(dtype=int)
    tail_target = frame["final_jackpot_target"].to_numpy(dtype=int)
    tail_types = frame["final_jackpot_type"].to_numpy(dtype=str)
    cold_target = frame["upset_jackpot_target"].to_numpy(dtype=int)
    final_home = frame["home_score"].to_numpy(dtype=int)
    final_away = frame["away_score"].to_numpy(dtype=int)
    current_home = frame["live_home_score"].to_numpy(dtype=int)
    current_away = frame["live_away_score"].to_numpy(dtype=int)
    remaining_home = frame["remaining_home_goals"].to_numpy(dtype=int)
    remaining_away = frame["remaining_away_goals"].to_numpy(dtype=int)
    base_home, base_away = _market_rates_for_frame(frame)
    state_features = dataset.features[:, _state_mask(dataset.feature_columns)]
    dynamic_features = dataset.features[:, _dynamic_feature_mask(dataset.feature_columns)]
    config = EventTailConfig(
        max_alert_coverage=0.08,
        minimum_alerts=6,
        minimum_lift=1.25,
        max_iter=180,
        min_samples_leaf=24,
    )

    collected: dict[int, dict[str, list[object]]] = {
        cutoff: {"target": [], "home": [], "away": [], "matrices": [], "rules": []}
        for cutoff in cutoffs_to_test
    }
    fold_details: list[dict[str, object]] = []

    for held_out in sorted(np.unique(league)):
        for cutoff in cutoffs_to_test:
            train_mask = np.logical_and(league != held_out, cutoffs == cutoff)
            test_mask = np.logical_and(league == held_out, cutoffs == cutoff)
            if int(train_mask.sum()) < 1000 or int(test_mask.sum()) < 250:
                continue
            order = np.argsort(
                frame.loc[train_mask, "date"].to_numpy(dtype="datetime64[ns]")
            )
            train_indices = np.where(train_mask)[0][order]
            test_indices = np.where(test_mask)[0]
            classifier = _fit_model(
                state_features[train_indices],
                tail_target[train_indices],
                tail_types[train_indices],
                final_home[train_indices],
                final_away[train_indices],
                config,
            )
            output = _predict_fold(
                classifier, state_features[test_indices], config
            )
            home_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_home[train_indices],
                base_home[train_indices],
                random_state=301 + cutoff,
            )
            away_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_away[train_indices],
                base_away[train_indices],
                random_state=311 + cutoff,
            )
            train_home_mean = home_model.predict(
                dynamic_features[train_indices], base_home[train_indices]
            )
            train_away_mean = away_model.predict(
                dynamic_features[train_indices], base_away[train_indices]
            )
            family = _choose_family(
                remaining_home[train_indices],
                remaining_away[train_indices],
                train_home_mean,
                train_away_mean,
            )
            matrices = _score_matrices(
                current_home[test_indices],
                current_away[test_indices],
                home_model.predict(
                    dynamic_features[test_indices], base_home[test_indices]
                ),
                away_model.predict(
                    dynamic_features[test_indices], base_away[test_indices]
                ),
                family=family[0],
                home_dispersion=family[1],
                away_dispersion=family[2],
            )
            test_frame = frame.iloc[test_indices]
            intersections = _intersection_rules(test_frame, output.alerts)
            parts = collected[cutoff]
            parts["target"].append(cold_target[test_indices])
            parts["home"].append(final_home[test_indices])
            parts["away"].append(final_away[test_indices])
            parts["matrices"].append(matrices)
            parts["rules"].append(intersections)
            fold_details.append(
                {
                    "held_out_league": int(held_out),
                    "cutoff": int(cutoff),
                    "test_matches": int(test_mask.sum()),
                    "tail_alerts": asdict(
                        alert_metrics(output.alerts, tail_target[test_indices])
                    ),
                    "cold_intersections": {
                        name: asdict(
                            alert_metrics(mask, cold_target[test_indices])
                        )
                        for name, mask in intersections.items()
                    },
                }
            )

    results: list[ColdTailCutoffResult] = []
    for cutoff, parts in collected.items():
        targets = np.concatenate(parts["target"])
        home_values = np.concatenate(parts["home"])
        away_values = np.concatenate(parts["away"])
        matrices = np.concatenate(parts["matrices"])
        rule_names = tuple(parts["rules"][0])
        rules: dict[str, ColdTailRuleResult] = {}
        for name in rule_names:
            mask = np.concatenate([rule[name] for rule in parts["rules"]])
            alert_result = alert_metrics(mask, targets)
            rules[name] = ColdTailRuleResult(
                alerts=alert_result.alerts,
                precision=alert_result.precision,
                recall=alert_result.recall,
                lift=alert_result.lift,
                exact_scores=(
                    _distribution_metrics(
                        matrices[mask], home_values[mask], away_values[mask]
                    )
                    if mask.any()
                    else None
                ),
            )
        results.append(ColdTailCutoffResult(int(cutoff), rules))
    return ColdTailIntersectionReport(tuple(results), tuple(fold_details))
