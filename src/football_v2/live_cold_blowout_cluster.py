from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .event_tail_model import AlertMetrics, alert_metrics
from .live_mechanism_rules import _rules, _select_rule
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
class ClusterScoreMetrics:
    all_alerts: ScoreDistributionMetrics | None
    true_cluster_only: ScoreDistributionMetrics | None


@dataclass(frozen=True)
class ColdBlowoutCutoffResult:
    cutoff: int
    matches: int
    base_rate: float
    selected_alerts: AlertMetrics
    selected_raw_scores: ClusterScoreMetrics
    selected_conditioned_scores: ClusterScoreMetrics
    fixed_rules: dict[str, AlertMetrics]


@dataclass(frozen=True)
class ColdBlowoutReport:
    cutoffs: tuple[ColdBlowoutCutoffResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoffs": [
                {
                    "cutoff": result.cutoff,
                    "matches": result.matches,
                    "base_rate": result.base_rate,
                    "selected_alerts": asdict(result.selected_alerts),
                    "selected_raw_scores": _cluster_metrics_to_dict(
                        result.selected_raw_scores
                    ),
                    "selected_conditioned_scores": _cluster_metrics_to_dict(
                        result.selected_conditioned_scores
                    ),
                    "fixed_rules": {
                        name: asdict(metrics)
                        for name, metrics in result.fixed_rules.items()
                    },
                }
                for result in self.cutoffs
            ],
            "fold_details": list(self.fold_details),
        }


def _cluster_metrics_to_dict(
    metrics: ClusterScoreMetrics,
) -> dict[str, object]:
    return {
        "all_alerts": (
            asdict(metrics.all_alerts) if metrics.all_alerts is not None else None
        ),
        "true_cluster_only": (
            asdict(metrics.true_cluster_only)
            if metrics.true_cluster_only is not None
            else None
        ),
    }


def _role_data(frame: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    underdog_home = (
        frame["market_home_prob"].to_numpy(dtype=float)
        < frame["market_away_prob"].to_numpy(dtype=float)
    )
    home_score = frame["home_score"].to_numpy(dtype=int)
    away_score = frame["away_score"].to_numpy(dtype=int)
    underdog_score = np.where(underdog_home, home_score, away_score)
    favorite_score = np.where(underdog_home, away_score, home_score)
    target = np.logical_and(
        underdog_score >= 3,
        underdog_score - favorite_score >= 2,
    )
    return underdog_home, underdog_score, target


def _condition_on_cold_blowout(
    matrices: np.ndarray,
    underdog_home: np.ndarray,
) -> np.ndarray:
    output = matrices.copy()
    home_index, away_index = np.indices(matrices.shape[1:])
    home_cluster = np.logical_and(
        home_index >= 3,
        home_index - away_index >= 2,
    )
    away_cluster = np.logical_and(
        away_index >= 3,
        away_index - home_index >= 2,
    )
    for row in range(len(output)):
        mask = home_cluster if underdog_home[row] else away_cluster
        mass = float(output[row][mask].sum())
        if mass <= 0:
            continue
        output[row][~mask] = 0.0
        output[row] /= output[row].sum()
    return output


def _score_metrics(
    matrices: np.ndarray,
    alerts: np.ndarray,
    target: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
) -> ClusterScoreMetrics:
    alert_mask = np.asarray(alerts, dtype=bool)
    true_mask = np.logical_and(alert_mask, target)
    return ClusterScoreMetrics(
        all_alerts=(
            _distribution_metrics(
                matrices[alert_mask], home[alert_mask], away[alert_mask]
            )
            if alert_mask.any()
            else None
        ),
        true_cluster_only=(
            _distribution_metrics(
                matrices[true_mask], home[true_mask], away[true_mask]
            )
            if true_mask.any()
            else None
        ),
    )


def _candidate_rules(frame: object) -> dict[str, np.ndarray]:
    rules = _rules(frame)
    return {
        name: rules[name]
        for name in (
            "underdog_lead_two",
            "underdog_two_goals_ahead",
            "underdog_lead_pressure",
            "underdog_lead_open_game",
            "underdog_two_goals_resistance",
            "underdog_lead_recent_chaos",
            "underdog_tied_dominant",
        )
    }


def leave_one_league_out_cold_blowout_test(
    dataset: LiveSnapshotDataset,
    *,
    cutoffs_to_test: tuple[int, ...] = (15, 30, 45),
) -> ColdBlowoutReport:
    frame = dataset.frame
    league = frame["division_id"].to_numpy(dtype=int)
    cutoffs = frame["snapshot_minute"].to_numpy(dtype=int)
    underdog_home, _, target = _role_data(frame)
    final_home = frame["home_score"].to_numpy(dtype=int)
    final_away = frame["away_score"].to_numpy(dtype=int)
    current_home = frame["live_home_score"].to_numpy(dtype=int)
    current_away = frame["live_away_score"].to_numpy(dtype=int)
    remaining_home = frame["remaining_home_goals"].to_numpy(dtype=int)
    remaining_away = frame["remaining_away_goals"].to_numpy(dtype=int)
    base_home, base_away = _market_rates_for_frame(frame)
    dynamic_features = dataset.features[:, _dynamic_feature_mask(dataset.feature_columns)]
    all_rules = _candidate_rules(frame)

    collected: dict[int, dict[str, list[object]]] = {
        cutoff: {
            "target": [],
            "home": [],
            "away": [],
            "underdog_home": [],
            "alerts": [],
            "raw": [],
            "conditioned": [],
        }
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
            training_rules = {
                name: values[train_indices] for name, values in all_rules.items()
            }
            selected_rule = _select_rule(training_rules, target[train_indices])
            alerts = (
                all_rules[selected_rule][test_indices]
                if selected_rule is not None
                else np.zeros(len(test_indices), dtype=bool)
            )

            home_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_home[train_indices],
                base_home[train_indices],
                random_state=501 + cutoff,
            )
            away_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_away[train_indices],
                base_away[train_indices],
                random_state=511 + cutoff,
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
            raw = _score_matrices(
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
            conditioned = _condition_on_cold_blowout(
                raw, underdog_home[test_indices]
            )
            parts = collected[cutoff]
            parts["target"].append(target[test_indices])
            parts["home"].append(final_home[test_indices])
            parts["away"].append(final_away[test_indices])
            parts["underdog_home"].append(underdog_home[test_indices])
            parts["alerts"].append(alerts)
            parts["raw"].append(raw)
            parts["conditioned"].append(conditioned)
            fold_details.append(
                {
                    "held_out_league": int(held_out),
                    "cutoff": int(cutoff),
                    "selected_rule": selected_rule,
                    "test_matches": int(test_mask.sum()),
                    "test_alerts": asdict(
                        alert_metrics(alerts, target[test_indices])
                    ),
                    "family": family[0],
                    "home_blend": home_model.blend,
                    "away_blend": away_model.blend,
                }
            )

    results: list[ColdBlowoutCutoffResult] = []
    for cutoff, parts in collected.items():
        targets = np.concatenate(parts["target"])
        home = np.concatenate(parts["home"])
        away = np.concatenate(parts["away"])
        alerts = np.concatenate(parts["alerts"])
        raw = np.concatenate(parts["raw"])
        conditioned = np.concatenate(parts["conditioned"])
        cutoff_mask = cutoffs == cutoff
        fixed_rules = {
            name: alert_metrics(values[cutoff_mask], target[cutoff_mask])
            for name, values in all_rules.items()
        }
        results.append(
            ColdBlowoutCutoffResult(
                cutoff=int(cutoff),
                matches=len(targets),
                base_rate=float(np.mean(targets)),
                selected_alerts=alert_metrics(alerts, targets),
                selected_raw_scores=_score_metrics(
                    raw, alerts, targets, home, away
                ),
                selected_conditioned_scores=_score_metrics(
                    conditioned, alerts, targets, home, away
                ),
                fixed_rules=fixed_rules,
            )
        )
    return ColdBlowoutReport(tuple(results), tuple(fold_details))
