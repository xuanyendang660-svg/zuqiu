from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .event_tail_model import EventTailConfig, _fit_model, _predict_fold, alert_metrics
from .live_cold_blowout_cluster import (
    ClusterScoreMetrics,
    _condition_on_cold_blowout,
    _role_data,
    _score_metrics,
)
from .live_oriented_score import _role_features
from .live_remaining_goals import (
    _choose_family,
    _dynamic_feature_mask,
    _fit_regressor,
    _market_rates_for_frame,
    _score_matrices,
)
from .live_snapshots import LiveSnapshotDataset


@dataclass(frozen=True)
class ColdBlowoutModelCutoffResult:
    cutoff: int
    matches: int
    state_alerts: dict[str, object]
    full_alerts: dict[str, object]
    state_raw_scores: ClusterScoreMetrics
    state_conditioned_scores: ClusterScoreMetrics
    full_raw_scores: ClusterScoreMetrics
    full_conditioned_scores: ClusterScoreMetrics


@dataclass(frozen=True)
class ColdBlowoutModelReport:
    cutoffs: tuple[ColdBlowoutModelCutoffResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoffs": [
                {
                    "cutoff": result.cutoff,
                    "matches": result.matches,
                    "state_alerts": result.state_alerts,
                    "full_alerts": result.full_alerts,
                    "state_raw_scores": _cluster_to_dict(
                        result.state_raw_scores
                    ),
                    "state_conditioned_scores": _cluster_to_dict(
                        result.state_conditioned_scores
                    ),
                    "full_raw_scores": _cluster_to_dict(
                        result.full_raw_scores
                    ),
                    "full_conditioned_scores": _cluster_to_dict(
                        result.full_conditioned_scores
                    ),
                }
                for result in self.cutoffs
            ],
            "fold_details": list(self.fold_details),
        }


def _cluster_to_dict(metrics: ClusterScoreMetrics) -> dict[str, object]:
    return {
        "all_alerts": (
            asdict(metrics.all_alerts) if metrics.all_alerts else None
        ),
        "true_cluster_only": (
            asdict(metrics.true_cluster_only)
            if metrics.true_cluster_only
            else None
        ),
    }


def _cold_types(frame: object) -> np.ndarray:
    underdog_home = (
        frame["market_home_prob"].to_numpy(dtype=float)
        < frame["market_away_prob"].to_numpy(dtype=float)
    )
    return np.where(underdog_home, "home_blowout", "away_blowout")


def _state_features(frame: object) -> np.ndarray:
    role = _role_features(frame)
    selected = np.column_stack(
        [
            frame["snapshot_minute"].to_numpy(dtype=float),
            frame["remaining_minutes"].to_numpy(dtype=float),
            frame["market_home_prob"].to_numpy(dtype=float),
            frame["market_draw_prob"].to_numpy(dtype=float),
            frame["market_away_prob"].to_numpy(dtype=float),
            frame["market_over25_prob"].to_numpy(dtype=float),
            frame["market_entropy"].to_numpy(dtype=float),
            frame["live_home_score"].to_numpy(dtype=float),
            frame["live_away_score"].to_numpy(dtype=float),
            frame["live_total_goals"].to_numpy(dtype=float),
            frame["goals_last10"].to_numpy(dtype=float),
            frame["shots_last10"].to_numpy(dtype=float),
            frame["xg_last10"].to_numpy(dtype=float),
        ]
    )
    return np.column_stack([selected, role])


def leave_one_league_out_cold_blowout_model_test(
    dataset: LiveSnapshotDataset,
    *,
    cutoffs_to_test: tuple[int, ...] = (15, 30, 45),
) -> ColdBlowoutModelReport:
    frame = dataset.frame
    underdog_home, _, target = _role_data(frame)
    target = target.astype(int)
    cold_types = _cold_types(frame)
    league = frame["division_id"].to_numpy(dtype=int)
    cutoffs = frame["snapshot_minute"].to_numpy(dtype=int)
    final_home = frame["home_score"].to_numpy(dtype=int)
    final_away = frame["away_score"].to_numpy(dtype=int)
    current_home = frame["live_home_score"].to_numpy(dtype=int)
    current_away = frame["live_away_score"].to_numpy(dtype=int)
    remaining_home = frame["remaining_home_goals"].to_numpy(dtype=int)
    remaining_away = frame["remaining_away_goals"].to_numpy(dtype=int)
    base_home, base_away = _market_rates_for_frame(frame)
    state_features = _state_features(frame)
    full_features = np.column_stack([dataset.features, state_features])
    dynamic_features = dataset.features[:, _dynamic_feature_mask(dataset.feature_columns)]
    config = EventTailConfig(
        max_alert_coverage=0.10,
        minimum_alerts=8,
        minimum_lift=1.50,
        max_iter=240,
        min_samples_leaf=22,
    )

    collected: dict[int, dict[str, list[object]]] = {
        cutoff: {
            "target": [],
            "home": [],
            "away": [],
            "underdog_home": [],
            "state_alerts": [],
            "full_alerts": [],
            "raw": [],
            "state_conditioned": [],
            "full_conditioned": [],
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

            state_model = _fit_model(
                state_features[train_indices],
                target[train_indices],
                cold_types[train_indices],
                final_home[train_indices],
                final_away[train_indices],
                config,
            )
            full_model = _fit_model(
                full_features[train_indices],
                target[train_indices],
                cold_types[train_indices],
                final_home[train_indices],
                final_away[train_indices],
                config,
            )
            state_output = _predict_fold(
                state_model, state_features[test_indices], config
            )
            full_output = _predict_fold(
                full_model, full_features[test_indices], config
            )

            home_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_home[train_indices],
                base_home[train_indices],
                random_state=601 + cutoff,
            )
            away_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_away[train_indices],
                base_away[train_indices],
                random_state=611 + cutoff,
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
            parts["state_alerts"].append(state_output.alerts)
            parts["full_alerts"].append(full_output.alerts)
            parts["raw"].append(raw)
            parts["state_conditioned"].append(conditioned)
            parts["full_conditioned"].append(conditioned)
            fold_details.append(
                {
                    "held_out_league": int(held_out),
                    "cutoff": int(cutoff),
                    "test_matches": int(test_mask.sum()),
                    "state_alerts": asdict(
                        alert_metrics(state_output.alerts, target[test_indices])
                    ),
                    "full_alerts": asdict(
                        alert_metrics(full_output.alerts, target[test_indices])
                    ),
                    "family": family[0],
                    "home_blend": home_model.blend,
                    "away_blend": away_model.blend,
                }
            )

    results: list[ColdBlowoutModelCutoffResult] = []
    for cutoff, parts in collected.items():
        targets = np.concatenate(parts["target"])
        home = np.concatenate(parts["home"])
        away = np.concatenate(parts["away"])
        state_alerts = np.concatenate(parts["state_alerts"])
        full_alerts = np.concatenate(parts["full_alerts"])
        raw = np.concatenate(parts["raw"])
        state_conditioned = np.concatenate(parts["state_conditioned"])
        full_conditioned = np.concatenate(parts["full_conditioned"])
        results.append(
            ColdBlowoutModelCutoffResult(
                cutoff=int(cutoff),
                matches=len(targets),
                state_alerts=asdict(alert_metrics(state_alerts, targets)),
                full_alerts=asdict(alert_metrics(full_alerts, targets)),
                state_raw_scores=_score_metrics(
                    raw, state_alerts, targets.astype(bool), home, away
                ),
                state_conditioned_scores=_score_metrics(
                    state_conditioned,
                    state_alerts,
                    targets.astype(bool),
                    home,
                    away,
                ),
                full_raw_scores=_score_metrics(
                    raw, full_alerts, targets.astype(bool), home, away
                ),
                full_conditioned_scores=_score_metrics(
                    full_conditioned,
                    full_alerts,
                    targets.astype(bool),
                    home,
                    away,
                ),
            )
        )
    return ColdBlowoutModelReport(tuple(results), tuple(fold_details))
