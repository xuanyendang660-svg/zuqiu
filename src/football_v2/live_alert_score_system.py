from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .event_tail_model import EventTailConfig, _fit_model, _predict_fold, alert_metrics
from .labels import ScoreArchetype, is_jackpot_tail, jackpot_tail_type
from .live_dynamic_model import _state_mask
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
class AlertScoreResult:
    cutoff: int
    alerts: int
    alert_precision: float | None
    alert_lift: float | None
    raw_scores: ScoreDistributionMetrics | None
    boosted_scores: ScoreDistributionMetrics | None
    true_tail_raw_scores: ScoreDistributionMetrics | None
    true_tail_boosted_scores: ScoreDistributionMetrics | None
    mean_tail_boost: float
    mean_type_boost: float


@dataclass(frozen=True)
class AlertScoreReport:
    cutoffs: tuple[AlertScoreResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoffs": [
                {
                    "cutoff": result.cutoff,
                    "alerts": result.alerts,
                    "alert_precision": result.alert_precision,
                    "alert_lift": result.alert_lift,
                    "raw_scores": (
                        asdict(result.raw_scores) if result.raw_scores else None
                    ),
                    "boosted_scores": (
                        asdict(result.boosted_scores)
                        if result.boosted_scores
                        else None
                    ),
                    "true_tail_raw_scores": (
                        asdict(result.true_tail_raw_scores)
                        if result.true_tail_raw_scores
                        else None
                    ),
                    "true_tail_boosted_scores": (
                        asdict(result.true_tail_boosted_scores)
                        if result.true_tail_boosted_scores
                        else None
                    ),
                    "mean_tail_boost": result.mean_tail_boost,
                    "mean_type_boost": result.mean_type_boost,
                }
                for result in self.cutoffs
            ],
            "fold_details": list(self.fold_details),
        }


def _adjust_matrices(
    matrices: np.ndarray,
    alerts: np.ndarray,
    predicted_types: np.ndarray,
    *,
    tail_boost: float,
    type_boost: float,
) -> np.ndarray:
    output = matrices.copy()
    maximum_home = matrices.shape[1]
    maximum_away = matrices.shape[2]
    tail_mask = np.zeros((maximum_home, maximum_away), dtype=bool)
    type_masks: dict[str, np.ndarray] = {
        archetype.value: np.zeros_like(tail_mask)
        for archetype in (
            ScoreArchetype.HOME_BLOWOUT,
            ScoreArchetype.AWAY_BLOWOUT,
            ScoreArchetype.SHOOTOUT,
        )
    }
    for home in range(maximum_home):
        for away in range(maximum_away):
            if is_jackpot_tail(home, away):
                tail_mask[home, away] = True
                type_masks[jackpot_tail_type(home, away).value][home, away] = True
    for row, alert in enumerate(alerts):
        if not alert:
            continue
        output[row][tail_mask] *= tail_boost
        predicted = str(predicted_types[row])
        if predicted in type_masks:
            output[row][type_masks[predicted]] *= type_boost
        output[row] /= output[row].sum()
    return output


def _selection_score(metrics: ScoreDistributionMetrics) -> tuple[float, ...]:
    return (
        metrics.exact_accuracy,
        metrics.top3_accuracy,
        metrics.top5_accuracy,
        -metrics.negative_log_likelihood,
        -metrics.mean_rank,
    )


def _choose_boosts(
    matrices: np.ndarray,
    alerts: np.ndarray,
    predicted_types: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> tuple[float, float]:
    if int(alerts.sum()) < 5:
        return 1.0, 1.0
    candidates: list[tuple[tuple[float, ...], float, float]] = []
    for tail_boost in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        for type_boost in (1.0, 1.5, 2.0, 3.0, 5.0):
            adjusted = _adjust_matrices(
                matrices,
                alerts,
                predicted_types,
                tail_boost=tail_boost,
                type_boost=type_boost,
            )
            metrics = _distribution_metrics(
                adjusted[alerts], home_goals[alerts], away_goals[alerts]
            )
            candidates.append(
                (_selection_score(metrics), tail_boost, type_boost)
            )
    _, tail_boost, type_boost = max(candidates, key=lambda item: item[0])
    return float(tail_boost), float(type_boost)


def _optional_metrics(
    matrices: np.ndarray,
    mask: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> ScoreDistributionMetrics | None:
    if not mask.any():
        return None
    return _distribution_metrics(
        matrices[mask], home_goals[mask], away_goals[mask]
    )


def leave_one_league_out_alert_score_test(
    dataset: LiveSnapshotDataset,
    *,
    cutoffs_to_test: tuple[int, ...] = (15, 30, 45),
) -> AlertScoreReport:
    frame = dataset.frame
    league = frame["division_id"].to_numpy(dtype=int)
    cutoffs = frame["snapshot_minute"].to_numpy(dtype=int)
    tail_target = frame["final_jackpot_target"].to_numpy(dtype=int)
    tail_types = frame["final_jackpot_type"].to_numpy(dtype=str)
    final_home = frame["home_score"].to_numpy(dtype=int)
    final_away = frame["away_score"].to_numpy(dtype=int)
    current_home = frame["live_home_score"].to_numpy(dtype=int)
    current_away = frame["live_away_score"].to_numpy(dtype=int)
    remaining_home = frame["remaining_home_goals"].to_numpy(dtype=int)
    remaining_away = frame["remaining_away_goals"].to_numpy(dtype=int)
    base_home, base_away = _market_rates_for_frame(frame)
    state_features = dataset.features[:, _state_mask(dataset.feature_columns)]
    dynamic_features = dataset.features[:, _dynamic_feature_mask(dataset.feature_columns)]

    collected: dict[int, dict[str, list[object]]] = {
        cutoff: {
            "alerts": [],
            "target": [],
            "home": [],
            "away": [],
            "raw": [],
            "boosted": [],
            "tail_boosts": [],
            "type_boosts": [],
        }
        for cutoff in cutoffs_to_test
    }
    fold_details: list[dict[str, object]] = []
    classifier_config = EventTailConfig(
        max_alert_coverage=0.08,
        minimum_alerts=6,
        minimum_lift=1.25,
        max_iter=180,
        min_samples_leaf=24,
    )

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
            calibration_size = max(250, int(len(train_indices) * 0.20))
            split = len(train_indices) - calibration_size
            fit_indices = train_indices[:split]
            calibration_indices = train_indices[split:]

            calibration_classifier = _fit_model(
                state_features[fit_indices],
                tail_target[fit_indices],
                tail_types[fit_indices],
                final_home[fit_indices],
                final_away[fit_indices],
                classifier_config,
            )
            calibration_output = _predict_fold(
                calibration_classifier,
                state_features[calibration_indices],
                classifier_config,
            )
            final_classifier = _fit_model(
                state_features[train_indices],
                tail_target[train_indices],
                tail_types[train_indices],
                final_home[train_indices],
                final_away[train_indices],
                classifier_config,
            )
            test_output = _predict_fold(
                final_classifier,
                state_features[test_indices],
                classifier_config,
            )

            calibration_home_model = _fit_regressor(
                dynamic_features[fit_indices],
                remaining_home[fit_indices],
                base_home[fit_indices],
                random_state=201 + cutoff,
            )
            calibration_away_model = _fit_regressor(
                dynamic_features[fit_indices],
                remaining_away[fit_indices],
                base_away[fit_indices],
                random_state=211 + cutoff,
            )
            calibration_home_mean = calibration_home_model.predict(
                dynamic_features[calibration_indices],
                base_home[calibration_indices],
            )
            calibration_away_mean = calibration_away_model.predict(
                dynamic_features[calibration_indices],
                base_away[calibration_indices],
            )
            calibration_family = _choose_family(
                remaining_home[fit_indices],
                remaining_away[fit_indices],
                calibration_home_model.predict(
                    dynamic_features[fit_indices], base_home[fit_indices]
                ),
                calibration_away_model.predict(
                    dynamic_features[fit_indices], base_away[fit_indices]
                ),
            )
            calibration_matrices = _score_matrices(
                current_home[calibration_indices],
                current_away[calibration_indices],
                calibration_home_mean,
                calibration_away_mean,
                family=calibration_family[0],
                home_dispersion=calibration_family[1],
                away_dispersion=calibration_family[2],
            )
            tail_boost, type_boost = _choose_boosts(
                calibration_matrices,
                calibration_output.alerts,
                calibration_output.type_predictions,
                final_home[calibration_indices],
                final_away[calibration_indices],
            )

            home_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_home[train_indices],
                base_home[train_indices],
                random_state=221 + cutoff,
            )
            away_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_away[train_indices],
                base_away[train_indices],
                random_state=231 + cutoff,
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
            test_home_mean = home_model.predict(
                dynamic_features[test_indices], base_home[test_indices]
            )
            test_away_mean = away_model.predict(
                dynamic_features[test_indices], base_away[test_indices]
            )
            raw_matrices = _score_matrices(
                current_home[test_indices],
                current_away[test_indices],
                test_home_mean,
                test_away_mean,
                family=family[0],
                home_dispersion=family[1],
                away_dispersion=family[2],
            )
            boosted_matrices = _adjust_matrices(
                raw_matrices,
                test_output.alerts,
                test_output.type_predictions,
                tail_boost=tail_boost,
                type_boost=type_boost,
            )
            parts = collected[cutoff]
            parts["alerts"].append(test_output.alerts)
            parts["target"].append(tail_target[test_indices])
            parts["home"].append(final_home[test_indices])
            parts["away"].append(final_away[test_indices])
            parts["raw"].append(raw_matrices)
            parts["boosted"].append(boosted_matrices)
            parts["tail_boosts"].append(tail_boost)
            parts["type_boosts"].append(type_boost)
            fold_details.append(
                {
                    "held_out_league": int(held_out),
                    "cutoff": int(cutoff),
                    "test_matches": int(test_mask.sum()),
                    "test_alerts": asdict(
                        alert_metrics(
                            test_output.alerts, tail_target[test_indices]
                        )
                    ),
                    "family": family[0],
                    "home_blend": home_model.blend,
                    "away_blend": away_model.blend,
                    "tail_boost": tail_boost,
                    "type_boost": type_boost,
                }
            )

    results: list[AlertScoreResult] = []
    for cutoff, parts in collected.items():
        alerts = np.concatenate(parts["alerts"])
        targets = np.concatenate(parts["target"])
        home_values = np.concatenate(parts["home"])
        away_values = np.concatenate(parts["away"])
        raw = np.concatenate(parts["raw"])
        boosted = np.concatenate(parts["boosted"])
        metrics = alert_metrics(alerts, targets)
        true_tail_alerts = np.logical_and(alerts, targets == 1)
        results.append(
            AlertScoreResult(
                cutoff=int(cutoff),
                alerts=int(alerts.sum()),
                alert_precision=metrics.precision,
                alert_lift=metrics.lift,
                raw_scores=_optional_metrics(
                    raw, alerts, home_values, away_values
                ),
                boosted_scores=_optional_metrics(
                    boosted, alerts, home_values, away_values
                ),
                true_tail_raw_scores=_optional_metrics(
                    raw, true_tail_alerts, home_values, away_values
                ),
                true_tail_boosted_scores=_optional_metrics(
                    boosted,
                    true_tail_alerts,
                    home_values,
                    away_values,
                ),
                mean_tail_boost=float(np.mean(parts["tail_boosts"])),
                mean_type_boost=float(np.mean(parts["type_boosts"])),
            )
        )
    return AlertScoreReport(tuple(results), tuple(fold_details))
