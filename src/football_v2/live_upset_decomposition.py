from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

from .event_tail_model import AlertMetrics, EventTailConfig, alert_metrics
from .live_snapshots import LiveSnapshotDataset
from .tail_score_classifier import TailExactScoreModel, score_ranking_metrics


@dataclass(frozen=True)
class DecomposedCutoffResult:
    cutoff: int
    matches: int
    full_alerts: AlertMetrics
    state_alerts: AlertMetrics
    full_exact: dict[str, float | int | None]
    state_exact: dict[str, float | int | None]


@dataclass(frozen=True)
class DecomposedUpsetReport:
    cutoffs: tuple[DecomposedCutoffResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
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


@dataclass
class _BinaryModel:
    imputer: SimpleImputer
    classifier: HistGradientBoostingClassifier

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.imputer.transform(features))[:, 1]


@dataclass(frozen=True)
class _Gate:
    tail_power: float
    upset_power: float
    threshold: float
    calibration: AlertMetrics


def _weights(target: np.ndarray) -> np.ndarray:
    rate = float(np.mean(target))
    if rate <= 0 or rate >= 1:
        return np.ones(len(target), dtype=float)
    return np.where(target == 1, min(12.0, (1.0 - rate) / rate), 1.0)


def _fit_binary(
    features: np.ndarray,
    target: np.ndarray,
    config: EventTailConfig,
    *,
    random_state: int,
) -> _BinaryModel:
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    transformed = imputer.fit_transform(features)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=config.max_iter,
        max_leaf_nodes=15,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=1.0,
        random_state=random_state,
    )
    classifier.fit(transformed, target, sample_weight=_weights(target))
    return _BinaryModel(imputer, classifier)


def _derived_features(frame: object) -> np.ndarray:
    home_underdog = (
        frame["market_home_prob"].to_numpy(dtype=float)
        < frame["market_away_prob"].to_numpy(dtype=float)
    )
    sign = np.where(home_underdog, 1.0, -1.0)
    score_diff = sign * (
        frame["live_home_score"].to_numpy(dtype=float)
        - frame["live_away_score"].to_numpy(dtype=float)
    )
    xg_diff = sign * (
        frame["live_home_xg"].to_numpy(dtype=float)
        - frame["live_away_xg"].to_numpy(dtype=float)
    )
    shot_diff = sign * (
        frame["live_home_shots"].to_numpy(dtype=float)
        - frame["live_away_shots"].to_numpy(dtype=float)
    )
    target_diff = sign * (
        frame["live_home_shots_on_target"].to_numpy(dtype=float)
        - frame["live_away_shots_on_target"].to_numpy(dtype=float)
    )
    red_diff = sign * (
        frame["live_home_red_cards"].to_numpy(dtype=float)
        - frame["live_away_red_cards"].to_numpy(dtype=float)
    )
    box_diff = sign * (
        frame["live_home_box_entries"].to_numpy(dtype=float)
        - frame["live_away_box_entries"].to_numpy(dtype=float)
    )
    return np.column_stack(
        [
            home_underdog.astype(float),
            score_diff,
            xg_diff,
            shot_diff,
            target_diff,
            red_diff,
            box_diff,
            score_diff * frame["remaining_minutes"].to_numpy(dtype=float) / 90.0,
        ]
    )


def _state_mask(columns: tuple[str, ...]) -> np.ndarray:
    keep = {
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
        [column.startswith("market_") or column in keep for column in columns],
        dtype=bool,
    )


def _targets(dataset: LiveSnapshotDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = dataset.frame
    home_underdog = (
        frame["market_home_prob"].to_numpy(dtype=float)
        < frame["market_away_prob"].to_numpy(dtype=float)
    )
    home_win = frame["home_score"].to_numpy(dtype=int) > frame["away_score"].to_numpy(
        dtype=int
    )
    away_win = frame["away_score"].to_numpy(dtype=int) > frame["home_score"].to_numpy(
        dtype=int
    )
    underdog_win = np.where(home_underdog, home_win, away_win).astype(int)
    tail = frame["final_jackpot_target"].to_numpy(dtype=int)
    combined = np.logical_and(tail == 1, underdog_win == 1).astype(int)
    return tail, underdog_win, combined


def _choose_gate(
    tail_probability: np.ndarray,
    upset_probability: np.ndarray,
    combined_target: np.ndarray,
) -> _Gate:
    candidates: list[tuple[tuple[float, float, float, float], _Gate]] = []
    midpoint = len(combined_target) // 2
    for tail_power in (0.5, 1.0, 1.5, 2.0):
        for upset_power in (0.5, 1.0, 1.5, 2.0):
            score = np.power(tail_probability, tail_power) * np.power(
                upset_probability, upset_power
            )
            for quantile in (0.97, 0.975, 0.98, 0.985, 0.99, 0.995):
                threshold = float(np.quantile(score, quantile))
                alerts = score >= threshold
                metrics = alert_metrics(alerts, combined_target)
                halves = (
                    alert_metrics(alerts[:midpoint], combined_target[:midpoint]),
                    alert_metrics(alerts[midpoint:], combined_target[midpoint:]),
                )
                robust = (
                    metrics.alerts >= 4
                    and metrics.coverage <= 0.035
                    and metrics.lift is not None
                    and metrics.lift >= 3.0
                    and all(
                        half.alerts >= 1
                        and half.lift is not None
                        and half.lift >= 1.5
                        for half in halves
                    )
                )
                ranking = (
                    1.0 if robust else 0.0,
                    metrics.lift if metrics.lift is not None else -1.0,
                    metrics.precision if metrics.precision is not None else -1.0,
                    metrics.recall,
                )
                candidates.append(
                    (
                        ranking,
                        _Gate(
                            tail_power=tail_power,
                            upset_power=upset_power,
                            threshold=threshold,
                            calibration=metrics,
                        ),
                    )
                )
    best_score, best_gate = max(candidates, key=lambda item: item[0])
    if best_score[0] == 0.0:
        return _Gate(1.0, 1.0, 1.0, alert_metrics(np.zeros(len(combined_target), dtype=bool), combined_target))
    return best_gate


def leave_one_league_out_decomposed_upset_test(
    dataset: LiveSnapshotDataset,
    *,
    config: EventTailConfig | None = None,
) -> DecomposedUpsetReport:
    config = config or EventTailConfig(
        max_iter=180,
        min_samples_leaf=20,
    )
    frame = dataset.frame
    league = frame["division_id"].to_numpy(dtype=int)
    cutoffs = frame["snapshot_minute"].to_numpy(dtype=int)
    home = frame["home_score"].to_numpy(dtype=int)
    away = frame["away_score"].to_numpy(dtype=int)
    tail, underdog_win, combined = _targets(dataset)
    derived = _derived_features(frame)
    full_features = np.column_stack([dataset.features, derived])
    state_mask = _state_mask(dataset.feature_columns)
    state_features = np.column_stack([dataset.features[:, state_mask], derived])

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
    fold_details: list[dict[str, object]] = []

    for held_out in sorted(np.unique(league)):
        train_mask = league != held_out
        test_mask = league == held_out
        train_order = np.lexsort(
            (
                frame.loc[train_mask, "snapshot_minute"].to_numpy(),
                frame.loc[train_mask, "match_id"].to_numpy(),
                frame.loc[train_mask, "date"].to_numpy(dtype="datetime64[ns]"),
            )
        )
        test_order = np.lexsort(
            (
                frame.loc[test_mask, "snapshot_minute"].to_numpy(),
                frame.loc[test_mask, "match_id"].to_numpy(),
                frame.loc[test_mask, "date"].to_numpy(dtype="datetime64[ns]"),
            )
        )
        calibration_size = max(500, int(train_mask.sum() * 0.20))
        split = int(train_mask.sum()) - calibration_size
        if split < 2500:
            continue

        train_full = full_features[train_mask][train_order]
        test_full = full_features[test_mask][test_order]
        train_state = state_features[train_mask][train_order]
        test_state = state_features[test_mask][test_order]
        train_tail = tail[train_mask][train_order]
        train_upset = underdog_win[train_mask][train_order]
        train_combined = combined[train_mask][train_order]
        test_combined = combined[test_mask][test_order]
        test_cutoffs = cutoffs[test_mask][test_order]
        test_home = home[test_mask][test_order]
        test_away = away[test_mask][test_order]

        full_tail_model = _fit_binary(
            train_full[:split], train_tail[:split], config, random_state=41
        )
        full_upset_model = _fit_binary(
            train_full[:split], train_upset[:split], config, random_state=42
        )
        state_tail_model = _fit_binary(
            train_state[:split], train_tail[:split], config, random_state=43
        )
        state_upset_model = _fit_binary(
            train_state[:split], train_upset[:split], config, random_state=44
        )
        full_gate = _choose_gate(
            full_tail_model.predict(train_full[split:]),
            full_upset_model.predict(train_full[split:]),
            train_combined[split:],
        )
        state_gate = _choose_gate(
            state_tail_model.predict(train_state[split:]),
            state_upset_model.predict(train_state[split:]),
            train_combined[split:],
        )
        full_score = np.power(
            full_tail_model.predict(test_full), full_gate.tail_power
        ) * np.power(full_upset_model.predict(test_full), full_gate.upset_power)
        state_score = np.power(
            state_tail_model.predict(test_state), state_gate.tail_power
        ) * np.power(state_upset_model.predict(test_state), state_gate.upset_power)
        full_alerts = full_score >= full_gate.threshold
        state_alerts = state_score >= state_gate.threshold

        score_model_full = TailExactScoreModel(
            random_state=51, n_estimators=120
        ).fit(
            train_full,
            home[train_mask][train_order],
            away[train_mask][train_order],
            train_tail,
        )
        score_model_state = TailExactScoreModel(
            random_state=52, n_estimators=120
        ).fit(
            train_state,
            home[train_mask][train_order],
            away[train_mask][train_order],
            train_tail,
        )
        full_distributions = score_model_full.predict_distribution(test_full)
        state_distributions = score_model_state.predict_distribution(test_state)

        fold_details.append(
            {
                "held_out_league": int(held_out),
                "test_snapshots": int(test_mask.sum()),
                "full_gate": {
                    "tail_power": full_gate.tail_power,
                    "upset_power": full_gate.upset_power,
                    "threshold": full_gate.threshold,
                    "calibration": asdict(full_gate.calibration),
                },
                "state_gate": {
                    "tail_power": state_gate.tail_power,
                    "upset_power": state_gate.upset_power,
                    "threshold": state_gate.threshold,
                    "calibration": asdict(state_gate.calibration),
                },
                "full_test": asdict(alert_metrics(full_alerts, test_combined)),
                "state_test": asdict(alert_metrics(state_alerts, test_combined)),
            }
        )
        for cutoff in collected:
            mask = test_cutoffs == cutoff
            collected[cutoff]["target"].append(test_combined[mask])
            collected[cutoff]["home"].append(test_home[mask])
            collected[cutoff]["away"].append(test_away[mask])
            collected[cutoff]["full_alerts"].append(full_alerts[mask])
            collected[cutoff]["state_alerts"].append(state_alerts[mask])
            collected[cutoff]["full_distributions"].append(full_distributions[mask])
            collected[cutoff]["state_distributions"].append(state_distributions[mask])

    results: list[DecomposedCutoffResult] = []
    for cutoff, parts in collected.items():
        target_values = np.concatenate(parts["target"])
        home_values = np.concatenate(parts["home"])
        away_values = np.concatenate(parts["away"])
        full_alert_values = np.concatenate(parts["full_alerts"])
        state_alert_values = np.concatenate(parts["state_alerts"])
        full_distributions = np.concatenate(parts["full_distributions"])
        state_distributions = np.concatenate(parts["state_distributions"])
        results.append(
            DecomposedCutoffResult(
                cutoff=cutoff,
                matches=len(target_values),
                full_alerts=alert_metrics(full_alert_values, target_values),
                state_alerts=alert_metrics(state_alert_values, target_values),
                full_exact=score_ranking_metrics(
                    full_distributions,
                    full_alert_values,
                    home_values,
                    away_values,
                ),
                state_exact=score_ranking_metrics(
                    state_distributions,
                    state_alert_values,
                    home_values,
                    away_values,
                ),
            )
        )
    return DecomposedUpsetReport(tuple(results), tuple(fold_details))
