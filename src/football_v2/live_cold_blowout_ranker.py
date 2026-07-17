from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from .event_tail_model import AlertMetrics, alert_metrics
from .live_cold_blowout_cluster import (
    ClusterScoreMetrics,
    _condition_on_cold_blowout,
    _role_data,
    _score_metrics,
)
from .live_mechanism_rules import _rules
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
class BudgetResult:
    budget: float
    alerts: AlertMetrics
    conditioned_scores: ClusterScoreMetrics


@dataclass(frozen=True)
class RankerResult:
    name: str
    budgets: tuple[BudgetResult, ...]


@dataclass(frozen=True)
class ColdBlowoutRankerReport:
    cutoff: int
    matches: int
    base_rate: float
    rankers: tuple[RankerResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoff": self.cutoff,
            "matches": self.matches,
            "base_rate": self.base_rate,
            "rankers": [
                {
                    "name": ranker.name,
                    "budgets": [
                        {
                            "budget": result.budget,
                            "alerts": asdict(result.alerts),
                            "conditioned_scores": _cluster_to_dict(
                                result.conditioned_scores
                            ),
                        }
                        for result in ranker.budgets
                    ],
                }
                for ranker in self.rankers
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


@dataclass
class _BinaryProbabilityModel:
    imputer: SimpleImputer
    classifier: HistGradientBoostingClassifier

    def predict(self, features: np.ndarray) -> np.ndarray:
        transformed = self.imputer.transform(features)
        return self.classifier.predict_proba(transformed)[:, 1]


def _sample_weights(target: np.ndarray) -> np.ndarray:
    rate = float(np.mean(target))
    if rate <= 0 or rate >= 1:
        return np.ones(len(target), dtype=float)
    return np.where(target == 1, min(10.0, (1.0 - rate) / rate), 1.0)


def _fit_binary_model(
    features: np.ndarray,
    target: np.ndarray,
    *,
    random_state: int,
) -> _BinaryProbabilityModel:
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    transformed = imputer.fit_transform(features)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.5,
        random_state=random_state,
    )
    classifier.fit(
        transformed,
        target,
        sample_weight=_sample_weights(target),
    )
    return _BinaryProbabilityModel(imputer, classifier)


def _role_targets(
    frame: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    underdog_home = (
        frame["market_home_prob"].to_numpy(dtype=float)
        < frame["market_away_prob"].to_numpy(dtype=float)
    )
    home_score = frame["home_score"].to_numpy(dtype=int)
    away_score = frame["away_score"].to_numpy(dtype=int)
    underdog_score = np.where(underdog_home, home_score, away_score)
    favorite_score = np.where(underdog_home, away_score, home_score)
    score_three_plus = (underdog_score >= 3).astype(int)
    win_two_plus = (underdog_score - favorite_score >= 2).astype(int)
    target = np.logical_and(score_three_plus == 1, win_two_plus == 1).astype(int)
    return score_three_plus, win_two_plus, target


def _mechanism_matrix(frame: object) -> np.ndarray:
    rules = _rules(frame)
    names = (
        "underdog_lead_two",
        "underdog_two_goals_ahead",
        "underdog_lead_pressure",
        "underdog_lead_open_game",
        "underdog_two_goals_resistance",
        "underdog_lead_recent_chaos",
        "underdog_tied_dominant",
    )
    matrix = np.column_stack(
        [np.asarray(rules[name], dtype=float) for name in names]
    )
    return np.column_stack(
        [
            matrix,
            matrix.sum(axis=1),
            np.max(matrix, axis=1),
        ]
    )


def _base_features(frame: object) -> np.ndarray:
    role = _role_features(frame)
    mechanisms = _mechanism_matrix(frame)
    direct = np.column_stack(
        [
            frame["snapshot_minute"].to_numpy(dtype=float),
            frame["remaining_minutes"].to_numpy(dtype=float),
            frame["market_home_prob"].to_numpy(dtype=float),
            frame["market_draw_prob"].to_numpy(dtype=float),
            frame["market_away_prob"].to_numpy(dtype=float),
            frame["market_over25_prob"].to_numpy(dtype=float),
            frame["market_entropy"].to_numpy(dtype=float),
            frame["market_underdog_prob"].to_numpy(dtype=float),
            frame["market_favorite_prob"].to_numpy(dtype=float),
            frame["live_home_score"].to_numpy(dtype=float),
            frame["live_away_score"].to_numpy(dtype=float),
            frame["live_total_goals"].to_numpy(dtype=float),
            frame["goals_last10"].to_numpy(dtype=float),
            frame["shots_last10"].to_numpy(dtype=float),
            frame["xg_last10"].to_numpy(dtype=float),
        ]
    )
    return np.column_stack([direct, role, mechanisms])


def _fit_probability_triplet(
    features: np.ndarray,
    score_three_plus: np.ndarray,
    win_two_plus: np.ndarray,
    target: np.ndarray,
    *,
    random_state: int,
) -> tuple[
    _BinaryProbabilityModel,
    _BinaryProbabilityModel,
    _BinaryProbabilityModel,
]:
    direct = _fit_binary_model(
        features,
        target,
        random_state=random_state,
    )
    score_model = _fit_binary_model(
        features,
        score_three_plus,
        random_state=random_state + 1,
    )
    margin_model = _fit_binary_model(
        features,
        win_two_plus,
        random_state=random_state + 2,
    )
    return direct, score_model, margin_model


def _meta_features(
    direct_probability: np.ndarray,
    score_probability: np.ndarray,
    margin_probability: np.ndarray,
    mechanisms: np.ndarray,
    frame: object,
) -> np.ndarray:
    product = score_probability * margin_probability
    minimum = np.minimum(score_probability, margin_probability)
    maximum = np.maximum(score_probability, margin_probability)
    role = _role_features(frame)
    compact_role = role[:, : min(18, role.shape[1])]
    return np.column_stack(
        [
            direct_probability,
            score_probability,
            margin_probability,
            product,
            minimum,
            maximum,
            direct_probability * product,
            mechanisms,
            compact_role,
        ]
    )


def _inner_oof_scores(
    features: np.ndarray,
    frame: object,
    league: np.ndarray,
    outer_train_mask: np.ndarray,
    score_three_plus: np.ndarray,
    win_two_plus: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outer_indices = np.where(outer_train_mask)[0]
    oof_meta = np.full((len(outer_indices), 1), np.nan, dtype=float)
    oof_target = target[outer_indices]
    position_by_index = {
        int(original): position for position, original in enumerate(outer_indices)
    }
    meta_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_held_out in sorted(np.unique(league[outer_train_mask])):
        inner_fit_mask = np.logical_and(
            outer_train_mask,
            league != inner_held_out,
        )
        inner_validation_mask = np.logical_and(
            outer_train_mask,
            league == inner_held_out,
        )
        fit_indices = np.where(inner_fit_mask)[0]
        validation_indices = np.where(inner_validation_mask)[0]
        if len(fit_indices) < 900 or len(validation_indices) < 250:
            continue
        models = _fit_probability_triplet(
            features[fit_indices],
            score_three_plus[fit_indices],
            win_two_plus[fit_indices],
            target[fit_indices],
            random_state=701 + int(inner_held_out) * 11,
        )
        direct_probability = models[0].predict(features[validation_indices])
        score_probability = models[1].predict(features[validation_indices])
        margin_probability = models[2].predict(features[validation_indices])
        mechanisms = _mechanism_matrix(frame.iloc[validation_indices])
        meta = _meta_features(
            direct_probability,
            score_probability,
            margin_probability,
            mechanisms,
            frame.iloc[validation_indices],
        )
        meta_rows.append((validation_indices, meta))
    if not meta_rows:
        raise RuntimeError("no inner league OOF folds were produced")
    meta_width = meta_rows[0][1].shape[1]
    oof_meta = np.full((len(outer_indices), meta_width), np.nan, dtype=float)
    for validation_indices, meta in meta_rows:
        positions = [position_by_index[int(index)] for index in validation_indices]
        oof_meta[positions] = meta
    valid = np.all(np.isfinite(oof_meta), axis=1)
    if int(valid.sum()) < 900:
        raise RuntimeError("insufficient complete OOF meta rows")
    return oof_meta[valid], oof_target[valid]


def _fit_meta_model(
    meta_features: np.ndarray,
    target: np.ndarray,
) -> tuple[SimpleImputer, LogisticRegression]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(meta_features)
    model = LogisticRegression(
        C=0.35,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=811,
    )
    model.fit(transformed, target)
    return imputer, model


def _budget_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    count = max(1, int(ceil(len(scores) * budget)))
    order = np.argsort(scores)[::-1]
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:count]] = True
    return mask


def _ranker_scores(
    name: str,
    direct_probability: np.ndarray,
    score_probability: np.ndarray,
    margin_probability: np.ndarray,
    stacked_probability: np.ndarray,
    mechanism_matrix: np.ndarray,
) -> np.ndarray:
    if name == "direct":
        return direct_probability
    if name == "decomposed":
        return score_probability * margin_probability
    if name == "mechanism":
        return mechanism_matrix.sum(axis=1)
    if name == "stacked":
        return stacked_probability
    raise ValueError(f"unknown ranker: {name}")


def leave_one_league_out_cold_blowout_ranker_test(
    dataset: LiveSnapshotDataset,
    *,
    cutoff: int = 30,
    budgets: tuple[float, ...] = (0.02, 0.04, 0.06),
) -> ColdBlowoutRankerReport:
    frame = dataset.frame
    cutoff_mask = frame["snapshot_minute"].to_numpy(dtype=int) == cutoff
    frame = frame.loc[cutoff_mask].reset_index(drop=True)
    dataset_features = dataset.features[cutoff_mask]
    dataset = LiveSnapshotDataset(frame, dataset.feature_columns)
    role_features = _base_features(frame)
    features = np.column_stack([dataset_features, role_features])
    score_three_plus, win_two_plus, target = _role_targets(frame)
    league = frame["division_id"].to_numpy(dtype=int)
    final_home = frame["home_score"].to_numpy(dtype=int)
    final_away = frame["away_score"].to_numpy(dtype=int)
    current_home = frame["live_home_score"].to_numpy(dtype=int)
    current_away = frame["live_away_score"].to_numpy(dtype=int)
    remaining_home = frame["remaining_home_goals"].to_numpy(dtype=int)
    remaining_away = frame["remaining_away_goals"].to_numpy(dtype=int)
    underdog_home, _, _ = _role_data(frame)
    base_home, base_away = _market_rates_for_frame(frame)
    dynamic_features = dataset.features[:, _dynamic_feature_mask(dataset.feature_columns)]
    mechanisms_all = _mechanism_matrix(frame)

    ranker_names = ("direct", "decomposed", "mechanism", "stacked")
    collected: dict[str, dict[float, dict[str, list[np.ndarray]]]] = {
        name: {
            budget: {
                "alerts": [],
                "target": [],
                "home": [],
                "away": [],
                "conditioned": [],
            }
            for budget in budgets
        }
        for name in ranker_names
    }
    fold_details: list[dict[str, object]] = []

    for outer_held_out in sorted(np.unique(league)):
        outer_train_mask = league != outer_held_out
        outer_test_mask = league == outer_held_out
        train_indices = np.where(outer_train_mask)[0]
        test_indices = np.where(outer_test_mask)[0]
        if len(train_indices) < 1200 or len(test_indices) < 250:
            continue

        oof_meta, oof_target = _inner_oof_scores(
            features,
            frame,
            league,
            outer_train_mask,
            score_three_plus,
            win_two_plus,
            target,
        )
        meta_imputer, meta_model = _fit_meta_model(oof_meta, oof_target)
        models = _fit_probability_triplet(
            features[train_indices],
            score_three_plus[train_indices],
            win_two_plus[train_indices],
            target[train_indices],
            random_state=901 + int(outer_held_out) * 13,
        )
        direct_probability = models[0].predict(features[test_indices])
        score_probability = models[1].predict(features[test_indices])
        margin_probability = models[2].predict(features[test_indices])
        test_mechanisms = mechanisms_all[test_indices]
        test_meta = _meta_features(
            direct_probability,
            score_probability,
            margin_probability,
            test_mechanisms,
            frame.iloc[test_indices],
        )
        stacked_probability = meta_model.predict_proba(
            meta_imputer.transform(test_meta)
        )[:, 1]

        home_model = _fit_regressor(
            dynamic_features[train_indices],
            remaining_home[train_indices],
            base_home[train_indices],
            random_state=1001 + cutoff,
        )
        away_model = _fit_regressor(
            dynamic_features[train_indices],
            remaining_away[train_indices],
            base_away[train_indices],
            random_state=1011 + cutoff,
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
        raw_matrices = _score_matrices(
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
            raw_matrices,
            underdog_home[test_indices],
        )

        fold_entry: dict[str, object] = {
            "held_out_league": int(outer_held_out),
            "test_matches": len(test_indices),
            "base_rate": float(np.mean(target[test_indices])),
            "rankers": {},
        }
        for name in ranker_names:
            scores = _ranker_scores(
                name,
                direct_probability,
                score_probability,
                margin_probability,
                stacked_probability,
                test_mechanisms,
            )
            fold_entry["rankers"][name] = {}
            for budget in budgets:
                alerts = _budget_mask(scores, budget)
                metrics = alert_metrics(alerts, target[test_indices])
                fold_entry["rankers"][name][str(budget)] = asdict(metrics)
                parts = collected[name][budget]
                parts["alerts"].append(alerts)
                parts["target"].append(target[test_indices])
                parts["home"].append(final_home[test_indices])
                parts["away"].append(final_away[test_indices])
                parts["conditioned"].append(conditioned)
        fold_details.append(fold_entry)

    ranker_results: list[RankerResult] = []
    for name in ranker_names:
        budget_results: list[BudgetResult] = []
        for budget in budgets:
            parts = collected[name][budget]
            alerts = np.concatenate(parts["alerts"])
            targets = np.concatenate(parts["target"])
            home = np.concatenate(parts["home"])
            away = np.concatenate(parts["away"])
            conditioned = np.concatenate(parts["conditioned"])
            budget_results.append(
                BudgetResult(
                    budget=budget,
                    alerts=alert_metrics(alerts, targets),
                    conditioned_scores=_score_metrics(
                        conditioned,
                        alerts,
                        targets.astype(bool),
                        home,
                        away,
                    ),
                )
            )
        ranker_results.append(RankerResult(name, tuple(budget_results)))

    return ColdBlowoutRankerReport(
        cutoff=cutoff,
        matches=len(frame),
        base_rate=float(np.mean(target)),
        rankers=tuple(ranker_results),
        fold_details=tuple(fold_details),
    )
