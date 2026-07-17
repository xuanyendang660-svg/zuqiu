from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import exp, lgamma, log

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from .live_mechanism_rules import _rules
from .live_snapshots import LiveSnapshotDataset
from .tail_score_classifier import score_ranking_metrics


@dataclass(frozen=True)
class ScoreDistributionMetrics:
    matches: int
    exact_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    negative_log_likelihood: float
    mean_rank: float
    median_rank: float


@dataclass(frozen=True)
class RemainingGoalModelResult:
    family: str
    home_blend: float
    away_blend: float
    home_dispersion: float
    away_dispersion: float
    all_matches: ScoreDistributionMetrics
    mechanism_rules: dict[str, ScoreDistributionMetrics]


@dataclass(frozen=True)
class RemainingGoalCutoffResult:
    cutoff: int
    market_baseline: RemainingGoalModelResult
    dynamic_model: RemainingGoalModelResult
    full_model: RemainingGoalModelResult


@dataclass(frozen=True)
class RemainingGoalReport:
    cutoffs: tuple[RemainingGoalCutoffResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoffs": [
                {
                    "cutoff": result.cutoff,
                    "market_baseline": _result_to_dict(result.market_baseline),
                    "dynamic_model": _result_to_dict(result.dynamic_model),
                    "full_model": _result_to_dict(result.full_model),
                }
                for result in self.cutoffs
            ],
            "fold_details": list(self.fold_details),
        }


def _result_to_dict(result: RemainingGoalModelResult) -> dict[str, object]:
    return {
        "family": result.family,
        "home_blend": result.home_blend,
        "away_blend": result.away_blend,
        "home_dispersion": result.home_dispersion,
        "away_dispersion": result.away_dispersion,
        "all_matches": asdict(result.all_matches),
        "mechanism_rules": {
            name: asdict(metrics) for name, metrics in result.mechanism_rules.items()
        },
    }


def _poisson_pmf(value: int, rate: float) -> float:
    safe_rate = max(float(rate), 1e-8)
    return exp(value * log(safe_rate) - safe_rate - lgamma(value + 1))


def _negative_binomial_pmf(value: int, mean: float, dispersion: float) -> float:
    safe_mean = max(float(mean), 1e-8)
    if dispersion <= 1e-8:
        return _poisson_pmf(value, safe_mean)
    size = 1.0 / dispersion
    probability = size / (size + safe_mean)
    return exp(
        lgamma(value + size)
        - lgamma(size)
        - lgamma(value + 1)
        + size * log(probability)
        + value * log(1.0 - probability)
    )


def _remaining_distribution(
    mean: float,
    *,
    family: str,
    dispersion: float,
    maximum: int,
) -> np.ndarray:
    if family == "negative_binomial":
        probabilities = np.array(
            [
                _negative_binomial_pmf(value, mean, dispersion)
                for value in range(maximum + 1)
            ],
            dtype=float,
        )
    elif family == "poisson":
        probabilities = np.array(
            [_poisson_pmf(value, mean) for value in range(maximum + 1)],
            dtype=float,
        )
    else:
        raise ValueError(f"unknown family: {family}")
    probabilities[-1] += max(0.0, 1.0 - float(probabilities.sum()))
    return probabilities / probabilities.sum()


@lru_cache(maxsize=1)
def _market_rate_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = np.linspace(0.15, 4.50, 88)
    candidates: list[tuple[float, float]] = []
    moments: list[tuple[float, float, float, float]] = []
    for home_rate in rates:
        home = np.array([_poisson_pmf(goal, home_rate) for goal in range(11)])
        for away_rate in rates:
            away = np.array([_poisson_pmf(goal, away_rate) for goal in range(11)])
            matrix = np.outer(home, away)
            matrix /= matrix.sum()
            home_index, away_index = np.indices(matrix.shape)
            candidates.append((float(home_rate), float(away_rate)))
            moments.append(
                (
                    float(matrix[home_index > away_index].sum()),
                    float(matrix[home_index == away_index].sum()),
                    float(matrix[home_index < away_index].sum()),
                    float(matrix[home_index + away_index > 2].sum()),
                )
            )
    return np.asarray(candidates), np.asarray(moments), rates


def infer_market_goal_rates(
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    over25_probability: float,
) -> tuple[float, float]:
    candidates, moments, _ = _market_rate_grid()
    target = np.array(
        [home_probability, draw_probability, away_probability, over25_probability],
        dtype=float,
    )
    finite = np.isfinite(target)
    if finite[:3].sum() < 3:
        return 1.45, 1.15
    weights = np.array([1.0, 1.25, 1.0, 0.85], dtype=float)[finite]
    loss = np.sum(weights * np.square(moments[:, finite] - target[finite]), axis=1)
    best = int(np.argmin(loss))
    return float(candidates[best, 0]), float(candidates[best, 1])


def _market_rates_for_frame(frame: object) -> tuple[np.ndarray, np.ndarray]:
    cache: dict[tuple[float, float, float, float], tuple[float, float]] = {}
    home_rates = np.empty(len(frame), dtype=float)
    away_rates = np.empty(len(frame), dtype=float)
    columns = zip(
        frame["market_home_prob"].to_numpy(dtype=float),
        frame["market_draw_prob"].to_numpy(dtype=float),
        frame["market_away_prob"].to_numpy(dtype=float),
        frame["market_over25_prob"].to_numpy(dtype=float),
        strict=True,
    )
    for index, values in enumerate(columns):
        key = tuple(round(float(value), 6) if np.isfinite(value) else np.nan for value in values)
        if key not in cache:
            cache[key] = infer_market_goal_rates(*values)
        home_rates[index], away_rates[index] = cache[key]
    remaining_fraction = np.clip(
        frame["remaining_minutes"].to_numpy(dtype=float) / 90.0,
        0.0,
        1.0,
    )
    return home_rates * remaining_fraction, away_rates * remaining_fraction


def _dynamic_feature_mask(columns: tuple[str, ...]) -> np.ndarray:
    fixed = {
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
        "shots_last10",
        "xg_last10",
    }
    return np.array(
        [
            column.startswith("market_")
            or column.startswith("live_home_")
            or column.startswith("live_away_")
            or column in fixed
            for column in columns
        ],
        dtype=bool,
    )


@dataclass
class _GoalRegressor:
    imputer: SimpleImputer
    model: HistGradientBoostingRegressor
    blend: float

    def predict(self, features: np.ndarray, base_rate: np.ndarray) -> np.ndarray:
        model_rate = np.clip(
            self.model.predict(self.imputer.transform(features)),
            0.01,
            7.0,
        )
        return np.clip(
            self.blend * model_rate + (1.0 - self.blend) * base_rate,
            0.01,
            7.0,
        )


def _poisson_nll(target: np.ndarray, mean: np.ndarray) -> float:
    safe = np.clip(mean, 1e-8, None)
    return float(
        np.mean(safe - target * np.log(safe) + np.vectorize(lgamma)(target + 1))
    )


def _fit_regressor(
    features: np.ndarray,
    target: np.ndarray,
    base_rate: np.ndarray,
    *,
    random_state: int,
) -> _GoalRegressor:
    calibration_size = max(300, int(len(features) * 0.20))
    split = len(features) - calibration_size
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    fit_features = imputer.fit_transform(features[:split])
    calibration_features = imputer.transform(features[split:])
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.045,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=24,
        l2_regularization=1.25,
        random_state=random_state,
    )
    model.fit(fit_features, target[:split])
    model_prediction = np.clip(model.predict(calibration_features), 0.01, 7.0)
    candidates = []
    for blend in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0):
        prediction = np.clip(
            blend * model_prediction + (1.0 - blend) * base_rate[split:],
            0.01,
            7.0,
        )
        candidates.append((_poisson_nll(target[split:], prediction), blend))
    blend = min(candidates)[1]
    all_features = imputer.fit_transform(features)
    model.fit(all_features, target)
    return _GoalRegressor(imputer, model, float(blend))


def _dispersion(target: np.ndarray, mean: np.ndarray) -> float:
    numerator = float(np.sum(np.square(target - mean) - mean))
    denominator = float(np.sum(np.square(mean)))
    if denominator <= 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 2.0))


def _family_nll(
    target: np.ndarray,
    mean: np.ndarray,
    *,
    family: str,
    dispersion: float,
) -> float:
    probabilities = [
        _negative_binomial_pmf(int(value), float(rate), dispersion)
        if family == "negative_binomial"
        else _poisson_pmf(int(value), float(rate))
        for value, rate in zip(target, mean, strict=True)
    ]
    return float(-np.mean(np.log(np.clip(probabilities, 1e-15, None))))


def _choose_family(
    home_target: np.ndarray,
    away_target: np.ndarray,
    home_mean: np.ndarray,
    away_mean: np.ndarray,
) -> tuple[str, float, float]:
    home_dispersion = _dispersion(home_target, home_mean)
    away_dispersion = _dispersion(away_target, away_mean)
    poisson = _family_nll(
        home_target, home_mean, family="poisson", dispersion=0.0
    ) + _family_nll(away_target, away_mean, family="poisson", dispersion=0.0)
    negative_binomial = _family_nll(
        home_target,
        home_mean,
        family="negative_binomial",
        dispersion=home_dispersion,
    ) + _family_nll(
        away_target,
        away_mean,
        family="negative_binomial",
        dispersion=away_dispersion,
    )
    if negative_binomial + 0.002 < poisson:
        return "negative_binomial", home_dispersion, away_dispersion
    return "poisson", 0.0, 0.0


def _score_matrices(
    current_home: np.ndarray,
    current_away: np.ndarray,
    home_mean: np.ndarray,
    away_mean: np.ndarray,
    *,
    family: str,
    home_dispersion: float,
    away_dispersion: float,
    maximum_final_goals: int = 8,
) -> np.ndarray:
    output = np.zeros(
        (len(home_mean), maximum_final_goals + 1, maximum_final_goals + 1),
        dtype=float,
    )
    for row, values in enumerate(
        zip(current_home, current_away, home_mean, away_mean, strict=True)
    ):
        home_score, away_score, home_rate, away_rate = values
        home_score = min(int(home_score), maximum_final_goals)
        away_score = min(int(away_score), maximum_final_goals)
        home_remaining = maximum_final_goals - home_score
        away_remaining = maximum_final_goals - away_score
        home_distribution = _remaining_distribution(
            home_rate,
            family=family,
            dispersion=home_dispersion,
            maximum=home_remaining,
        )
        away_distribution = _remaining_distribution(
            away_rate,
            family=family,
            dispersion=away_dispersion,
            maximum=away_remaining,
        )
        output[row, home_score:, away_score:] = np.outer(
            home_distribution, away_distribution
        )
        output[row] /= output[row].sum()
    return output


def _distribution_metrics(
    matrices: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> ScoreDistributionMetrics:
    exact: list[bool] = []
    top3: list[bool] = []
    top5: list[bool] = []
    ranks: list[int] = []
    nll: list[float] = []
    for matrix, home, away in zip(matrices, home_goals, away_goals, strict=True):
        actual = (
            min(int(home), matrix.shape[0] - 1),
            min(int(away), matrix.shape[1] - 1),
        )
        order = np.argsort(matrix.ravel())[::-1]
        actual_index = np.ravel_multi_index(actual, matrix.shape)
        rank = int(np.where(order == actual_index)[0][0]) + 1
        predicted = np.unravel_index(int(order[0]), matrix.shape)
        exact.append(predicted == actual)
        top3.append(actual_index in set(order[:3]))
        top5.append(actual_index in set(order[:5]))
        ranks.append(rank)
        nll.append(-log(max(float(matrix[actual]), 1e-15)))
    return ScoreDistributionMetrics(
        matches=len(matrices),
        exact_accuracy=float(np.mean(exact)),
        top3_accuracy=float(np.mean(top3)),
        top5_accuracy=float(np.mean(top5)),
        negative_log_likelihood=float(np.mean(nll)),
        mean_rank=float(np.mean(ranks)),
        median_rank=float(np.median(ranks)),
    )


def _rule_metrics(
    matrices: np.ndarray,
    frame: object,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> dict[str, ScoreDistributionMetrics]:
    rules = _rules(frame)
    selected_names = (
        "underdog_lead_two",
        "underdog_two_goals_ahead",
        "underdog_lead_open_game",
        "underdog_two_goals_resistance",
        "underdog_lead_favorite_red",
    )
    result: dict[str, ScoreDistributionMetrics] = {}
    for name in selected_names:
        mask = np.asarray(rules[name], dtype=bool)
        if int(mask.sum()) == 0:
            continue
        result[name] = _distribution_metrics(
            matrices[mask], home_goals[mask], away_goals[mask]
        )
    union = np.logical_or.reduce([np.asarray(rules[name], dtype=bool) for name in selected_names])
    if union.any():
        result["mechanism_union"] = _distribution_metrics(
            matrices[union], home_goals[union], away_goals[union]
        )
    return result


def leave_one_league_out_remaining_goal_test(
    dataset: LiveSnapshotDataset,
) -> RemainingGoalReport:
    frame = dataset.frame
    league = frame["division_id"].to_numpy(dtype=int)
    cutoffs = frame["snapshot_minute"].to_numpy(dtype=int)
    full_features = dataset.features
    dynamic_mask = _dynamic_feature_mask(dataset.feature_columns)
    dynamic_features = full_features[:, dynamic_mask]
    base_home, base_away = _market_rates_for_frame(frame)
    current_home = frame["live_home_score"].to_numpy(dtype=int)
    current_away = frame["live_away_score"].to_numpy(dtype=int)
    remaining_home = frame["remaining_home_goals"].to_numpy(dtype=int)
    remaining_away = frame["remaining_away_goals"].to_numpy(dtype=int)
    final_home = frame["home_score"].to_numpy(dtype=int)
    final_away = frame["away_score"].to_numpy(dtype=int)

    collected: dict[int, dict[str, list[object]]] = {
        cutoff: {
            "frame": [],
            "home": [],
            "away": [],
            "market": [],
            "dynamic": [],
            "full": [],
            "dynamic_families": [],
            "full_families": [],
            "dynamic_parameters": [],
            "full_parameters": [],
        }
        for cutoff in sorted(np.unique(cutoffs))
    }
    fold_details: list[dict[str, object]] = []

    for held_out in sorted(np.unique(league)):
        for cutoff in sorted(np.unique(cutoffs)):
            train_mask = np.logical_and(league != held_out, cutoffs == cutoff)
            test_mask = np.logical_and(league == held_out, cutoffs == cutoff)
            if int(train_mask.sum()) < 1000 or int(test_mask.sum()) < 250:
                continue
            order = np.argsort(
                frame.loc[train_mask, "date"].to_numpy(dtype="datetime64[ns]")
            )
            train_indices = np.where(train_mask)[0][order]
            test_indices = np.where(test_mask)[0]

            dynamic_home_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_home[train_indices],
                base_home[train_indices],
                random_state=101 + cutoff,
            )
            dynamic_away_model = _fit_regressor(
                dynamic_features[train_indices],
                remaining_away[train_indices],
                base_away[train_indices],
                random_state=111 + cutoff,
            )
            full_home_model = _fit_regressor(
                full_features[train_indices],
                remaining_home[train_indices],
                base_home[train_indices],
                random_state=121 + cutoff,
            )
            full_away_model = _fit_regressor(
                full_features[train_indices],
                remaining_away[train_indices],
                base_away[train_indices],
                random_state=131 + cutoff,
            )

            dynamic_train_home = dynamic_home_model.predict(
                dynamic_features[train_indices], base_home[train_indices]
            )
            dynamic_train_away = dynamic_away_model.predict(
                dynamic_features[train_indices], base_away[train_indices]
            )
            full_train_home = full_home_model.predict(
                full_features[train_indices], base_home[train_indices]
            )
            full_train_away = full_away_model.predict(
                full_features[train_indices], base_away[train_indices]
            )
            dynamic_family = _choose_family(
                remaining_home[train_indices],
                remaining_away[train_indices],
                dynamic_train_home,
                dynamic_train_away,
            )
            full_family = _choose_family(
                remaining_home[train_indices],
                remaining_away[train_indices],
                full_train_home,
                full_train_away,
            )

            dynamic_test_home = dynamic_home_model.predict(
                dynamic_features[test_indices], base_home[test_indices]
            )
            dynamic_test_away = dynamic_away_model.predict(
                dynamic_features[test_indices], base_away[test_indices]
            )
            full_test_home = full_home_model.predict(
                full_features[test_indices], base_home[test_indices]
            )
            full_test_away = full_away_model.predict(
                full_features[test_indices], base_away[test_indices]
            )
            market_matrices = _score_matrices(
                current_home[test_indices],
                current_away[test_indices],
                base_home[test_indices],
                base_away[test_indices],
                family="poisson",
                home_dispersion=0.0,
                away_dispersion=0.0,
            )
            dynamic_matrices = _score_matrices(
                current_home[test_indices],
                current_away[test_indices],
                dynamic_test_home,
                dynamic_test_away,
                family=dynamic_family[0],
                home_dispersion=dynamic_family[1],
                away_dispersion=dynamic_family[2],
            )
            full_matrices = _score_matrices(
                current_home[test_indices],
                current_away[test_indices],
                full_test_home,
                full_test_away,
                family=full_family[0],
                home_dispersion=full_family[1],
                away_dispersion=full_family[2],
            )

            parts = collected[cutoff]
            parts["frame"].append(frame.iloc[test_indices])
            parts["home"].append(final_home[test_indices])
            parts["away"].append(final_away[test_indices])
            parts["market"].append(market_matrices)
            parts["dynamic"].append(dynamic_matrices)
            parts["full"].append(full_matrices)
            parts["dynamic_families"].append(dynamic_family[0])
            parts["full_families"].append(full_family[0])
            parts["dynamic_parameters"].append(
                (
                    dynamic_home_model.blend,
                    dynamic_away_model.blend,
                    dynamic_family[1],
                    dynamic_family[2],
                )
            )
            parts["full_parameters"].append(
                (
                    full_home_model.blend,
                    full_away_model.blend,
                    full_family[1],
                    full_family[2],
                )
            )
            fold_details.append(
                {
                    "held_out_league": int(held_out),
                    "cutoff": int(cutoff),
                    "test_matches": int(test_mask.sum()),
                    "dynamic_family": dynamic_family[0],
                    "dynamic_home_blend": dynamic_home_model.blend,
                    "dynamic_away_blend": dynamic_away_model.blend,
                    "full_family": full_family[0],
                    "full_home_blend": full_home_model.blend,
                    "full_away_blend": full_away_model.blend,
                }
            )

    results: list[RemainingGoalCutoffResult] = []
    for cutoff, parts in collected.items():
        cutoff_frame = __import__("pandas").concat(parts["frame"], ignore_index=True)
        home_values = np.concatenate(parts["home"])
        away_values = np.concatenate(parts["away"])
        market_matrices = np.concatenate(parts["market"])
        dynamic_matrices = np.concatenate(parts["dynamic"])
        full_matrices = np.concatenate(parts["full"])
        dynamic_parameters = np.asarray(parts["dynamic_parameters"], dtype=float)
        full_parameters = np.asarray(parts["full_parameters"], dtype=float)
        dynamic_family = max(
            set(parts["dynamic_families"]), key=parts["dynamic_families"].count
        )
        full_family = max(set(parts["full_families"]), key=parts["full_families"].count)
        results.append(
            RemainingGoalCutoffResult(
                cutoff=int(cutoff),
                market_baseline=RemainingGoalModelResult(
                    family="poisson",
                    home_blend=0.0,
                    away_blend=0.0,
                    home_dispersion=0.0,
                    away_dispersion=0.0,
                    all_matches=_distribution_metrics(
                        market_matrices, home_values, away_values
                    ),
                    mechanism_rules=_rule_metrics(
                        market_matrices, cutoff_frame, home_values, away_values
                    ),
                ),
                dynamic_model=RemainingGoalModelResult(
                    family=dynamic_family,
                    home_blend=float(np.mean(dynamic_parameters[:, 0])),
                    away_blend=float(np.mean(dynamic_parameters[:, 1])),
                    home_dispersion=float(np.mean(dynamic_parameters[:, 2])),
                    away_dispersion=float(np.mean(dynamic_parameters[:, 3])),
                    all_matches=_distribution_metrics(
                        dynamic_matrices, home_values, away_values
                    ),
                    mechanism_rules=_rule_metrics(
                        dynamic_matrices, cutoff_frame, home_values, away_values
                    ),
                ),
                full_model=RemainingGoalModelResult(
                    family=full_family,
                    home_blend=float(np.mean(full_parameters[:, 0])),
                    away_blend=float(np.mean(full_parameters[:, 1])),
                    home_dispersion=float(np.mean(full_parameters[:, 2])),
                    away_dispersion=float(np.mean(full_parameters[:, 3])),
                    all_matches=_distribution_metrics(
                        full_matrices, home_values, away_values
                    ),
                    mechanism_rules=_rule_metrics(
                        full_matrices, cutoff_frame, home_values, away_values
                    ),
                ),
            )
        )
    return RemainingGoalReport(tuple(results), tuple(fold_details))
