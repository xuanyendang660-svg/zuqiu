from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, lgamma, log

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer

from .labels import ScoreArchetype, classify_score
from .statsbomb_events import EventDataset


_TAIL_TYPES = (
    ScoreArchetype.HOME_BLOWOUT,
    ScoreArchetype.AWAY_BLOWOUT,
    ScoreArchetype.SHOOTOUT,
)


@dataclass(frozen=True)
class AlertMetrics:
    matches: int
    alerts: int
    coverage: float
    base_rate: float
    precision: float | None
    recall: float
    lift: float | None


@dataclass(frozen=True)
class EventTailReport:
    matches: int
    folds: int
    first_test_date: str
    last_test_date: str
    event_alerts: AlertMetrics
    result_only_alerts: AlertMetrics
    event_tail_type_accuracy: float | None
    event_alert_exact_accuracy: float | None
    event_alert_top5_accuracy: float | None
    event_alert_mean_rank: float | None
    result_only_alert_exact_accuracy: float | None
    thresholds: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "folds": self.folds,
            "first_test_date": self.first_test_date,
            "last_test_date": self.last_test_date,
            "event_alerts": asdict(self.event_alerts),
            "result_only_alerts": asdict(self.result_only_alerts),
            "event_tail_type_accuracy": self.event_tail_type_accuracy,
            "event_alert_exact_accuracy": self.event_alert_exact_accuracy,
            "event_alert_top5_accuracy": self.event_alert_top5_accuracy,
            "event_alert_mean_rank": self.event_alert_mean_rank,
            "result_only_alert_exact_accuracy": self.result_only_alert_exact_accuracy,
            "thresholds": list(self.thresholds),
        }


@dataclass(frozen=True)
class EventTailConfig:
    max_alert_coverage: float = 0.08
    minimum_alerts: int = 8
    minimum_lift: float = 1.25
    learning_rate: float = 0.045
    max_iter: int = 220
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 18
    l2_regularization: float = 1.0
    random_state: int = 42
    max_goals: int = 7


@dataclass
class _FittedTailModel:
    imputer: SimpleImputer
    classifier: HistGradientBoostingClassifier
    threshold: float
    type_imputer: SimpleImputer
    type_classifier: RandomForestClassifier | None
    home_goal_model: HistGradientBoostingRegressor
    away_goal_model: HistGradientBoostingRegressor


@dataclass(frozen=True)
class _FoldOutput:
    probabilities: np.ndarray
    alerts: np.ndarray
    type_predictions: np.ndarray
    distributions: np.ndarray
    threshold: float


def _poisson_pmf(value: int, rate: float) -> float:
    return exp(value * log(rate) - rate - lgamma(value + 1))


def _score_matrix(home_rate: float, away_rate: float, max_goals: int) -> np.ndarray:
    home = np.array([_poisson_pmf(goal, home_rate) for goal in range(max_goals + 1)])
    away = np.array([_poisson_pmf(goal, away_rate) for goal in range(max_goals + 1)])
    matrix = np.outer(home, away)
    return matrix / matrix.sum()


def _tail_mask(archetype: ScoreArchetype, max_goals: int) -> np.ndarray:
    mask = np.zeros((max_goals + 1, max_goals + 1), dtype=bool)
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            mask[home, away] = classify_score(home, away) is archetype
    return mask


def _sample_weights(target: np.ndarray) -> np.ndarray:
    positive_rate = float(np.mean(target))
    if positive_rate <= 0 or positive_rate >= 1:
        return np.ones(len(target), dtype=float)
    positive_weight = min(8.0, (1.0 - positive_rate) / positive_rate)
    return np.where(target == 1, positive_weight, 1.0)


def alert_metrics(alerts: np.ndarray, target: np.ndarray) -> AlertMetrics:
    alerts = np.asarray(alerts, dtype=bool)
    target = np.asarray(target, dtype=bool)
    count = len(target)
    alert_count = int(alerts.sum())
    positives = int(target.sum())
    true_positive = int(np.logical_and(alerts, target).sum())
    base_rate = positives / count if count else 0.0
    precision = true_positive / alert_count if alert_count else None
    recall = true_positive / positives if positives else 0.0
    lift = precision / base_rate if precision is not None and base_rate > 0 else None
    return AlertMetrics(
        matches=count,
        alerts=alert_count,
        coverage=alert_count / count if count else 0.0,
        base_rate=base_rate,
        precision=precision,
        recall=recall,
        lift=lift,
    )


def _threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    quantiles = np.linspace(0.80, 0.995, 40)
    return np.unique(np.concatenate([np.quantile(probabilities, quantiles), [1.0]]))


def _choose_threshold(
    probabilities: np.ndarray,
    target: np.ndarray,
    config: EventTailConfig,
) -> float:
    midpoint = max(1, len(target) // 2)
    candidates: list[tuple[tuple[float, float, float, float], float]] = []
    for threshold in _threshold_candidates(probabilities):
        alerts = probabilities >= threshold
        full = alert_metrics(alerts, target)
        halves = (
            alert_metrics(alerts[:midpoint], target[:midpoint]),
            alert_metrics(alerts[midpoint:], target[midpoint:]),
        )
        robust = (
            full.alerts >= config.minimum_alerts
            and full.coverage <= config.max_alert_coverage
            and full.lift is not None
            and full.lift >= config.minimum_lift
            and all(
                half.alerts >= max(2, config.minimum_alerts // 4)
                and half.coverage <= config.max_alert_coverage * 1.5
                and half.lift is not None
                and half.lift >= 1.0
                for half in halves
            )
        )
        score = (
            1.0 if robust else 0.0,
            full.lift if full.lift is not None else -1.0,
            full.precision if full.precision is not None else -1.0,
            full.recall,
        )
        candidates.append((score, float(threshold)))
    best_score, best_threshold = max(candidates, key=lambda item: item[0])
    return best_threshold if best_score[0] == 1.0 else 1.0


def _fit_model(
    features: np.ndarray,
    target: np.ndarray,
    tail_types: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    config: EventTailConfig,
) -> _FittedTailModel:
    calibration_size = max(80, int(len(features) * 0.20))
    split = len(features) - calibration_size
    if split < 250:
        raise ValueError("insufficient rows for event-tail calibration")

    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x_fit = imputer.fit_transform(features[:split])
    x_calibration = imputer.transform(features[split:])
    classifier = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        random_state=config.random_state,
    )
    classifier.fit(x_fit, target[:split], sample_weight=_sample_weights(target[:split]))
    calibration_probabilities = classifier.predict_proba(x_calibration)[:, 1]
    threshold = _choose_threshold(calibration_probabilities, target[split:], config)

    x_all = imputer.fit_transform(features)
    classifier.fit(x_all, target, sample_weight=_sample_weights(target))

    type_imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    tail_mask = target == 1
    type_classifier: RandomForestClassifier | None = None
    if int(tail_mask.sum()) >= 30 and len(np.unique(tail_types[tail_mask])) >= 2:
        type_x = type_imputer.fit_transform(features[tail_mask])
        type_classifier = RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=config.random_state,
            n_jobs=-1,
        )
        type_classifier.fit(type_x, tail_types[tail_mask])
    else:
        type_imputer.fit(features)

    home_goal_model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.04,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=18,
        l2_regularization=1.0,
        random_state=config.random_state,
    )
    away_goal_model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.04,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=18,
        l2_regularization=1.0,
        random_state=config.random_state + 1,
    )
    home_goal_model.fit(x_all, home_goals)
    away_goal_model.fit(x_all, away_goals)
    return _FittedTailModel(
        imputer=imputer,
        classifier=classifier,
        threshold=threshold,
        type_imputer=type_imputer,
        type_classifier=type_classifier,
        home_goal_model=home_goal_model,
        away_goal_model=away_goal_model,
    )


def _predict_fold(
    model: _FittedTailModel,
    features: np.ndarray,
    config: EventTailConfig,
) -> _FoldOutput:
    x = model.imputer.transform(features)
    probabilities = model.classifier.predict_proba(x)[:, 1]
    alerts = probabilities >= model.threshold
    if model.type_classifier is not None:
        type_x = model.type_imputer.transform(features)
        type_predictions = model.type_classifier.predict(type_x).astype(str)
    else:
        type_predictions = np.full(len(features), ScoreArchetype.SHOOTOUT.value, dtype=object)

    home_rates = np.clip(model.home_goal_model.predict(x), 0.15, 6.5)
    away_rates = np.clip(model.away_goal_model.predict(x), 0.15, 6.5)
    distributions = np.empty(
        (len(features), config.max_goals + 1, config.max_goals + 1), dtype=float
    )
    for index, (home_rate, away_rate) in enumerate(
        zip(home_rates, away_rates, strict=True)
    ):
        matrix = _score_matrix(float(home_rate), float(away_rate), config.max_goals)
        if alerts[index]:
            try:
                archetype = ScoreArchetype(str(type_predictions[index]))
            except ValueError:
                archetype = ScoreArchetype.SHOOTOUT
            mask = _tail_mask(archetype, config.max_goals)
            if matrix[mask].sum() > 0:
                matrix = np.where(mask, matrix, 0.0)
                matrix /= matrix.sum()
        distributions[index] = matrix
    return _FoldOutput(probabilities, alerts, type_predictions, distributions, model.threshold)


def _result_only_columns(feature_columns: tuple[str, ...]) -> np.ndarray:
    allowed_tokens = (
        "goals_",
        "tail_for",
        "tail_against",
        "collapse_conceded",
        "rest_days",
        "history",
        "xi_continuity",
        "xi_experience",
        "formation_change",
        "month_",
    )
    return np.array(
        [any(token in column for token in allowed_tokens) for column in feature_columns],
        dtype=bool,
    )


def _score_metrics(
    distributions: np.ndarray,
    alerts: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if not alerts.any():
        return None, None, None
    exact_hits: list[bool] = []
    top5_hits: list[bool] = []
    ranks: list[int] = []
    for matrix, alert, home, away in zip(
        distributions, alerts, home_goals, away_goals, strict=True
    ):
        if not alert:
            continue
        order = np.argsort(matrix.ravel())[::-1]
        predicted = np.unravel_index(int(order[0]), matrix.shape)
        actual = (int(home), int(away))
        actual_index = np.ravel_multi_index(actual, matrix.shape)
        rank = int(np.where(order == actual_index)[0][0]) + 1
        exact_hits.append(predicted == actual)
        top5_hits.append(actual_index in set(order[:5]))
        ranks.append(rank)
    return float(np.mean(exact_hits)), float(np.mean(top5_hits)), float(np.mean(ranks))


def walk_forward_event_tail_test(
    dataset: EventDataset,
    *,
    folds: int = 2,
    initial_train_fraction: float = 0.55,
    config: EventTailConfig | None = None,
) -> EventTailReport:
    config = config or EventTailConfig()
    count = len(dataset.frame)
    initial = int(count * initial_train_fraction)
    if initial < 350 or initial >= count:
        raise ValueError("insufficient event rows for walk-forward test")
    window = max(1, (count - initial) // folds)
    result_columns = _result_only_columns(dataset.feature_columns)

    event_outputs: list[_FoldOutput] = []
    result_outputs: list[_FoldOutput] = []
    target_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    home_parts: list[np.ndarray] = []
    away_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []

    for fold in range(folds):
        train_end = initial + fold * window
        test_end = count if fold == folds - 1 else min(count, train_end + window)
        if test_end <= train_end:
            continue
        train_features = dataset.features[:train_end]
        test_features = dataset.features[train_end:test_end]
        train_target = dataset.tail_target[:train_end]
        train_types = dataset.tail_type[:train_end]
        train_home = dataset.home_goals[:train_end]
        train_away = dataset.away_goals[:train_end]

        event_model = _fit_model(
            train_features,
            train_target,
            train_types,
            train_home,
            train_away,
            config,
        )
        result_model = _fit_model(
            train_features[:, result_columns],
            train_target,
            train_types,
            train_home,
            train_away,
            config,
        )
        event_outputs.append(_predict_fold(event_model, test_features, config))
        result_outputs.append(
            _predict_fold(result_model, test_features[:, result_columns], config)
        )
        target_parts.append(dataset.tail_target[train_end:test_end])
        type_parts.append(dataset.tail_type[train_end:test_end])
        home_parts.append(dataset.home_goals[train_end:test_end])
        away_parts.append(dataset.away_goals[train_end:test_end])
        date_parts.append(dataset.frame.iloc[train_end:test_end]["date"].to_numpy())

    target = np.concatenate(target_parts)
    actual_types = np.concatenate(type_parts)
    home = np.concatenate(home_parts)
    away = np.concatenate(away_parts)
    dates = np.concatenate(date_parts)
    event_alerts = np.concatenate([output.alerts for output in event_outputs])
    result_alerts = np.concatenate([output.alerts for output in result_outputs])
    event_types = np.concatenate([output.type_predictions for output in event_outputs])
    event_distributions = np.concatenate([output.distributions for output in event_outputs])
    result_distributions = np.concatenate([output.distributions for output in result_outputs])

    event_type_mask = np.logical_and(event_alerts, target == 1)
    type_accuracy = (
        float(np.mean(event_types[event_type_mask] == actual_types[event_type_mask]))
        if event_type_mask.any()
        else None
    )
    event_exact, event_top5, event_rank = _score_metrics(
        event_distributions, event_alerts, home, away
    )
    result_exact, _, _ = _score_metrics(result_distributions, result_alerts, home, away)
    return EventTailReport(
        matches=len(target),
        folds=folds,
        first_test_date=str(np.min(dates).astype("datetime64[D]")),
        last_test_date=str(np.max(dates).astype("datetime64[D]")),
        event_alerts=alert_metrics(event_alerts, target),
        result_only_alerts=alert_metrics(result_alerts, target),
        event_tail_type_accuracy=type_accuracy,
        event_alert_exact_accuracy=event_exact,
        event_alert_top5_accuracy=event_top5,
        event_alert_mean_rank=event_rank,
        result_only_alert_exact_accuracy=result_exact,
        thresholds=tuple(output.threshold for output in event_outputs),
    )
