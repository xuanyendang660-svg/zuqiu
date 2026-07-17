from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

from .labels import ScoreArchetype, classify_score, label_to_score, score_to_label
from .metrics import DistributionMetrics, evaluate_matrices


@dataclass(frozen=True)
class ResidualConfig:
    max_goals: int = 7
    n_estimators: int = 220
    min_samples_leaf: int = 3
    random_state: int = 42
    calibration_fraction: float = 0.20
    alpha_grid: tuple[float, ...] = (0.50, 0.75, 1.0)
    beta_grid: tuple[float, ...] = (0.0, 0.10)
    tail_boost_grid: tuple[float, ...] = (1.25, 1.50)
    gate_ratio_grid: tuple[float, ...] = (1.50, 2.50, 4.00)
    min_tail_probability_grid: tuple[float, ...] = (0.15, 0.25)
    probability_floor: float = 1e-8


@dataclass(frozen=True)
class ResidualSelection:
    alpha: float
    beta: float
    tail_boost: float
    gate_ratio: float
    min_tail_probability: float
    calibration_override_rate: float
    calibration_metrics: DistributionMetrics
    calibration_market_metrics: DistributionMetrics


class MarketResidualScoreModel:
    """Selectively override a market/DC score grid on high-confidence tail cases.

    Most matches remain exactly equal to the market-conditioned Dixon-Coles base.
    A learned residual is activated only when its tail probability is both high
    in absolute terms and high relative to the market tail mass. This prevents
    tail recall from being purchased by degrading ordinary matches globally.
    """

    _TAILS = {
        ScoreArchetype.HOME_BLOWOUT,
        ScoreArchetype.AWAY_BLOWOUT,
        ScoreArchetype.SHOOTOUT,
    }

    def __init__(self, config: ResidualConfig | None = None) -> None:
        self.config = config or ResidualConfig()
        if self.config.max_goals < 5:
            raise ValueError("max_goals must be at least 5")
        if not 0.10 <= self.config.calibration_fraction <= 0.40:
            raise ValueError("calibration_fraction must be between 0.10 and 0.40")
        forest_args = {
            "n_estimators": self.config.n_estimators,
            "min_samples_leaf": self.config.min_samples_leaf,
            "class_weight": "balanced_subsample",
            "random_state": self.config.random_state,
            "n_jobs": -1,
        }
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        self.exact_model = RandomForestClassifier(**forest_args)
        self.archetype_model = RandomForestClassifier(**forest_args)
        self._score_grid = tuple(
            (home, away)
            for home in range(self.config.max_goals + 1)
            for away in range(self.config.max_goals + 1)
        )
        self._archetype_masks = {
            archetype: self._mask_for(archetype) for archetype in ScoreArchetype
        }
        self._tail_mask = np.logical_or.reduce(
            [self._archetype_masks[archetype] for archetype in self._TAILS]
        )

    def _mask_for(self, archetype: ScoreArchetype) -> np.ndarray:
        mask = np.zeros(
            (self.config.max_goals + 1, self.config.max_goals + 1), dtype=bool
        )
        for score in self._score_grid:
            if classify_score(*score) is archetype:
                mask[score] = True
        return mask

    @staticmethod
    def _features(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("features must be a non-empty 2D array")
        if np.any(np.isinf(array)):
            raise ValueError("features contain infinite values")
        return array

    def _fit_probability_models(
        self,
        features: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
    ) -> None:
        x = self.imputer.fit_transform(features)
        exact_labels = np.array(
            [
                score_to_label(int(home), int(away))
                for home, away in zip(home_goals, away_goals, strict=True)
            ]
        )
        archetype_labels = np.array(
            [
                classify_score(int(home), int(away)).value
                for home, away in zip(home_goals, away_goals, strict=True)
            ]
        )
        self.exact_model.fit(x, exact_labels)
        self.archetype_model.fit(x, archetype_labels)
        self.n_raw_features_in_ = features.shape[1]

    def _probability_tables(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, list[str]]:
        x = self.imputer.transform(features)
        exact = self.exact_model.predict_proba(x)
        exact_classes = [label_to_score(str(value)) for value in self.exact_model.classes_]
        archetype = self.archetype_model.predict_proba(x)
        archetype_classes = [str(value) for value in self.archetype_model.classes_]
        return exact, exact_classes, archetype, archetype_classes

    def _combine(
        self,
        base_distributions: np.ndarray,
        exact_probabilities: np.ndarray,
        exact_classes: list[tuple[int, int]],
        archetype_probabilities: np.ndarray,
        archetype_classes: list[str],
        *,
        alpha: float,
        beta: float,
        tail_boost: float,
        gate_ratio: float,
        min_tail_probability: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(base_distributions) != len(exact_probabilities):
            raise ValueError("base distributions and features must have matching rows")
        output = np.empty_like(base_distributions, dtype=float)
        override = np.zeros(len(base_distributions), dtype=bool)
        epsilon = self.config.probability_floor
        archetype_indices = {label: index for index, label in enumerate(archetype_classes)}

        for row, base in enumerate(base_distributions):
            market = np.asarray(base, dtype=float).copy()
            market = np.clip(market, epsilon, None)
            market /= market.sum()
            learned_tail_probability = sum(
                float(archetype_probabilities[row, archetype_indices[archetype.value]])
                for archetype in self._TAILS
                if archetype.value in archetype_indices
            )
            market_tail_probability = float(market[self._tail_mask].sum())
            tail_ratio = learned_tail_probability / max(market_tail_probability, epsilon)
            should_override = (
                learned_tail_probability >= min_tail_probability
                and tail_ratio >= gate_ratio
            )
            if not should_override:
                output[row] = market
                continue

            override[row] = True
            matrix = market.copy()
            exact_model_grid = np.full(matrix.shape, epsilon, dtype=float)
            for score, probability in zip(
                exact_classes, exact_probabilities[row], strict=True
            ):
                exact_model_grid[score] += float(probability)
            exact_model_grid /= exact_model_grid.sum()
            if beta > 0:
                exact_ratio = np.clip(exact_model_grid / matrix, 1e-3, 1e3)
                matrix *= np.power(exact_ratio, beta)
                matrix /= matrix.sum()

            archetype_targets = {
                label: float(probability)
                for label, probability in zip(
                    archetype_classes, archetype_probabilities[row], strict=True
                )
            }
            for archetype in ScoreArchetype:
                mask = self._archetype_masks[archetype]
                base_mass = float(matrix[mask].sum())
                learned_mass = archetype_targets.get(archetype.value, epsilon)
                if archetype in self._TAILS:
                    learned_mass *= tail_boost
                if base_mass > 0:
                    ratio = np.clip(learned_mass / base_mass, 1e-3, 1e3)
                    matrix[mask] *= ratio**alpha
            matrix /= matrix.sum()
            output[row] = matrix
        return output, override

    @staticmethod
    def _is_feasible(
        candidate: DistributionMetrics, market: DistributionMetrics
    ) -> bool:
        return (
            candidate.exact_accuracy >= market.exact_accuracy - 0.002
            and candidate.top_k_accuracy >= market.top_k_accuracy - 0.005
            and candidate.direction_accuracy >= market.direction_accuracy - 0.010
            and candidate.negative_log_likelihood <= market.negative_log_likelihood + 0.010
        )

    @staticmethod
    def _tail_value(metrics: DistributionMetrics) -> float:
        return metrics.extreme_tail_recall if metrics.extreme_tail_recall is not None else -1.0

    def fit(
        self,
        features: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        base_distributions: np.ndarray,
    ) -> "MarketResidualScoreModel":
        raw = self._features(features)
        home = np.asarray(home_goals, dtype=int)
        away = np.asarray(away_goals, dtype=int)
        base = np.asarray(base_distributions, dtype=float)
        if len(raw) != len(home) or len(home) != len(away) or len(home) != len(base):
            raise ValueError("training arrays must have matching rows")
        expected_shape = (
            self.config.max_goals + 1,
            self.config.max_goals + 1,
        )
        if base.shape[1:] != expected_shape:
            raise ValueError("base score grid differs from model configuration")

        calibration_rows = max(100, int(len(raw) * self.config.calibration_fraction))
        split = len(raw) - calibration_rows
        if split < 500:
            raise ValueError("insufficient data for residual calibration")

        self._fit_probability_models(raw[:split], home[:split], away[:split])
        exact, exact_classes, archetype, archetype_classes = self._probability_tables(
            raw[split:]
        )
        market_metrics = evaluate_matrices(base[split:], home[split:], away[split:])
        candidates: list[
            tuple[ResidualSelection, tuple[float, float, float, float, float, float]]
        ] = []

        baseline_selection = ResidualSelection(
            alpha=0.0,
            beta=0.0,
            tail_boost=1.0,
            gate_ratio=float("inf"),
            min_tail_probability=1.0,
            calibration_override_rate=0.0,
            calibration_metrics=market_metrics,
            calibration_market_metrics=market_metrics,
        )
        candidates.append(
            (
                baseline_selection,
                (
                    1.0,
                    self._tail_value(market_metrics),
                    market_metrics.exact_accuracy,
                    market_metrics.top_k_accuracy,
                    -market_metrics.negative_log_likelihood,
                    0.0,
                ),
            )
        )

        for alpha in self.config.alpha_grid:
            for beta in self.config.beta_grid:
                for tail_boost in self.config.tail_boost_grid:
                    for gate_ratio in self.config.gate_ratio_grid:
                        for minimum in self.config.min_tail_probability_grid:
                            adjusted, override = self._combine(
                                base[split:],
                                exact,
                                exact_classes,
                                archetype,
                                archetype_classes,
                                alpha=alpha,
                                beta=beta,
                                tail_boost=tail_boost,
                                gate_ratio=gate_ratio,
                                min_tail_probability=minimum,
                            )
                            metrics = evaluate_matrices(
                                adjusted, home[split:], away[split:]
                            )
                            selection = ResidualSelection(
                                alpha=alpha,
                                beta=beta,
                                tail_boost=tail_boost,
                                gate_ratio=gate_ratio,
                                min_tail_probability=minimum,
                                calibration_override_rate=float(np.mean(override)),
                                calibration_metrics=metrics,
                                calibration_market_metrics=market_metrics,
                            )
                            feasible = self._is_feasible(metrics, market_metrics)
                            score = (
                                1.0 if feasible else 0.0,
                                self._tail_value(metrics),
                                metrics.exact_accuracy,
                                metrics.top_k_accuracy,
                                -metrics.negative_log_likelihood,
                                -float(np.mean(override)),
                            )
                            candidates.append((selection, score))

        self.selection_ = max(candidates, key=lambda item: item[1])[0]
        self._fit_probability_models(raw, home, away)
        return self

    def _check(self) -> None:
        check_is_fitted(self.imputer)
        check_is_fitted(self.exact_model)
        check_is_fitted(self.archetype_model)
        if not hasattr(self, "selection_"):
            raise ValueError("model has not selected residual parameters")

    def predict_distribution(
        self, features: np.ndarray, base_distributions: np.ndarray
    ) -> np.ndarray:
        self._check()
        raw = self._features(features)
        if raw.shape[1] != self.n_raw_features_in_:
            raise ValueError("feature count differs from training data")
        exact, exact_classes, archetype, archetype_classes = self._probability_tables(raw)
        result, _ = self._combine(
            np.asarray(base_distributions, dtype=float),
            exact,
            exact_classes,
            archetype,
            archetype_classes,
            alpha=self.selection_.alpha,
            beta=self.selection_.beta,
            tail_boost=self.selection_.tail_boost,
            gate_ratio=self.selection_.gate_ratio,
            min_tail_probability=self.selection_.min_tail_probability,
        )
        return result

    def override_mask(
        self, features: np.ndarray, base_distributions: np.ndarray
    ) -> np.ndarray:
        self._check()
        raw = self._features(features)
        exact, exact_classes, archetype, archetype_classes = self._probability_tables(raw)
        _, override = self._combine(
            np.asarray(base_distributions, dtype=float),
            exact,
            exact_classes,
            archetype,
            archetype_classes,
            alpha=self.selection_.alpha,
            beta=self.selection_.beta,
            tail_boost=self.selection_.tail_boost,
            gate_ratio=self.selection_.gate_ratio,
            min_tail_probability=self.selection_.min_tail_probability,
        )
        return override
