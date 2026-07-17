from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

from .labels import ScoreArchetype, classify_score, label_to_score, score_to_label


@dataclass(frozen=True)
class ModelConfig:
    max_goals: int = 7
    n_estimators: int = 300
    min_samples_leaf: int = 1
    random_state: int = 42
    probability_floor: float = 1e-9


class TailAwareExactScoreModel:
    """Hierarchical, data-trained exact-score model.

    Two classifiers are fitted from the same historical feature matrix:

    1. exact score classes;
    2. score-shape archetypes, including one-sided home/away blowouts.

    Their probabilities are reconciled so an archetype such as away_blowout can
    transfer real mass to 0-4, 1-4, 1-5 and similar scorelines. No hand-written
    upset or tail weight is accepted by this class.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        if self.config.max_goals < 5:
            raise ValueError("max_goals must be at least 5 to represent extreme tails")
        forest_args = {
            "n_estimators": self.config.n_estimators,
            "min_samples_leaf": self.config.min_samples_leaf,
            "class_weight": "balanced_subsample",
            "random_state": self.config.random_state,
            "n_jobs": -1,
        }
        self.imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        self.exact_model = RandomForestClassifier(**forest_args)
        self.archetype_model = RandomForestClassifier(**forest_args)
        self._score_grid = tuple(
            (home, away)
            for home in range(self.config.max_goals + 1)
            for away in range(self.config.max_goals + 1)
        )

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

    def fit(
        self,
        features: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
    ) -> "TailAwareExactScoreModel":
        raw = self._features(features)
        home = np.asarray(home_goals, dtype=int)
        away = np.asarray(away_goals, dtype=int)
        if home.ndim != 1 or away.ndim != 1 or len(home) != len(away) or len(home) != len(raw):
            raise ValueError("features and goal arrays must have matching rows")
        if np.any(home < 0) or np.any(away < 0):
            raise ValueError("goals cannot be negative")
        if np.any(home > self.config.max_goals) or np.any(away > self.config.max_goals):
            raise ValueError("training goals exceed configured score grid")

        x = self.imputer.fit_transform(raw)
        exact_labels = np.array(
            [score_to_label(int(h), int(a)) for h, a in zip(home, away, strict=True)]
        )
        archetype_labels = np.array(
            [classify_score(int(h), int(a)).value for h, a in zip(home, away, strict=True)]
        )
        self.exact_model.fit(x, exact_labels)
        self.archetype_model.fit(x, archetype_labels)
        self.n_raw_features_in_ = raw.shape[1]
        self.n_features_in_ = x.shape[1]
        return self

    def _check(self) -> None:
        check_is_fitted(self.imputer)
        check_is_fitted(self.exact_model)
        check_is_fitted(self.archetype_model)

    def predict_distribution(self, features: np.ndarray) -> np.ndarray:
        self._check()
        raw = self._features(features)
        if raw.shape[1] != self.n_raw_features_in_:
            raise ValueError("feature count differs from training data")
        x = self.imputer.transform(raw)

        exact_probabilities = self.exact_model.predict_proba(x)
        archetype_probabilities = self.archetype_model.predict_proba(x)
        exact_classes = [label_to_score(str(label)) for label in self.exact_model.classes_]
        archetype_classes = [str(label) for label in self.archetype_model.classes_]

        output = np.zeros(
            (len(x), self.config.max_goals + 1, self.config.max_goals + 1),
            dtype=float,
        )
        for row in range(len(x)):
            distribution = np.full(
                (self.config.max_goals + 1, self.config.max_goals + 1),
                self.config.probability_floor,
                dtype=float,
            )
            for score, probability in zip(
                exact_classes, exact_probabilities[row], strict=True
            ):
                distribution[score] += float(probability)
            distribution /= distribution.sum()

            target_by_archetype = {
                label: float(probability)
                for label, probability in zip(
                    archetype_classes, archetype_probabilities[row], strict=True
                )
            }
            for archetype in ScoreArchetype:
                mask = np.zeros_like(distribution, dtype=bool)
                for score in self._score_grid:
                    if classify_score(*score) is archetype:
                        mask[score] = True
                current_mass = float(distribution[mask].sum())
                target_mass = target_by_archetype.get(archetype.value, 0.0)
                if current_mass > 0:
                    distribution[mask] *= target_mass / current_mass
            distribution /= distribution.sum()
            output[row] = distribution
        return output

    def predict_score(self, features: np.ndarray) -> list[tuple[int, int]]:
        distributions = self.predict_distribution(features)
        scores: list[tuple[int, int]] = []
        for matrix in distributions:
            index = int(np.argmax(matrix))
            scores.append(tuple(int(value) for value in np.unravel_index(index, matrix.shape)))
        return scores

    def predict_top_k(
        self,
        features: np.ndarray,
        *,
        k: int = 5,
    ) -> list[list[tuple[tuple[int, int], float]]]:
        if k <= 0:
            raise ValueError("k must be positive")
        distributions = self.predict_distribution(features)
        result: list[list[tuple[tuple[int, int], float]]] = []
        for matrix in distributions:
            order = np.argsort(matrix.ravel())[::-1][:k]
            result.append(
                [
                    (
                        tuple(
                            int(value)
                            for value in np.unravel_index(index, matrix.shape)
                        ),
                        float(matrix.ravel()[index]),
                    )
                    for index in order
                ]
            )
        return result
