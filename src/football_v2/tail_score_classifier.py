from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

from .labels import score_to_label, label_to_score


@dataclass
class TailExactScoreModel:
    max_goals: int = 7
    random_state: int = 42

    def fit(
        self,
        features: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        tail_target: np.ndarray,
    ) -> "TailExactScoreModel":
        mask = np.asarray(tail_target, dtype=bool)
        if int(mask.sum()) < 40:
            raise ValueError("insufficient tail matches for exact-score model")
        raw = np.asarray(features, dtype=float)[mask]
        home = np.minimum(np.asarray(home_goals, dtype=int)[mask], self.max_goals)
        away = np.minimum(np.asarray(away_goals, dtype=int)[mask], self.max_goals)
        labels = np.array(
            [score_to_label(int(h), int(a)) for h, a in zip(home, away, strict=True)]
        )
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        x = self.imputer.fit_transform(raw)
        self.classifier = ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.classifier.fit(x, labels)
        self.n_raw_features_in_ = raw.shape[1]
        return self

    def predict_distribution(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] != self.n_raw_features_in_:
            raise ValueError("feature count differs from training data")
        probabilities = self.classifier.predict_proba(self.imputer.transform(raw))
        output = np.full(
            (len(raw), self.max_goals + 1, self.max_goals + 1),
            1e-9,
            dtype=float,
        )
        classes = [label_to_score(str(label)) for label in self.classifier.classes_]
        for row in range(len(raw)):
            for score, probability in zip(classes, probabilities[row], strict=True):
                output[row][score] += float(probability)
            output[row] /= output[row].sum()
        return output


def score_ranking_metrics(
    distributions: np.ndarray,
    alerts: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> dict[str, float | int | None]:
    selected = np.asarray(alerts, dtype=bool)
    if not selected.any():
        return {
            "alerts": 0,
            "exact_accuracy": None,
            "top3_accuracy": None,
            "top5_accuracy": None,
            "mean_rank": None,
            "median_rank": None,
        }
    exact: list[bool] = []
    top3: list[bool] = []
    top5: list[bool] = []
    ranks: list[int] = []
    for matrix, alert, home, away in zip(
        distributions, selected, home_goals, away_goals, strict=True
    ):
        if not alert:
            continue
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
    return {
        "alerts": len(ranks),
        "exact_accuracy": float(np.mean(exact)),
        "top3_accuracy": float(np.mean(top3)),
        "top5_accuracy": float(np.mean(top5)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }
