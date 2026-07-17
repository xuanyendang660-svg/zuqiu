from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

from .live_mechanism_rules import _rules
from .live_remaining_goals import (
    ScoreDistributionMetrics,
    _distribution_metrics,
    _market_rates_for_frame,
    _score_matrices,
)
from .live_snapshots import LiveSnapshotDataset


@dataclass(frozen=True)
class OrientedScoreCutoffResult:
    cutoff: int
    matches: int
    market_all: ScoreDistributionMetrics
    oriented_all: ScoreDistributionMetrics
    market_cold_tail: ScoreDistributionMetrics | None
    oriented_cold_tail: ScoreDistributionMetrics | None
    market_cold_path: ScoreDistributionMetrics | None
    oriented_cold_path: ScoreDistributionMetrics | None
    mean_blend: float


@dataclass(frozen=True)
class OrientedScoreReport:
    cutoffs: tuple[OrientedScoreCutoffResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoffs": [
                {
                    "cutoff": result.cutoff,
                    "matches": result.matches,
                    "market_all": asdict(result.market_all),
                    "oriented_all": asdict(result.oriented_all),
                    "market_cold_tail": (
                        asdict(result.market_cold_tail)
                        if result.market_cold_tail
                        else None
                    ),
                    "oriented_cold_tail": (
                        asdict(result.oriented_cold_tail)
                        if result.oriented_cold_tail
                        else None
                    ),
                    "market_cold_path": (
                        asdict(result.market_cold_path)
                        if result.market_cold_path
                        else None
                    ),
                    "oriented_cold_path": (
                        asdict(result.oriented_cold_path)
                        if result.oriented_cold_path
                        else None
                    ),
                    "mean_blend": result.mean_blend,
                }
                for result in self.cutoffs
            ],
            "fold_details": list(self.fold_details),
        }


@dataclass
class _OrientedClassifier:
    imputer: SimpleImputer
    classifier: ExtraTreesClassifier
    classes: tuple[tuple[int, int], ...]

    def predict_final_matrices(
        self,
        features: np.ndarray,
        current_home: np.ndarray,
        current_away: np.ndarray,
        underdog_home: np.ndarray,
        *,
        maximum_final_goals: int = 8,
    ) -> np.ndarray:
        probabilities = self.classifier.predict_proba(
            self.imputer.transform(features)
        )
        output = np.full(
            (
                len(features),
                maximum_final_goals + 1,
                maximum_final_goals + 1,
            ),
            1e-12,
            dtype=float,
        )
        for row, class_probabilities in enumerate(probabilities):
            for (underdog_remaining, favorite_remaining), probability in zip(
                self.classes, class_probabilities, strict=True
            ):
                if underdog_home[row]:
                    final_home = min(
                        maximum_final_goals,
                        int(current_home[row]) + underdog_remaining,
                    )
                    final_away = min(
                        maximum_final_goals,
                        int(current_away[row]) + favorite_remaining,
                    )
                else:
                    final_home = min(
                        maximum_final_goals,
                        int(current_home[row]) + favorite_remaining,
                    )
                    final_away = min(
                        maximum_final_goals,
                        int(current_away[row]) + underdog_remaining,
                    )
                output[row, final_home, final_away] += float(probability)
            output[row] /= output[row].sum()
        return output


def _underlying_role(frame: object) -> np.ndarray:
    return (
        frame["market_home_prob"].to_numpy(dtype=float)
        < frame["market_away_prob"].to_numpy(dtype=float)
    )


def _role_features(frame: object) -> np.ndarray:
    underdog_home = _underlying_role(frame)

    def role_values(home_name: str, away_name: str) -> tuple[np.ndarray, np.ndarray]:
        home = frame[home_name].to_numpy(dtype=float)
        away = frame[away_name].to_numpy(dtype=float)
        underdog = np.where(underdog_home, home, away)
        favorite = np.where(underdog_home, away, home)
        return underdog, favorite

    features: list[np.ndarray] = [
        underdog_home.astype(float),
        frame["market_underdog_prob"].to_numpy(dtype=float),
        frame["market_favorite_prob"].to_numpy(dtype=float),
        frame["market_draw_prob"].to_numpy(dtype=float),
        frame["market_over25_prob"].to_numpy(dtype=float),
        frame["market_entropy"].to_numpy(dtype=float),
        frame["snapshot_minute"].to_numpy(dtype=float),
        frame["remaining_minutes"].to_numpy(dtype=float),
        frame["goals_last10"].to_numpy(dtype=float),
        frame["shots_last10"].to_numpy(dtype=float),
        frame["xg_last10"].to_numpy(dtype=float),
    ]
    pairs = (
        ("live_home_score", "live_away_score"),
        ("live_home_xg", "live_away_xg"),
        ("live_home_shots", "live_away_shots"),
        ("live_home_shots_on_target", "live_away_shots_on_target"),
        ("live_home_big_chances", "live_away_big_chances"),
        ("live_home_counter_xg", "live_away_counter_xg"),
        ("live_home_set_piece_xg", "live_away_set_piece_xg"),
        ("live_home_box_entries", "live_away_box_entries"),
        ("live_home_final_third_entries", "live_away_final_third_entries"),
        ("live_home_progressive_actions", "live_away_progressive_actions"),
        ("live_home_pressures", "live_away_pressures"),
        ("live_home_high_recoveries", "live_away_high_recoveries"),
        ("live_home_turnovers", "live_away_turnovers"),
        ("live_home_yellow_cards", "live_away_yellow_cards"),
        ("live_home_red_cards", "live_away_red_cards"),
    )
    for home_name, away_name in pairs:
        underdog, favorite = role_values(home_name, away_name)
        features.extend((underdog, favorite, underdog - favorite))
    underdog_score, favorite_score = role_values(
        "live_home_score", "live_away_score"
    )
    features.extend(
        (
            underdog_score + favorite_score,
            (underdog_score - favorite_score)
            * frame["remaining_minutes"].to_numpy(dtype=float)
            / 90.0,
        )
    )
    return np.column_stack(features)


def _remaining_targets(
    frame: object,
    *,
    maximum_remaining: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    underdog_home = _underlying_role(frame)
    home_remaining = frame["remaining_home_goals"].to_numpy(dtype=int)
    away_remaining = frame["remaining_away_goals"].to_numpy(dtype=int)
    underdog_remaining = np.where(
        underdog_home, home_remaining, away_remaining
    )
    favorite_remaining = np.where(
        underdog_home, away_remaining, home_remaining
    )
    return (
        np.minimum(underdog_remaining, maximum_remaining),
        np.minimum(favorite_remaining, maximum_remaining),
    )


def _fit_classifier(
    features: np.ndarray,
    underdog_remaining: np.ndarray,
    favorite_remaining: np.ndarray,
    *,
    random_state: int,
) -> _OrientedClassifier:
    labels = np.array(
        [
            f"{int(underdog)}-{int(favorite)}"
            for underdog, favorite in zip(
                underdog_remaining, favorite_remaining, strict=True
            )
        ]
    )
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    transformed = imputer.fit_transform(features)
    classifier = ExtraTreesClassifier(
        n_estimators=240,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(transformed, labels)
    classes = tuple(
        tuple(int(value) for value in str(label).split("-", maxsplit=1))
        for label in classifier.classes_
    )
    return _OrientedClassifier(imputer, classifier, classes)


def _blend(
    baseline: np.ndarray,
    classifier: np.ndarray,
    weight: float,
) -> np.ndarray:
    output = (1.0 - weight) * baseline + weight * classifier
    return output / output.sum(axis=(1, 2), keepdims=True)


def _mean_nll(
    matrices: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
) -> float:
    values = []
    for matrix, home_goal, away_goal in zip(
        matrices, home, away, strict=True
    ):
        actual = (
            min(int(home_goal), matrix.shape[0] - 1),
            min(int(away_goal), matrix.shape[1] - 1),
        )
        values.append(-log(max(float(matrix[actual]), 1e-15)))
    return float(np.mean(values))


def _cold_path_mask(frame: object) -> np.ndarray:
    rules = _rules(frame)
    return np.logical_or.reduce(
        [
            rules["underdog_lead_two"],
            rules["underdog_two_goals_ahead"],
            rules["underdog_lead_open_game"],
            rules["underdog_two_goals_resistance"],
            rules["underdog_tied_dominant"],
        ]
    )


def _choose_weight(
    baseline: np.ndarray,
    classifier: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
    cold_path: np.ndarray,
) -> float:
    baseline_nll = _mean_nll(baseline, home, away)
    candidates: list[tuple[tuple[float, ...], float]] = []
    for weight in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0):
        matrices = _blend(baseline, classifier, weight)
        overall = _distribution_metrics(matrices, home, away)
        if cold_path.any():
            path = _distribution_metrics(
                matrices[cold_path], home[cold_path], away[cold_path]
            )
        else:
            path = overall
        feasible = overall.negative_log_likelihood <= baseline_nll + 0.035
        score = (
            1.0 if feasible else 0.0,
            path.top5_accuracy,
            path.top3_accuracy,
            path.exact_accuracy,
            overall.exact_accuracy,
            -overall.negative_log_likelihood,
        )
        candidates.append((score, weight))
    return float(max(candidates, key=lambda item: item[0])[1])


def _optional_metrics(
    matrices: np.ndarray,
    mask: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
) -> ScoreDistributionMetrics | None:
    if not mask.any():
        return None
    return _distribution_metrics(matrices[mask], home[mask], away[mask])


def leave_one_league_out_oriented_score_test(
    dataset: LiveSnapshotDataset,
    *,
    cutoffs_to_test: tuple[int, ...] = (30, 45),
) -> OrientedScoreReport:
    frame = dataset.frame
    role_features = _role_features(frame)
    underdog_remaining, favorite_remaining = _remaining_targets(frame)
    underdog_home = _underlying_role(frame)
    league = frame["division_id"].to_numpy(dtype=int)
    cutoffs = frame["snapshot_minute"].to_numpy(dtype=int)
    current_home = frame["live_home_score"].to_numpy(dtype=int)
    current_away = frame["live_away_score"].to_numpy(dtype=int)
    final_home = frame["home_score"].to_numpy(dtype=int)
    final_away = frame["away_score"].to_numpy(dtype=int)
    cold_tail = frame["upset_jackpot_target"].to_numpy(dtype=bool)
    base_home, base_away = _market_rates_for_frame(frame)

    collected: dict[int, dict[str, list[object]]] = {
        cutoff: {
            "market": [],
            "oriented": [],
            "home": [],
            "away": [],
            "cold_tail": [],
            "cold_path": [],
            "weights": [],
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
            train_order = np.argsort(
                frame.loc[train_mask, "date"].to_numpy(dtype="datetime64[ns]")
            )
            train_indices = np.where(train_mask)[0][train_order]
            test_indices = np.where(test_mask)[0]
            calibration_size = max(250, int(len(train_indices) * 0.20))
            split = len(train_indices) - calibration_size
            fit_indices = train_indices[:split]
            calibration_indices = train_indices[split:]

            calibration_model = _fit_classifier(
                role_features[fit_indices],
                underdog_remaining[fit_indices],
                favorite_remaining[fit_indices],
                random_state=401 + cutoff,
            )
            calibration_classifier = calibration_model.predict_final_matrices(
                role_features[calibration_indices],
                current_home[calibration_indices],
                current_away[calibration_indices],
                underdog_home[calibration_indices],
            )
            calibration_market = _score_matrices(
                current_home[calibration_indices],
                current_away[calibration_indices],
                base_home[calibration_indices],
                base_away[calibration_indices],
                family="poisson",
                home_dispersion=0.0,
                away_dispersion=0.0,
            )
            calibration_path = _cold_path_mask(frame.iloc[calibration_indices])
            weight = _choose_weight(
                calibration_market,
                calibration_classifier,
                final_home[calibration_indices],
                final_away[calibration_indices],
                calibration_path,
            )

            final_model = _fit_classifier(
                role_features[train_indices],
                underdog_remaining[train_indices],
                favorite_remaining[train_indices],
                random_state=411 + cutoff,
            )
            classifier_matrices = final_model.predict_final_matrices(
                role_features[test_indices],
                current_home[test_indices],
                current_away[test_indices],
                underdog_home[test_indices],
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
            oriented_matrices = _blend(
                market_matrices, classifier_matrices, weight
            )
            test_path = _cold_path_mask(frame.iloc[test_indices])
            parts = collected[cutoff]
            parts["market"].append(market_matrices)
            parts["oriented"].append(oriented_matrices)
            parts["home"].append(final_home[test_indices])
            parts["away"].append(final_away[test_indices])
            parts["cold_tail"].append(cold_tail[test_indices])
            parts["cold_path"].append(test_path)
            parts["weights"].append(weight)
            fold_details.append(
                {
                    "held_out_league": int(held_out),
                    "cutoff": int(cutoff),
                    "test_matches": int(test_mask.sum()),
                    "calibrated_weight": weight,
                    "cold_path_matches": int(test_path.sum()),
                    "cold_tail_matches": int(cold_tail[test_indices].sum()),
                }
            )

    results: list[OrientedScoreCutoffResult] = []
    for cutoff, parts in collected.items():
        market = np.concatenate(parts["market"])
        oriented = np.concatenate(parts["oriented"])
        home = np.concatenate(parts["home"])
        away = np.concatenate(parts["away"])
        cold_tail_mask = np.concatenate(parts["cold_tail"])
        cold_path_mask = np.concatenate(parts["cold_path"])
        results.append(
            OrientedScoreCutoffResult(
                cutoff=int(cutoff),
                matches=len(home),
                market_all=_distribution_metrics(market, home, away),
                oriented_all=_distribution_metrics(oriented, home, away),
                market_cold_tail=_optional_metrics(
                    market, cold_tail_mask, home, away
                ),
                oriented_cold_tail=_optional_metrics(
                    oriented, cold_tail_mask, home, away
                ),
                market_cold_path=_optional_metrics(
                    market, cold_path_mask, home, away
                ),
                oriented_cold_path=_optional_metrics(
                    oriented, cold_path_mask, home, away
                ),
                mean_blend=float(np.mean(parts["weights"])),
            )
        )
    return OrientedScoreReport(tuple(results), tuple(fold_details))
