from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log1p, sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from .halftime_two_nil import build_oriented_halftime_frame


@dataclass(frozen=True)
class StateRule:
    task: str
    family: str
    columns: tuple[str, ...]
    values: tuple[object, ...]
    prediction: str
    train_matches: int
    train_accuracy: float
    calibration_matches: int
    calibration_accuracy: float
    calibration_baseline_accuracy: float
    calibration_improvement: float
    priority: float


@dataclass(frozen=True)
class TaskMetrics:
    matches: int
    predictions: int
    coverage: float
    accuracy: float | None
    baseline_accuracy: float | None
    improvement: float | None
    wilson_95_low: float | None
    wilson_95_high: float | None


@dataclass(frozen=True)
class TaskReport:
    task: str
    selected_rules: tuple[StateRule, ...]
    train: TaskMetrics
    calibration: TaskMetrics
    test: TaskMetrics
    test_by_league: dict[str, TaskMetrics]
    test_by_era: dict[str, TaskMetrics]
    prediction_distribution: dict[str, int]
    actual_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "selected_rules": [asdict(rule) for rule in self.selected_rules],
            "train": asdict(self.train),
            "calibration": asdict(self.calibration),
            "test": asdict(self.test),
            "test_by_league": {
                key: asdict(value) for key, value in self.test_by_league.items()
            },
            "test_by_era": {
                key: asdict(value) for key, value in self.test_by_era.items()
            },
            "prediction_distribution": self.prediction_distribution,
            "actual_distribution": self.actual_distribution,
        }


@dataclass(frozen=True)
class HalftimeStateSearchReport:
    rows: int
    train_rows: int
    calibration_rows: int
    test_rows: int
    split_dates: dict[str, str]
    tasks: dict[str, TaskReport]

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "train_rows": self.train_rows,
            "calibration_rows": self.calibration_rows,
            "test_rows": self.test_rows,
            "split_dates": self.split_dates,
            "tasks": {
                key: value.to_dict() for key, value in self.tasks.items()
            },
        }


def _mode(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    if counts.empty:
        raise ValueError("cannot calculate a mode from an empty series")
    maximum = int(counts.max())
    return str(sorted(counts[counts == maximum].index)[0])


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    spread = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def prepare_state_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = build_oriented_halftime_frame(frame)
    data["home_underdog"] = data["market_home_prob"] < data["market_away_prob"]
    data["ht_udg_cap"] = data["halftime_underdog"].clip(upper=3).astype(int)
    data["ht_fav_cap"] = data["halftime_favourite"].clip(upper=3).astype(int)
    data["ht_total_cap"] = (
        data["halftime_underdog"] + data["halftime_favourite"]
    ).clip(upper=5).astype(int)
    data["underdog_prob_bin"] = pd.cut(
        data["underdog_prob"],
        bins=[-np.inf, 0.20, 0.25, 0.30, 0.35, 0.40, np.inf],
        labels=["p20", "p25", "p30", "p35", "p40", "p50"],
        include_lowest=True,
    ).astype(str)
    data["market_gap_bin"] = pd.cut(
        data["market_gap"],
        bins=[-np.inf, 0.05, 0.10, 0.15, 0.20, 0.30, np.inf],
        labels=["g05", "g10", "g15", "g20", "g30", "g50"],
        include_lowest=True,
    ).astype(str)
    data["exact_score_target"] = data["final_oriented_score"].astype(str)
    data["direction_target"] = np.where(
        data["final_underdog"] > data["final_favourite"],
        "underdog_win",
        np.where(
            data["final_underdog"] == data["final_favourite"],
            "draw",
            "favourite_win",
        ),
    )
    final_total = data["final_underdog"] + data["final_favourite"]
    data["total_family_target"] = np.select(
        [final_total <= 1, final_total <= 3],
        ["0-1", "2-3"],
        default="4+",
    )
    additional_total = final_total - (
        data["halftime_underdog"] + data["halftime_favourite"]
    )
    data["additional_goal_target"] = np.select(
        [additional_total <= 0, additional_total == 1, additional_total == 2],
        ["0", "1", "2"],
        default="3+",
    )
    data["season_start_year"] = data["date"].dt.year - (
        data["date"].dt.month < 7
    ).astype(int)
    return data


def _baseline_map(train: pd.DataFrame, target: str) -> dict[tuple[object, ...], str]:
    columns = ("ht_udg_cap", "ht_fav_cap")
    output: dict[tuple[object, ...], str] = {}
    for values, group in train.groupby(list(columns), dropna=False):
        key = values if isinstance(values, tuple) else (values,)
        output[tuple(key)] = _mode(group[target])
    return output


def _baseline_prediction(
    frame: pd.DataFrame,
    mapping: dict[tuple[object, ...], str],
) -> pd.Series:
    keys = list(
        zip(
            frame["ht_udg_cap"].tolist(),
            frame["ht_fav_cap"].tolist(),
            strict=True,
        )
    )
    return pd.Series([mapping.get(tuple(key)) for key in keys], index=frame.index)


def _match_mask(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    values: tuple[object, ...],
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in zip(columns, values, strict=True):
        mask &= frame[column].astype(str) == str(value)
    return mask


def _candidate_families() -> tuple[tuple[str, tuple[str, ...]], ...]:
    base = ("ht_udg_cap", "ht_fav_cap")
    return (
        ("score_state", base),
        ("score_state_prob", base + ("underdog_prob_bin",)),
        ("score_state_gap", base + ("market_gap_bin",)),
        ("score_state_home", base + ("home_underdog",)),
        (
            "score_state_prob_home",
            base + ("underdog_prob_bin", "home_underdog"),
        ),
        (
            "score_state_gap_home",
            base + ("market_gap_bin", "home_underdog"),
        ),
        (
            "total_state_prob",
            ("ht_total_cap", "underdog_prob_bin", "home_underdog"),
        ),
    )


def _discover_rules(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    task: str,
    target: str,
    minimum_train: int,
    minimum_calibration: int,
    minimum_train_accuracy: float,
    minimum_calibration_accuracy: float,
    minimum_improvement: float,
) -> tuple[StateRule, ...]:
    baseline_mapping = _baseline_map(train, target)
    calibration_baseline = _baseline_prediction(calibration, baseline_mapping)
    candidates: list[StateRule] = []

    for family, columns in _candidate_families():
        grouped = train.groupby(list(columns), dropna=False)
        for raw_values, group in grouped:
            values = raw_values if isinstance(raw_values, tuple) else (raw_values,)
            values = tuple(values)
            if len(group) < minimum_train:
                continue
            prediction = _mode(group[target])
            train_accuracy = float((group[target].astype(str) == prediction).mean())
            if train_accuracy < minimum_train_accuracy:
                continue
            mask = _match_mask(calibration, columns, values)
            calibration_group = calibration.loc[mask]
            if len(calibration_group) < minimum_calibration:
                continue
            calibration_accuracy = float(
                (calibration_group[target].astype(str) == prediction).mean()
            )
            baseline_values = calibration_baseline.loc[mask]
            valid_baseline = baseline_values.notna()
            baseline_accuracy = (
                float(
                    (
                        baseline_values.loc[valid_baseline].astype(str)
                        == calibration_group.loc[valid_baseline, target].astype(str)
                    ).mean()
                )
                if bool(valid_baseline.any())
                else 0.0
            )
            improvement = calibration_accuracy - baseline_accuracy
            if calibration_accuracy < minimum_calibration_accuracy:
                continue
            if improvement < minimum_improvement:
                continue
            priority = (
                calibration_accuracy
                + 0.50 * improvement
                + 0.01 * log1p(len(calibration_group))
            )
            candidates.append(
                StateRule(
                    task=task,
                    family=family,
                    columns=columns,
                    values=values,
                    prediction=prediction,
                    train_matches=len(group),
                    train_accuracy=train_accuracy,
                    calibration_matches=len(calibration_group),
                    calibration_accuracy=calibration_accuracy,
                    calibration_baseline_accuracy=baseline_accuracy,
                    calibration_improvement=improvement,
                    priority=priority,
                )
            )

    candidates.sort(
        key=lambda rule: (
            rule.priority,
            rule.calibration_matches,
            rule.train_matches,
        ),
        reverse=True,
    )
    selected: list[StateRule] = []
    calibration_claimed = pd.Series(False, index=calibration.index)
    for candidate in candidates:
        mask = _match_mask(calibration, candidate.columns, candidate.values)
        new_mask = mask & ~calibration_claimed
        if int(new_mask.sum()) < max(5, minimum_calibration // 2):
            continue
        new_accuracy = float(
            (
                calibration.loc[new_mask, target].astype(str)
                == candidate.prediction
            ).mean()
        )
        if new_accuracy < minimum_calibration_accuracy:
            continue
        selected.append(candidate)
        calibration_claimed |= mask
        if len(selected) >= 30:
            break
    return tuple(selected)


def _apply_rules(
    frame: pd.DataFrame,
    rules: Iterable[StateRule],
) -> pd.Series:
    prediction = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    for rule in rules:
        mask = _match_mask(frame, rule.columns, rule.values) & prediction.isna()
        prediction.loc[mask] = rule.prediction
    return prediction


def _task_metrics(
    frame: pd.DataFrame,
    target: str,
    prediction: pd.Series,
    baseline: pd.Series,
) -> TaskMetrics:
    mask = prediction.notna()
    predictions = int(mask.sum())
    if predictions == 0:
        return TaskMetrics(
            matches=len(frame),
            predictions=0,
            coverage=0.0,
            accuracy=None,
            baseline_accuracy=None,
            improvement=None,
            wilson_95_low=None,
            wilson_95_high=None,
        )
    actual = frame.loc[mask, target].astype(str)
    predicted = prediction.loc[mask].astype(str)
    successes = int((actual == predicted).sum())
    accuracy = successes / predictions
    baseline_values = baseline.loc[mask]
    baseline_mask = baseline_values.notna()
    baseline_accuracy = (
        float(
            (
                baseline_values.loc[baseline_mask].astype(str)
                == actual.loc[baseline_mask]
            ).mean()
        )
        if bool(baseline_mask.any())
        else None
    )
    low, high = _wilson(successes, predictions)
    return TaskMetrics(
        matches=len(frame),
        predictions=predictions,
        coverage=predictions / len(frame) if len(frame) else 0.0,
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        improvement=(
            accuracy - baseline_accuracy if baseline_accuracy is not None else None
        ),
        wilson_95_low=low,
        wilson_95_high=high,
    )


def _build_task_report(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    task: str,
    target: str,
    minimum_train: int,
    minimum_calibration: int,
    minimum_train_accuracy: float,
    minimum_calibration_accuracy: float,
    minimum_improvement: float,
) -> TaskReport:
    rules = _discover_rules(
        train,
        calibration,
        task=task,
        target=target,
        minimum_train=minimum_train,
        minimum_calibration=minimum_calibration,
        minimum_train_accuracy=minimum_train_accuracy,
        minimum_calibration_accuracy=minimum_calibration_accuracy,
        minimum_improvement=minimum_improvement,
    )
    baseline_mapping = _baseline_map(train, target)
    frames = {"train": train, "calibration": calibration, "test": test}
    predictions = {name: _apply_rules(value, rules) for name, value in frames.items()}
    baselines = {
        name: _baseline_prediction(value, baseline_mapping)
        for name, value in frames.items()
    }
    test_prediction = predictions["test"]
    predicted_mask = test_prediction.notna()
    return TaskReport(
        task=task,
        selected_rules=rules,
        train=_task_metrics(
            train, target, predictions["train"], baselines["train"]
        ),
        calibration=_task_metrics(
            calibration,
            target,
            predictions["calibration"],
            baselines["calibration"],
        ),
        test=_task_metrics(test, target, test_prediction, baselines["test"]),
        test_by_league={
            str(division): _task_metrics(
                group,
                target,
                test_prediction.loc[group.index],
                baselines["test"].loc[group.index],
            )
            for division, group in test.groupby("division")
        },
        test_by_era={
            str(era): _task_metrics(
                group,
                target,
                test_prediction.loc[group.index],
                baselines["test"].loc[group.index],
            )
            for era, group in test.groupby("era")
        },
        prediction_distribution={
            str(key): int(value)
            for key, value in test_prediction.loc[predicted_mask]
            .value_counts()
            .items()
        },
        actual_distribution={
            str(key): int(value)
            for key, value in test.loc[predicted_mask, target]
            .astype(str)
            .value_counts()
            .items()
        },
    )


def run_halftime_state_search(
    frame: pd.DataFrame,
    *,
    train_end_year: int = 2011,
    calibration_end_year: int = 2017,
) -> HalftimeStateSearchReport:
    data = prepare_state_frame(frame)
    train = data.loc[data["season_start_year"] <= train_end_year].copy()
    calibration = data.loc[
        (data["season_start_year"] > train_end_year)
        & (data["season_start_year"] <= calibration_end_year)
    ].copy()
    test = data.loc[data["season_start_year"] > calibration_end_year].copy()
    if min(len(train), len(calibration), len(test)) == 0:
        raise ValueError("train, calibration, and test splits must all be non-empty")

    task_specs = {
        "exact_score": {
            "target": "exact_score_target",
            "minimum_train": 60,
            "minimum_calibration": 15,
            "minimum_train_accuracy": 0.18,
            "minimum_calibration_accuracy": 0.20,
            "minimum_improvement": 0.025,
        },
        "direction": {
            "target": "direction_target",
            "minimum_train": 80,
            "minimum_calibration": 20,
            "minimum_train_accuracy": 0.55,
            "minimum_calibration_accuracy": 0.58,
            "minimum_improvement": 0.025,
        },
        "total_family": {
            "target": "total_family_target",
            "minimum_train": 80,
            "minimum_calibration": 20,
            "minimum_train_accuracy": 0.50,
            "minimum_calibration_accuracy": 0.53,
            "minimum_improvement": 0.025,
        },
        "additional_goals": {
            "target": "additional_goal_target",
            "minimum_train": 80,
            "minimum_calibration": 20,
            "minimum_train_accuracy": 0.40,
            "minimum_calibration_accuracy": 0.43,
            "minimum_improvement": 0.025,
        },
    }
    reports = {
        task: _build_task_report(
            train,
            calibration,
            test,
            task=task,
            **spec,
        )
        for task, spec in task_specs.items()
    }
    return HalftimeStateSearchReport(
        rows=len(data),
        train_rows=len(train),
        calibration_rows=len(calibration),
        test_rows=len(test),
        split_dates={
            "train": f"season_start_year <= {train_end_year}",
            "calibration": (
                f"{train_end_year} < season_start_year <= {calibration_end_year}"
            ),
            "test": f"season_start_year > {calibration_end_year}",
        },
        tasks=reports,
    )
