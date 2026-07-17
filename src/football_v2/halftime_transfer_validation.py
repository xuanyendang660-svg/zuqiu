from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import pandas as pd

from .halftime_state_search import (
    StateRule,
    TaskMetrics,
    _apply_rules,
    _baseline_map,
    _baseline_prediction,
    _match_mask,
    _task_metrics,
    prepare_state_frame,
    run_halftime_state_search,
)


@dataclass(frozen=True)
class RuleTransferMetrics:
    family: str
    values: tuple[object, ...]
    prediction: str
    matches: int
    accuracy: float | None
    baseline_accuracy: float | None
    improvement: float | None
    positive_leagues: int
    evaluated_leagues: int
    wilson_95_low: float | None
    wilson_95_high: float | None


@dataclass(frozen=True)
class TransferValidationReport:
    discovered_rules: tuple[StateRule, ...]
    transfer_rule_metrics: tuple[RuleTransferMetrics, ...]
    retained_rules: tuple[StateRule, ...]
    transfer_combined: TaskMetrics
    final_shadow: TaskMetrics
    final_shadow_by_league: dict[str, TaskMetrics]
    final_shadow_by_rule: tuple[RuleTransferMetrics, ...]
    split_definition: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "discovered_rules": [asdict(rule) for rule in self.discovered_rules],
            "transfer_rule_metrics": [
                asdict(metrics) for metrics in self.transfer_rule_metrics
            ],
            "retained_rules": [asdict(rule) for rule in self.retained_rules],
            "transfer_combined": asdict(self.transfer_combined),
            "final_shadow": asdict(self.final_shadow),
            "final_shadow_by_league": {
                key: asdict(value)
                for key, value in self.final_shadow_by_league.items()
            },
            "final_shadow_by_rule": [
                asdict(metrics) for metrics in self.final_shadow_by_rule
            ],
            "split_definition": self.split_definition,
        }


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


def _rule_metrics(
    frame: pd.DataFrame,
    rule: StateRule,
    baseline: pd.Series,
    *,
    target: str = "exact_score_target",
) -> RuleTransferMetrics:
    mask = _match_mask(frame, rule.columns, rule.values)
    group = frame.loc[mask]
    matches = len(group)
    if matches == 0:
        return RuleTransferMetrics(
            family=rule.family,
            values=rule.values,
            prediction=rule.prediction,
            matches=0,
            accuracy=None,
            baseline_accuracy=None,
            improvement=None,
            positive_leagues=0,
            evaluated_leagues=0,
            wilson_95_low=None,
            wilson_95_high=None,
        )
    actual = group[target].astype(str)
    successes = int((actual == rule.prediction).sum())
    accuracy = successes / matches
    baseline_values = baseline.loc[mask]
    valid = baseline_values.notna()
    baseline_accuracy = (
        float(
            (
                baseline_values.loc[valid].astype(str)
                == group.loc[valid, target].astype(str)
            ).mean()
        )
        if bool(valid.any())
        else None
    )
    positive_leagues = 0
    evaluated_leagues = 0
    for _, league_group in group.groupby("division"):
        if len(league_group) < 5:
            continue
        evaluated_leagues += 1
        league_accuracy = float(
            (league_group[target].astype(str) == rule.prediction).mean()
        )
        league_baseline = baseline.loc[league_group.index]
        league_valid = league_baseline.notna()
        if not bool(league_valid.any()):
            continue
        league_baseline_accuracy = float(
            (
                league_baseline.loc[league_valid].astype(str)
                == league_group.loc[league_valid, target].astype(str)
            ).mean()
        )
        if league_accuracy > league_baseline_accuracy:
            positive_leagues += 1
    low, high = _wilson(successes, matches)
    return RuleTransferMetrics(
        family=rule.family,
        values=rule.values,
        prediction=rule.prediction,
        matches=matches,
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        improvement=(
            accuracy - baseline_accuracy
            if baseline_accuracy is not None
            else None
        ),
        positive_leagues=positive_leagues,
        evaluated_leagues=evaluated_leagues,
        wilson_95_low=low,
        wilson_95_high=high,
    )


def _retain_rule(metrics: RuleTransferMetrics) -> bool:
    return bool(
        metrics.matches >= 25
        and metrics.accuracy is not None
        and metrics.baseline_accuracy is not None
        and metrics.accuracy >= 0.20
        and metrics.improvement is not None
        and metrics.improvement >= 0.015
        and metrics.positive_leagues >= 2
    )


def validate_halftime_rule_transfer(
    discovery_frame: pd.DataFrame,
    transfer_frame: pd.DataFrame,
    shadow_frame: pd.DataFrame,
    *,
    discovery_train_end_year: int = 2011,
    discovery_calibration_end_year: int = 2017,
    transfer_start_year: int = 2018,
    transfer_end_year: int = 2021,
    shadow_start_year: int = 2022,
) -> TransferValidationReport:
    discovery = prepare_state_frame(discovery_frame)
    transfer = prepare_state_frame(transfer_frame)
    shadow = prepare_state_frame(shadow_frame)

    discovery_report = run_halftime_state_search(
        discovery_frame,
        train_end_year=discovery_train_end_year,
        calibration_end_year=discovery_calibration_end_year,
    )
    rules = discovery_report.tasks["exact_score"].selected_rules
    discovery_train = discovery.loc[
        discovery["season_start_year"] <= discovery_train_end_year
    ]
    baseline_mapping = _baseline_map(discovery_train, "exact_score_target")

    transfer = transfer.loc[
        (transfer["season_start_year"] >= transfer_start_year)
        & (transfer["season_start_year"] <= transfer_end_year)
    ].copy()
    shadow = shadow.loc[
        shadow["season_start_year"] >= shadow_start_year
    ].copy()
    if transfer.empty or shadow.empty:
        raise ValueError("transfer and shadow datasets must be non-empty")

    transfer_baseline = _baseline_prediction(transfer, baseline_mapping)
    transfer_metrics = tuple(
        _rule_metrics(transfer, rule, transfer_baseline)
        for rule in rules
    )
    retained = tuple(
        rule
        for rule, metrics in zip(rules, transfer_metrics, strict=True)
        if _retain_rule(metrics)
    )

    transfer_prediction = _apply_rules(transfer, retained)
    transfer_combined = _task_metrics(
        transfer,
        "exact_score_target",
        transfer_prediction,
        transfer_baseline,
    )

    shadow_baseline = _baseline_prediction(shadow, baseline_mapping)
    shadow_prediction = _apply_rules(shadow, retained)
    shadow_metrics = _task_metrics(
        shadow,
        "exact_score_target",
        shadow_prediction,
        shadow_baseline,
    )
    shadow_by_league = {
        str(division): _task_metrics(
            group,
            "exact_score_target",
            shadow_prediction.loc[group.index],
            shadow_baseline.loc[group.index],
        )
        for division, group in shadow.groupby("division")
    }
    shadow_rule_metrics = tuple(
        _rule_metrics(shadow, rule, shadow_baseline)
        for rule in retained
    )
    return TransferValidationReport(
        discovered_rules=rules,
        transfer_rule_metrics=transfer_metrics,
        retained_rules=retained,
        transfer_combined=transfer_combined,
        final_shadow=shadow_metrics,
        final_shadow_by_league=shadow_by_league,
        final_shadow_by_rule=shadow_rule_metrics,
        split_definition={
            "discovery_train": (
                f"top-five leagues, season_start_year <= {discovery_train_end_year}"
            ),
            "discovery_calibration": (
                "top-five leagues, "
                f"{discovery_train_end_year} < season_start_year "
                f"<= {discovery_calibration_end_year}"
            ),
            "transfer_filter": (
                "second divisions, "
                f"{transfer_start_year} <= season_start_year <= {transfer_end_year}"
            ),
            "final_shadow": (
                "other first divisions, "
                f"season_start_year >= {shadow_start_year}"
            ),
        },
    )
