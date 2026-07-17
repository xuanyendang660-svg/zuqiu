from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .halftime_core_predictor import CORE_RULES, predict_core_halftime_exact_score
from .halftime_final_audit import PairedComparison, paired_comparison
from .halftime_state_search import (
    _baseline_map,
    _baseline_prediction,
    prepare_state_frame,
)
from .halftime_transfer_validation import validate_halftime_rule_transfer


@dataclass(frozen=True)
class CoreEvaluation:
    matches: int
    predictions: int
    coverage: float
    paired: PairedComparison
    positive_leagues: int
    evaluated_leagues: int
    by_league: dict[str, PairedComparison | None]


@dataclass(frozen=True)
class CoreProductionGate:
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class CoreMethodAudit:
    core_rule_count: int
    core_selection_matches_transfer_only_criteria: bool
    replication_shadow: CoreEvaluation
    untouched_second_shadow: CoreEvaluation
    gate: CoreProductionGate
    method_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "core_rule_count": self.core_rule_count,
            "core_selection_matches_transfer_only_criteria": (
                self.core_selection_matches_transfer_only_criteria
            ),
            "replication_shadow": _evaluation_to_dict(self.replication_shadow),
            "untouched_second_shadow": _evaluation_to_dict(
                self.untouched_second_shadow
            ),
            "gate": asdict(self.gate),
            "method_status": self.method_status,
        }


def _evaluation_to_dict(value: CoreEvaluation) -> dict[str, object]:
    return {
        "matches": value.matches,
        "predictions": value.predictions,
        "coverage": value.coverage,
        "paired": asdict(value.paired),
        "positive_leagues": value.positive_leagues,
        "evaluated_leagues": value.evaluated_leagues,
        "by_league": {
            key: asdict(metrics) if metrics is not None else None
            for key, metrics in value.by_league.items()
        },
    }


def _predict_core(frame: pd.DataFrame) -> pd.Series:
    predictions: list[str | None] = []
    for row in frame.itertuples(index=False):
        result = predict_core_halftime_exact_score(
            home_probability=float(row.market_home_prob),
            draw_probability=float(row.market_draw_prob),
            away_probability=float(row.market_away_prob),
            halftime_home_score=int(row.halftime_home),
            halftime_away_score=int(row.halftime_away),
        )
        predictions.append(result.oriented_score if result.accepted else None)
    return pd.Series(predictions, index=frame.index, dtype=object)


def _evaluate(
    frame: pd.DataFrame,
    baseline_mapping: dict[tuple[object, ...], str],
    *,
    random_state: int,
) -> CoreEvaluation:
    prediction = _predict_core(frame)
    baseline = _baseline_prediction(frame, baseline_mapping)
    paired = paired_comparison(
        frame["exact_score_target"],
        prediction,
        baseline,
        random_state=random_state,
    )
    prediction_mask = prediction.notna() & baseline.notna()
    by_league: dict[str, PairedComparison | None] = {}
    positive_leagues = 0
    evaluated_leagues = 0
    for division, group in frame.groupby("division"):
        group_mask = prediction_mask.loc[group.index]
        if int(group_mask.sum()) < 30:
            by_league[str(division)] = None
            continue
        metrics = paired_comparison(
            group["exact_score_target"],
            prediction.loc[group.index],
            baseline.loc[group.index],
            bootstrap_samples=20_000,
            random_state=random_state + evaluated_leagues + 1,
        )
        by_league[str(division)] = metrics
        evaluated_leagues += 1
        if metrics.improvement > 0.0:
            positive_leagues += 1
    predictions = int(prediction_mask.sum())
    return CoreEvaluation(
        matches=len(frame),
        predictions=predictions,
        coverage=predictions / len(frame) if len(frame) else 0.0,
        paired=paired,
        positive_leagues=positive_leagues,
        evaluated_leagues=evaluated_leagues,
        by_league=by_league,
    )


def _core_signature() -> tuple[tuple[object, ...], ...]:
    output: list[tuple[object, ...]] = []
    for rule in CORE_RULES:
        values: list[object] = [rule.halftime_underdog, rule.halftime_favourite]
        if rule.family == "score_state_prob":
            values.append(rule.probability_bin)
        elif rule.family == "score_state_gap":
            values.append(rule.market_gap_bin)
        output.append((rule.family, tuple(values), rule.oriented_prediction))
    return tuple(output)


def _transfer_selected_core(transfer_report: object) -> tuple[tuple[object, ...], ...]:
    output: list[tuple[object, ...]] = []
    for rule, metrics in zip(
        transfer_report.discovered_rules,
        transfer_report.transfer_rule_metrics,
        strict=True,
    ):
        selected = bool(
            metrics.matches >= 500
            and metrics.accuracy is not None
            and metrics.accuracy >= 0.27
            and metrics.improvement is not None
            and metrics.improvement >= 0.07
            and metrics.evaluated_leagues >= 5
            and metrics.positive_leagues == metrics.evaluated_leagues
        )
        if selected:
            output.append((rule.family, tuple(rule.values), rule.prediction))
    return tuple(output)


def run_core_method_audit(
    discovery_frame: pd.DataFrame,
    transfer_frame: pd.DataFrame,
    replication_shadow_frame: pd.DataFrame,
    untouched_second_shadow_frame: pd.DataFrame,
) -> CoreMethodAudit:
    transfer_report = validate_halftime_rule_transfer(
        discovery_frame,
        transfer_frame,
        replication_shadow_frame,
    )
    selection_match = _core_signature() == _transfer_selected_core(transfer_report)

    discovery = prepare_state_frame(discovery_frame)
    replication = prepare_state_frame(replication_shadow_frame)
    untouched = prepare_state_frame(untouched_second_shadow_frame)
    discovery_train = discovery.loc[discovery["season_start_year"] <= 2011]
    replication = replication.loc[replication["season_start_year"] >= 2022].copy()
    untouched = untouched.loc[untouched["season_start_year"] >= 2018].copy()
    baseline_mapping = _baseline_map(discovery_train, "exact_score_target")

    replication_evaluation = _evaluate(
        replication,
        baseline_mapping,
        random_state=20260717,
    )
    untouched_evaluation = _evaluate(
        untouched,
        baseline_mapping,
        random_state=20260718,
    )
    checks = {
        "core_selection_matches_transfer_only_criteria": selection_match,
        "replication_predictions_at_least_500": (
            replication_evaluation.predictions >= 500
        ),
        "replication_improvement_at_least_4pp": (
            replication_evaluation.paired.improvement >= 0.04
        ),
        "replication_p_value_below_0_01": (
            replication_evaluation.paired.mcnemar_exact_p_value < 0.01
        ),
        "second_shadow_predictions_at_least_1000": (
            untouched_evaluation.predictions >= 1000
        ),
        "second_shadow_coverage_at_least_8pct": (
            untouched_evaluation.coverage >= 0.08
        ),
        "second_shadow_improvement_at_least_3pp": (
            untouched_evaluation.paired.improvement >= 0.03
        ),
        "second_shadow_bootstrap_interval_above_zero": (
            untouched_evaluation.paired.improvement_bootstrap_95_low > 0.0
        ),
        "second_shadow_p_value_below_0_01": (
            untouched_evaluation.paired.mcnemar_exact_p_value < 0.01
        ),
        "second_shadow_positive_in_at_least_four_leagues": (
            untouched_evaluation.positive_leagues >= 4
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    gate = CoreProductionGate(
        passed=not failures,
        checks=checks,
        failures=failures,
    )
    return CoreMethodAudit(
        core_rule_count=len(CORE_RULES),
        core_selection_matches_transfer_only_criteria=selection_match,
        replication_shadow=replication_evaluation,
        untouched_second_shadow=untouched_evaluation,
        gate=gate,
        method_status=(
            "validated_core_selective_halftime_exact_score_method"
            if gate.passed
            else "rejected_by_second_shadow_gate"
        ),
    )
