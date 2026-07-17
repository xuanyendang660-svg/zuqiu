from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .halftime_selective_predictor import (
    FROZEN_RULES,
    predict_halftime_exact_score,
)
from .halftime_state_search import (
    _baseline_map,
    _baseline_prediction,
    prepare_state_frame,
)
from .halftime_transfer_validation import validate_halftime_rule_transfer


@dataclass(frozen=True)
class PairedComparison:
    predictions: int
    both_correct: int
    model_only_correct: int
    baseline_only_correct: int
    neither_correct: int
    model_accuracy: float
    baseline_accuracy: float
    improvement: float
    improvement_bootstrap_95_low: float
    improvement_bootstrap_95_high: float
    mcnemar_exact_p_value: float


@dataclass(frozen=True)
class FrozenEvaluation:
    matches: int
    predictions: int
    coverage: float
    paired: PairedComparison
    positive_leagues: int
    evaluated_leagues: int
    by_league: dict[str, PairedComparison | None]
    by_rule: dict[str, PairedComparison | None]


@dataclass(frozen=True)
class ProductionGate:
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class FinalMethodAudit:
    frozen_rule_count: int
    frozen_rules_match_transfer_selection: bool
    transfer: FrozenEvaluation
    shadow: FrozenEvaluation
    gate: ProductionGate
    method_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "frozen_rule_count": self.frozen_rule_count,
            "frozen_rules_match_transfer_selection": (
                self.frozen_rules_match_transfer_selection
            ),
            "transfer": _evaluation_to_dict(self.transfer),
            "shadow": _evaluation_to_dict(self.shadow),
            "gate": asdict(self.gate),
            "method_status": self.method_status,
        }


def _comparison_to_dict(value: PairedComparison | None) -> dict[str, object] | None:
    return asdict(value) if value is not None else None


def _evaluation_to_dict(value: FrozenEvaluation) -> dict[str, object]:
    return {
        "matches": value.matches,
        "predictions": value.predictions,
        "coverage": value.coverage,
        "paired": asdict(value.paired),
        "positive_leagues": value.positive_leagues,
        "evaluated_leagues": value.evaluated_leagues,
        "by_league": {
            key: _comparison_to_dict(metrics)
            for key, metrics in value.by_league.items()
        },
        "by_rule": {
            key: _comparison_to_dict(metrics)
            for key, metrics in value.by_rule.items()
        },
    }


def paired_comparison(
    actual: pd.Series,
    model_prediction: pd.Series,
    baseline_prediction: pd.Series,
    *,
    bootstrap_samples: int = 50_000,
    random_state: int = 20260717,
) -> PairedComparison:
    mask = model_prediction.notna() & baseline_prediction.notna()
    actual_values = actual.loc[mask].astype(str).to_numpy()
    model_values = model_prediction.loc[mask].astype(str).to_numpy()
    baseline_values = baseline_prediction.loc[mask].astype(str).to_numpy()
    predictions = len(actual_values)
    if predictions <= 0:
        raise ValueError("paired comparison requires at least one prediction")

    model_correct = model_values == actual_values
    baseline_correct = baseline_values == actual_values
    both = int(np.logical_and(model_correct, baseline_correct).sum())
    model_only = int(np.logical_and(model_correct, ~baseline_correct).sum())
    baseline_only = int(np.logical_and(~model_correct, baseline_correct).sum())
    neither = int(np.logical_and(~model_correct, ~baseline_correct).sum())
    model_accuracy = float(model_correct.mean())
    baseline_accuracy = float(baseline_correct.mean())
    improvement = model_accuracy - baseline_accuracy

    probabilities = np.array(
        [both, model_only, baseline_only, neither],
        dtype=float,
    ) / predictions
    rng = np.random.default_rng(random_state)
    bootstrap = rng.multinomial(
        predictions,
        probabilities,
        size=bootstrap_samples,
    )
    differences = (bootstrap[:, 1] - bootstrap[:, 2]) / predictions
    low, high = np.quantile(differences, [0.025, 0.975])
    discordant = model_only + baseline_only
    p_value = (
        float(
            binomtest(
                min(model_only, baseline_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant > 0
        else 1.0
    )
    return PairedComparison(
        predictions=predictions,
        both_correct=both,
        model_only_correct=model_only,
        baseline_only_correct=baseline_only,
        neither_correct=neither,
        model_accuracy=model_accuracy,
        baseline_accuracy=baseline_accuracy,
        improvement=improvement,
        improvement_bootstrap_95_low=float(low),
        improvement_bootstrap_95_high=float(high),
        mcnemar_exact_p_value=p_value,
    )


def _predict_frame(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    predictions: list[str | None] = []
    rule_ids: list[str | None] = []
    for row in frame.itertuples(index=False):
        result = predict_halftime_exact_score(
            home_probability=float(row.market_home_prob),
            draw_probability=float(row.market_draw_prob),
            away_probability=float(row.market_away_prob),
            halftime_home_score=int(row.halftime_home),
            halftime_away_score=int(row.halftime_away),
        )
        predictions.append(result.oriented_score if result.accepted else None)
        rule_ids.append(result.rule_id if result.accepted else None)
    return (
        pd.Series(predictions, index=frame.index, dtype=object),
        pd.Series(rule_ids, index=frame.index, dtype=object),
    )


def _evaluate(
    frame: pd.DataFrame,
    baseline_mapping: dict[tuple[object, ...], str],
) -> FrozenEvaluation:
    prediction, rule_ids = _predict_frame(frame)
    baseline = _baseline_prediction(frame, baseline_mapping)
    mask = prediction.notna() & baseline.notna()
    paired = paired_comparison(
        frame["exact_score_target"],
        prediction,
        baseline,
    )
    by_league: dict[str, PairedComparison | None] = {}
    positive_leagues = 0
    evaluated_leagues = 0
    for division, group in frame.groupby("division"):
        group_mask = mask.loc[group.index]
        if int(group_mask.sum()) < 20:
            by_league[str(division)] = None
            continue
        metrics = paired_comparison(
            group["exact_score_target"],
            prediction.loc[group.index],
            baseline.loc[group.index],
            bootstrap_samples=10_000,
            random_state=20260717 + evaluated_leagues,
        )
        by_league[str(division)] = metrics
        evaluated_leagues += 1
        if metrics.improvement > 0.0:
            positive_leagues += 1

    by_rule: dict[str, PairedComparison | None] = {}
    for rule in FROZEN_RULES:
        rule_mask = rule_ids == rule.rule_id
        if int(rule_mask.sum()) < 10:
            by_rule[rule.rule_id] = None
            continue
        by_rule[rule.rule_id] = paired_comparison(
            frame.loc[rule_mask, "exact_score_target"],
            prediction.loc[rule_mask],
            baseline.loc[rule_mask],
            bootstrap_samples=10_000,
            random_state=20260717 + int(rule.rule_id[-2:]),
        )

    predictions = int(mask.sum())
    return FrozenEvaluation(
        matches=len(frame),
        predictions=predictions,
        coverage=predictions / len(frame) if len(frame) else 0.0,
        paired=paired,
        positive_leagues=positive_leagues,
        evaluated_leagues=evaluated_leagues,
        by_league=by_league,
        by_rule=by_rule,
    )


def _rule_signature_from_frozen() -> tuple[tuple[object, ...], ...]:
    output: list[tuple[object, ...]] = []
    for rule in FROZEN_RULES:
        values: list[object] = [
            rule.halftime_underdog,
            rule.halftime_favourite,
        ]
        if rule.family in {"score_state_prob", "score_state_prob_home"}:
            values.append(rule.probability_bin)
        elif rule.family in {"score_state_gap", "score_state_gap_home"}:
            values.append(rule.market_gap_bin)
        if rule.family in {"score_state_prob_home", "score_state_gap_home"}:
            values.append(rule.home_underdog)
        output.append((rule.family, tuple(values), rule.oriented_prediction))
    return tuple(output)


def _rule_signature_from_selection(selection: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            rule.family,
            tuple(rule.values),
            rule.prediction,
        )
        for rule in selection
    )


def run_final_method_audit(
    discovery_frame: pd.DataFrame,
    transfer_frame: pd.DataFrame,
    shadow_frame: pd.DataFrame,
) -> FinalMethodAudit:
    transfer_report = validate_halftime_rule_transfer(
        discovery_frame,
        transfer_frame,
        shadow_frame,
    )
    frozen_match = (
        _rule_signature_from_frozen()
        == _rule_signature_from_selection(transfer_report.retained_rules)
    )

    discovery = prepare_state_frame(discovery_frame)
    transfer = prepare_state_frame(transfer_frame)
    shadow = prepare_state_frame(shadow_frame)
    discovery_train = discovery.loc[discovery["season_start_year"] <= 2011]
    transfer = transfer.loc[
        (transfer["season_start_year"] >= 2018)
        & (transfer["season_start_year"] <= 2021)
    ].copy()
    shadow = shadow.loc[shadow["season_start_year"] >= 2022].copy()
    baseline_mapping = _baseline_map(discovery_train, "exact_score_target")
    transfer_evaluation = _evaluate(transfer, baseline_mapping)
    shadow_evaluation = _evaluate(shadow, baseline_mapping)

    checks = {
        "frozen_rules_match_transfer_selection": frozen_match,
        "transfer_predictions_at_least_1000": transfer_evaluation.predictions >= 1000,
        "transfer_improvement_at_least_4pp": (
            transfer_evaluation.paired.improvement >= 0.04
        ),
        "transfer_p_value_below_0_01": (
            transfer_evaluation.paired.mcnemar_exact_p_value < 0.01
        ),
        "shadow_predictions_at_least_1500": shadow_evaluation.predictions >= 1500,
        "shadow_coverage_at_least_20pct": shadow_evaluation.coverage >= 0.20,
        "shadow_improvement_at_least_2pp": (
            shadow_evaluation.paired.improvement >= 0.02
        ),
        "shadow_bootstrap_interval_above_zero": (
            shadow_evaluation.paired.improvement_bootstrap_95_low > 0.0
        ),
        "shadow_p_value_below_0_05": (
            shadow_evaluation.paired.mcnemar_exact_p_value < 0.05
        ),
        "shadow_positive_in_at_least_four_leagues": (
            shadow_evaluation.positive_leagues >= 4
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    gate = ProductionGate(
        passed=not failures,
        checks=checks,
        failures=failures,
    )
    return FinalMethodAudit(
        frozen_rule_count=len(FROZEN_RULES),
        frozen_rules_match_transfer_selection=frozen_match,
        transfer=transfer_evaluation,
        shadow=shadow_evaluation,
        gate=gate,
        method_status=(
            "validated_selective_halftime_exact_score_method"
            if gate.passed
            else "rejected_by_production_gate"
        ),
    )
