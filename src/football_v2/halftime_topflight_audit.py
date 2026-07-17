from __future__ import annotations

from dataclasses import asdict, dataclass

from .halftime_core_audit import (
    CoreEvaluation,
    _core_signature,
    _evaluate,
    _transfer_selected_core,
)
from .halftime_state_search import _baseline_map, prepare_state_frame
from .halftime_transfer_validation import validate_halftime_rule_transfer


@dataclass(frozen=True)
class TopFlightGate:
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class TopFlightDomainAudit:
    core_selection_matches_transfer_only_criteria: bool
    modern_topflight_replication: CoreEvaluation
    historical_topflight_blind_test: CoreEvaluation
    gate: TopFlightGate
    method_status: str
    supported_domain: str

    def to_dict(self) -> dict[str, object]:
        return {
            "core_selection_matches_transfer_only_criteria": (
                self.core_selection_matches_transfer_only_criteria
            ),
            "modern_topflight_replication": _evaluation_to_dict(
                self.modern_topflight_replication
            ),
            "historical_topflight_blind_test": _evaluation_to_dict(
                self.historical_topflight_blind_test
            ),
            "gate": asdict(self.gate),
            "method_status": self.method_status,
            "supported_domain": self.supported_domain,
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


def run_topflight_domain_audit(
    discovery_frame: object,
    transfer_frame: object,
    modern_topflight_frame: object,
    historical_topflight_frame: object,
) -> TopFlightDomainAudit:
    transfer_report = validate_halftime_rule_transfer(
        discovery_frame,
        transfer_frame,
        modern_topflight_frame,
    )
    selection_match = _core_signature() == _transfer_selected_core(transfer_report)

    discovery = prepare_state_frame(discovery_frame)
    modern = prepare_state_frame(modern_topflight_frame)
    historical = prepare_state_frame(historical_topflight_frame)
    discovery_train = discovery.loc[discovery["season_start_year"] <= 2011]
    modern = modern.loc[modern["season_start_year"] >= 2022].copy()
    historical = historical.loc[
        (historical["season_start_year"] >= 2002)
        & (historical["season_start_year"] <= 2017)
    ].copy()
    baseline_mapping = _baseline_map(discovery_train, "exact_score_target")

    modern_evaluation = _evaluate(
        modern,
        baseline_mapping,
        random_state=20260719,
    )
    historical_evaluation = _evaluate(
        historical,
        baseline_mapping,
        random_state=20260720,
    )
    checks = {
        "core_selection_matches_transfer_only_criteria": selection_match,
        "modern_predictions_at_least_500": modern_evaluation.predictions >= 500,
        "modern_improvement_at_least_4pp": (
            modern_evaluation.paired.improvement >= 0.04
        ),
        "modern_bootstrap_interval_above_zero": (
            modern_evaluation.paired.improvement_bootstrap_95_low > 0.0
        ),
        "modern_p_value_below_0_01": (
            modern_evaluation.paired.mcnemar_exact_p_value < 0.01
        ),
        "historical_predictions_at_least_1500": (
            historical_evaluation.predictions >= 1500
        ),
        "historical_coverage_at_least_8pct": (
            historical_evaluation.coverage >= 0.08
        ),
        "historical_improvement_at_least_3pp": (
            historical_evaluation.paired.improvement >= 0.03
        ),
        "historical_bootstrap_interval_above_zero": (
            historical_evaluation.paired.improvement_bootstrap_95_low > 0.0
        ),
        "historical_p_value_below_0_001": (
            historical_evaluation.paired.mcnemar_exact_p_value < 0.001
        ),
        "historical_positive_in_at_least_four_leagues": (
            historical_evaluation.positive_leagues >= 4
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    gate = TopFlightGate(
        passed=not failures,
        checks=checks,
        failures=failures,
    )
    return TopFlightDomainAudit(
        core_selection_matches_transfer_only_criteria=selection_match,
        modern_topflight_replication=modern_evaluation,
        historical_topflight_blind_test=historical_evaluation,
        gate=gate,
        method_status=(
            "validated_topflight_selective_halftime_exact_score_method"
            if gate.passed
            else "rejected_by_topflight_domain_gate"
        ),
        supported_domain=(
            "top-flight domestic leagues only; halftime 0-0; strong pre-match favourite"
        ),
    )
