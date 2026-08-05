from __future__ import annotations

from itertools import combinations

from .models import BetCandidate, ParlayCandidate, Policy


def build_two_leg_parlays(
    singles: tuple[BetCandidate, ...],
    policy: Policy,
) -> tuple[ParlayCandidate, ...]:
    """Build parlays only from legs that already qualify as standalone bets."""
    qualified = [item for item in singles if item.decision == "BET"]
    parlays: list[ParlayCandidate] = []

    for first, second in combinations(qualified, 2):
        if first.event_id == second.event_id and not policy.allow_same_event_parlay:
            continue

        combined_odds = first.odds * second.odds
        joint_probability = first.model_probability * second.model_probability
        conservative_joint = (
            first.conservative_probability * second.conservative_probability
        )
        expected_value = joint_probability * combined_odds - 1.0
        conservative_ev = conservative_joint * combined_odds - 1.0

        if conservative_ev < policy.min_parlay_ev:
            continue

        recommended_stake = min(
            policy.max_parlay_stake_fraction,
            first.recommended_stake_fraction / 2.0,
            second.recommended_stake_fraction / 2.0,
        )
        can_execute = (
            first.execution_stake_fraction > 0.0
            and second.execution_stake_fraction > 0.0
        )
        execution_stake = recommended_stake if can_execute else 0.0

        parlays.append(
            ParlayCandidate(
                decision="BET",
                legs=(first, second),
                combined_odds=combined_odds,
                joint_probability=joint_probability,
                conservative_joint_probability=conservative_joint,
                expected_value=expected_value,
                conservative_expected_value=conservative_ev,
                recommended_stake_fraction=recommended_stake,
                execution_stake_fraction=execution_stake,
                reasons=(
                    "both_legs_pass_single_bet_gates",
                    "independent_event_pricing" if first.event_id != second.event_id else "same_event_explicitly_allowed",
                    "conservative_joint_ev_passed",
                ),
            )
        )

    parlays.sort(
        key=lambda item: (
            item.conservative_expected_value,
            item.expected_value,
        ),
        reverse=True,
    )
    return tuple(parlays)
