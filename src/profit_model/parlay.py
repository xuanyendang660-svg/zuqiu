"""Two-leg parlay engine.

Only independently selected positive-EV legs from different matches are allowed.
Same-game parlays stay blocked until a validated joint-probability model exists.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

from .config import RiskConfig
from .models import ParlayDecision, SingleBetDecision
from .pricing import expected_value, fair_odds, kelly_fraction


def build_best_two_leg_parlay(
    decisions: Iterable[SingleBetDecision],
    config: RiskConfig | None = None,
    quoted_odds: dict[frozenset[str], float] | None = None,
) -> ParlayDecision:
    config = config or RiskConfig()
    eligible = [decision for decision in decisions if decision.action == "BET"]

    candidates: list[ParlayDecision] = []
    for first, second in combinations(eligible, 2):
        if (
            first.offer.match_id == second.offer.match_id
            and not config.allow_same_match_parlay
        ):
            continue

        key = frozenset((first.offer.offer_id, second.offer.offer_id))
        combined_odds = (
            quoted_odds[key]
            if quoted_odds is not None and key in quoted_odds
            else first.offer.decimal_odds * second.offer.decimal_odds
        )
        if combined_odds <= 1.0:
            continue

        joint_probability = (
            first.conservative_probability * second.conservative_probability
        )
        ev = expected_value(joint_probability, combined_odds)
        if ev < config.min_parlay_ev:
            continue

        full_kelly = kelly_fraction(joint_probability, combined_odds)
        stake = min(
            config.max_parlay_stake_fraction,
            full_kelly * config.fractional_kelly,
        )
        if stake <= 0.0:
            continue

        candidates.append(
            ParlayDecision(
                action="PARLAY",
                legs=(first, second),
                joint_probability=joint_probability,
                combined_odds=combined_odds,
                expected_value=ev,
                fair_odds=fair_odds(joint_probability),
                stake_fraction=stake,
                reason="TWO_INDEPENDENT_POSITIVE_EV_LEGS",
            )
        )

    if not candidates:
        return ParlayDecision(
            action="NO_PARLAY",
            legs=(),
            joint_probability=0.0,
            combined_odds=0.0,
            expected_value=0.0,
            fair_odds=None,
            stake_fraction=0.0,
            reason="NO_TWO_LEG_PAIR_PASSED_EV_AND_INDEPENDENCE_RULES",
        )

    return max(
        candidates,
        key=lambda item: (
            item.expected_value,
            item.joint_probability,
            item.combined_odds,
        ),
    )
