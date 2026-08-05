"""Positive-EV single-bet selector with an explicit NO_BET path."""

from __future__ import annotations

from collections.abc import Iterable

from .config import RiskConfig
from .models import BetOffer, SingleBetDecision
from .pricing import expected_value, fair_odds, implied_probability, kelly_fraction


def conservative_probability(offer: BetOffer, config: RiskConfig) -> float:
    if offer.model_probability_lower is not None:
        return min(offer.model_probability, offer.model_probability_lower)
    return max(0.001, offer.model_probability - config.uncertainty_haircut)


def evaluate_offer(
    offer: BetOffer,
    config: RiskConfig | None = None,
) -> SingleBetDecision:
    config = config or RiskConfig()
    probability = conservative_probability(offer, config)
    break_even = implied_probability(offer.decimal_odds)
    edge = probability - break_even
    ev = expected_value(probability, offer.decimal_odds)
    price = fair_odds(probability)

    reason: str | None = None
    if not config.min_decimal_odds <= offer.decimal_odds <= config.max_decimal_odds:
        reason = "PRICE_OUTSIDE_ALLOWED_RANGE"
    elif edge < config.min_probability_edge:
        reason = "EDGE_BELOW_THRESHOLD"
    elif ev < config.min_single_ev:
        reason = "EV_BELOW_THRESHOLD"
    elif (
        offer.market_fair_probability is not None
        and probability <= offer.market_fair_probability
    ):
        reason = "NO_EDGE_OVER_DEVIGGED_MARKET"

    if reason is not None:
        return SingleBetDecision(
            offer=offer,
            action="NO_BET",
            conservative_probability=probability,
            break_even_probability=break_even,
            probability_edge=edge,
            expected_value=ev,
            fair_odds=price,
            stake_fraction=0.0,
            reason=reason,
        )

    full_kelly = kelly_fraction(probability, offer.decimal_odds)
    stake = min(
        config.max_single_stake_fraction,
        full_kelly * config.fractional_kelly,
    )
    if stake <= 0.0:
        return SingleBetDecision(
            offer=offer,
            action="NO_BET",
            conservative_probability=probability,
            break_even_probability=break_even,
            probability_edge=edge,
            expected_value=ev,
            fair_odds=price,
            stake_fraction=0.0,
            reason="KELLY_NOT_POSITIVE",
        )

    return SingleBetDecision(
        offer=offer,
        action="BET",
        conservative_probability=probability,
        break_even_probability=break_even,
        probability_edge=edge,
        expected_value=ev,
        fair_odds=price,
        stake_fraction=stake,
        reason="POSITIVE_EV_AFTER_HAIRCUT",
    )


def select_singles(
    offers: Iterable[BetOffer],
    config: RiskConfig | None = None,
) -> tuple[list[SingleBetDecision], list[SingleBetDecision]]:
    """Evaluate all offers and keep at most one active bet per match."""
    config = config or RiskConfig()
    evaluated = [evaluate_offer(offer, config) for offer in offers]

    by_match: dict[str, list[SingleBetDecision]] = {}
    rejected: list[SingleBetDecision] = []
    for decision in evaluated:
        if decision.action == "BET":
            by_match.setdefault(decision.offer.match_id, []).append(decision)
        else:
            rejected.append(decision)

    selected: list[SingleBetDecision] = []
    for match_id in sorted(by_match):
        match_decisions = sorted(
            by_match[match_id],
            key=lambda item: (
                item.expected_value,
                item.probability_edge,
                item.conservative_probability,
            ),
            reverse=True,
        )
        selected.append(match_decisions[0])
        for duplicate in match_decisions[1:]:
            rejected.append(
                SingleBetDecision(
                    offer=duplicate.offer,
                    action="NO_BET",
                    conservative_probability=duplicate.conservative_probability,
                    break_even_probability=duplicate.break_even_probability,
                    probability_edge=duplicate.probability_edge,
                    expected_value=duplicate.expected_value,
                    fair_odds=duplicate.fair_odds,
                    stake_fraction=0.0,
                    reason="ONE_BET_PER_MATCH_RULE",
                )
            )

    selected.sort(key=lambda item: item.expected_value, reverse=True)
    rejected.sort(key=lambda item: item.offer.offer_id)
    return selected, rejected
