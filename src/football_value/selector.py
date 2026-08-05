from __future__ import annotations

from dataclasses import replace

from .devig import devig_probabilities
from .models import BetCandidate, Market, Policy


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full Kelly fraction for a binary wager, clipped at zero."""
    net_odds = decimal_odds - 1.0
    if net_odds <= 0.0:
        return 0.0
    value = (probability * decimal_odds - 1.0) / net_odds
    return max(0.0, value)


def deployment_allowed(policy: Policy, settled_bets: int) -> bool:
    return not policy.shadow_mode and settled_bets >= policy.min_settled_bets


def evaluate_market(
    market: Market,
    policy: Policy,
    *,
    settled_bets: int = 0,
) -> tuple[BetCandidate, ...]:
    market.validate()
    policy.validate()

    market_probabilities = devig_probabilities(
        [outcome.odds for outcome in market.outcomes], market.devig_method
    )
    can_execute = deployment_allowed(policy, settled_bets)
    candidates: list[BetCandidate] = []

    for outcome, market_probability in zip(
        market.outcomes, market_probabilities, strict=True
    ):
        conservative_probability = max(
            1e-9,
            outcome.model_probability
            - policy.uncertainty_multiplier * outcome.uncertainty,
        )
        fair_odds = 1.0 / outcome.model_probability
        edge_pp = outcome.model_probability - market_probability
        expected_value = outcome.model_probability * outcome.odds - 1.0
        conservative_ev = conservative_probability * outcome.odds - 1.0
        full_kelly = kelly_fraction(conservative_probability, outcome.odds)
        recommended_stake = min(
            full_kelly * policy.fractional_kelly,
            policy.max_stake_fraction,
            policy.max_event_risk_fraction,
        )

        failed: list[str] = []
        if outcome.data_quality < policy.min_data_quality:
            failed.append("data_quality_below_threshold")
        if edge_pp < policy.min_edge_pp:
            failed.append("model_edge_below_threshold")
        if expected_value < policy.min_ev:
            failed.append("raw_ev_below_threshold")
        if conservative_ev <= policy.min_conservative_ev:
            failed.append("conservative_ev_not_positive")
        if recommended_stake <= 0.0:
            failed.append("kelly_not_positive")

        decision = "NO_BET" if failed else "BET"
        reasons = tuple(failed) if failed else ("all_value_gates_passed",)
        execution_stake = recommended_stake if decision == "BET" and can_execute else 0.0

        candidates.append(
            BetCandidate(
                decision=decision,
                event_id=outcome.event_id,
                market_id=outcome.market_id,
                market_type=outcome.market_type,
                selection=outcome.selection,
                odds=outcome.odds,
                market_probability=market_probability,
                model_probability=outcome.model_probability,
                conservative_probability=conservative_probability,
                fair_odds=fair_odds,
                edge_pp=edge_pp,
                expected_value=expected_value,
                conservative_expected_value=conservative_ev,
                recommended_stake_fraction=recommended_stake,
                execution_stake_fraction=execution_stake,
                data_quality=outcome.data_quality,
                reasons=reasons,
            )
        )

    return tuple(candidates)


def select_portfolio(
    markets: tuple[Market, ...],
    policy: Policy,
    *,
    settled_bets: int = 0,
) -> tuple[BetCandidate, ...]:
    """Choose at most one qualified single per event and apply a daily risk cap."""
    qualified: list[BetCandidate] = []
    for market in markets:
        qualified.extend(
            candidate
            for candidate in evaluate_market(
                market, policy, settled_bets=settled_bets
            )
            if candidate.decision == "BET"
        )

    qualified.sort(
        key=lambda item: (
            item.conservative_expected_value,
            item.edge_pp,
            item.data_quality,
        ),
        reverse=True,
    )

    selected: list[BetCandidate] = []
    used_events: set[str] = set()
    for candidate in qualified:
        if candidate.event_id in used_events:
            continue
        selected.append(candidate)
        used_events.add(candidate.event_id)

    total_recommended = sum(item.recommended_stake_fraction for item in selected)
    if total_recommended <= policy.daily_risk_fraction or total_recommended == 0.0:
        return tuple(selected)

    scale = policy.daily_risk_fraction / total_recommended
    return tuple(
        replace(
            item,
            recommended_stake_fraction=item.recommended_stake_fraction * scale,
            execution_stake_fraction=item.execution_stake_fraction * scale,
            reasons=item.reasons + ("scaled_to_daily_risk_cap",),
        )
        for item in selected
    )
