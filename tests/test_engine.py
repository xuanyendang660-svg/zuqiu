from __future__ import annotations

from football_value import (
    BetCandidate,
    Market,
    OutcomeQuote,
    Policy,
    SettledBet,
    build_two_leg_parlays,
    devig_probabilities,
    evaluate_backtest,
    evaluate_market,
    select_portfolio,
)


def market(event_id: str, market_id: str, first_probability: float) -> Market:
    return Market(
        event_id=event_id,
        market_id=market_id,
        market_type="total",
        outcomes=(
            OutcomeQuote(
                event_id=event_id,
                market_id=market_id,
                market_type="total",
                selection="over",
                odds=2.05,
                model_probability=first_probability,
                uncertainty=0.01,
                data_quality=0.95,
            ),
            OutcomeQuote(
                event_id=event_id,
                market_id=market_id,
                market_type="total",
                selection="under",
                odds=1.82,
                model_probability=1.0 - first_probability,
                uncertainty=0.01,
                data_quality=0.95,
            ),
        ),
    )


def test_devig_probabilities_sum_to_one() -> None:
    for method in ("proportional", "power"):
        probabilities = devig_probabilities([1.91, 1.97], method)
        assert abs(sum(probabilities) - 1.0) < 1e-10
        assert all(0.0 < value < 1.0 for value in probabilities)


def test_positive_single_is_selected_but_shadow_stake_is_zero() -> None:
    policy = Policy(shadow_mode=True)
    evaluated = evaluate_market(market("a", "a-ou", 0.55), policy)
    selected = [item for item in evaluated if item.decision == "BET"]
    assert len(selected) == 1
    assert selected[0].selection == "over"
    assert selected[0].recommended_stake_fraction > 0.0
    assert selected[0].execution_stake_fraction == 0.0


def test_low_quality_market_is_no_bet() -> None:
    quote_market = Market(
        event_id="b",
        market_id="b-ou",
        market_type="total",
        outcomes=(
            OutcomeQuote("b", "b-ou", "total", "over", 2.10, 0.56, 0.01, 0.50),
            OutcomeQuote("b", "b-ou", "total", "under", 1.80, 0.44, 0.01, 0.50),
        ),
    )
    evaluated = evaluate_market(quote_market, Policy())
    assert all(item.decision == "NO_BET" for item in evaluated)
    assert "data_quality_below_threshold" in evaluated[0].reasons


def test_parlay_uses_only_two_qualified_different_events() -> None:
    policy = Policy(shadow_mode=True)
    singles = select_portfolio(
        (market("a", "a-ou", 0.55), market("b", "b-ou", 0.56)),
        policy,
    )
    parlays = build_two_leg_parlays(singles, policy)
    assert len(singles) == 2
    assert len(parlays) == 1
    assert parlays[0].legs[0].event_id != parlays[0].legs[1].event_id
    assert parlays[0].conservative_expected_value >= policy.min_parlay_ev
    assert parlays[0].execution_stake_fraction == 0.0


def test_same_event_parlay_is_rejected_by_default() -> None:
    base = dict(
        decision="BET",
        event_id="same",
        market_type="total",
        odds=2.0,
        market_probability=0.50,
        model_probability=0.56,
        conservative_probability=0.55,
        fair_odds=1 / 0.56,
        edge_pp=0.06,
        expected_value=0.12,
        conservative_expected_value=0.10,
        recommended_stake_fraction=0.005,
        execution_stake_fraction=0.0,
        data_quality=0.95,
        reasons=("all_value_gates_passed",),
    )
    first = BetCandidate(market_id="m1", selection="over", **base)
    second = BetCandidate(market_id="m2", selection="home", **base)
    assert build_two_leg_parlays((first, second), Policy()) == ()


def test_backtest_reports_roi_clv_and_drawdown() -> None:
    records = (
        SettledBet("1", 1.0, 1.0, True, 2.0, 1.90),
        SettledBet("2", 1.0, -1.0, False, 2.0, 2.05),
        SettledBet("3", 1.0, -1.0, False, 2.0, 2.10),
    )
    report = evaluate_backtest(records)
    assert report.bets == 3
    assert report.total_profit == -1.0
    assert report.roi == -1.0 / 3.0
    assert report.longest_losing_streak == 2
    assert report.max_drawdown == 2.0
    assert report.average_clv is not None
