"""Profitability-first football value betting engine."""

from .backtest import BacktestReport, closing_line_value, evaluate_backtest
from .devig import devig_probabilities, power_devig, proportional_devig
from .models import (
    BetCandidate,
    Market,
    OutcomeQuote,
    ParlayCandidate,
    Policy,
    SettledBet,
)
from .parlay import build_two_leg_parlays
from .selector import deployment_allowed, evaluate_market, select_portfolio

__all__ = [
    "BacktestReport",
    "BetCandidate",
    "Market",
    "OutcomeQuote",
    "ParlayCandidate",
    "Policy",
    "SettledBet",
    "build_two_leg_parlays",
    "closing_line_value",
    "deployment_allowed",
    "devig_probabilities",
    "evaluate_backtest",
    "evaluate_market",
    "power_devig",
    "proportional_devig",
    "select_portfolio",
]
