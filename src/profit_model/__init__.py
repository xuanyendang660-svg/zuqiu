"""Profit-oriented football betting and two-leg parlay selection."""

from .config import RiskConfig
from .models import BetOffer, ParlayDecision, SingleBetDecision
from .parlay import build_best_two_leg_parlay
from .selector import evaluate_offer, select_singles

__all__ = [
    "BetOffer",
    "ParlayDecision",
    "RiskConfig",
    "SingleBetDecision",
    "build_best_two_leg_parlay",
    "evaluate_offer",
    "select_singles",
]
