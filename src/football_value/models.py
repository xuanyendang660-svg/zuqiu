from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Decision = Literal["BET", "NO_BET"]


@dataclass(frozen=True)
class OutcomeQuote:
    event_id: str
    market_id: str
    market_type: str
    selection: str
    odds: float
    model_probability: float
    uncertainty: float = 0.0
    data_quality: float = 1.0
    bookmaker: str = ""
    line: float | None = None

    def validate(self) -> None:
        if not self.event_id or not self.market_id or not self.selection:
            raise ValueError("event_id, market_id and selection are required")
        if self.odds <= 1.0:
            raise ValueError("decimal odds must be greater than 1.0")
        if not 0.0 < self.model_probability < 1.0:
            raise ValueError("model_probability must be between 0 and 1")
        if not 0.0 <= self.uncertainty < 1.0:
            raise ValueError("uncertainty must be between 0 and 1")
        if not 0.0 <= self.data_quality <= 1.0:
            raise ValueError("data_quality must be between 0 and 1")


@dataclass(frozen=True)
class Market:
    event_id: str
    market_id: str
    market_type: str
    outcomes: tuple[OutcomeQuote, ...]
    devig_method: str = "proportional"

    def validate(self) -> None:
        if len(self.outcomes) < 2:
            raise ValueError("a market needs at least two mutually exclusive outcomes")
        if any(outcome.event_id != self.event_id for outcome in self.outcomes):
            raise ValueError("all outcomes must use the market event_id")
        if any(outcome.market_id != self.market_id for outcome in self.outcomes):
            raise ValueError("all outcomes must use the market market_id")
        for outcome in self.outcomes:
            outcome.validate()


@dataclass(frozen=True)
class Policy:
    min_edge_pp: float = 0.025
    min_ev: float = 0.03
    min_conservative_ev: float = 0.0
    min_data_quality: float = 0.80
    uncertainty_multiplier: float = 1.0
    fractional_kelly: float = 0.15
    max_stake_fraction: float = 0.005
    max_parlay_stake_fraction: float = 0.0025
    daily_risk_fraction: float = 0.02
    max_event_risk_fraction: float = 0.01
    min_parlay_ev: float = 0.05
    allow_same_event_parlay: bool = False
    shadow_mode: bool = True
    min_settled_bets: int = 500

    def validate(self) -> None:
        bounded = {
            "min_edge_pp": self.min_edge_pp,
            "min_data_quality": self.min_data_quality,
            "uncertainty_multiplier": self.uncertainty_multiplier,
            "fractional_kelly": self.fractional_kelly,
            "max_stake_fraction": self.max_stake_fraction,
            "max_parlay_stake_fraction": self.max_parlay_stake_fraction,
            "daily_risk_fraction": self.daily_risk_fraction,
            "max_event_risk_fraction": self.max_event_risk_fraction,
        }
        if any(value < 0.0 for value in bounded.values()):
            raise ValueError("policy values cannot be negative")
        if self.min_data_quality > 1.0:
            raise ValueError("min_data_quality cannot exceed 1")
        if self.max_stake_fraction > self.max_event_risk_fraction:
            raise ValueError("single stake cap cannot exceed event risk cap")
        if self.max_parlay_stake_fraction > self.daily_risk_fraction:
            raise ValueError("parlay stake cap cannot exceed daily risk cap")
        if self.min_settled_bets < 0:
            raise ValueError("min_settled_bets cannot be negative")


@dataclass(frozen=True)
class BetCandidate:
    decision: Decision
    event_id: str
    market_id: str
    market_type: str
    selection: str
    odds: float
    market_probability: float
    model_probability: float
    conservative_probability: float
    fair_odds: float
    edge_pp: float
    expected_value: float
    conservative_expected_value: float
    recommended_stake_fraction: float
    execution_stake_fraction: float
    data_quality: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParlayCandidate:
    decision: Decision
    legs: tuple[BetCandidate, BetCandidate]
    combined_odds: float
    joint_probability: float
    conservative_joint_probability: float
    expected_value: float
    conservative_expected_value: float
    recommended_stake_fraction: float
    execution_stake_fraction: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SettledBet:
    bet_id: str
    stake: float
    profit: float
    won: bool | None
    taken_odds: float
    closing_odds: float | None = None

    def validate(self) -> None:
        if self.stake <= 0:
            raise ValueError("stake must be positive")
        if self.taken_odds <= 1.0:
            raise ValueError("taken_odds must be greater than 1")
        if self.closing_odds is not None and self.closing_odds <= 1.0:
            raise ValueError("closing_odds must be greater than 1")
