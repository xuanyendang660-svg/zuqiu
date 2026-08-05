"""Risk and selection configuration.

Defaults are intentionally conservative starting values. They are not evidence of
profitability and must be calibrated with forward, time-sealed data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    min_single_ev: float = 0.03
    min_probability_edge: float = 0.025
    uncertainty_haircut: float = 0.03
    min_decimal_odds: float = 1.45
    max_decimal_odds: float = 3.50
    fractional_kelly: float = 0.20
    max_single_stake_fraction: float = 0.005

    min_parlay_ev: float = 0.07
    max_parlay_stake_fraction: float = 0.0025
    allow_same_match_parlay: bool = False

    def __post_init__(self) -> None:
        probability_fields = (
            "min_probability_edge",
            "uncertainty_haircut",
            "fractional_kelly",
            "max_single_stake_fraction",
            "max_parlay_stake_fraction",
        )
        for field_name in probability_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.min_single_ev < 0.0 or self.min_parlay_ev < 0.0:
            raise ValueError("EV thresholds cannot be negative")
        if self.min_decimal_odds <= 1.0:
            raise ValueError("min_decimal_odds must be above 1.0")
        if self.max_decimal_odds < self.min_decimal_odds:
            raise ValueError("max_decimal_odds must be >= min_decimal_odds")
