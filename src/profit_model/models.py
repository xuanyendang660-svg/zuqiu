"""Typed data contracts for single bets and two-leg parlays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


Action = Literal["BET", "NO_BET"]
ParlayAction = Literal["PARLAY", "NO_PARLAY"]


@dataclass(frozen=True)
class BetOffer:
    match_id: str
    market: str
    selection: str
    decimal_odds: float
    model_probability: float
    model_probability_lower: float | None = None
    market_fair_probability: float | None = None
    line: float | None = None
    bookmaker: str | None = None
    kickoff_utc: str | None = None

    def __post_init__(self) -> None:
        if not self.match_id.strip():
            raise ValueError("match_id is required")
        if not self.market.strip() or not self.selection.strip():
            raise ValueError("market and selection are required")
        if self.decimal_odds <= 1.0:
            raise ValueError("decimal_odds must be above 1.0")
        for field_name in (
            "model_probability",
            "model_probability_lower",
            "market_fair_probability",
        ):
            value = getattr(self, field_name)
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")

    @property
    def offer_id(self) -> str:
        line_text = "" if self.line is None else f"@{self.line:g}"
        return f"{self.match_id}|{self.market}|{self.selection}{line_text}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BetOffer":
        return cls(
            match_id=str(data["match_id"]),
            market=str(data["market"]),
            selection=str(data["selection"]),
            decimal_odds=float(data["decimal_odds"]),
            model_probability=float(data["model_probability"]),
            model_probability_lower=(
                None
                if data.get("model_probability_lower") is None
                else float(data["model_probability_lower"])
            ),
            market_fair_probability=(
                None
                if data.get("market_fair_probability") is None
                else float(data["market_fair_probability"])
            ),
            line=None if data.get("line") is None else float(data["line"]),
            bookmaker=data.get("bookmaker"),
            kickoff_utc=data.get("kickoff_utc"),
        )

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["offer_id"] = self.offer_id
        return output


@dataclass(frozen=True)
class SingleBetDecision:
    offer: BetOffer
    action: Action
    conservative_probability: float
    break_even_probability: float
    probability_edge: float
    expected_value: float
    fair_odds: float
    stake_fraction: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "offer": self.offer.to_dict(),
            "action": self.action,
            "conservative_probability": self.conservative_probability,
            "break_even_probability": self.break_even_probability,
            "probability_edge": self.probability_edge,
            "expected_value": self.expected_value,
            "fair_odds": self.fair_odds,
            "stake_fraction": self.stake_fraction,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ParlayDecision:
    action: ParlayAction
    legs: tuple[SingleBetDecision, ...]
    joint_probability: float
    combined_odds: float
    expected_value: float
    fair_odds: float | None
    stake_fraction: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "legs": [leg.to_dict() for leg in self.legs],
            "joint_probability": self.joint_probability,
            "combined_odds": self.combined_odds,
            "expected_value": self.expected_value,
            "fair_odds": self.fair_odds,
            "stake_fraction": self.stake_fraction,
            "reason": self.reason,
        }
