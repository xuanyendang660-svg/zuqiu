"""Independent settlement and performance reporting for singles and parlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Result = Literal["win", "loss", "push"]
BetKind = Literal["single", "parlay"]


@dataclass(frozen=True)
class SettledBet:
    bet_id: str
    kind: BetKind
    stake: float
    placed_odds: float
    result: Result
    closing_odds: float | None = None

    def __post_init__(self) -> None:
        if not self.bet_id.strip():
            raise ValueError("bet_id is required")
        if self.kind not in ("single", "parlay"):
            raise ValueError("kind must be single or parlay")
        if self.stake <= 0.0:
            raise ValueError("stake must be positive")
        if self.placed_odds <= 1.0:
            raise ValueError("placed_odds must be above 1.0")
        if self.result not in ("win", "loss", "push"):
            raise ValueError("result must be win, loss, or push")
        if self.closing_odds is not None and self.closing_odds <= 1.0:
            raise ValueError("closing_odds must be above 1.0")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SettledBet":
        return cls(
            bet_id=str(data["bet_id"]),
            kind=str(data["kind"]),  # type: ignore[arg-type]
            stake=float(data["stake"]),
            placed_odds=float(data["placed_odds"]),
            result=str(data["result"]),  # type: ignore[arg-type]
            closing_odds=(
                None if data.get("closing_odds") is None else float(data["closing_odds"])
            ),
        )

    @property
    def profit(self) -> float:
        if self.result == "win":
            return self.stake * (self.placed_odds - 1.0)
        if self.result == "loss":
            return -self.stake
        return 0.0

    @property
    def clv(self) -> float | None:
        if self.closing_odds is None:
            return None
        return self.placed_odds / self.closing_odds - 1.0


def summarize_records(records: list[SettledBet]) -> dict[str, Any]:
    return {
        "overall": _summarize_slice(records),
        "singles": _summarize_slice(
            [record for record in records if record.kind == "single"]
        ),
        "parlays": _summarize_slice(
            [record for record in records if record.kind == "parlay"]
        ),
    }


def _summarize_slice(records: list[SettledBet]) -> dict[str, Any]:
    total_stake = sum(record.stake for record in records)
    total_profit = sum(record.profit for record in records)
    wins = sum(record.result == "win" for record in records)
    losses = sum(record.result == "loss" for record in records)
    pushes = sum(record.result == "push" for record in records)
    decided = wins + losses

    clv_values = [record.clv for record in records if record.clv is not None]
    cumulative_profit = 0.0
    peak = 0.0
    max_drawdown = 0.0
    current_losing_streak = 0
    longest_losing_streak = 0

    for record in records:
        cumulative_profit += record.profit
        peak = max(peak, cumulative_profit)
        max_drawdown = max(max_drawdown, peak - cumulative_profit)
        if record.result == "loss":
            current_losing_streak += 1
            longest_losing_streak = max(
                longest_losing_streak,
                current_losing_streak,
            )
        elif record.result == "win":
            current_losing_streak = 0

    return {
        "bets": len(records),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": None if total_stake == 0.0 else total_profit / total_stake,
        "hit_rate": None if decided == 0 else wins / decided,
        "average_clv": (
            None if not clv_values else sum(clv_values) / len(clv_values)
        ),
        "max_drawdown_units": max_drawdown,
        "longest_losing_streak": longest_losing_streak,
    }
