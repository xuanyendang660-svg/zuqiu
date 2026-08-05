from __future__ import annotations

from dataclasses import dataclass

from .models import SettledBet


@dataclass(frozen=True)
class BacktestReport:
    bets: int
    settled_decisions: int
    total_stake: float
    total_profit: float
    roi: float
    hit_rate: float
    average_clv: float | None
    max_drawdown: float
    longest_losing_streak: int


def closing_line_value(taken_odds: float, closing_odds: float) -> float:
    if taken_odds <= 1.0 or closing_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1")
    return taken_odds / closing_odds - 1.0


def evaluate_backtest(records: tuple[SettledBet, ...]) -> BacktestReport:
    for record in records:
        record.validate()

    total_stake = sum(record.stake for record in records)
    total_profit = sum(record.profit for record in records)
    roi = total_profit / total_stake if total_stake else 0.0

    decisions = [record for record in records if record.won is not None]
    wins = sum(1 for record in decisions if record.won is True)
    hit_rate = wins / len(decisions) if decisions else 0.0

    clv_values = [
        closing_line_value(record.taken_odds, record.closing_odds)
        for record in records
        if record.closing_odds is not None
    ]
    average_clv = sum(clv_values) / len(clv_values) if clv_values else None

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    current_losing_streak = 0
    longest_losing_streak = 0

    for record in records:
        cumulative += record.profit
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

        if record.profit < 0:
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)
        elif record.profit > 0:
            current_losing_streak = 0

    return BacktestReport(
        bets=len(records),
        settled_decisions=len(decisions),
        total_stake=total_stake,
        total_profit=total_profit,
        roi=roi,
        hit_rate=hit_rate,
        average_clv=average_clv,
        max_drawdown=max_drawdown,
        longest_losing_streak=longest_losing_streak,
    )
