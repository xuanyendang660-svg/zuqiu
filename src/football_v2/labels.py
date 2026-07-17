from __future__ import annotations

from enum import StrEnum


class ScoreArchetype(StrEnum):
    LOW_EVENT = "low_event"
    NORMAL = "normal"
    SHOOTOUT = "shootout"
    HOME_CONTROL = "home_control"
    AWAY_CONTROL = "away_control"
    HOME_BLOWOUT = "home_blowout"
    AWAY_BLOWOUT = "away_blowout"


def classify_score(home_goals: int, away_goals: int) -> ScoreArchetype:
    """Classify a scoreline by its realised match shape."""
    if min(home_goals, away_goals) < 0:
        raise ValueError("goals cannot be negative")

    total = home_goals + away_goals
    margin = home_goals - away_goals

    if home_goals >= 4 and margin >= 3:
        return ScoreArchetype.HOME_BLOWOUT
    if away_goals >= 4 and margin <= -3:
        return ScoreArchetype.AWAY_BLOWOUT
    if home_goals >= 2 and away_goals >= 2 and total >= 5:
        return ScoreArchetype.SHOOTOUT
    if home_goals >= 2 and away_goals == 0:
        return ScoreArchetype.HOME_CONTROL
    if away_goals >= 2 and home_goals == 0:
        return ScoreArchetype.AWAY_CONTROL
    if total <= 1:
        return ScoreArchetype.LOW_EVENT
    return ScoreArchetype.NORMAL


def is_jackpot_tail(home_goals: int, away_goals: int) -> bool:
    """Rare score tail: 5+ by one team, four-goal margin, or six-goal mutual game."""
    if min(home_goals, away_goals) < 0:
        raise ValueError("goals cannot be negative")
    total = home_goals + away_goals
    margin = abs(home_goals - away_goals)
    return (
        max(home_goals, away_goals) >= 5
        or margin >= 4
        or (total >= 6 and min(home_goals, away_goals) >= 2)
    )


def jackpot_tail_type(home_goals: int, away_goals: int) -> ScoreArchetype:
    if not is_jackpot_tail(home_goals, away_goals):
        return ScoreArchetype.NORMAL
    if home_goals - away_goals >= 3:
        return ScoreArchetype.HOME_BLOWOUT
    if away_goals - home_goals >= 3:
        return ScoreArchetype.AWAY_BLOWOUT
    return ScoreArchetype.SHOOTOUT


def score_to_label(home_goals: int, away_goals: int) -> str:
    return f"{home_goals}-{away_goals}"


def label_to_score(label: str) -> tuple[int, int]:
    parts = label.split("-")
    if len(parts) != 2:
        raise ValueError(f"invalid score label: {label!r}")
    return int(parts[0]), int(parts[1])
