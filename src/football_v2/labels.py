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
    """Classify a scoreline by its realised match shape.

    Extreme tails are classified before ordinary totals so 6-1 and 1-5 are
    treated as one-sided collapses rather than generic high-scoring matches.
    """
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


def score_to_label(home_goals: int, away_goals: int) -> str:
    return f"{home_goals}-{away_goals}"


def label_to_score(label: str) -> tuple[int, int]:
    parts = label.split("-")
    if len(parts) != 2:
        raise ValueError(f"invalid score label: {label!r}")
    return int(parts[0]), int(parts[1])
