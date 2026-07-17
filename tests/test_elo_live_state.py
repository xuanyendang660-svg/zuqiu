from __future__ import annotations

import pandas as pd

from football_v2.elo_live_state import (
    build_elo_live_state_frame,
    evaluate_elo_live_state,
)
from football_v2.statsbomb_events import EventMatch, TeamEventSummary


def _match(
    match_id: int,
    date: str,
    home_score: int,
    away_score: int,
    home_minutes: list[int],
    away_minutes: list[int],
) -> EventMatch:
    home = TeamEventSummary(1, "Home")
    away = TeamEventSummary(2, "Away")
    home.goals = home_score
    away.goals = away_score
    home.goal_minutes = home_minutes
    away.goal_minutes = away_minutes
    return EventMatch(
        match_id=match_id,
        date=pd.Timestamp(date),
        competition_id=1,
        season_id=1,
        home_team_id=1,
        home_team_name="Home",
        away_team_id=2,
        away_team_name="Away",
        home_score=home_score,
        away_score=away_score,
        home=home,
        away=away,
    )


def test_elo_live_state_detects_away_underdog_two_nil() -> None:
    matches = [
        _match(1, "2020-01-01", 1, 0, [20], []),
        _match(2, "2020-01-08", 2, 0, [15, 70], []),
        _match(3, "2020-01-15", 1, 1, [40], [60]),
        _match(4, "2020-01-22", 0, 3, [], [10, 25, 70]),
    ]

    frame = build_elo_live_state_frame(matches, cutoff=30)
    report = evaluate_elo_live_state(frame, cutoff=30)

    alert_row = frame.loc[frame["match_id"] == 4].iloc[0]
    assert bool(alert_row["home_underdog"]) is False
    assert bool(alert_row["alert"]) is True
    assert bool(alert_row["target"]) is True
    assert report.filters["all"].alerts == 1
    assert report.filters["all"].successes == 1
    assert report.filters["all"].precision == 1.0
    assert report.success_score_distribution == {"3-0": 1}
