from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from football_v2.live_score_integrity import repair_live_score_integrity
from football_v2.live_snapshots import LiveSnapshotDataset
from football_v2.wyscout_events import WyscoutIndexRecord


def _event(
    minute: float,
    team_id: int,
    *,
    goal: bool = False,
    own_goal: bool = False,
) -> dict[str, object]:
    tags = []
    if goal:
        tags.append({"id": 101})
    if own_goal:
        tags.append({"id": 102})
    return {
        "matchPeriod": "1H" if minute <= 45 else "2H",
        "eventSec": minute * 60 if minute <= 45 else (minute - 45) * 60,
        "teamId": team_id,
        "eventName": "Shot",
        "subEventName": "Shot",
        "tags": tags,
        "positions": [{"x": 90, "y": 50}, {"x": 100, "y": 50}],
    }


def test_repair_credits_own_goal_to_opponent(tmp_path: Path) -> None:
    event_path = tmp_path / "1.json"
    event_path.write_text(
        json.dumps(
            [
                _event(10, 11, goal=True, own_goal=True),
                _event(35, 11, goal=True),
                _event(70, 22, goal=True),
            ]
        ),
        encoding="utf-8",
    )
    record = WyscoutIndexRecord(
        match_id=1,
        path=event_path,
        date=pd.Timestamp("2017-08-12"),
        source="matches_England.json",
        home_name="Home",
        away_name="Away",
        home_score=1,
        away_score=2,
    )
    frame = pd.DataFrame(
        [
            {
                "match_id": 1,
                "snapshot_minute": 15.0,
                "home_score": 1,
                "away_score": 2,
                "live_home_score": 1.0,
                "live_away_score": 0.0,
                "live_score_diff": 1.0,
                "live_total_goals": 1.0,
                "live_home_goals": 1.0,
                "live_away_goals": 0.0,
                "remaining_home_goals": 0,
                "remaining_away_goals": 2,
            },
            {
                "match_id": 1,
                "snapshot_minute": 45.0,
                "home_score": 1,
                "away_score": 2,
                "live_home_score": 2.0,
                "live_away_score": 0.0,
                "live_score_diff": 2.0,
                "live_total_goals": 2.0,
                "live_home_goals": 2.0,
                "live_away_goals": 0.0,
                "remaining_home_goals": 0,
                "remaining_away_goals": 2,
            },
        ]
    )
    dataset = LiveSnapshotDataset(
        frame,
        (
            "live_home_score",
            "live_away_score",
            "live_score_diff",
            "live_total_goals",
            "live_home_goals",
            "live_away_goals",
        ),
    )

    repaired = repair_live_score_integrity(dataset, [record], cutoffs=(15, 45))

    assert repaired.frame.loc[0, "live_home_score"] == 0
    assert repaired.frame.loc[0, "live_away_score"] == 1
    assert repaired.frame.loc[1, "live_home_score"] == 1
    assert repaired.frame.loc[1, "live_away_score"] == 1
    assert repaired.frame.loc[1, "remaining_home_goals"] == 0
    assert repaired.frame.loc[1, "remaining_away_goals"] == 1


def test_repair_rejects_event_final_mismatch(tmp_path: Path) -> None:
    event_path = tmp_path / "2.json"
    event_path.write_text(json.dumps([_event(10, 11, goal=True)]), encoding="utf-8")
    record = WyscoutIndexRecord(
        match_id=2,
        path=event_path,
        date=pd.Timestamp("2017-08-12"),
        source="matches_England.json",
        home_name="Home",
        away_name="Away",
        home_score=2,
        away_score=0,
    )
    frame = pd.DataFrame(
        [
            {
                "match_id": 2,
                "snapshot_minute": 15.0,
                "home_score": 2,
                "away_score": 0,
                "live_home_score": 1.0,
                "live_away_score": 0.0,
                "live_score_diff": 1.0,
                "live_total_goals": 1.0,
                "remaining_home_goals": 1,
                "remaining_away_goals": 0,
            }
        ]
    )
    dataset = LiveSnapshotDataset(frame, ("live_home_score", "live_away_score"))

    try:
        repair_live_score_integrity(dataset, [record], cutoffs=(15,))
    except RuntimeError as exc:
        assert "event score 1-0 != index score 2-0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected integrity failure")
