from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from football_v2.strict_side_resolution import resolve_wyscout_sides_strict
from football_v2.wyscout_events import WyscoutIndexRecord


def _write_events(
    path: Path,
    events: list[tuple[int, tuple[int, ...]]],
) -> None:
    payload = [
        {
            "eventId": 10,
            "subEventId": 100,
            "teamId": team_id,
            "matchPeriod": "1H",
            "eventSec": index * 60,
            "eventName": "Shot",
            "subEventName": "Shot",
            "tags": [{"id": tag} for tag in tags],
            "positions": [{"x": 90, "y": 50}, {"x": 100, "y": 50}],
        }
        for index, (team_id, tags) in enumerate(events, start=1)
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strict_resolver_credits_own_goal_to_opponent(tmp_path: Path) -> None:
    path = tmp_path / "1.json"
    _write_events(path, [(11, (102,)), (22, tuple())])
    record = WyscoutIndexRecord(
        match_id=1,
        path=path,
        date=pd.Timestamp("2017-08-12"),
        source="matches_England.json",
        home_name="Alpha",
        away_name="Beta",
        home_score=0,
        away_score=1,
    )

    resolved = resolve_wyscout_sides_strict([record])

    assert resolved[0].home_team_id == 11
    assert resolved[0].away_team_id == 22


def test_strict_resolver_propagates_team_ids_into_draw(tmp_path: Path) -> None:
    first_path = tmp_path / "1.json"
    second_path = tmp_path / "2.json"
    _write_events(first_path, [(11, (101,)), (22, tuple())])
    _write_events(second_path, [(22, (101,)), (11, (101,))])
    records = [
        WyscoutIndexRecord(
            match_id=1,
            path=first_path,
            date=pd.Timestamp("2017-08-12"),
            source="matches_England.json",
            home_name="Alpha",
            away_name="Beta",
            home_score=1,
            away_score=0,
        ),
        WyscoutIndexRecord(
            match_id=2,
            path=second_path,
            date=pd.Timestamp("2017-08-19"),
            source="matches_England.json",
            home_name="Beta",
            away_name="Alpha",
            home_score=1,
            away_score=1,
        ),
    ]

    resolved = resolve_wyscout_sides_strict(records)
    by_match = {item.index.match_id: item for item in resolved}

    assert by_match[2].home_team_id == 22
    assert by_match[2].away_team_id == 11
