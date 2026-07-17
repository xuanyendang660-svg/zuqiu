from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .live_snapshots import LiveSnapshotDataset, _absolute_minute
from .wyscout_events import (
    WyscoutIndexRecord,
    _GOAL_TAG,
    _OWN_GOAL_TAG,
    _event_tags,
    _payload_events,
    resolve_wyscout_sides,
)


def _score_timeline(
    record: WyscoutIndexRecord,
    home_team_id: int,
    away_team_id: int,
    cutoffs: tuple[int, ...],
) -> tuple[dict[int, tuple[int, int]], tuple[int, int]]:
    payload = json.loads(Path(record.path).read_text(encoding="utf-8"))
    events = sorted(_payload_events(payload), key=_absolute_minute)
    score = {home_team_id: 0, away_team_id: 0}
    timeline: dict[int, tuple[int, int]] = {}
    cutoff_index = 0

    for event in events + [{"matchPeriod": "2H", "eventSec": 46 * 60}]:
        minute = _absolute_minute(event)
        while cutoff_index < len(cutoffs) and minute > cutoffs[cutoff_index]:
            cutoff = cutoffs[cutoff_index]
            timeline[cutoff] = (score[home_team_id], score[away_team_id])
            cutoff_index += 1

        team_value = event.get("teamId")
        if team_value is None:
            continue
        team_id = int(team_value)
        if team_id not in score:
            continue
        tags = _event_tags(event)
        is_own_goal = _OWN_GOAL_TAG in tags
        is_goal = _GOAL_TAG in tags or is_own_goal
        if not is_goal:
            continue

        if is_own_goal:
            scoring_team = away_team_id if team_id == home_team_id else home_team_id
        else:
            scoring_team = team_id
        score[scoring_team] += 1

    while cutoff_index < len(cutoffs):
        cutoff = cutoffs[cutoff_index]
        timeline[cutoff] = (score[home_team_id], score[away_team_id])
        cutoff_index += 1

    return timeline, (score[home_team_id], score[away_team_id])


def repair_live_score_integrity(
    dataset: LiveSnapshotDataset,
    index_records: list[WyscoutIndexRecord],
    *,
    cutoffs: tuple[int, ...],
    side_map: dict[int, tuple[int, int]] | None = None,
) -> LiveSnapshotDataset:
    if not cutoffs:
        raise ValueError("cutoffs must not be empty")

    frame = dataset.frame.copy()
    records = {record.match_id: record for record in index_records}
    resolved_side_map = side_map or {
        item.index.match_id: (item.home_team_id, item.away_team_id)
        for item in resolve_wyscout_sides(index_records)
    }
    corrections: dict[tuple[int, int], tuple[int, int]] = {}
    failures: list[str] = []

    for match_id in sorted(frame["match_id"].astype(int).unique()):
        record = records.get(match_id)
        sides = resolved_side_map.get(match_id)
        if record is None or sides is None:
            failures.append(f"{match_id}: missing record or side mapping")
            continue
        timeline, final_score = _score_timeline(
            record,
            sides[0],
            sides[1],
            cutoffs,
        )
        expected = (int(record.home_score), int(record.away_score))
        if final_score != expected:
            failures.append(
                f"{match_id}: event score {final_score[0]}-{final_score[1]} "
                f"!= index score {expected[0]}-{expected[1]}"
            )
            continue
        for cutoff, score in timeline.items():
            corrections[(match_id, cutoff)] = score

    if failures:
        sample = "; ".join(failures[:10])
        raise RuntimeError(
            f"live score integrity failed for {len(failures)} matches: {sample}"
        )

    for index, row in frame.iterrows():
        key = (int(row["match_id"]), int(row["snapshot_minute"]))
        score = corrections.get(key)
        if score is None:
            raise RuntimeError(f"missing corrected live score for match/cutoff {key}")
        home_live, away_live = score
        home_final = int(row["home_score"])
        away_final = int(row["away_score"])
        if home_live > home_final or away_live > away_final:
            raise RuntimeError(
                f"impossible live score at {key}: {home_live}-{away_live} "
                f"exceeds final {home_final}-{away_final}"
            )
        frame.at[index, "live_home_score"] = float(home_live)
        frame.at[index, "live_away_score"] = float(away_live)
        frame.at[index, "live_score_diff"] = float(home_live - away_live)
        frame.at[index, "live_total_goals"] = float(home_live + away_live)
        if "live_home_goals" in frame.columns:
            frame.at[index, "live_home_goals"] = float(home_live)
        if "live_away_goals" in frame.columns:
            frame.at[index, "live_away_goals"] = float(away_live)
        frame.at[index, "remaining_home_goals"] = int(home_final - home_live)
        frame.at[index, "remaining_away_goals"] = int(away_final - away_live)

    return LiveSnapshotDataset(frame.reset_index(drop=True), dataset.feature_columns)
