from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import jackpot_tail_type, is_jackpot_tail
from .statsbomb_events import EventDataset
from .wyscout_events import (
    WyscoutIndexRecord,
    _ACCURATE_TAG,
    _GOAL_TAG,
    _RED_CARD_TAGS,
    _YELLOW_CARD_TAG,
    _event_tags,
    _payload_events,
    _shot_quality,
    resolve_wyscout_sides,
)


@dataclass(frozen=True)
class LiveSnapshotDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]

    @property
    def features(self) -> np.ndarray:
        return self.frame.loc[:, self.feature_columns].to_numpy(dtype=float)


_LIVE_METRICS = (
    "goals",
    "shots",
    "shots_on_target",
    "xg",
    "big_chances",
    "counter_xg",
    "set_piece_xg",
    "box_entries",
    "final_third_entries",
    "progressive_actions",
    "pressures",
    "high_recoveries",
    "turnovers",
    "passes",
    "completed_passes",
    "fouls",
    "yellow_cards",
    "red_cards",
)


def _empty_summary() -> dict[str, float]:
    return {metric: 0.0 for metric in _LIVE_METRICS}


def _absolute_minute(event: dict[str, object]) -> float:
    period = str(event.get("matchPeriod") or "1H")
    seconds = float(event.get("eventSec") or 0.0)
    if period == "2H":
        seconds += 45 * 60
    elif period in {"E1", "ET1"}:
        seconds += 90 * 60
    elif period in {"E2", "ET2"}:
        seconds += 105 * 60
    return seconds / 60.0


def _update_summary(
    summary: dict[str, float],
    event: dict[str, object],
    *,
    quick_transition: bool,
) -> None:
    event_name = str(event.get("eventName") or "")
    sub_event = str(event.get("subEventName") or "")
    tags = _event_tags(event)
    positions = event.get("positions") or []
    start = positions[0] if positions and isinstance(positions[0], dict) else {}
    end = positions[-1] if positions and isinstance(positions[-1], dict) else start
    x = float(start.get("x") or 0.0)
    y = float(start.get("y") or 50.0)
    end_x = float(end.get("x") or x)
    end_y = float(end.get("y") or y)
    goal = _GOAL_TAG in tags
    accurate = _ACCURATE_TAG in tags
    is_shot = event_name == "Shot" or sub_event == "Penalty"

    if is_shot:
        xg = _shot_quality(x, y, 403 in tags)
        summary["shots"] += 1.0
        summary["shots_on_target"] += float(goal or accurate)
        summary["xg"] += xg
        summary["big_chances"] += float(xg >= 0.18)
        summary["counter_xg"] += xg if quick_transition else 0.0
        summary["set_piece_xg"] += xg if event_name == "Free Kick" else 0.0
        summary["goals"] += float(goal)
    if event_name == "Pass":
        summary["passes"] += 1.0
        summary["completed_passes"] += float(accurate)
    if event_name in {"Duel", "Others on the ball"}:
        summary["pressures"] += 1.0
    if event_name in {"Duel", "Others on the ball"} and accurate and x >= 67:
        summary["high_recoveries"] += 1.0
    if (event_name == "Pass" and not accurate) or 1302 in tags:
        summary["turnovers"] += 1.0
    if event_name == "Foul":
        summary["fouls"] += 1.0
    summary["yellow_cards"] += float(_YELLOW_CARD_TAG in tags)
    summary["red_cards"] += float(bool(tags & _RED_CARD_TAGS))
    summary["progressive_actions"] += float(end_x - x >= 20)
    summary["final_third_entries"] += float(x < 67 <= end_x)
    start_box = x >= 85 and 20 <= y <= 80
    end_box = end_x >= 85 and 20 <= end_y <= 80
    summary["box_entries"] += float(not start_box and end_box)


def _snapshot_rows(
    record: WyscoutIndexRecord,
    home_team_id: int,
    away_team_id: int,
    cutoffs: tuple[int, ...],
) -> list[dict[str, float]]:
    payload = json.loads(Path(record.path).read_text(encoding="utf-8"))
    events = sorted(_payload_events(payload), key=_absolute_minute)
    summaries = {
        home_team_id: _empty_summary(),
        away_team_id: _empty_summary(),
    }
    cutoff_index = 0
    rows: list[dict[str, float]] = []
    previous_team: int | None = None
    previous_minute = -999.0
    last_goal_minute = -999.0
    recent_events: list[tuple[float, int, bool, float]] = []

    for event in events + [{"matchPeriod": "2H", "eventSec": 46 * 60}]:
        minute = _absolute_minute(event)
        while cutoff_index < len(cutoffs) and minute > cutoffs[cutoff_index]:
            cutoff = cutoffs[cutoff_index]
            home = summaries[home_team_id]
            away = summaries[away_team_id]
            window = [item for item in recent_events if cutoff - 10 <= item[0] <= cutoff]
            rows.append(
                {
                    "snapshot_minute": float(cutoff),
                    "remaining_minutes": float(max(0, 90 - cutoff)),
                    "live_home_score": home["goals"],
                    "live_away_score": away["goals"],
                    "live_score_diff": home["goals"] - away["goals"],
                    "live_total_goals": home["goals"] + away["goals"],
                    "minutes_since_goal": float(
                        cutoff - last_goal_minute if last_goal_minute > -900 else cutoff
                    ),
                    "goals_last10": float(sum(item[2] for item in window)),
                    "shots_last10": float(len(window)),
                    "xg_last10": float(sum(item[3] for item in window)),
                    **{
                        f"live_home_{metric}": value
                        for metric, value in home.items()
                    },
                    **{
                        f"live_away_{metric}": value
                        for metric, value in away.items()
                    },
                }
            )
            cutoff_index += 1
        if cutoff_index >= len(cutoffs):
            break
        team_value = event.get("teamId")
        if team_value is None:
            continue
        team_id = int(team_value)
        if team_id not in summaries:
            continue
        quick_transition = (
            previous_team is not None
            and previous_team != team_id
            and minute - previous_minute <= 0.20
        )
        before_goals = summaries[team_id]["goals"]
        before_xg = summaries[team_id]["xg"]
        _update_summary(summaries[team_id], event, quick_transition=quick_transition)
        goal_added = summaries[team_id]["goals"] > before_goals
        xg_added = summaries[team_id]["xg"] - before_xg
        is_shot = str(event.get("eventName") or "") == "Shot" or str(
            event.get("subEventName") or ""
        ) == "Penalty"
        if is_shot:
            recent_events.append((minute, team_id, goal_added, xg_added))
        if goal_added:
            last_goal_minute = minute
        previous_team = team_id
        previous_minute = minute
    return rows


def build_live_snapshot_dataset(
    repository_root: str | Path,
    market_event_dataset: EventDataset,
    index_records: list[WyscoutIndexRecord],
    *,
    cutoffs: tuple[int, ...] = (15, 30, 45),
) -> LiveSnapshotDataset:
    if not cutoffs or any(cutoff <= 0 or cutoff >= 90 for cutoff in cutoffs):
        raise ValueError("cutoffs must be between 1 and 89")
    index_map = {record.match_id: record for record in index_records}
    side_map = {
        item.index.match_id: (item.home_team_id, item.away_team_id)
        for item in resolve_wyscout_sides(index_records)
    }
    pre_match = {
        int(row["match_id"]): row
        for row in market_event_dataset.frame.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for match_id, base in pre_match.items():
        record = index_map.get(match_id)
        sides = side_map.get(match_id)
        if record is None or sides is None:
            continue
        snapshots = _snapshot_rows(record, sides[0], sides[1], cutoffs)
        for snapshot in snapshots:
            home_score = int(base["home_score"])
            away_score = int(base["away_score"])
            live_home = int(snapshot["live_home_score"])
            live_away = int(snapshot["live_away_score"])
            rows.append(
                {
                    **base,
                    **snapshot,
                    "remaining_home_goals": max(0, home_score - live_home),
                    "remaining_away_goals": max(0, away_score - live_away),
                    "final_jackpot_target": int(
                        is_jackpot_tail(home_score, away_score)
                    ),
                    "final_jackpot_type": jackpot_tail_type(
                        home_score, away_score
                    ).value,
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["date", "match_id", "snapshot_minute"]
    ).reset_index(drop=True)
    non_features = {
        "match_id",
        "date",
        "competition_id_raw",
        "season_id_raw",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tail_type",
        "tail_target",
        "upset_jackpot_target",
        "remaining_home_goals",
        "remaining_away_goals",
        "final_jackpot_target",
        "final_jackpot_type",
    }
    feature_columns = tuple(
        column for column in frame.columns if column not in non_features
    )
    return LiveSnapshotDataset(frame, feature_columns)
