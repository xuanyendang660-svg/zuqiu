from __future__ import annotations

from pathlib import Path

import pandas as pd

from .labels import jackpot_tail_type, is_jackpot_tail
from .live_score_integrity import repair_live_score_integrity
from .live_snapshots import LiveSnapshotDataset, _snapshot_rows
from .statsbomb_events import EventDataset
from .strict_side_resolution import resolve_wyscout_sides_strict
from .wyscout_events import WyscoutIndexRecord


def build_strict_live_snapshot_dataset(
    repository_root: str | Path,
    market_event_dataset: EventDataset,
    index_records: list[WyscoutIndexRecord],
    *,
    cutoffs: tuple[int, ...] = (15, 30, 45),
) -> LiveSnapshotDataset:
    del repository_root
    if not cutoffs or any(cutoff <= 0 or cutoff >= 90 for cutoff in cutoffs):
        raise ValueError("cutoffs must be between 1 and 89")

    index_map = {record.match_id: record for record in index_records}
    strict_sides = resolve_wyscout_sides_strict(
        index_records,
        drop_invalid=True,
    )
    side_map = {
        item.index.match_id: (item.home_team_id, item.away_team_id)
        for item in strict_sides
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
    dataset = LiveSnapshotDataset(frame, feature_columns)
    return repair_live_score_integrity(
        dataset,
        index_records,
        cutoffs=cutoffs,
        side_map=side_map,
    )
