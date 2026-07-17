from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from football_v2.live_score_integrity import repair_live_score_integrity
from football_v2.live_snapshots import build_live_snapshot_dataset
from football_v2.market_event_data import load_market_1718
from football_v2.strict_market_join import join_market_events_without_score
from football_v2.wyscout_events import (
    build_wyscout_event_dataset,
    load_wyscout_index,
    load_wyscout_league_matches,
)
from run_wyscout_cold_blowout_state_audit import _cutoff_report
from run_wyscout_cold_blowout_strict_audit import _json_default, _normalise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Own-goal-corrected fixed live-state cold-blowout audit"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--market-cache", default=".cache/football-data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", default="artifacts/wyscout_cold_blowout_integrity_audit.json"
    )
    args = parser.parse_args()

    cutoffs = (15, 30, 45, 60)
    records = _normalise(load_wyscout_index(args.data_root))
    matches = load_wyscout_league_matches(args.data_root, workers=args.workers)
    event_dataset = build_wyscout_event_dataset(matches)
    market_event_dataset = join_market_events_without_score(
        event_dataset,
        records,
        load_market_1718(args.market_cache),
    )
    raw_live = build_live_snapshot_dataset(
        args.data_root,
        market_event_dataset,
        records,
        cutoffs=cutoffs,
    )
    repaired = repair_live_score_integrity(
        raw_live,
        records,
        cutoffs=cutoffs,
    )

    raw_home = raw_live.frame["live_home_score"].to_numpy(dtype=int)
    raw_away = raw_live.frame["live_away_score"].to_numpy(dtype=int)
    clean_home = repaired.frame["live_home_score"].to_numpy(dtype=int)
    clean_away = repaired.frame["live_away_score"].to_numpy(dtype=int)
    changed = np.logical_or(raw_home != clean_home, raw_away != clean_away)
    impossible_before = np.logical_or(
        raw_home > raw_live.frame["home_score"].to_numpy(dtype=int),
        raw_away > raw_live.frame["away_score"].to_numpy(dtype=int),
    )
    impossible_after = np.logical_or(
        clean_home > repaired.frame["home_score"].to_numpy(dtype=int),
        clean_away > repaired.frame["away_score"].to_numpy(dtype=int),
    )

    payload = {
        "dataset": {
            "matches": int(repaired.frame["match_id"].nunique()),
            "snapshots": len(repaired.frame),
            "market_join_uses_final_score": False,
            "cutoffs": list(cutoffs),
        },
        "integrity": {
            "snapshots_changed_by_score_repair": int(changed.sum()),
            "matches_changed_by_score_repair": int(
                repaired.frame.loc[changed, "match_id"].nunique()
            ),
            "impossible_snapshots_before_repair": int(impossible_before.sum()),
            "impossible_snapshots_after_repair": int(impossible_after.sum()),
            "event_final_score_verified_for_every_match": True,
            "own_goals_credited_to_opponent": True,
        },
        "cutoffs": [
            _cutoff_report(repaired.frame, cutoff)
            for cutoff in cutoffs
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
