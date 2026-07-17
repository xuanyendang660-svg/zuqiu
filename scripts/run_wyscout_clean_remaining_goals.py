from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.live_remaining_goals import leave_one_league_out_remaining_goal_test
from football_v2.market_event_data import load_market_1718
from football_v2.strict_live_snapshots import build_strict_live_snapshot_dataset
from football_v2.strict_market_join import join_market_events_without_score
from football_v2.strict_wyscout_loader import load_wyscout_league_matches_strict
from football_v2.wyscout_events import build_wyscout_event_dataset, load_wyscout_index
from run_wyscout_cold_blowout_strict_audit import _json_default, _normalise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict clean-data remaining-goal exact-score benchmark"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--market-cache", default=".cache/football-data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", default="artifacts/wyscout_clean_remaining_goals.json"
    )
    args = parser.parse_args()

    cutoffs = (15, 30, 45, 60)
    records = _normalise(load_wyscout_index(args.data_root))
    matches = load_wyscout_league_matches_strict(
        args.data_root,
        workers=args.workers,
    )
    event_dataset = build_wyscout_event_dataset(matches)
    market_event_dataset = join_market_events_without_score(
        event_dataset,
        records,
        load_market_1718(args.market_cache),
    )
    live_dataset = build_strict_live_snapshot_dataset(
        args.data_root,
        market_event_dataset,
        records,
        cutoffs=cutoffs,
    )
    report = leave_one_league_out_remaining_goal_test(live_dataset)
    payload = {
        "remaining_goal_benchmark": report.to_dict(),
        "dataset": {
            "matches": int(live_dataset.frame["match_id"].nunique()),
            "snapshots": len(live_dataset.frame),
            "features": len(live_dataset.feature_columns),
            "cutoffs": list(cutoffs),
            "market_join_uses_final_score": False,
            "strict_side_resolution": True,
            "event_final_score_verified": True,
            "validation": "leave-one-league-out",
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
