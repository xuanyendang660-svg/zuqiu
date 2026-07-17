from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from football_v2.live_mechanism_rules import leave_one_league_out_mechanism_test
from football_v2.live_snapshots import build_live_snapshot_dataset
from football_v2.market_event_data import join_market_events, load_market_1718
from football_v2.wyscout_events import (
    WyscoutIndexRecord,
    build_wyscout_event_dataset,
    load_wyscout_index,
    load_wyscout_league_matches,
)


def _normalise(records: list[WyscoutIndexRecord]) -> list[WyscoutIndexRecord]:
    return [
        WyscoutIndexRecord(
            match_id=record.match_id,
            path=record.path,
            date=pd.Timestamp(record.date).normalize(),
            source=record.source,
            home_name=record.home_name,
            away_name=record.away_name,
            home_score=record.home_score,
            away_score=record.away_score,
        )
        for record in records
    ]


def _json_default(value: object) -> object:
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live upset-blowout mechanism test")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--market-cache", default=".cache/football-data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="artifacts/wyscout_mechanism_tail.json")
    args = parser.parse_args()

    records = _normalise(load_wyscout_index(args.data_root))
    matches = load_wyscout_league_matches(args.data_root, workers=args.workers)
    event_dataset = build_wyscout_event_dataset(matches)
    market_event_dataset = join_market_events(
        event_dataset,
        records,
        load_market_1718(args.market_cache),
    )
    live_dataset = build_live_snapshot_dataset(
        args.data_root,
        market_event_dataset,
        records,
        cutoffs=(15, 30, 45),
    )
    report = leave_one_league_out_mechanism_test(live_dataset)
    payload = {
        "mechanism_upset_jackpot": report.to_dict(),
        "dataset": {
            "matches": int(live_dataset.frame["match_id"].nunique()),
            "snapshots": len(live_dataset.frame),
            "base_rate": float(live_dataset.frame["upset_jackpot_target"].mean()),
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
