from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.event_tail_model import EventTailConfig, walk_forward_event_tail_test
from football_v2.statsbomb_events import (
    CompetitionSeason,
    build_event_dataset,
    download_statsbomb_matches,
)


def _selections(value: str) -> list[CompetitionSeason]:
    result: list[CompetitionSeason] = []
    for item in value.split(","):
        competition, season = item.strip().split(":", maxsplit=1)
        result.append(CompetitionSeason(int(competition), int(season)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="StatsBomb event-tail backtest")
    parser.add_argument("--selections", default="9:27,9:281")
    parser.add_argument("--cache-dir", default=".cache/statsbomb")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-matches", type=int, default=650)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--output", default="artifacts/statsbomb_event_tail.json")
    args = parser.parse_args()

    matches = download_statsbomb_matches(
        _selections(args.selections),
        cache_dir=args.cache_dir,
        workers=args.workers,
        max_matches=args.max_matches,
    )
    dataset = build_event_dataset(matches)
    report = walk_forward_event_tail_test(
        dataset,
        folds=args.folds,
        initial_train_fraction=args.initial_train_fraction,
        config=EventTailConfig(),
    )
    payload = report.to_dict()
    payload["dataset"] = {
        "rows": len(dataset.frame),
        "features": len(dataset.feature_columns),
        "selections": args.selections,
        "source": "StatsBomb Open Data",
        "first_date": str(dataset.frame["date"].min().date()),
        "last_date": str(dataset.frame["date"].max().date()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
