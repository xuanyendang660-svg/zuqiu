from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.event_tail_model import EventTailConfig, walk_forward_event_tail_test
from football_v2.wyscout_events import build_wyscout_event_dataset, load_wyscout_league_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Wyscout full-season event-tail backtest")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-matches", type=int)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.55)
    parser.add_argument("--output", default="artifacts/wyscout_event_tail.json")
    args = parser.parse_args()

    matches = load_wyscout_league_matches(
        args.data_root,
        workers=args.workers,
        max_matches=args.max_matches,
    )
    dataset = build_wyscout_event_dataset(matches)
    report = walk_forward_event_tail_test(
        dataset,
        folds=args.folds,
        initial_train_fraction=args.initial_train_fraction,
        config=EventTailConfig(
            max_alert_coverage=0.06,
            minimum_alerts=6,
            minimum_lift=1.35,
            max_iter=260,
            min_samples_leaf=24,
        ),
    )
    payload = report.to_dict()
    payload["dataset"] = {
        "rows": len(dataset.frame),
        "features": len(dataset.feature_columns),
        "source": "Public Wyscout 2017/18 top-five leagues",
        "first_date": str(dataset.frame["date"].min().date()),
        "last_date": str(dataset.frame["date"].max().date()),
        "jackpot_tail_rate": float(dataset.frame["tail_target"].mean()),
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
