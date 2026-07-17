from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.event_tail_model import EventTailConfig, walk_forward_event_tail_test
from football_v2.market_event_data import join_market_events, load_market_1718
from football_v2.market_event_experiment import walk_forward_market_event_test
from football_v2.wyscout_events import (
    build_wyscout_event_dataset,
    load_wyscout_index,
    load_wyscout_league_matches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wyscout full-season tail backtests")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--market-cache", default=".cache/football-data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-matches", type=int)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.55)
    parser.add_argument("--output", default="artifacts/wyscout_event_tail.json")
    args = parser.parse_args()

    index_records = load_wyscout_index(args.data_root)
    matches = load_wyscout_league_matches(
        args.data_root,
        workers=args.workers,
        max_matches=args.max_matches,
    )
    event_dataset = build_wyscout_event_dataset(matches)
    config = EventTailConfig(
        max_alert_coverage=0.06,
        minimum_alerts=6,
        minimum_lift=1.35,
        max_iter=260,
        min_samples_leaf=24,
    )
    event_report = walk_forward_event_tail_test(
        event_dataset,
        folds=args.folds,
        initial_train_fraction=args.initial_train_fraction,
        config=config,
    )

    market = load_market_1718(args.market_cache)
    market_event_dataset = join_market_events(event_dataset, index_records, market)
    jackpot_report = walk_forward_market_event_test(
        market_event_dataset,
        target_column="tail_target",
        folds=args.folds,
        initial_train_fraction=args.initial_train_fraction,
        config=config,
    )
    upset_report = walk_forward_market_event_test(
        market_event_dataset,
        target_column="upset_jackpot_target",
        folds=args.folds,
        initial_train_fraction=args.initial_train_fraction,
        config=EventTailConfig(
            max_alert_coverage=0.04,
            minimum_alerts=4,
            minimum_lift=1.50,
            max_iter=280,
            min_samples_leaf=20,
        ),
    )
    payload = {
        "event_only_experiment": event_report.to_dict(),
        "market_event_jackpot": jackpot_report.to_dict(),
        "market_event_upset_jackpot": upset_report.to_dict(),
        "dataset": {
            "event_rows": len(event_dataset.frame),
            "market_join_rows": len(market_event_dataset.frame),
            "event_features": len(event_dataset.feature_columns),
            "combined_features": len(market_event_dataset.feature_columns),
            "source": "Public Wyscout 2017/18 top-five leagues + football-data odds",
            "first_date": str(market_event_dataset.frame["date"].min().date()),
            "last_date": str(market_event_dataset.frame["date"].max().date()),
            "jackpot_tail_rate": float(market_event_dataset.frame["tail_target"].mean()),
            "upset_jackpot_rate": float(
                market_event_dataset.frame["upset_jackpot_target"].mean()
            ),
        },
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
