from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.halftime_state_search import run_halftime_state_search
from football_v2.halftime_two_nil import load_halftime_market_data, season_codes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover halftime state rules with train/calibration/test separation"
    )
    parser.add_argument("--cache-dir", default=".cache/football-data-halftime")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--train-end-year", type=int, default=2011)
    parser.add_argument("--calibration-end-year", type=int, default=2017)
    parser.add_argument(
        "--divisions",
        default="E0,D1,F1,I1,SP1",
    )
    parser.add_argument(
        "--output", default="artifacts/halftime_state_search.json"
    )
    args = parser.parse_args()

    divisions = tuple(
        item.strip() for item in args.divisions.split(",") if item.strip()
    )
    frame, loading = load_halftime_market_data(
        args.cache_dir,
        seasons=season_codes(args.start_year, args.end_year),
        divisions=divisions,
    )
    report = run_halftime_state_search(
        frame,
        train_end_year=args.train_end_year,
        calibration_end_year=args.calibration_end_year,
    )
    payload = {
        "state_search": report.to_dict(),
        "dataset": {
            "source": "football-data.co.uk",
            "matches": len(frame),
            "divisions": list(divisions),
            "start_year": args.start_year,
            "end_year": args.end_year,
            "candidate_universe_frozen_before_test": True,
            "test_split_used_once_for_final_evaluation": True,
            **loading,
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
