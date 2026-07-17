from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.halftime_two_nil import (
    evaluate_halftime_two_nil,
    load_halftime_market_data,
    season_codes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate pre-match underdog 2-0 halftime states across major leagues"
    )
    parser.add_argument("--cache-dir", default=".cache/football-data-halftime")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--divisions",
        default="E0,D1,F1,I1,SP1",
        help="Comma-separated football-data division codes",
    )
    parser.add_argument(
        "--output", default="artifacts/halftime_two_nil_validation.json"
    )
    args = parser.parse_args()

    seasons = season_codes(args.start_year, args.end_year)
    divisions = tuple(
        item.strip() for item in args.divisions.split(",") if item.strip()
    )
    frame, loading = load_halftime_market_data(
        args.cache_dir,
        seasons=seasons,
        divisions=divisions,
    )
    report = evaluate_halftime_two_nil(frame)
    payload = {
        "validation": report.to_dict(),
        "dataset": {
            "source": "football-data.co.uk",
            "start_year": args.start_year,
            "end_year": args.end_year,
            "divisions": list(divisions),
            "matches": len(frame),
            "first_date": str(frame["date"].min().date()),
            "last_date": str(frame["date"].max().date()),
            "underdog_definition": "lower de-vigged pre-match home/away win probability",
            "alert": "underdog leads 2-0 at halftime",
            "third_goal_target": "underdog finishes with at least 3 goals",
            "blowout_target": "underdog finishes with at least 3 goals and wins by at least 2",
            "final_score_not_used_to_define_underdog": True,
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
