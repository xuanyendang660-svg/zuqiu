from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.elo_live_state import (
    build_elo_live_state_frame,
    evaluate_elo_live_state,
)
from football_v2.statsbomb_discovery import discover_male_competition_seasons
from football_v2.statsbomb_events import download_statsbomb_matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-source Elo-defined 30-minute two-nil validation"
    )
    parser.add_argument("--cache-dir", default=".cache/statsbomb")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-matches", type=int, default=1800)
    parser.add_argument(
        "--output", default="artifacts/statsbomb_elo_live_state.json"
    )
    args = parser.parse_args()

    selections = discover_male_competition_seasons(args.cache_dir)
    matches = download_statsbomb_matches(
        selections,
        cache_dir=args.cache_dir,
        workers=args.workers,
        max_matches=args.max_matches,
    )
    frame = build_elo_live_state_frame(matches, cutoff=30)
    report = evaluate_elo_live_state(frame, cutoff=30)
    payload = {
        "live_state_validation": report.to_dict(),
        "dataset": {
            "source": "StatsBomb Open Data",
            "matches": len(matches),
            "selection_count": len(selections),
            "first_date": str(frame["date"].min().date()),
            "last_date": str(frame["date"].max().date()),
            "underdog_definition": "pre-match rolling Elo with 65-point home advantage",
            "alert": "underdog leads 2-0 at 30 minutes",
            "target": "underdog finishes with at least 3 goals and wins by at least 2",
            "final_score_not_used_to_define_underdog": True,
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
