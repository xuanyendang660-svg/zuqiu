from __future__ import annotations

import argparse
import json

from football_v2.halftime_topflight_predictor import (
    predict_topflight_halftime_exact_score_from_odds,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validated top-flight halftime exact-score predictor"
    )
    parser.add_argument("--home-odds", type=float, required=True)
    parser.add_argument("--draw-odds", type=float, required=True)
    parser.add_argument("--away-odds", type=float, required=True)
    parser.add_argument("--halftime-home", type=int, required=True)
    parser.add_argument("--halftime-away", type=int, required=True)
    parser.add_argument(
        "--top-flight",
        action="store_true",
        help="Confirm this is a domestic top-flight league match",
    )
    args = parser.parse_args()

    result = predict_topflight_halftime_exact_score_from_odds(
        is_top_flight_domestic_league=args.top_flight,
        home_odds=args.home_odds,
        draw_odds=args.draw_odds,
        away_odds=args.away_odds,
        halftime_home_score=args.halftime_home,
        halftime_away_score=args.halftime_away,
    )
    print(
        json.dumps(
            {
                "accepted": result.accepted,
                "home_score": result.home_score,
                "away_score": result.away_score,
                "rule_id": result.rule_id,
                "underdog_side": result.underdog_side,
                "transfer_matches": result.transfer_matches,
                "transfer_accuracy": result.transfer_accuracy,
                "transfer_baseline_accuracy": result.transfer_baseline_accuracy,
                "reason": result.reason,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
