"""Command line interface for the exact-score model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import ExactScoreModel
from .schema import FinalPick, MatchInput


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one exact football score with adaptive, balanced, aggressive, or contrarian logic.")
    parser.add_argument("match_json", type=Path, help="Path to one match JSON, a list of matches, or {'matches': [...]} JSON.")
    parser.add_argument("--pretty", action="store_true", help="Print a compact human-readable summary.")
    parser.add_argument(
        "--mode",
        choices=["adaptive", "contrarian", "aggressive", "balanced"],
        default="adaptive",
        help="adaptive classifies normal/draw/upset/exchange first; contrarian fades public scores; balanced is conservative.",
    )
    parser.add_argument("--max-total-goals", type=int, default=10, help="Maximum total goals included in score search.")
    parser.add_argument("--max-team-goals", type=int, default=9, help="Maximum goals allowed for either team.")
    parser.add_argument(
        "--score-profile",
        choices=["scorebook", "full"],
        default="scorebook",
        help="scorebook uses the curated score list; full uses every score inside the goal limits.",
    )
    parser.add_argument(
        "--no-slate-diversity",
        action="store_true",
        help="Disable batch de-clumping and keep every match's isolated top score.",
    )
    args = parser.parse_args()

    payload = json.loads(args.match_json.read_text(encoding="utf-8"))
    matches = _matches_from_payload(payload)
    model = ExactScoreModel(
        mode=args.mode,
        max_total_goals=args.max_total_goals,
        max_team_goals=args.max_team_goals,
        score_profile=args.score_profile,
    )
    picks = model.predict_many(matches, diversify=not args.no_slate_diversity)

    if args.pretty:
        _print_pretty(picks)
        return

    output = picks[0].to_dict() if len(picks) == 1 else [pick.to_dict() for pick in picks]
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _matches_from_payload(payload: object) -> list[MatchInput]:
    if isinstance(payload, list):
        return [MatchInput.from_dict(item) for item in payload]
    if isinstance(payload, dict) and "matches" in payload:
        matches = payload["matches"]
        if not isinstance(matches, list):
            raise ValueError("'matches' must be a list.")
        return [MatchInput.from_dict(item) for item in matches]
    if isinstance(payload, dict):
        return [MatchInput.from_dict(payload)]
    raise ValueError("Input JSON must be one match object, a match list, or {'matches': [...]}.")


def _print_pretty(picks: list[FinalPick]) -> None:
    if len(picks) == 1:
        pick = picks[0]
        print(f"{pick.home_team} vs {pick.away_team}")
        print(f"唯一比分: {pick.score} | 信号等级: {pick.confidence_band}")
        print(f"xG估计: {pick.expected_home_goals:.2f}-{pick.expected_away_goals:.2f}")
        print("各总球层冠军:")
        for candidate in pick.layer_winners:
            print(
                f"  T={candidate.total_goals}: {candidate.score} "
                f"real={candidate.real_probability:.3%} edge={candidate.edge_ratio:.2f} "
                f"value={candidate.final_value:.5f}"
            )
        return

    print("多场唯一精准比分结果")
    print("场次 | 比赛 | 唯一比分 | 信号 | xG估计")
    print("-" * 72)
    for index, pick in enumerate(picks, start=1):
        print(
            f"{index:02d} | {pick.home_team} vs {pick.away_team} | "
            f"{pick.score} | {pick.confidence_band} | "
            f"{pick.expected_home_goals:.2f}-{pick.expected_away_goals:.2f}"
        )


if __name__ == "__main__":
    main()
