"""CLI for positive-EV singles and two-leg parlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import RiskConfig
from .models import BetOffer
from .parlay import build_best_two_leg_parlay
from .selector import select_singles


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select positive-EV football singles and at most one two-leg parlay. "
            "Outputs NO_BET/NO_PARLAY when thresholds are not met."
        )
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    config = _config_from_payload(payload.get("config", {}))
    offers = [BetOffer.from_dict(item) for item in payload.get("offers", [])]
    quoted_odds = _quoted_odds_from_payload(payload.get("quoted_parlays", []))

    selected, rejected = select_singles(offers, config)
    parlay = build_best_two_leg_parlay(
        selected,
        config=config,
        quoted_odds=quoted_odds or None,
    )

    output = {
        "mode": "PROFIT_SHADOW",
        "singles": [decision.to_dict() for decision in selected],
        "no_bets": [decision.to_dict() for decision in rejected],
        "parlay": parlay.to_dict(),
        "warning": (
            "Selection is based on supplied probabilities. Profitability is not "
            "established until time-sealed forward testing beats market baselines."
        ),
    }

    if args.pretty:
        _print_pretty(output)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


def _config_from_payload(data: dict[str, Any]) -> RiskConfig:
    allowed = set(RiskConfig.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return RiskConfig(**data)


def _quoted_odds_from_payload(
    rows: list[dict[str, Any]],
) -> dict[frozenset[str], float]:
    output: dict[frozenset[str], float] = {}
    for row in rows:
        legs = row.get("legs", [])
        if len(legs) != 2:
            raise ValueError("Each quoted parlay must contain exactly two leg IDs")
        odds = float(row["decimal_odds"])
        if odds <= 1.0:
            raise ValueError("Quoted parlay odds must be above 1.0")
        output[frozenset(str(leg) for leg in legs)] = odds
    return output


def _print_pretty(output: dict[str, Any]) -> None:
    singles = output["singles"]
    if not singles:
        print("单关: NO_BET")
    else:
        print("单关:")
        for decision in singles:
            offer = decision["offer"]
            print(
                f"- {offer['match_id']} | {offer['market']} {offer['selection']} "
                f"@ {offer['decimal_odds']:.2f} | EV {decision['expected_value']:.2%} "
                f"| 仓位 {decision['stake_fraction']:.2%}"
            )

    parlay = output["parlay"]
    if parlay["action"] == "NO_PARLAY":
        print(f"串关: NO_PARLAY ({parlay['reason']})")
        return

    leg_names = " + ".join(
        leg["offer"]["offer_id"] for leg in parlay["legs"]
    )
    print(
        f"2串1: {leg_names} | @ {parlay['combined_odds']:.2f} "
        f"| EV {parlay['expected_value']:.2%} "
        f"| 仓位 {parlay['stake_fraction']:.2%}"
    )


if __name__ == "__main__":
    main()
