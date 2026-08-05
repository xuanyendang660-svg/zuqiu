from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .backtest import evaluate_backtest
from .models import Market, OutcomeQuote, Policy, SettledBet
from .parlay import build_two_leg_parlays
from .selector import deployment_allowed, evaluate_market, select_portfolio


def _read_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON must be an object")
    return payload


def _policy(payload: dict[str, Any]) -> Policy:
    allowed = {item.name for item in fields(Policy)}
    values = {key: value for key, value in payload.items() if key in allowed}
    policy = Policy(**values)
    policy.validate()
    return policy


def _markets(payload: list[dict[str, Any]]) -> tuple[Market, ...]:
    markets: list[Market] = []
    for item in payload:
        event_id = str(item["event_id"])
        market_id = str(item["market_id"])
        market_type = str(item["market_type"])
        outcomes = tuple(
            OutcomeQuote(
                event_id=event_id,
                market_id=market_id,
                market_type=market_type,
                selection=str(outcome["selection"]),
                odds=float(outcome["odds"]),
                model_probability=float(outcome["model_probability"]),
                uncertainty=float(outcome.get("uncertainty", 0.0)),
                data_quality=float(outcome.get("data_quality", 1.0)),
                bookmaker=str(outcome.get("bookmaker", "")),
                line=(float(outcome["line"]) if outcome.get("line") is not None else None),
            )
            for outcome in item["outcomes"]
        )
        markets.append(
            Market(
                event_id=event_id,
                market_id=market_id,
                market_type=market_type,
                outcomes=outcomes,
                devig_method=str(item.get("devig_method", "proportional")),
            )
        )
    return tuple(markets)


def _dump(payload: dict[str, Any], pretty: bool) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if pretty else None,
            sort_keys=pretty,
        )
    )


def run_select(path: str, pretty: bool) -> None:
    payload = _read_json(path)
    policy = _policy(dict(payload.get("policy", {})))
    settled_bets = int(payload.get("settled_bets", 0))
    markets = _markets(list(payload.get("markets", [])))

    audits = {
        market.market_id: [asdict(item) for item in evaluate_market(
            market, policy, settled_bets=settled_bets
        )]
        for market in markets
    }
    singles = select_portfolio(markets, policy, settled_bets=settled_bets)
    parlays = build_two_leg_parlays(singles, policy)
    can_deploy = deployment_allowed(policy, settled_bets)

    _dump(
        {
            "deployment_allowed": can_deploy,
            "mode": "live" if can_deploy else "shadow",
            "settled_bets": settled_bets,
            "single_status": "BET" if singles else "NO_BET",
            "parlay_status": "BET" if parlays else "NO_PARLAY",
            "singles": [asdict(item) for item in singles],
            "two_leg_parlays": [asdict(item) for item in parlays],
            "market_audit": audits,
        },
        pretty,
    )


def run_backtest(path: str, pretty: bool) -> None:
    payload = _read_json(path)
    records = tuple(
        SettledBet(
            bet_id=str(item["bet_id"]),
            stake=float(item["stake"]),
            profit=float(item["profit"]),
            won=(None if item.get("won") is None else bool(item["won"])),
            taken_odds=float(item["taken_odds"]),
            closing_odds=(
                float(item["closing_odds"])
                if item.get("closing_odds") is not None
                else None
            ),
        )
        for item in payload.get("records", [])
    )
    _dump(asdict(evaluate_backtest(records)), pretty)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-value")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="evaluate singles and 2-leg parlays")
    select_parser.add_argument("input")
    select_parser.add_argument("--pretty", action="store_true")

    backtest_parser = subparsers.add_parser("backtest", help="evaluate settled bets")
    backtest_parser.add_argument("input")
    backtest_parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "select":
        run_select(args.input, args.pretty)
    elif args.command == "backtest":
        run_backtest(args.input, args.pretty)


if __name__ == "__main__":
    main()
