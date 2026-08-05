"""CLI for independent single/parlay settlement reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import SettledBet, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report ROI, CLV, drawdown and losing streaks separately."
    )
    parser.add_argument("records_json", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.records_json.read_text(encoding="utf-8"))
    rows = payload["records"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Input must be a record list or {'records': [...]}")

    records = [SettledBet.from_dict(row) for row in rows]
    print(json.dumps(summarize_records(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
