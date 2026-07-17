from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.strict_side_resolution import (
    _credited_goal_counts,
    _event_clock_seconds,
    _is_primary_scoring_event,
    _processed_v2_ordered_sides,
    _team_ids,
)
from football_v2.wyscout_events import (
    _GOAL_TAG,
    _event_tags,
    _payload_events,
    load_wyscout_index,
)


def _event_row(event: dict[str, object], sides: tuple[int, int]) -> dict[str, object]:
    team_id = int(event.get("teamId") or -1)
    return {
        "clock": _event_clock_seconds(event),
        "period": str(event.get("matchPeriod")),
        "event_sec": float(event.get("eventSec") or 0.0),
        "team_id": team_id,
        "side": "home" if team_id == sides[0] else "away" if team_id == sides[1] else "unknown",
        "event_id": int(event.get("eventId") or -1),
        "sub_event_id": int(event.get("subEventId") or -1),
        "event_name": str(event.get("eventName")),
        "sub_event_name": str(event.get("subEventName")),
        "tags": sorted(_event_tags(event)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit goal-event pairing gaps")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--output", default="artifacts/wyscout_goal_pairing_audit.json"
    )
    args = parser.parse_args()

    records = load_wyscout_index(args.data_root)
    failures: list[dict[str, object]] = []
    all_keeper_gaps: list[float] = []
    for record in records:
        payload = json.loads(Path(record.path).read_text(encoding="utf-8"))
        events = _payload_events(payload)
        sides = _processed_v2_ordered_sides(payload)
        if sides is None:
            continue
        team_ids = _team_ids(events)
        counts = _credited_goal_counts(events, team_ids)
        actual = (counts.get(sides[0], 0), counts.get(sides[1], 0))
        expected = (int(record.home_score), int(record.away_score))

        primary_events = [event for event in events if _is_primary_scoring_event(event)]
        keeper_events = [
            event
            for event in events
            if int(event.get("eventId") or -1) == 9
            and _GOAL_TAG in _event_tags(event)
        ]
        primary_clocks = [_event_clock_seconds(event) for event in primary_events]
        keeper_rows: list[dict[str, object]] = []
        for event in keeper_events:
            clock = _event_clock_seconds(event)
            nearest = min((abs(clock - value) for value in primary_clocks), default=None)
            if nearest is not None:
                all_keeper_gaps.append(float(nearest))
            row = _event_row(event, sides)
            row["nearest_primary_gap"] = nearest
            row["nearest_primary"] = (
                _event_row(
                    min(primary_events, key=lambda item: abs(_event_clock_seconds(item) - clock)),
                    sides,
                )
                if primary_events
                else None
            )
            keeper_rows.append(row)

        if actual != expected:
            failures.append(
                {
                    "match_id": record.match_id,
                    "label": f"{record.home_name} - {record.away_name}",
                    "expected": list(expected),
                    "credited": list(actual),
                    "primary_events": [_event_row(event, sides) for event in primary_events],
                    "keeper_events": keeper_rows,
                }
            )

    ordered_gaps = sorted(all_keeper_gaps)
    payload = {
        "matches": len(records),
        "failures": failures,
        "keeper_primary_gap_summary": {
            "count": len(ordered_gaps),
            "maximum": max(ordered_gaps, default=None),
            "p95": ordered_gaps[int(0.95 * (len(ordered_gaps) - 1))] if ordered_gaps else None,
            "p99": ordered_gaps[int(0.99 * (len(ordered_gaps) - 1))] if ordered_gaps else None,
            "gaps_over_8": sum(value > 8.0 for value in ordered_gaps),
            "gaps_over_15": sum(value > 15.0 for value in ordered_gaps),
            "gaps_over_30": sum(value > 30.0 for value in ordered_gaps),
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
