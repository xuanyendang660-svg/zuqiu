from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from football_v2.strict_side_resolution import (
    _credited_goal_counts,
    _processed_v2_ordered_sides,
    _team_ids,
)
from football_v2.wyscout_events import (
    _GOAL_TAG,
    _OWN_GOAL_TAG,
    _event_tags,
    _payload_events,
    load_wyscout_index,
)


def _signature(event: dict[str, object]) -> str:
    tags = _event_tags(event)
    return "|".join(
        [
            f"event={int(event.get('eventId') or -1)}",
            f"sub={int(event.get('subEventId') or -1)}",
            f"name={event.get('eventName')}",
            f"subname={event.get('subEventName')}",
            f"goal={int(_GOAL_TAG in tags)}",
            f"own={int(_OWN_GOAL_TAG in tags)}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Wyscout goal event encodings")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--output", default="artifacts/wyscout_goal_encoding_audit.json"
    )
    args = parser.parse_args()

    records = load_wyscout_index(args.data_root)
    signature_counts: Counter[str] = Counter()
    mismatch_signature_counts: Counter[str] = Counter()
    deficits: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    mismatches = 0

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
        tagged_events = []
        for event in events:
            tags = _event_tags(event)
            if _GOAL_TAG not in tags and _OWN_GOAL_TAG not in tags:
                continue
            signature = _signature(event)
            signature_counts[signature] += 1
            tagged_events.append(
                {
                    "minute": str(event.get("matchPeriod"))
                    + ":"
                    + str(event.get("eventSec")),
                    "team_id": int(event.get("teamId") or -1),
                    "side": (
                        "home"
                        if int(event.get("teamId") or -1) == sides[0]
                        else "away"
                        if int(event.get("teamId") or -1) == sides[1]
                        else "unknown"
                    ),
                    "signature": signature,
                    "tags": sorted(tags),
                }
            )
        if actual == expected:
            continue
        mismatches += 1
        deficits[f"{expected[0] - actual[0]},{expected[1] - actual[1]}"] += 1
        for item in tagged_events:
            mismatch_signature_counts[str(item["signature"])] += 1
        if len(examples) < 30:
            examples.append(
                {
                    "match_id": record.match_id,
                    "label": f"{record.home_name} - {record.away_name}",
                    "expected": list(expected),
                    "shot_only_count": list(actual),
                    "ordered_sides": list(sides),
                    "tagged_events": tagged_events,
                }
            )

    payload = {
        "matches": len(records),
        "mismatches": mismatches,
        "deficits": dict(deficits.most_common()),
        "all_goal_tag_signatures": dict(signature_counts.most_common()),
        "mismatch_goal_tag_signatures": dict(
            mismatch_signature_counts.most_common()
        ),
        "examples": examples,
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
