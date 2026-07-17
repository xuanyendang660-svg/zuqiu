from __future__ import annotations

import json
from pathlib import Path

from .wyscout_events import (
    WyscoutIndexRecord,
    WyscoutSideRecord,
    _GOAL_TAG,
    _OWN_GOAL_TAG,
    _event_tags,
    _payload_events,
    _team_side_metadata,
)


def _team_ids(events: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(event["teamId"])
                for event in events
                if event.get("teamId") is not None
            }
        )
    )


def _processed_v2_ordered_sides(payload: object) -> tuple[int, int] | None:
    """Read the home-away order written by the processed-v2 generator."""

    if not isinstance(payload, dict):
        return None
    teams = payload.get("teams")
    if not isinstance(teams, dict) or len(teams) != 2:
        return None
    ordered_ids: list[int] = []
    for key, value in teams.items():
        if not isinstance(value, dict) or not isinstance(value.get("team"), dict):
            return None
        try:
            team_id = int(key)
        except (TypeError, ValueError):
            raw_id = value["team"].get("wyId")
            if raw_id is None:
                return None
            team_id = int(raw_id)
        ordered_ids.append(team_id)
    return ordered_ids[0], ordered_ids[1]


def _event_clock_seconds(event: dict[str, object]) -> float:
    period = str(event.get("matchPeriod") or "1H")
    offset = {
        "1H": 0.0,
        "2H": 45.0 * 60.0,
        "E1": 90.0 * 60.0,
        "ET1": 90.0 * 60.0,
        "E2": 105.0 * 60.0,
        "ET2": 105.0 * 60.0,
    }.get(period, 0.0)
    return offset + float(event.get("eventSec") or 0.0)


def _is_primary_scoring_event(event: dict[str, object]) -> bool:
    """Identify scoring actions while excluding goalkeeper save events."""

    tags = _event_tags(event)
    if _OWN_GOAL_TAG in tags:
        return True
    if _GOAL_TAG not in tags:
        return False
    event_id = int(event.get("eventId") or -1)
    sub_event_id = int(event.get("subEventId") or -1)
    return event_id == 10 or (
        event_id == 3 and sub_event_id in {30, 33, 35}
    )


def _opponent(team_id: int, team_ids: tuple[int, int]) -> int:
    first, second = team_ids
    if team_id == first:
        return second
    if team_id == second:
        return first
    raise ValueError(f"team {team_id} not in match teams")


def _credited_goal_events(
    events: list[dict[str, object]],
    team_ids: tuple[int, ...],
) -> list[tuple[float, int, str]]:
    """Return one credited scoring event per actual goal.

    Most goals have both a scoring action and a goalkeeper event tagged 101.
    The goalkeeper event is ignored when a primary scoring action exists nearby.
    A tiny number of feeds omit the shot and retain only the failed save; those
    orphan goalkeeper events are credited to the opponent.
    """

    if len(team_ids) != 2:
        return []
    pair = (int(team_ids[0]), int(team_ids[1]))
    primary: list[tuple[float, int, str]] = []
    goalkeeper_candidates: list[tuple[float, int]] = []

    for event in events:
        team_value = event.get("teamId")
        if team_value is None:
            continue
        team_id = int(team_value)
        if team_id not in pair:
            continue
        tags = _event_tags(event)
        clock = _event_clock_seconds(event)
        if _is_primary_scoring_event(event):
            scoring_team = (
                _opponent(team_id, pair)
                if _OWN_GOAL_TAG in tags
                else team_id
            )
            source = "own_goal" if _OWN_GOAL_TAG in tags else "scoring_action"
            primary.append((clock, scoring_team, source))
        elif int(event.get("eventId") or -1) == 9 and _GOAL_TAG in tags:
            goalkeeper_candidates.append((clock, team_id))

    credits = list(primary)
    primary_times = [clock for clock, _, _ in primary]
    orphan_credits: list[tuple[float, int, str]] = []
    for clock, goalkeeper_team in goalkeeper_candidates:
        if any(abs(clock - primary_clock) <= 8.0 for primary_clock in primary_times):
            continue
        scoring_team = _opponent(goalkeeper_team, pair)
        duplicate = any(
            abs(clock - existing_clock) <= 8.0 and existing_team == scoring_team
            for existing_clock, existing_team, _ in orphan_credits
        )
        if not duplicate:
            orphan_credits.append((clock, scoring_team, "orphan_goalkeeper"))
    credits.extend(orphan_credits)
    credits.sort(key=lambda item: item[0])
    return credits


def _credited_goal_counts(
    events: list[dict[str, object]],
    team_ids: tuple[int, ...],
) -> dict[int, int]:
    counts = {team_id: 0 for team_id in team_ids}
    for _, scoring_team, _ in _credited_goal_events(events, team_ids):
        counts[scoring_team] += 1
    return counts


def _register_mapping(
    record: WyscoutIndexRecord,
    sides: tuple[int, int],
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
) -> None:
    home_id, away_id = sides
    pairs = ((record.home_name, home_id), (record.away_name, away_id))
    for name, team_id in pairs:
        known_id = name_to_id.get(name)
        known_name = id_to_name.get(team_id)
        if known_id is not None and known_id != team_id:
            raise RuntimeError(
                f"team name {name!r} maps to both {known_id} and {team_id}"
            )
        if known_name is not None and known_name != name:
            raise RuntimeError(
                f"team id {team_id} maps to both {known_name!r} and {name!r}"
            )
        name_to_id[name] = team_id
        id_to_name[team_id] = name


def resolve_wyscout_sides_strict(
    records: list[WyscoutIndexRecord],
) -> list[WyscoutSideRecord]:
    summaries: dict[
        int,
        tuple[tuple[int, ...], dict[int, int], tuple[int, int] | None],
    ] = {}
    resolved: dict[int, tuple[int, int]] = {}
    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}

    for record in records:
        payload = json.loads(Path(record.path).read_text(encoding="utf-8"))
        events = _payload_events(payload)
        team_ids = _team_ids(events)
        counts = _credited_goal_counts(events, team_ids)
        explicit_sides = _processed_v2_ordered_sides(payload)
        metadata_sides = _team_side_metadata(payload)
        preferred_sides = explicit_sides or metadata_sides
        summaries[record.match_id] = (team_ids, counts, preferred_sides)

        sides: tuple[int, int] | None = None
        if preferred_sides is not None:
            sides = (int(preferred_sides[0]), int(preferred_sides[1]))
        elif len(team_ids) == 2:
            first, second = team_ids
            possibilities = [
                (home_id, away_id)
                for home_id, away_id in ((first, second), (second, first))
                if counts.get(home_id, 0) == int(record.home_score)
                and counts.get(away_id, 0) == int(record.away_score)
            ]
            if len(possibilities) == 1:
                sides = possibilities[0]

        if sides is not None:
            resolved[record.match_id] = sides
            _register_mapping(record, sides, name_to_id, id_to_name)

    while True:
        progress = False
        for record in records:
            if record.match_id in resolved:
                continue
            team_ids, _, _ = summaries[record.match_id]
            if len(team_ids) != 2:
                continue
            first, second = team_ids
            known_home = name_to_id.get(record.home_name)
            known_away = name_to_id.get(record.away_name)
            sides: tuple[int, int] | None = None
            if known_home in team_ids and known_away in team_ids:
                sides = (int(known_home), int(known_away))
            elif known_home in team_ids:
                away_id = second if first == known_home else first
                sides = (int(known_home), int(away_id))
            elif known_away in team_ids:
                home_id = second if first == known_away else first
                sides = (int(home_id), int(known_away))
            else:
                first_name = id_to_name.get(first)
                second_name = id_to_name.get(second)
                if first_name == record.home_name or second_name == record.away_name:
                    sides = (first, second)
                elif second_name == record.home_name or first_name == record.away_name:
                    sides = (second, first)
            if sides is None:
                continue
            resolved[record.match_id] = sides
            _register_mapping(record, sides, name_to_id, id_to_name)
            progress = True
        if not progress:
            break

    unresolved = [record for record in records if record.match_id not in resolved]
    if unresolved:
        sample = "; ".join(
            f"{record.match_id}:{record.home_name}-{record.away_name}"
            for record in unresolved[:10]
        )
        raise RuntimeError(
            f"strict side resolution left {len(unresolved)} matches unresolved: {sample}"
        )

    output: list[WyscoutSideRecord] = []
    failures: list[str] = []
    for record in records:
        home_id, away_id = resolved[record.match_id]
        team_ids, counts, _ = summaries[record.match_id]
        if home_id not in team_ids or away_id not in team_ids:
            failures.append(f"{record.match_id}: resolved IDs absent from events")
            continue
        actual = (counts.get(home_id, 0), counts.get(away_id, 0))
        expected = (int(record.home_score), int(record.away_score))
        if actual != expected:
            failures.append(
                f"{record.match_id}: credited score {actual[0]}-{actual[1]} "
                f"!= index score {expected[0]}-{expected[1]}"
            )
            continue
        output.append(WyscoutSideRecord(record, home_id, away_id))

    if failures:
        sample = "; ".join(failures[:10])
        raise RuntimeError(
            f"strict side resolution failed validation for {len(failures)} matches: {sample}"
        )
    return output
