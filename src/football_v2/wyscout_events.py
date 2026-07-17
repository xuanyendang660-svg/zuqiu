from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .labels import jackpot_tail_type, is_jackpot_tail
from .statsbomb_events import EventDataset, EventMatch, TeamEventSummary, build_event_dataset


@dataclass(frozen=True)
class WyscoutIndexRecord:
    match_id: int
    path: Path
    date: pd.Timestamp
    source: str
    home_name: str
    away_name: str
    home_score: int
    away_score: int


@dataclass(frozen=True)
class WyscoutSideRecord:
    index: WyscoutIndexRecord
    home_team_id: int
    away_team_id: int


_INDEX_PATTERN = re.compile(
    r"^\|\[(?P<id>\d+)\]\(files/\d+\.json\)\|"
    r"(?P<home>.*?) - (?P<away>.*?), (?P<hg>\d+) - (?P<ag>\d+)"
    r"(?: \([A-Z]\))?\|(?P<date>.*?)\|(?P<source>.*?)\|$"
)
_ALLOWED_SOURCES = {
    "matches_England.json",
    "matches_France.json",
    "matches_Germany.json",
    "matches_Italy.json",
    "matches_Spain.json",
}
_GOAL_TAG = 101
_OWN_GOAL_TAG = 102
_ACCURATE_TAG = 1801
_RED_CARD_TAGS = {1701, 1703}
_YELLOW_CARD_TAG = 1702


def _parse_date(value: str) -> pd.Timestamp:
    cleaned = re.sub(r"\s+GMT[+-]\d+(?::\d+)?$", "", value.strip())
    cleaned = cleaned.replace(" at ", " ")
    return pd.Timestamp(pd.to_datetime(cleaned, errors="raise")).tz_localize(None)


def load_wyscout_index(repository_root: str | Path) -> list[WyscoutIndexRecord]:
    processed = Path(repository_root) / "processed"
    records: list[WyscoutIndexRecord] = []
    for line in (processed / "README.md").read_text(encoding="utf-8").splitlines():
        match = _INDEX_PATTERN.match(line.strip())
        if match is None or match.group("source") not in _ALLOWED_SOURCES:
            continue
        match_id = int(match.group("id"))
        records.append(
            WyscoutIndexRecord(
                match_id=match_id,
                path=processed / "files" / f"{match_id}.json",
                date=_parse_date(match.group("date")),
                source=match.group("source"),
                home_name=match.group("home").strip(),
                away_name=match.group("away").strip(),
                home_score=int(match.group("hg")),
                away_score=int(match.group("ag")),
            )
        )
    records.sort(key=lambda item: (item.date, item.match_id))
    if len(records) < 1800:
        raise RuntimeError(f"expected full top-five leagues; found {len(records)} matches")
    return records


def _event_tags(event: dict[str, object]) -> set[int]:
    return {
        int(tag["id"])
        for tag in event.get("tags", [])
        if isinstance(tag, dict) and tag.get("id") is not None
    }


def _payload_events(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        events = payload.get("events")
    else:
        events = payload
    if not isinstance(events, list):
        raise ValueError("Wyscout payload does not contain an event list")
    return [event for event in events if isinstance(event, dict)]


def _team_side_metadata(payload: object) -> tuple[int, int] | None:
    if not isinstance(payload, dict):
        return None
    candidates: list[object] = [payload.get("teamsData"), payload.get("teams")]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend([metadata.get("teamsData"), metadata.get("teams")])
    match = payload.get("match")
    if isinstance(match, dict):
        candidates.extend([match.get("teamsData"), match.get("teams")])
    for candidate in candidates:
        if isinstance(candidate, dict):
            home_id: int | None = None
            away_id: int | None = None
            for key, value in candidate.items():
                if not isinstance(value, dict):
                    continue
                side = str(value.get("side") or value.get("role") or "").lower()
                team_id = int(value.get("teamId") or value.get("id") or key)
                if side == "home":
                    home_id = team_id
                elif side == "away":
                    away_id = team_id
            if home_id is not None and away_id is not None:
                return home_id, away_id
        elif isinstance(candidate, list):
            home_id = away_id = None
            for value in candidate:
                if not isinstance(value, dict):
                    continue
                side = str(value.get("side") or value.get("role") or "").lower()
                team_id = int(value.get("teamId") or value.get("id"))
                if side == "home":
                    home_id = team_id
                elif side == "away":
                    away_id = team_id
            if home_id is not None and away_id is not None:
                return home_id, away_id
    return None


def _goal_counts(events: list[dict[str, object]]) -> tuple[dict[int, int], set[int]]:
    counts: dict[int, int] = {}
    team_ids: set[int] = set()
    for event in events:
        if event.get("teamId") is None:
            continue
        team_id = int(event["teamId"])
        team_ids.add(team_id)
        tags = _event_tags(event)
        if _GOAL_TAG in tags and _OWN_GOAL_TAG not in tags:
            counts[team_id] = counts.get(team_id, 0) + 1
    return counts, team_ids


def resolve_wyscout_sides(records: list[WyscoutIndexRecord]) -> list[WyscoutSideRecord]:
    summaries: dict[int, tuple[tuple[int, ...], dict[int, int], tuple[int, int] | None]] = {}
    team_name_to_id: dict[str, int] = {}
    team_id_to_name: dict[int, str] = {}
    resolved: dict[int, tuple[int, int]] = {}

    for record in records:
        payload = json.loads(record.path.read_text(encoding="utf-8"))
        events = _payload_events(payload)
        counts, team_ids = _goal_counts(events)
        metadata_sides = _team_side_metadata(payload)
        summaries[record.match_id] = (tuple(sorted(team_ids)), counts, metadata_sides)
        if metadata_sides is not None:
            resolved[record.match_id] = metadata_sides
            team_id_to_name[metadata_sides[0]] = record.home_name
            team_id_to_name[metadata_sides[1]] = record.away_name
            team_name_to_id[record.home_name] = metadata_sides[0]
            team_name_to_id[record.away_name] = metadata_sides[1]
            continue
        if len(team_ids) != 2:
            continue
        first, second = sorted(team_ids)
        possibilities = []
        for home_id, away_id in ((first, second), (second, first)):
            if (
                counts.get(home_id, 0) == record.home_score
                and counts.get(away_id, 0) == record.away_score
            ):
                possibilities.append((home_id, away_id))
        if len(possibilities) == 1:
            home_id, away_id = possibilities[0]
            resolved[record.match_id] = (home_id, away_id)
            team_id_to_name[home_id] = record.home_name
            team_id_to_name[away_id] = record.away_name
            team_name_to_id[record.home_name] = home_id
            team_name_to_id[record.away_name] = away_id

    for _ in range(4):
        progress = False
        for record in records:
            if record.match_id in resolved:
                continue
            team_ids, _, _ = summaries[record.match_id]
            if len(team_ids) != 2:
                continue
            known_home = team_name_to_id.get(record.home_name)
            known_away = team_name_to_id.get(record.away_name)
            if known_home in team_ids and known_away in team_ids:
                resolved[record.match_id] = (int(known_home), int(known_away))
                progress = True
            elif known_home in team_ids:
                away_id = next(team_id for team_id in team_ids if team_id != known_home)
                resolved[record.match_id] = (int(known_home), away_id)
                team_name_to_id[record.away_name] = away_id
                team_id_to_name[away_id] = record.away_name
                progress = True
            elif known_away in team_ids:
                home_id = next(team_id for team_id in team_ids if team_id != known_away)
                resolved[record.match_id] = (home_id, int(known_away))
                team_name_to_id[record.home_name] = home_id
                team_id_to_name[home_id] = record.home_name
                progress = True
        if not progress:
            break

    output: list[WyscoutSideRecord] = []
    for record in records:
        team_ids, _, metadata_sides = summaries[record.match_id]
        sides = resolved.get(record.match_id) or metadata_sides
        if sides is None:
            if len(team_ids) != 2:
                continue
            sides = (team_ids[0], team_ids[1])
        output.append(WyscoutSideRecord(record, int(sides[0]), int(sides[1])))
    if len(output) < 1800:
        raise RuntimeError(f"resolved sides for only {len(output)} matches")
    return output


def _shot_quality(x: float, y: float, header: bool) -> float:
    distance = float(np.hypot(100.0 - x, 0.70 * (y - 50.0))) / 100.0
    logit = -1.45 - 7.0 * distance - (0.35 if header else 0.0)
    return float(1.0 / (1.0 + np.exp(-logit)))


def _first_eleven(events: list[dict[str, object]], team_id: int) -> frozenset[int]:
    players: list[int] = []
    for event in events:
        if int(event.get("teamId") or -1) != team_id:
            continue
        player_id = int(event.get("playerId") or 0)
        if player_id > 0 and player_id not in players:
            players.append(player_id)
            if len(players) == 11:
                break
    return frozenset(players)


def parse_wyscout_file(record: WyscoutSideRecord) -> EventMatch:
    payload = json.loads(record.index.path.read_text(encoding="utf-8"))
    events = _payload_events(payload)
    summaries = {
        record.home_team_id: TeamEventSummary(
            record.home_team_id, record.index.home_name
        ),
        record.away_team_id: TeamEventSummary(
            record.away_team_id, record.index.away_name
        ),
    }
    summaries[record.home_team_id].starting_xi = _first_eleven(
        events, record.home_team_id
    )
    summaries[record.away_team_id].starting_xi = _first_eleven(
        events, record.away_team_id
    )
    previous_team: int | None = None
    previous_second = -10_000.0

    for event in events:
        if event.get("teamId") is None:
            continue
        team_id = int(event["teamId"])
        if team_id not in summaries:
            continue
        summary = summaries[team_id]
        event_name = str(event.get("eventName") or "")
        sub_event = str(event.get("subEventName") or "")
        tags = _event_tags(event)
        positions = event.get("positions") or []
        start = positions[0] if positions and isinstance(positions[0], dict) else {}
        end = positions[-1] if positions and isinstance(positions[-1], dict) else start
        x = float(start.get("x") or 0.0)
        y = float(start.get("y") or 50.0)
        end_x = float(end.get("x") or x)
        end_y = float(end.get("y") or y)
        period = str(event.get("matchPeriod") or "1H")
        event_second = float(event.get("eventSec") or 0.0)
        absolute_second = event_second + (45 * 60 if period == "2H" else 0)
        goal = _GOAL_TAG in tags
        accurate = _ACCURATE_TAG in tags
        quick_transition = (
            previous_team is not None
            and previous_team != team_id
            and absolute_second - previous_second <= 12.0
        )
        is_shot = event_name == "Shot" or sub_event == "Penalty"

        if is_shot:
            xg = _shot_quality(x, y, 403 in tags)
            summary.xg += xg
            summary.shots += 1
            summary.shots_on_target += int(goal or accurate)
            summary.big_chances += int(xg >= 0.18)
            summary.counter_xg += xg if quick_transition else 0.0
            summary.set_piece_xg += xg if event_name == "Free Kick" else 0.0
            if goal:
                summary.goal_minutes.append(int(absolute_second // 60))
        if event_name == "Pass":
            summary.passes += 1
            summary.completed_passes += int(accurate)
        if event_name in {"Duel", "Others on the ball"}:
            summary.pressures += 1
        if event_name in {"Duel", "Others on the ball"} and accurate and x >= 67:
            summary.high_recoveries += 1
        if (event_name == "Pass" and not accurate) or 1302 in tags:
            summary.turnovers += 1
        if event_name == "Foul":
            summary.fouls += 1
        if tags & (_RED_CARD_TAGS | {_YELLOW_CARD_TAG}):
            summary.cards += 1
        summary.progressive_actions += int(end_x - x >= 20)
        summary.final_third_entries += int(x < 67 <= end_x)
        start_box = x >= 85 and 20 <= y <= 80
        end_box = end_x >= 85 and 20 <= end_y <= 80
        summary.box_entries += int(not start_box and end_box)
        previous_team = team_id
        previous_second = absolute_second

    home = summaries[record.home_team_id]
    away = summaries[record.away_team_id]
    home.goals = record.index.home_score
    away.goals = record.index.away_score
    return EventMatch(
        match_id=record.index.match_id,
        date=record.index.date,
        competition_id=abs(hash(record.index.source)) % 1_000_000,
        season_id=201718,
        home_team_id=record.home_team_id,
        home_team_name=record.index.home_name,
        away_team_id=record.away_team_id,
        away_team_name=record.index.away_name,
        home_score=record.index.home_score,
        away_score=record.index.away_score,
        home=home,
        away=away,
    )


def load_wyscout_league_matches(
    repository_root: str | Path,
    *,
    workers: int = 8,
    max_matches: int | None = None,
) -> list[EventMatch]:
    records = load_wyscout_index(repository_root)
    sides = resolve_wyscout_sides(records)
    if max_matches is not None:
        sides = sides[:max_matches]
    matches: list[EventMatch] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parse_wyscout_file, record): record for record in sides}
        for future in as_completed(futures):
            record = futures[future]
            try:
                matches.append(future.result())
            except Exception as exc:  # pragma: no cover - integration diagnostics
                failures.append(f"{record.index.path.name}: {exc}")
    if len(matches) < 1700:
        sample = "; ".join(failures[:5])
        raise RuntimeError(f"only parsed {len(matches)} Wyscout matches; {sample}")
    matches.sort(key=lambda match: (match.date, match.match_id))
    return matches


def build_wyscout_event_dataset(matches: list[EventMatch]) -> EventDataset:
    dataset = build_event_dataset(matches)
    frame = dataset.frame.copy()
    frame["tail_target"] = [
        int(is_jackpot_tail(int(home), int(away)))
        for home, away in zip(
            frame["home_score"], frame["away_score"], strict=True
        )
    ]
    frame["tail_type"] = [
        jackpot_tail_type(int(home), int(away)).value
        for home, away in zip(
            frame["home_score"], frame["away_score"], strict=True
        )
    ]
    return EventDataset(frame, dataset.feature_columns)
