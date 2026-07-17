from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .labels import ScoreArchetype, classify_score

_RAW_ROOT = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


@dataclass(frozen=True)
class CompetitionSeason:
    competition_id: int
    season_id: int


@dataclass
class TeamEventSummary:
    team_id: int
    team_name: str
    goals: int = 0
    xg: float = 0.0
    shots: int = 0
    shots_on_target: int = 0
    big_chances: int = 0
    counter_xg: float = 0.0
    set_piece_xg: float = 0.0
    box_entries: int = 0
    final_third_entries: int = 0
    progressive_actions: int = 0
    pressures: int = 0
    high_recoveries: int = 0
    turnovers: int = 0
    passes: int = 0
    completed_passes: int = 0
    under_pressure_actions: int = 0
    fouls: int = 0
    cards: int = 0
    starting_xi: frozenset[int] = frozenset()
    formation: int = 0
    goal_minutes: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class EventMatch:
    match_id: int
    date: pd.Timestamp
    competition_id: int
    season_id: int
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_score: int
    away_score: int
    home: TeamEventSummary
    away: TeamEventSummary


@dataclass(frozen=True)
class EventDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]

    @property
    def features(self) -> np.ndarray:
        return self.frame.loc[:, self.feature_columns].to_numpy(dtype=float)

    @property
    def tail_target(self) -> np.ndarray:
        return self.frame["tail_target"].to_numpy(dtype=int)

    @property
    def tail_type(self) -> np.ndarray:
        return self.frame["tail_type"].to_numpy(dtype=str)

    @property
    def home_goals(self) -> np.ndarray:
        return self.frame["home_score"].to_numpy(dtype=int)

    @property
    def away_goals(self) -> np.ndarray:
        return self.frame["away_score"].to_numpy(dtype=int)


@dataclass
class RollingTeamState:
    rows: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=10))
    previous_xi: frozenset[int] = frozenset()
    player_starts: dict[int, int] = field(default_factory=dict)
    last_date: pd.Timestamp | None = None
    formation: int = 0


_EVENT_FEATURES = (
    "goals",
    "xg",
    "shots",
    "shots_on_target",
    "big_chances",
    "counter_xg",
    "set_piece_xg",
    "box_entries",
    "final_third_entries",
    "progressive_actions",
    "pressures",
    "high_recoveries",
    "turnovers",
    "pass_completion",
    "under_pressure_actions",
    "fouls",
    "cards",
    "collapse_conceded",
    "tail_for",
    "tail_against",
)


def _fetch_json(url: str, *, attempts: int = 4, timeout: int = 60) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "football-v2-research"})
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}") from last_error


def download_statsbomb_matches(
    selections: Iterable[CompetitionSeason],
    *,
    cache_dir: str | Path,
    workers: int = 12,
    max_matches: int | None = None,
) -> list[EventMatch]:
    cache = Path(cache_dir)
    match_dir = cache / "matches"
    event_dir = cache / "events"
    match_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, Any]] = []
    for selection in selections:
        path = match_dir / f"{selection.competition_id}-{selection.season_id}.json"
        if path.exists():
            matches = json.loads(path.read_text(encoding="utf-8"))
        else:
            url = (
                f"{_RAW_ROOT}/matches/{selection.competition_id}/"
                f"{selection.season_id}.json"
            )
            matches = _fetch_json(url)
            path.write_text(json.dumps(matches), encoding="utf-8")
        metadata.extend(matches)
    metadata.sort(key=lambda row: (row["match_date"], row["match_id"]))
    if max_matches is not None:
        metadata = metadata[-max_matches:]

    def fetch_event_file(match: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        match_id = int(match["match_id"])
        path = event_dir / f"{match_id}.json"
        if path.exists():
            events = json.loads(path.read_text(encoding="utf-8"))
        else:
            events = _fetch_json(f"{_RAW_ROOT}/events/{match_id}.json")
            path.write_text(json.dumps(events), encoding="utf-8")
        return match, events

    completed: list[EventMatch] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_event_file, match): match for match in metadata}
        for future in as_completed(futures):
            match, events = future.result()
            completed.append(parse_statsbomb_match(match, events))
    completed.sort(key=lambda item: (item.date, item.match_id))
    return completed


def _team_id(event: dict[str, Any]) -> int | None:
    team = event.get("team")
    return int(team["id"]) if isinstance(team, dict) and "id" in team else None


def _end_location(event: dict[str, Any]) -> list[float] | None:
    event_type = event.get("type", {}).get("name")
    if event_type == "Pass":
        return event.get("pass", {}).get("end_location")
    if event_type == "Carry":
        return event.get("carry", {}).get("end_location")
    return None


def _is_box(location: list[float] | None) -> bool:
    return bool(location and len(location) >= 2 and location[0] >= 102 and 18 <= location[1] <= 62)


def parse_statsbomb_match(
    match: dict[str, Any], events: list[dict[str, Any]]
) -> EventMatch:
    home_id = int(match["home_team"]["home_team_id"])
    away_id = int(match["away_team"]["away_team_id"])
    summaries = {
        home_id: TeamEventSummary(home_id, str(match["home_team"]["home_team_name"])),
        away_id: TeamEventSummary(away_id, str(match["away_team"]["away_team_name"])),
    }

    for event in events:
        team_id = _team_id(event)
        if team_id not in summaries:
            continue
        summary = summaries[team_id]
        event_type = event.get("type", {}).get("name", "")
        play_pattern = event.get("play_pattern", {}).get("name", "")
        location = event.get("location")
        end_location = _end_location(event)

        if event_type == "Starting XI":
            lineup = event.get("tactics", {}).get("lineup", [])
            summary.starting_xi = frozenset(
                int(player["player"]["id"])
                for player in lineup
                if player.get("player", {}).get("id") is not None
            )
            summary.formation = int(event.get("tactics", {}).get("formation") or 0)
        elif event_type == "Shot":
            shot = event.get("shot", {})
            xg = float(shot.get("statsbomb_xg") or 0.0)
            outcome = shot.get("outcome", {}).get("name", "")
            summary.xg += xg
            summary.shots += 1
            summary.shots_on_target += int(outcome in {"Goal", "Saved", "Saved to Post"})
            summary.big_chances += int(xg >= 0.20)
            summary.counter_xg += xg if play_pattern == "From Counter" else 0.0
            summary.set_piece_xg += xg if play_pattern not in {"Regular Play", "From Counter"} else 0.0
            if outcome == "Goal":
                summary.goals += 1
                summary.goal_minutes.append(int(event.get("minute") or 0))
        elif event_type == "Own Goal For":
            summary.goals += 1
            summary.goal_minutes.append(int(event.get("minute") or 0))

        if event_type == "Pass":
            summary.passes += 1
            summary.completed_passes += int("outcome" not in event.get("pass", {}))
        if event.get("under_pressure"):
            summary.under_pressure_actions += 1
        if event_type == "Pressure":
            summary.pressures += 1
        if event_type in {"Ball Recovery", "Interception"} and location and location[0] >= 80:
            summary.high_recoveries += 1
        if event_type in {"Miscontrol", "Dispossessed"}:
            summary.turnovers += 1
        if event_type == "Foul Committed":
            summary.fouls += 1
            summary.cards += int(bool(event.get("foul_committed", {}).get("card")))
        if event_type == "Bad Behaviour":
            summary.cards += int(bool(event.get("bad_behaviour", {}).get("card")))

        if end_location and location:
            start_x = float(location[0])
            end_x = float(end_location[0])
            summary.progressive_actions += int(end_x - start_x >= 20)
            summary.final_third_entries += int(start_x < 80 <= end_x)
            summary.box_entries += int(not _is_box(location) and _is_box(end_location))

    home = summaries[home_id]
    away = summaries[away_id]
    home.goals = int(match.get("home_score", home.goals))
    away.goals = int(match.get("away_score", away.goals))
    return EventMatch(
        match_id=int(match["match_id"]),
        date=pd.Timestamp(match["match_date"]),
        competition_id=int(match["competition"]["competition_id"]),
        season_id=int(match["season"]["season_id"]),
        home_team_id=home_id,
        home_team_name=home.team_name,
        away_team_id=away_id,
        away_team_name=away.team_name,
        home_score=home.goals,
        away_score=away.goals,
        home=home,
        away=away,
    )


def _collapse_conceded(opponent_goal_minutes: list[int]) -> float:
    ordered = sorted(opponent_goal_minutes)
    return float(any(second - first <= 15 for first, second in zip(ordered, ordered[1:])))


def _summary_row(team: TeamEventSummary, opponent: TeamEventSummary, score: tuple[int, int]) -> dict[str, float]:
    goals_for, goals_against = score
    archetype = classify_score(goals_for, goals_against)
    tail_for = archetype in {ScoreArchetype.HOME_BLOWOUT, ScoreArchetype.AWAY_BLOWOUT, ScoreArchetype.SHOOTOUT}
    return {
        "goals": float(goals_for),
        "xg": team.xg,
        "shots": float(team.shots),
        "shots_on_target": float(team.shots_on_target),
        "big_chances": float(team.big_chances),
        "counter_xg": team.counter_xg,
        "set_piece_xg": team.set_piece_xg,
        "box_entries": float(team.box_entries),
        "final_third_entries": float(team.final_third_entries),
        "progressive_actions": float(team.progressive_actions),
        "pressures": float(team.pressures),
        "high_recoveries": float(team.high_recoveries),
        "turnovers": float(team.turnovers),
        "pass_completion": team.completed_passes / team.passes if team.passes else np.nan,
        "under_pressure_actions": float(team.under_pressure_actions),
        "fouls": float(team.fouls),
        "cards": float(team.cards),
        "collapse_conceded": _collapse_conceded(opponent.goal_minutes),
        "tail_for": float(tail_for),
        "tail_against": float(
            classify_score(goals_against, goals_for)
            in {ScoreArchetype.HOME_BLOWOUT, ScoreArchetype.AWAY_BLOWOUT, ScoreArchetype.SHOOTOUT}
        ),
    }


def _rolling_stats(state: RollingTeamState, prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in _EVENT_FEATURES:
        values = [row[name] for row in state.rows if np.isfinite(row[name])]
        result[f"{prefix}_{name}_mean"] = float(np.mean(values)) if values else np.nan
        result[f"{prefix}_{name}_std"] = float(np.std(values)) if values else np.nan
    return result


def build_event_dataset(matches: list[EventMatch]) -> EventDataset:
    states: dict[int, RollingTeamState] = defaultdict(RollingTeamState)
    rows: list[dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: (item.date, item.match_id)):
        home_state = states[match.home_team_id]
        away_state = states[match.away_team_id]
        home_xi = match.home.starting_xi
        away_xi = match.away.starting_xi

        def continuity(current: frozenset[int], previous: frozenset[int]) -> float:
            return len(current & previous) / len(current | previous) if current and previous else np.nan

        def experience(current: frozenset[int], state: RollingTeamState) -> float:
            return float(np.mean([state.player_starts.get(player, 0) for player in current])) if current else np.nan

        features = {
            **_rolling_stats(home_state, "home"),
            **_rolling_stats(away_state, "away"),
            "home_xi_continuity": continuity(home_xi, home_state.previous_xi),
            "away_xi_continuity": continuity(away_xi, away_state.previous_xi),
            "home_xi_experience": experience(home_xi, home_state),
            "away_xi_experience": experience(away_xi, away_state),
            "home_formation_change": float(bool(home_state.formation and home_state.formation != match.home.formation)),
            "away_formation_change": float(bool(away_state.formation and away_state.formation != match.away.formation)),
            "home_rest_days": float((match.date - home_state.last_date).days) if home_state.last_date is not None else np.nan,
            "away_rest_days": float((match.date - away_state.last_date).days) if away_state.last_date is not None else np.nan,
            "home_history": float(len(home_state.rows)),
            "away_history": float(len(away_state.rows)),
            "competition_id": float(match.competition_id),
            "season_id": float(match.season_id),
            "month_sin": float(np.sin(2 * np.pi * (match.date.month - 1) / 12)),
            "month_cos": float(np.cos(2 * np.pi * (match.date.month - 1) / 12)),
        }
        for name in _EVENT_FEATURES:
            features[f"diff_{name}_mean"] = features[f"home_{name}_mean"] - features[f"away_{name}_mean"]

        archetype = classify_score(match.home_score, match.away_score)
        rows.append(
            {
                "match_id": match.match_id,
                "date": match.date,
                "competition_id_raw": match.competition_id,
                "season_id_raw": match.season_id,
                "home_team": match.home_team_name,
                "away_team": match.away_team_name,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "tail_type": archetype.value,
                "tail_target": int(
                    archetype
                    in {ScoreArchetype.HOME_BLOWOUT, ScoreArchetype.AWAY_BLOWOUT, ScoreArchetype.SHOOTOUT}
                ),
                **features,
            }
        )

        home_state.rows.append(
            _summary_row(match.home, match.away, (match.home_score, match.away_score))
        )
        away_state.rows.append(
            _summary_row(match.away, match.home, (match.away_score, match.home_score))
        )
        home_state.previous_xi = home_xi
        away_state.previous_xi = away_xi
        for player in home_xi:
            home_state.player_starts[player] = home_state.player_starts.get(player, 0) + 1
        for player in away_xi:
            away_state.player_starts[player] = away_state.player_starts.get(player, 0) + 1
        home_state.last_date = match.date
        away_state.last_date = match.date
        home_state.formation = match.home.formation
        away_state.formation = match.away.formation

    frame = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    non_features = {
        "match_id",
        "date",
        "competition_id_raw",
        "season_id_raw",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tail_type",
        "tail_target",
    }
    feature_columns = tuple(column for column in frame.columns if column not in non_features)
    return EventDataset(frame, feature_columns)
