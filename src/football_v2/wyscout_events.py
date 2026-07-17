from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import jackpot_tail_type, is_jackpot_tail
from .statsbomb_events import EventDataset, EventMatch, TeamEventSummary, build_event_dataset


def _numeric_id(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return abs(hash(str(value))) % 2_000_000_000


def _formation_number(value: object) -> int:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return int(digits[:5]) if digits else 0


def _shot_quality(x: float, y: float, body_part: str) -> float:
    if not np.isfinite(x) or not np.isfinite(y):
        return 0.06
    distance = float(np.hypot(1.0 - x, 0.70 * (y - 0.50)))
    logit = -1.55 - 7.2 * distance
    if "HEAD" in body_part.upper():
        logit -= 0.35
    return float(1.0 / (1.0 + np.exp(-logit)))


def _as_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).upper()


def _time_minutes(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(pd.Timedelta(value).total_seconds() // 60)
    except (TypeError, ValueError):
        return 0


def parse_wyscout_file(path: str | Path) -> EventMatch:
    try:
        from kloppy import wyscout
    except ImportError as exc:  # pragma: no cover - integration dependency
        raise RuntimeError("install kloppy to parse Wyscout event files") from exc

    dataset = wyscout.load(event_data=str(path), coordinates="wyscout")
    metadata = dataset.metadata
    if metadata.score is None or metadata.date is None or len(metadata.teams) != 2:
        raise ValueError(f"missing Wyscout metadata in {path}")
    home_team, away_team = metadata.teams
    home_id = _numeric_id(home_team.team_id)
    away_id = _numeric_id(away_team.team_id)
    summaries = {
        home_id: TeamEventSummary(home_id, str(home_team.name)),
        away_id: TeamEventSummary(away_id, str(away_team.name)),
    }
    summaries[home_id].starting_xi = frozenset(
        _numeric_id(player.player_id) for player in home_team.players if player.starting
    )
    summaries[away_id].starting_xi = frozenset(
        _numeric_id(player.player_id) for player in away_team.players if player.starting
    )
    summaries[home_id].formation = _formation_number(home_team.starting_formation)
    summaries[away_id].formation = _formation_number(away_team.starting_formation)

    frame = dataset.to_df()
    previous_team: int | None = None
    previous_second = -10_000.0
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        team_value = row_dict.get("team_id")
        if team_value is None:
            continue
        team_id = _numeric_id(team_value)
        if team_id not in summaries:
            continue
        summary = summaries[team_id]
        event_type = _as_text(row_dict.get("event_type"))
        result = _as_text(row_dict.get("result"))
        success = (
            bool(row_dict.get("success"))
            if row_dict.get("success") is not None
            else False
        )
        x = (
            float(row_dict.get("coordinates_x"))
            if pd.notna(row_dict.get("coordinates_x"))
            else np.nan
        )
        y = (
            float(row_dict.get("coordinates_y"))
            if pd.notna(row_dict.get("coordinates_y"))
            else np.nan
        )
        end_x = (
            float(row_dict.get("end_coordinates_x"))
            if pd.notna(row_dict.get("end_coordinates_x"))
            else np.nan
        )
        end_y = (
            float(row_dict.get("end_coordinates_y"))
            if pd.notna(row_dict.get("end_coordinates_y"))
            else np.nan
        )
        timestamp = row_dict.get("timestamp")
        second = (
            float(pd.Timedelta(timestamp).total_seconds())
            if timestamp is not None
            else 0.0
        )
        body_part = _as_text(row_dict.get("body_part_type"))
        set_piece = _as_text(row_dict.get("set_piece_type"))
        counter = bool(row_dict.get("is_counter_attack"))
        possession_flip = previous_team is not None and previous_team != team_id
        quick_transition = possession_flip and second - previous_second <= 12.0

        if event_type == "SHOT":
            xg = _shot_quality(x, y, body_part)
            summary.xg += xg
            summary.shots += 1
            summary.shots_on_target += int(
                result in {"GOAL", "SAVED", "SAVED_TO_POST"}
            )
            summary.big_chances += int(xg >= 0.18)
            summary.counter_xg += xg if counter or quick_transition else 0.0
            summary.set_piece_xg += xg if set_piece else 0.0
            if result == "GOAL":
                summary.goal_minutes.append(_time_minutes(timestamp))
        if event_type == "PASS":
            summary.passes += 1
            summary.completed_passes += int(
                success or result in {"COMPLETE", "SUCCESS"}
            )
        if event_type in {"DUEL", "INTERCEPTION", "RECOVERY", "TACKLE"}:
            summary.pressures += 1
        if (
            event_type in {"INTERCEPTION", "RECOVERY", "TACKLE"}
            and np.isfinite(x)
            and x >= 0.67
        ):
            summary.high_recoveries += 1
        if (event_type == "PASS" and not success) or event_type in {
            "MISCONTROL",
            "DISPOSSESSED",
        }:
            summary.turnovers += 1
        if event_type in {"FOUL", "FOUL_COMMITTED"}:
            summary.fouls += 1
        if _as_text(row_dict.get("card_type")):
            summary.cards += 1
        if np.isfinite(x) and np.isfinite(end_x):
            summary.progressive_actions += int(end_x - x >= 0.20)
            summary.final_third_entries += int(x < 0.67 <= end_x)
            start_box = (
                x >= 0.85 and 0.20 <= y <= 0.80 if np.isfinite(y) else False
            )
            end_box = (
                end_x >= 0.85 and 0.20 <= end_y <= 0.80
                if np.isfinite(end_y)
                else False
            )
            summary.box_entries += int(not start_box and end_box)

        previous_team = team_id
        previous_second = second

    home_score = int(metadata.score.home)
    away_score = int(metadata.score.away)
    summaries[home_id].goals = home_score
    summaries[away_id].goals = away_score
    game_id = _numeric_id(metadata.game_id or Path(path).stem)
    attributes = metadata.attributes or {}
    competition = attributes.get("competition_id") or attributes.get("competition") or 0
    season = attributes.get("season_id") or attributes.get("season") or 0
    return EventMatch(
        match_id=game_id,
        date=pd.Timestamp(metadata.date).tz_localize(None),
        competition_id=_numeric_id(competition),
        season_id=_numeric_id(season),
        home_team_id=home_id,
        home_team_name=str(home_team.name),
        away_team_id=away_id,
        away_team_name=str(away_team.name),
        home_score=home_score,
        away_score=away_score,
        home=summaries[home_id],
        away=summaries[away_id],
    )


def _league_file_ids(index_path: Path) -> list[str]:
    allowed = {
        "matches_England.json",
        "matches_France.json",
        "matches_Germany.json",
        "matches_Italy.json",
        "matches_Spain.json",
    }
    ids: list[str] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not any(source in line for source in allowed):
            continue
        marker = line.split("]", maxsplit=1)[0]
        match_id = "".join(character for character in marker if character.isdigit())
        if match_id:
            ids.append(match_id)
    return ids


def load_wyscout_league_matches(
    repository_root: str | Path,
    *,
    workers: int = 8,
    max_matches: int | None = None,
) -> list[EventMatch]:
    root = Path(repository_root)
    processed = root / "processed"
    index = processed / "README.md"
    file_ids = _league_file_ids(index)
    if max_matches is not None:
        file_ids = file_ids[:max_matches]
    paths = [processed / "files" / f"{match_id}.json" for match_id in file_ids]
    matches: list[EventMatch] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parse_wyscout_file, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                matches.append(future.result())
            except Exception as exc:  # pragma: no cover - integration diagnostics
                failures.append(f"{path.name}: {exc}")
    if len(matches) < 1000:
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
