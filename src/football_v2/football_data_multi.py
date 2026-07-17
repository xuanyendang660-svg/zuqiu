from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .baselines import multiplicative_devig
from .statsbomb_events import EventDataset
from .wyscout_events import WyscoutIndexRecord


_SOURCE_DIVISION = {
    "matches_England.json": "E0",
    "matches_France.json": "F1",
    "matches_Germany.json": "D1",
    "matches_Italy.json": "I1",
    "matches_Spain.json": "SP1",
}
_DIVISION_ID = {division: index + 1 for index, division in enumerate(sorted(set(_SOURCE_DIVISION.values())))}
_BASE_URL = "https://www.football-data.co.uk/mmz4281/1718"


def _download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "football-v2-research"})
    with urlopen(request, timeout=90) as response:
        path.write_bytes(response.read())


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def load_football_data_1718(cache_dir: str | Path) -> pd.DataFrame:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for division in sorted(_DIVISION_ID):
        path = cache / f"1718-{division}.csv"
        if not path.exists():
            _download(f"{_BASE_URL}/{division}.csv", path)
        frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        frame["division"] = division
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    home_col = _first_existing(data, ("AvgH", "BbAvH", "B365H", "PSH"))
    draw_col = _first_existing(data, ("AvgD", "BbAvD", "B365D", "PSD"))
    away_col = _first_existing(data, ("AvgA", "BbAvA", "B365A", "PSA"))
    over_col = _first_existing(data, ("Avg>2.5", "BbAv>2.5", "B365>2.5", "P>2.5"))
    under_col = _first_existing(data, ("Avg<2.5", "BbAv<2.5", "B365<2.5", "P<2.5"))
    ah_col = _first_existing(data, ("AHh", "BbAHh"))
    if not all((home_col, draw_col, away_col)):
        raise RuntimeError("football-data files do not contain a complete 1X2 market")

    rows: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        values = row._asdict()
        try:
            odds = {
                "home": float(values[home_col]),
                "draw": float(values[draw_col]),
                "away": float(values[away_col]),
            }
            fair = multiplicative_devig(odds)
        except (TypeError, ValueError, KeyError):
            fair = {"home": np.nan, "draw": np.nan, "away": np.nan}
        try:
            over = float(values[over_col]) if over_col else np.nan
            under = float(values[under_col]) if under_col else np.nan
            over_probability = (
                multiplicative_devig({"over": over, "under": under})["over"]
                if min(over, under) > 1.0
                else np.nan
            )
        except (TypeError, ValueError, KeyError):
            over_probability = np.nan
        probabilities = np.array(
            [fair["home"], fair["draw"], fair["away"]], dtype=float
        )
        finite = probabilities[np.isfinite(probabilities) & (probabilities > 0)]
        entropy = float(-np.sum(finite * np.log(finite))) if len(finite) else np.nan
        favorite_side = (
            1.0
            if fair["home"] > fair["away"]
            else -1.0
            if fair["away"] > fair["home"]
            else 0.0
        )
        rows.append(
            {
                "division": values["division"],
                "division_id": float(_DIVISION_ID[str(values["division"])]),
                "date": values["date"],
                "home_team_market": str(values["HomeTeam"]),
                "away_team_market": str(values["AwayTeam"]),
                "home_score": int(values["FTHG"]),
                "away_score": int(values["FTAG"]),
                "market_home_prob": fair["home"],
                "market_draw_prob": fair["draw"],
                "market_away_prob": fair["away"],
                "market_over25_prob": over_probability,
                "market_entropy": entropy,
                "market_favorite_prob": float(np.nanmax(probabilities)),
                "market_underdog_prob": float(np.nanmin(probabilities[[0, 2]])),
                "market_favorite_side": favorite_side,
                "ah_line": float(values[ah_col]) if ah_col and pd.notna(values[ah_col]) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "division"]).reset_index(drop=True)


def _normal_name(value: str) -> str:
    aliases = {
        "manchester united": "man united",
        "manchester city": "man city",
        "tottenham hotspur": "tottenham",
        "west bromwich albion": "west brom",
        "paris saint germain": "paris sg",
        "internazionale": "inter",
        "athletic club": "ath bilbao",
        "atletico madrid": "ath madrid",
        "real betis balompie": "betis",
        "borussia monchengladbach": "monchengladbach",
        "bayern munchen": "bayern munich",
        "koln": "cologne",
        "hellas verona": "verona",
        "spal 2013": "spal",
    }
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return aliases.get(text, text)


def _team_similarity(first: str, second: str) -> float:
    a = _normal_name(first)
    b = _normal_name(second)
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def join_wyscout_market(
    event_dataset: EventDataset,
    index_records: list[WyscoutIndexRecord],
    market: pd.DataFrame,
) -> EventDataset:
    index_by_id = {record.match_id: record for record in index_records}
    market_groups: dict[tuple[str, pd.Timestamp, int, int], pd.DataFrame] = {}
    for key, group in market.groupby(["division", "date", "home_score", "away_score"]):
        market_groups[key] = group

    joined_rows: list[dict[str, object]] = []
    failures: list[int] = []
    for row in event_dataset.frame.to_dict(orient="records"):
        match_id = int(row["match_id"])
        index = index_by_id.get(match_id)
        if index is None:
            failures.append(match_id)
            continue
        division = _SOURCE_DIVISION[index.source]
        candidates: list[pd.DataFrame] = []
        for delta in (-1, 0, 1):
            key = (
                division,
                pd.Timestamp(index.date) + pd.Timedelta(days=delta),
                int(index.home_score),
                int(index.away_score),
            )
            if key in market_groups:
                candidates.append(market_groups[key])
        if not candidates:
            failures.append(match_id)
            continue
        candidate_frame = pd.concat(candidates, ignore_index=True)
        scores = [
            _team_similarity(index.home_name, str(candidate["home_team_market"]))
            + _team_similarity(index.away_name, str(candidate["away_team_market"]))
            for candidate in candidate_frame.to_dict(orient="records")
        ]
        best_index = int(np.argmax(scores))
        if scores[best_index] < 1.15:
            failures.append(match_id)
            continue
        selected = candidate_frame.iloc[best_index]
        market_features = {
            column: selected[column]
            for column in (
                "division_id",
                "market_home_prob",
                "market_draw_prob",
                "market_away_prob",
                "market_over25_prob",
                "market_entropy",
                "market_favorite_prob",
                "market_underdog_prob",
                "market_favorite_side",
                "ah_line",
            )
        }
        home_win = int(row["home_score"]) > int(row["away_score"])
        away_win = int(row["away_score"]) > int(row["home_score"])
        underdog_win = (
            home_win
            and float(selected["market_home_prob"]) < float(selected["market_away_prob"])
        ) or (
            away_win
            and float(selected["market_away_prob"]) < float(selected["market_home_prob"])
        )
        joined_rows.append(
            {
                **row,
                **market_features,
                "upset_jackpot_target": int(bool(row["tail_target"]) and underdog_win),
            }
        )
    frame = pd.DataFrame(joined_rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(frame) < 1700:
        raise RuntimeError(
            f"joined only {len(frame)} of {len(event_dataset.frame)} Wyscout matches; "
            f"failures={len(failures)}"
        )
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
        "upset_jackpot_target",
    }
    feature_columns = tuple(column for column in frame.columns if column not in non_features)
    return EventDataset(frame, feature_columns)
