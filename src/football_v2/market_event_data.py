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


SOURCE_DIVISION = {
    "matches_England.json": "E0",
    "matches_France.json": "F1",
    "matches_Germany.json": "D1",
    "matches_Italy.json": "I1",
    "matches_Spain.json": "SP1",
}
DIVISION_ID = {code: index + 1 for index, code in enumerate(sorted(set(SOURCE_DIVISION.values())))}
BASE_URL = "https://www.football-data.co.uk/mmz4281/1718"


def _download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "football-v2-research"})
    with urlopen(request, timeout=90) as response:
        path.write_bytes(response.read())


def _first(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def load_market_1718(cache_dir: str | Path) -> pd.DataFrame:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for division in sorted(DIVISION_ID):
        path = cache / f"1718-{division}.csv"
        if not path.exists():
            _download(f"{BASE_URL}/{division}.csv", path)
        frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        frame["division"] = division
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    home_col = _first(data, ("AvgH", "BbAvH", "B365H", "PSH"))
    draw_col = _first(data, ("AvgD", "BbAvD", "B365D", "PSD"))
    away_col = _first(data, ("AvgA", "BbAvA", "B365A", "PSA"))
    over_col = _first(data, ("Avg>2.5", "BbAv>2.5", "B365>2.5", "P>2.5"))
    under_col = _first(data, ("Avg<2.5", "BbAv<2.5", "B365<2.5", "P<2.5"))
    ah_col = _first(data, ("AHh", "BbAHh"))
    if not all((home_col, draw_col, away_col)):
        raise RuntimeError("no complete 1X2 market columns")

    output: list[dict[str, object]] = []
    for _, row in data.iterrows():
        try:
            fair = multiplicative_devig(
                {
                    "home": float(row[home_col]),
                    "draw": float(row[draw_col]),
                    "away": float(row[away_col]),
                }
            )
        except (TypeError, ValueError):
            fair = {"home": np.nan, "draw": np.nan, "away": np.nan}
        try:
            over = float(row[over_col]) if over_col else np.nan
            under = float(row[under_col]) if under_col else np.nan
            over_probability = multiplicative_devig({"over": over, "under": under})["over"]
        except (TypeError, ValueError):
            over_probability = np.nan
        probabilities = np.array([fair["home"], fair["draw"], fair["away"]], dtype=float)
        finite = probabilities[np.isfinite(probabilities) & (probabilities > 0)]
        favorite_side = 1.0 if fair["home"] > fair["away"] else -1.0 if fair["away"] > fair["home"] else 0.0
        output.append(
            {
                "division": str(row["division"]),
                "division_id": float(DIVISION_ID[str(row["division"])]),
                "date": pd.Timestamp(row["date"]),
                "home_team_market": str(row["HomeTeam"]),
                "away_team_market": str(row["AwayTeam"]),
                "home_score": int(row["FTHG"]),
                "away_score": int(row["FTAG"]),
                "market_home_prob": fair["home"],
                "market_draw_prob": fair["draw"],
                "market_away_prob": fair["away"],
                "market_over25_prob": over_probability,
                "market_entropy": float(-np.sum(finite * np.log(finite))) if len(finite) else np.nan,
                "market_favorite_prob": float(np.nanmax(probabilities)),
                "market_underdog_prob": float(np.nanmin(probabilities[[0, 2]])),
                "market_favorite_side": favorite_side,
                "ah_line": float(row[ah_col]) if ah_col and pd.notna(row[ah_col]) else np.nan,
            }
        )
    return pd.DataFrame(output).sort_values(["date", "division"]).reset_index(drop=True)


def _normal(value: str) -> str:
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


def _similarity(first: str, second: str) -> float:
    a = _normal(first)
    b = _normal(second)
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def join_market_events(
    event_dataset: EventDataset,
    index_records: list[WyscoutIndexRecord],
    market: pd.DataFrame,
) -> EventDataset:
    index_by_id = {record.match_id: record for record in index_records}
    grouped = {
        key: group
        for key, group in market.groupby(["division", "date", "home_score", "away_score"])
    }
    rows: list[dict[str, object]] = []
    for event_row in event_dataset.frame.to_dict(orient="records"):
        index = index_by_id.get(int(event_row["match_id"]))
        if index is None:
            continue
        division = SOURCE_DIVISION[index.source]
        candidates: list[pd.DataFrame] = []
        for delta in (-1, 0, 1):
            key = (
                division,
                pd.Timestamp(index.date) + pd.Timedelta(days=delta),
                int(index.home_score),
                int(index.away_score),
            )
            if key in grouped:
                candidates.append(grouped[key])
        if not candidates:
            continue
        candidate_frame = pd.concat(candidates, ignore_index=True)
        scores = [
            _similarity(index.home_name, str(candidate["home_team_market"]))
            + _similarity(index.away_name, str(candidate["away_team_market"]))
            for candidate in candidate_frame.to_dict(orient="records")
        ]
        best = int(np.argmax(scores))
        if scores[best] < 1.15:
            continue
        selected = candidate_frame.iloc[best]
        market_features = {
            name: selected[name]
            for name in (
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
        home_win = int(event_row["home_score"]) > int(event_row["away_score"])
        away_win = int(event_row["away_score"]) > int(event_row["home_score"])
        underdog_win = (
            home_win and float(selected["market_home_prob"]) < float(selected["market_away_prob"])
        ) or (
            away_win and float(selected["market_away_prob"]) < float(selected["market_home_prob"])
        )
        rows.append(
            {
                **event_row,
                **market_features,
                "upset_jackpot_target": int(bool(event_row["tail_target"]) and underdog_win),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(frame) < 1700:
        raise RuntimeError(f"market join produced only {len(frame)} matches")
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
    return EventDataset(frame, tuple(column for column in frame.columns if column not in non_features))
