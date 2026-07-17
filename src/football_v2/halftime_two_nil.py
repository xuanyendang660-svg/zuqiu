from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


_DEFAULT_DIVISIONS = ("E0", "D1", "F1", "I1", "SP1")
_BASE_URL = "https://www.football-data.co.uk/mmz4281"


@dataclass(frozen=True)
class ConditionalMetrics:
    matches: int
    alerts: int
    third_goal_successes: int
    blowout_successes: int
    third_goal_precision: float | None
    blowout_precision: float | None
    blowout_base_rate: float
    blowout_lift: float | None
    blowout_wilson_95_low: float | None
    blowout_wilson_95_high: float | None


@dataclass(frozen=True)
class HalftimeTwoNilReport:
    rows: int
    alerts: int
    overall: ConditionalMetrics
    filters: dict[str, ConditionalMetrics]
    by_league: dict[str, ConditionalMetrics]
    by_era: dict[str, ConditionalMetrics]
    alert_score_distribution: dict[str, int]
    blowout_score_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "alerts": self.alerts,
            "overall": asdict(self.overall),
            "filters": {
                name: asdict(metrics) for name, metrics in self.filters.items()
            },
            "by_league": {
                name: asdict(metrics) for name, metrics in self.by_league.items()
            },
            "by_era": {
                name: asdict(metrics) for name, metrics in self.by_era.items()
            },
            "alert_score_distribution": self.alert_score_distribution,
            "blowout_score_distribution": self.blowout_score_distribution,
        }


def season_codes(start_year: int = 2000, end_year: int = 2025) -> tuple[str, ...]:
    if end_year < start_year:
        raise ValueError("end_year must be at least start_year")
    return tuple(
        f"{year % 100:02d}{(year + 1) % 100:02d}"
        for year in range(start_year, end_year + 1)
    )


def _download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "football-v2-research"})
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with urlopen(request, timeout=90) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _first_triplet(columns: Iterable[str]) -> tuple[str, str, str] | None:
    available = set(columns)
    candidates = (
        ("AvgCH", "AvgCD", "AvgCA"),
        ("PSCH", "PSCD", "PSCA"),
        ("AvgH", "AvgD", "AvgA"),
        ("BbAvH", "BbAvD", "BbAvA"),
        ("B365H", "B365D", "B365A"),
        ("PSH", "PSD", "PSA"),
    )
    return next((triplet for triplet in candidates if set(triplet) <= available), None)


def _safe_inverse(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.divide(
        1.0,
        values,
        out=np.full_like(values, np.nan, dtype=float),
        where=valid,
    )


def _load_file(path: Path, season: str, division: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    required = {
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "HTHG",
        "HTAG",
    }
    if not required <= set(frame.columns):
        return pd.DataFrame()
    triplet = _first_triplet(frame.columns)
    if triplet is None:
        return pd.DataFrame()

    home_odds = pd.to_numeric(frame[triplet[0]], errors="coerce").to_numpy(dtype=float)
    draw_odds = pd.to_numeric(frame[triplet[1]], errors="coerce").to_numpy(dtype=float)
    away_odds = pd.to_numeric(frame[triplet[2]], errors="coerce").to_numpy(dtype=float)
    valid = np.logical_and.reduce(
        [
            np.isfinite(home_odds),
            np.isfinite(draw_odds),
            np.isfinite(away_odds),
            home_odds > 1.0,
            draw_odds > 1.0,
            away_odds > 1.0,
        ]
    )
    inverse = np.column_stack(
        [
            _safe_inverse(home_odds, valid),
            _safe_inverse(draw_odds, valid),
            _safe_inverse(away_odds, valid),
        ]
    )
    total = np.nansum(inverse, axis=1)
    probabilities = np.divide(
        inverse,
        total[:, None],
        out=np.full_like(inverse, np.nan),
        where=valid[:, None] & (total[:, None] > 0),
    )
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce"),
            "season": season,
            "division": division,
            "home_team": frame["HomeTeam"].astype(str),
            "away_team": frame["AwayTeam"].astype(str),
            "home_score": pd.to_numeric(frame["FTHG"], errors="coerce"),
            "away_score": pd.to_numeric(frame["FTAG"], errors="coerce"),
            "halftime_home": pd.to_numeric(frame["HTHG"], errors="coerce"),
            "halftime_away": pd.to_numeric(frame["HTAG"], errors="coerce"),
            "market_home_prob": probabilities[:, 0],
            "market_draw_prob": probabilities[:, 1],
            "market_away_prob": probabilities[:, 2],
        }
    ).dropna()
    for column in (
        "home_score",
        "away_score",
        "halftime_home",
        "halftime_away",
    ):
        output[column] = output[column].astype(int)
    return output.reset_index(drop=True)


def load_halftime_market_data(
    cache_dir: str | Path,
    *,
    seasons: Iterable[str] | None = None,
    divisions: Iterable[str] = _DEFAULT_DIVISIONS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    selected_seasons = tuple(seasons or season_codes())
    selected_divisions = tuple(divisions)
    frames: list[pd.DataFrame] = []
    loaded: list[str] = []
    skipped: list[str] = []

    for season in selected_seasons:
        for division in selected_divisions:
            key = f"{season}-{division}"
            path = cache / f"{key}.csv"
            if not path.exists():
                try:
                    _download(f"{_BASE_URL}/{season}/{division}.csv", path)
                except (HTTPError, URLError, TimeoutError):
                    skipped.append(key)
                    continue
            try:
                frame = _load_file(path, season, division)
            except (OSError, UnicodeError, pd.errors.ParserError):
                path.unlink(missing_ok=True)
                skipped.append(key)
                continue
            if frame.empty:
                path.unlink(missing_ok=True)
                skipped.append(key)
                continue
            loaded.append(key)
            frames.append(frame)

    if not frames:
        raise RuntimeError("no football-data halftime files were loaded")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data.sort_values(["date", "division", "home_team"]).reset_index(drop=True)
    return data, {
        "files_loaded": len(loaded),
        "files_skipped": len(skipped),
        "loaded": loaded,
        "skipped": skipped,
    }


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    spread = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _metrics(frame: pd.DataFrame) -> ConditionalMetrics:
    matches = len(frame)
    alerts = int(frame["alert"].sum()) if matches else 0
    alert_frame = frame.loc[frame["alert"]]
    third_goal_successes = int(alert_frame["third_goal_target"].sum())
    blowout_successes = int(alert_frame["blowout_target"].sum())
    base_rate = float(frame["blowout_target"].mean()) if matches else 0.0
    third_precision = third_goal_successes / alerts if alerts else None
    blowout_precision = blowout_successes / alerts if alerts else None
    lift = (
        blowout_precision / base_rate
        if blowout_precision is not None and base_rate > 0
        else None
    )
    low, high = _wilson(blowout_successes, alerts)
    return ConditionalMetrics(
        matches=matches,
        alerts=alerts,
        third_goal_successes=third_goal_successes,
        blowout_successes=blowout_successes,
        third_goal_precision=third_precision,
        blowout_precision=blowout_precision,
        blowout_base_rate=base_rate,
        blowout_lift=lift,
        blowout_wilson_95_low=low,
        blowout_wilson_95_high=high,
    )


def build_oriented_halftime_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    home_underdog = data["market_home_prob"] < data["market_away_prob"]
    data["underdog_prob"] = np.where(
        home_underdog,
        data["market_home_prob"],
        data["market_away_prob"],
    )
    data["favourite_prob"] = np.where(
        home_underdog,
        data["market_away_prob"],
        data["market_home_prob"],
    )
    data["market_gap"] = data["favourite_prob"] - data["underdog_prob"]
    data["halftime_underdog"] = np.where(
        home_underdog,
        data["halftime_home"],
        data["halftime_away"],
    ).astype(int)
    data["halftime_favourite"] = np.where(
        home_underdog,
        data["halftime_away"],
        data["halftime_home"],
    ).astype(int)
    data["final_underdog"] = np.where(
        home_underdog,
        data["home_score"],
        data["away_score"],
    ).astype(int)
    data["final_favourite"] = np.where(
        home_underdog,
        data["away_score"],
        data["home_score"],
    ).astype(int)
    data["alert"] = (data["halftime_underdog"] == 2) & (
        data["halftime_favourite"] == 0
    )
    data["third_goal_target"] = data["final_underdog"] >= 3
    data["blowout_target"] = (data["final_underdog"] >= 3) & (
        data["final_underdog"] - data["final_favourite"] >= 2
    )
    data["final_oriented_score"] = (
        data["final_underdog"].astype(str)
        + "-"
        + data["final_favourite"].astype(str)
    )
    start_year = data["date"].dt.year
    data["era"] = pd.cut(
        start_year,
        bins=[1999, 2008, 2017, 2026],
        labels=["2000-2008", "2009-2017", "2018-2026"],
        include_lowest=True,
    ).astype(str)
    return data


def evaluate_halftime_two_nil(frame: pd.DataFrame) -> HalftimeTwoNilReport:
    data = build_oriented_halftime_frame(frame)
    filters = {
        "all": pd.Series(True, index=data.index),
        "underdog_prob_le_035": data["underdog_prob"] <= 0.35,
        "underdog_prob_le_030": data["underdog_prob"] <= 0.30,
        "market_gap_ge_010": data["market_gap"] >= 0.10,
        "market_gap_ge_020": data["market_gap"] >= 0.20,
    }
    alert_frame = data.loc[data["alert"]]
    blowout_frame = alert_frame.loc[alert_frame["blowout_target"]]
    return HalftimeTwoNilReport(
        rows=len(data),
        alerts=len(alert_frame),
        overall=_metrics(data),
        filters={
            name: _metrics(data.loc[mask].reset_index(drop=True))
            for name, mask in filters.items()
        },
        by_league={
            division: _metrics(group.reset_index(drop=True))
            for division, group in data.groupby("division")
        },
        by_era={
            era: _metrics(group.reset_index(drop=True))
            for era, group in data.groupby("era")
        },
        alert_score_distribution=dict(
            Counter(alert_frame["final_oriented_score"]).most_common()
        ),
        blowout_score_distribution=dict(
            Counter(blowout_frame["final_oriented_score"]).most_common()
        ),
    )
