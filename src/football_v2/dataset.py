from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import log
from typing import Iterable

import numpy as np
import pandas as pd

from .baselines import multiplicative_devig


@dataclass(frozen=True)
class HistoricalDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]

    @property
    def features(self) -> np.ndarray:
        return self.frame.loc[:, self.feature_columns].to_numpy(dtype=float)

    @property
    def home_goals(self) -> np.ndarray:
        return self.frame["target_home_goals"].to_numpy(dtype=int)

    @property
    def away_goals(self) -> np.ndarray:
        return self.frame["target_away_goals"].to_numpy(dtype=int)


@dataclass
class TeamHistory:
    goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    points: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    btts: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    scored_three_plus: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    conceded_three_plus: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    clean_sheet: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    home_goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    home_goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    away_goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    away_goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    last_date: pd.Timestamp | None = None
    matches: int = 0

    def update(self, *, goals_for: int, goals_against: int, venue: str, date: pd.Timestamp) -> None:
        self.goals_for.append(float(goals_for))
        self.goals_against.append(float(goals_against))
        self.points.append(3.0 if goals_for > goals_against else 1.0 if goals_for == goals_against else 0.0)
        self.btts.append(float(goals_for > 0 and goals_against > 0))
        self.scored_three_plus.append(float(goals_for >= 3))
        self.conceded_three_plus.append(float(goals_against >= 3))
        self.clean_sheet.append(float(goals_against == 0))
        if venue == "home":
            self.home_goals_for.append(float(goals_for))
            self.home_goals_against.append(float(goals_against))
        elif venue == "away":
            self.away_goals_for.append(float(goals_for))
            self.away_goals_against.append(float(goals_against))
        else:
            raise ValueError(f"unsupported venue: {venue!r}")
        self.last_date = date
        self.matches += 1


@dataclass(frozen=True)
class MarketColumns:
    home: str | None
    draw: str | None
    away: str | None
    over25: str | None
    under25: str | None
    ah_line: str | None


_FEATURE_COLUMNS = (
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "market_over25_prob",
    "market_entropy",
    "ah_line",
    "elo_home",
    "elo_away",
    "elo_diff",
    "market_minus_elo_home",
    "home_gf10",
    "home_ga10",
    "home_ppg10",
    "home_btts10",
    "home_scored3_10",
    "home_conceded3_10",
    "home_clean10",
    "home_goal_variance10",
    "away_gf10",
    "away_ga10",
    "away_ppg10",
    "away_btts10",
    "away_scored3_10",
    "away_conceded3_10",
    "away_clean10",
    "away_goal_variance10",
    "home_venue_gf10",
    "home_venue_ga10",
    "away_venue_gf10",
    "away_venue_ga10",
    "home_rest_days",
    "away_rest_days",
    "home_history_matches",
    "away_history_matches",
    "league_home_goals",
    "league_away_goals",
    "league_total_goals",
    "month_sin",
    "month_cos",
)


def load_premier_league_frame(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """Load bundled real EPL results and bookmaker odds.

    The optional dependency ships the data inside its Python package, so this
    function performs no scraping and has no network dependency after install.
    """
    try:
        import premier_league_data as pl
    except ImportError as exc:  # pragma: no cover - exercised in real-data CI
        raise RuntimeError(
            "install the real-data extra: pip install -e '.[realdata]'"
        ) from exc

    results = pl.load_results()
    odds = pl.load_results_with_odds()
    identity = ["match_id", "season", "date", "home_team", "away_team"]
    odds_only = odds.drop(columns=[column for column in identity if column != "match_id"], errors="ignore")
    frame = results.merge(odds_only, on="match_id", how="left", validate="one_to_one")
    if seasons is not None:
        selected = set(seasons)
        frame = frame[frame["season"].isin(selected)]
    frame = frame.dropna(subset=["date", "home_team", "away_team", "fthg", "ftag"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame.sort_values(["date", "match_id"]).reset_index(drop=True)


def _find_three_way_columns(columns: Iterable[str]) -> tuple[str | None, str | None, str | None]:
    available = set(columns)
    preferred = (
        "market_avg",
        "pinnacle",
        "bet365",
        "market_max",
        "betbrain_avg",
    )
    for suffix in ("_close", ""):
        for prefix in preferred:
            candidate = (
                f"{prefix}_1x2_home{suffix}",
                f"{prefix}_1x2_draw{suffix}",
                f"{prefix}_1x2_away{suffix}",
            )
            if set(candidate).issubset(available):
                return candidate
    for column in sorted(available):
        if column.endswith("_1x2_home_close"):
            prefix = column.removesuffix("_1x2_home_close")
            candidate = (
                column,
                f"{prefix}_1x2_draw_close",
                f"{prefix}_1x2_away_close",
            )
            if set(candidate).issubset(available):
                return candidate
    return None, None, None


def _find_two_way_columns(columns: Iterable[str]) -> tuple[str | None, str | None]:
    available = set(columns)
    preferred = ("market_avg", "pinnacle", "bet365", "market_max", "betbrain_avg")
    for suffix in ("_close", ""):
        for prefix in preferred:
            candidate = (f"{prefix}_over25{suffix}", f"{prefix}_under25{suffix}")
            if set(candidate).issubset(available):
                return candidate
    for column in sorted(available):
        if column.endswith("_over25_close"):
            prefix = column.removesuffix("_over25_close")
            candidate = (column, f"{prefix}_under25_close")
            if set(candidate).issubset(available):
                return candidate
    return None, None


def select_market_columns(frame: pd.DataFrame) -> MarketColumns:
    home, draw, away = _find_three_way_columns(frame.columns)
    over25, under25 = _find_two_way_columns(frame.columns)
    ah_line = "ah_line" if "ah_line" in frame.columns else None
    return MarketColumns(home, draw, away, over25, under25, ah_line)


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _mean(values: deque[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _variance(values_a: deque[float], values_b: deque[float]) -> float:
    if not values_a:
        return np.nan
    totals = np.asarray(values_a, dtype=float) + np.asarray(values_b, dtype=float)
    return float(np.var(totals))


def _rest_days(history: TeamHistory, date: pd.Timestamp) -> float:
    if history.last_date is None:
        return np.nan
    return float(np.clip((date - history.last_date).days, 0, 30))


def _elo_expectation(home_elo: float, away_elo: float, home_advantage: float = 65.0) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - home_advantage) / 400.0)))


def _market_features(row: pd.Series, columns: MarketColumns) -> tuple[float, ...]:
    home_price = _safe_float(row.get(columns.home)) if columns.home else None
    draw_price = _safe_float(row.get(columns.draw)) if columns.draw else None
    away_price = _safe_float(row.get(columns.away)) if columns.away else None
    if home_price and draw_price and away_price and min(home_price, draw_price, away_price) > 1.0:
        fair = multiplicative_devig(
            {"home": home_price, "draw": draw_price, "away": away_price}
        )
        home_probability = fair["home"]
        draw_probability = fair["draw"]
        away_probability = fair["away"]
        entropy = -sum(probability * log(probability) for probability in fair.values())
    else:
        home_probability = draw_probability = away_probability = entropy = np.nan

    over_price = _safe_float(row.get(columns.over25)) if columns.over25 else None
    under_price = _safe_float(row.get(columns.under25)) if columns.under25 else None
    if over_price and under_price and min(over_price, under_price) > 1.0:
        over_probability = multiplicative_devig(
            {"over": over_price, "under": under_price}
        )["over"]
    else:
        over_probability = np.nan
    ah_line = _safe_float(row.get(columns.ah_line)) if columns.ah_line else None
    return (
        home_probability,
        draw_probability,
        away_probability,
        over_probability,
        entropy,
        ah_line if ah_line is not None else np.nan,
    )


def build_historical_dataset(
    frame: pd.DataFrame,
    *,
    max_goals: int = 7,
    elo_k: float = 20.0,
) -> HistoricalDataset:
    """Convert chronologically ordered matches to strictly pre-match features."""
    required = {"date", "home_team", "away_team", "fthg", "ftag"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if max_goals < 5:
        raise ValueError("max_goals must be at least 5")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values(["date", "match_id"] if "match_id" in data.columns else ["date"])
    market_columns = select_market_columns(data)
    histories: dict[str, TeamHistory] = {}
    elo: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    league_matches = 0
    league_home_goals = 0.0
    league_away_goals = 0.0

    for _, match in data.iterrows():
        date = pd.Timestamp(match["date"])
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        home_goals = int(match["fthg"])
        away_goals = int(match["ftag"])
        home_history = histories.setdefault(home_team, TeamHistory())
        away_history = histories.setdefault(away_team, TeamHistory())
        home_elo = elo.setdefault(home_team, 1500.0)
        away_elo = elo.setdefault(away_team, 1500.0)
        elo_home_probability = _elo_expectation(home_elo, away_elo)
        market = _market_features(match, market_columns)
        month_angle = 2.0 * np.pi * (date.month - 1) / 12.0
        league_home_average = league_home_goals / league_matches if league_matches else np.nan
        league_away_average = league_away_goals / league_matches if league_matches else np.nan

        feature_values = {
            "market_home_prob": market[0],
            "market_draw_prob": market[1],
            "market_away_prob": market[2],
            "market_over25_prob": market[3],
            "market_entropy": market[4],
            "ah_line": market[5],
            "elo_home": home_elo,
            "elo_away": away_elo,
            "elo_diff": home_elo - away_elo,
            "market_minus_elo_home": market[0] - elo_home_probability
            if np.isfinite(market[0])
            else np.nan,
            "home_gf10": _mean(home_history.goals_for),
            "home_ga10": _mean(home_history.goals_against),
            "home_ppg10": _mean(home_history.points),
            "home_btts10": _mean(home_history.btts),
            "home_scored3_10": _mean(home_history.scored_three_plus),
            "home_conceded3_10": _mean(home_history.conceded_three_plus),
            "home_clean10": _mean(home_history.clean_sheet),
            "home_goal_variance10": _variance(
                home_history.goals_for, home_history.goals_against
            ),
            "away_gf10": _mean(away_history.goals_for),
            "away_ga10": _mean(away_history.goals_against),
            "away_ppg10": _mean(away_history.points),
            "away_btts10": _mean(away_history.btts),
            "away_scored3_10": _mean(away_history.scored_three_plus),
            "away_conceded3_10": _mean(away_history.conceded_three_plus),
            "away_clean10": _mean(away_history.clean_sheet),
            "away_goal_variance10": _variance(
                away_history.goals_for, away_history.goals_against
            ),
            "home_venue_gf10": _mean(home_history.home_goals_for),
            "home_venue_ga10": _mean(home_history.home_goals_against),
            "away_venue_gf10": _mean(away_history.away_goals_for),
            "away_venue_ga10": _mean(away_history.away_goals_against),
            "home_rest_days": _rest_days(home_history, date),
            "away_rest_days": _rest_days(away_history, date),
            "home_history_matches": float(home_history.matches),
            "away_history_matches": float(away_history.matches),
            "league_home_goals": league_home_average,
            "league_away_goals": league_away_average,
            "league_total_goals": league_home_average + league_away_average
            if league_matches
            else np.nan,
            "month_sin": float(np.sin(month_angle)),
            "month_cos": float(np.cos(month_angle)),
        }
        rows.append(
            {
                "match_id": match.get("match_id", f"{date.date()}-{home_team}-{away_team}"),
                "season": match.get("season", "unknown"),
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                **feature_values,
                "target_home_goals": min(home_goals, max_goals),
                "target_away_goals": min(away_goals, max_goals),
                "actual_home_goals": home_goals,
                "actual_away_goals": away_goals,
            }
        )

        actual_home_score = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        delta = elo_k * (actual_home_score - elo_home_probability)
        elo[home_team] = home_elo + delta
        elo[away_team] = away_elo - delta
        home_history.update(
            goals_for=home_goals,
            goals_against=away_goals,
            venue="home",
            date=date,
        )
        away_history.update(
            goals_for=away_goals,
            goals_against=home_goals,
            venue="away",
            date=date,
        )
        league_matches += 1
        league_home_goals += home_goals
        league_away_goals += away_goals

    output = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    return HistoricalDataset(output, _FEATURE_COLUMNS)


def time_split(
    dataset: HistoricalDataset,
    *,
    test_fraction: float = 0.20,
    minimum_train_matches: int = 1000,
) -> tuple[HistoricalDataset, HistoricalDataset]:
    if not 0.05 <= test_fraction <= 0.50:
        raise ValueError("test_fraction must be between 0.05 and 0.50")
    split = int(len(dataset.frame) * (1.0 - test_fraction))
    if split < minimum_train_matches or split >= len(dataset.frame):
        raise ValueError("dataset is too small for the requested time split")
    train = dataset.frame.iloc[:split].reset_index(drop=True)
    test = dataset.frame.iloc[split:].reset_index(drop=True)
    return (
        HistoricalDataset(train, dataset.feature_columns),
        HistoricalDataset(test, dataset.feature_columns),
    )
