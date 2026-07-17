from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable

import pandas as pd

from .statsbomb_events import EventMatch


@dataclass(frozen=True)
class LiveStateMetrics:
    matches: int
    alerts: int
    successes: int
    base_rate: float
    precision: float | None
    recall: float
    lift: float | None
    precision_wilson_95_low: float | None
    precision_wilson_95_high: float | None


@dataclass(frozen=True)
class EloLiveStateReport:
    cutoff: int
    rows: int
    filters: dict[str, LiveStateMetrics]
    by_time: dict[str, LiveStateMetrics]
    by_competition: dict[str, LiveStateMetrics]
    alert_score_distribution: dict[str, int]
    success_score_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoff": self.cutoff,
            "rows": self.rows,
            "filters": {
                name: asdict(metrics) for name, metrics in self.filters.items()
            },
            "by_time": {
                name: asdict(metrics) for name, metrics in self.by_time.items()
            },
            "by_competition": {
                name: asdict(metrics)
                for name, metrics in self.by_competition.items()
            },
            "alert_score_distribution": self.alert_score_distribution,
            "success_score_distribution": self.success_score_distribution,
        }


def _expected_home(home_rating: float, away_rating: float, home_advantage: float) -> float:
    return float(
        1.0
        / (
            1.0
            + 10.0
            ** ((away_rating - (home_rating + home_advantage)) / 400.0)
        )
    )


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


def _metrics(frame: pd.DataFrame) -> LiveStateMetrics:
    matches = len(frame)
    alerts = int(frame["alert"].sum()) if matches else 0
    successes = int(frame.loc[frame["alert"], "target"].sum()) if alerts else 0
    positives = int(frame["target"].sum()) if matches else 0
    base_rate = positives / matches if matches else 0.0
    precision = successes / alerts if alerts else None
    recall = successes / positives if positives else 0.0
    lift = precision / base_rate if precision is not None and base_rate > 0 else None
    low, high = _wilson(successes, alerts)
    return LiveStateMetrics(
        matches=matches,
        alerts=alerts,
        successes=successes,
        base_rate=base_rate,
        precision=precision,
        recall=recall,
        lift=lift,
        precision_wilson_95_low=low,
        precision_wilson_95_high=high,
    )


def _goals_by_cutoff(minutes: list[int], cutoff: int) -> int:
    return sum(int(minute) <= cutoff for minute in minutes)


def build_elo_live_state_frame(
    matches: Iterable[EventMatch],
    *,
    cutoff: int = 30,
    initial_rating: float = 1500.0,
    home_advantage: float = 65.0,
    k_factor: float = 20.0,
) -> pd.DataFrame:
    ratings: dict[int, float] = defaultdict(lambda: initial_rating)
    history: dict[int, int] = defaultdict(int)
    rows: list[dict[str, object]] = []

    ordered = sorted(matches, key=lambda match: (match.date, match.match_id))
    for match in ordered:
        home_rating = float(ratings[match.home_team_id])
        away_rating = float(ratings[match.away_team_id])
        expected_home = _expected_home(home_rating, away_rating, home_advantage)
        home_underdog = expected_home < 0.5
        underdog_rating = home_rating if home_underdog else away_rating
        favourite_effective_rating = (
            away_rating
            if home_underdog
            else home_rating + home_advantage
        )
        underdog_effective_rating = (
            home_rating + home_advantage
            if home_underdog
            else away_rating
        )
        elo_gap = favourite_effective_rating - underdog_effective_rating

        live_home = _goals_by_cutoff(match.home.goal_minutes, cutoff)
        live_away = _goals_by_cutoff(match.away.goal_minutes, cutoff)
        live_underdog = live_home if home_underdog else live_away
        live_favourite = live_away if home_underdog else live_home
        final_underdog = match.home_score if home_underdog else match.away_score
        final_favourite = match.away_score if home_underdog else match.home_score
        alert = live_underdog == 2 and live_favourite == 0
        target = final_underdog >= 3 and final_underdog - final_favourite >= 2
        rows.append(
            {
                "match_id": match.match_id,
                "date": pd.Timestamp(match.date),
                "competition_id": int(match.competition_id),
                "season_id": int(match.season_id),
                "home_underdog": bool(home_underdog),
                "elo_gap": float(max(0.0, elo_gap)),
                "underdog_history": int(
                    history[
                        match.home_team_id if home_underdog else match.away_team_id
                    ]
                ),
                "favourite_history": int(
                    history[
                        match.away_team_id if home_underdog else match.home_team_id
                    ]
                ),
                "minimum_history": int(
                    min(
                        history[match.home_team_id],
                        history[match.away_team_id],
                    )
                ),
                "alert": bool(alert),
                "target": bool(target),
                "final_oriented_score": f"{final_underdog}-{final_favourite}",
                "underdog_rating": underdog_rating,
                "expected_home": expected_home,
            }
        )

        if match.home_score > match.away_score:
            actual_home = 1.0
        elif match.home_score < match.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5
        update = k_factor * (actual_home - expected_home)
        ratings[match.home_team_id] = home_rating + update
        ratings[match.away_team_id] = away_rating - update
        history[match.home_team_id] += 1
        history[match.away_team_id] += 1

    return pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)


def evaluate_elo_live_state(
    frame: pd.DataFrame,
    *,
    cutoff: int = 30,
) -> EloLiveStateReport:
    if frame.empty:
        raise ValueError("live-state frame is empty")
    filters = {
        "all": pd.Series(True, index=frame.index),
        "history_3": frame["minimum_history"] >= 3,
        "history_5": frame["minimum_history"] >= 5,
        "history_3_gap_50": (frame["minimum_history"] >= 3)
        & (frame["elo_gap"] >= 50.0),
        "history_3_gap_100": (frame["minimum_history"] >= 3)
        & (frame["elo_gap"] >= 100.0),
    }
    filter_metrics = {
        name: _metrics(frame.loc[mask].reset_index(drop=True))
        for name, mask in filters.items()
    }

    midpoint = pd.Timestamp(frame["date"].quantile(0.5))
    history_mask = frame["minimum_history"] >= 3
    by_time = {
        "early_half_history_3": _metrics(
            frame.loc[history_mask & (frame["date"] <= midpoint)].reset_index(drop=True)
        ),
        "late_half_history_3": _metrics(
            frame.loc[history_mask & (frame["date"] > midpoint)].reset_index(drop=True)
        ),
    }
    by_competition = {
        str(int(competition)): _metrics(group.reset_index(drop=True))
        for competition, group in frame.loc[history_mask].groupby("competition_id")
        if len(group) >= 20
    }
    alert_frame = frame.loc[history_mask & frame["alert"]]
    success_frame = alert_frame.loc[alert_frame["target"]]
    return EloLiveStateReport(
        cutoff=cutoff,
        rows=len(frame),
        filters=filter_metrics,
        by_time=by_time,
        by_competition=by_competition,
        alert_score_distribution=dict(
            Counter(alert_frame["final_oriented_score"]).most_common()
        ),
        success_score_distribution=dict(
            Counter(success_frame["final_oriented_score"]).most_common()
        ),
    )
