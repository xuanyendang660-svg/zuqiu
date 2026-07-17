from __future__ import annotations

import numpy as np
import pandas as pd

from .market_event_data import SOURCE_DIVISION, _similarity
from .statsbomb_events import EventDataset
from .wyscout_events import WyscoutIndexRecord


_MARKET_FEATURES = (
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


def join_market_events_without_score(
    event_dataset: EventDataset,
    index_records: list[WyscoutIndexRecord],
    market: pd.DataFrame,
    *,
    minimum_rows: int = 1700,
    minimum_similarity: float = 1.15,
) -> EventDataset:
    """Join market rows without using the final score as a lookup key.

    The regular research join uses final score to reduce ambiguous historical
    matches. That is convenient for data assembly but is outcome-dependent and
    therefore unsuitable for a strict leakage audit. This join uses only the
    competition, date window and team-name similarity.
    """

    index_by_id = {record.match_id: record for record in index_records}
    grouped = {
        key: group.reset_index(drop=True)
        for key, group in market.groupby(["division", "date"])
    }
    used_market_rows: set[tuple[str, pd.Timestamp, str, str]] = set()
    rows: list[dict[str, object]] = []

    for event_row in event_dataset.frame.to_dict(orient="records"):
        index = index_by_id.get(int(event_row["match_id"]))
        if index is None:
            continue
        division = SOURCE_DIVISION[index.source]
        candidates: list[pd.DataFrame] = []
        for delta in (-1, 0, 1):
            date = pd.Timestamp(index.date).normalize() + pd.Timedelta(days=delta)
            group = grouped.get((division, date))
            if group is not None:
                candidates.append(group)
        if not candidates:
            continue

        candidate_frame = pd.concat(candidates, ignore_index=True)
        candidate_records = candidate_frame.to_dict(orient="records")
        similarities = np.asarray(
            [
                _similarity(index.home_name, str(candidate["home_team_market"]))
                + _similarity(index.away_name, str(candidate["away_team_market"]))
                for candidate in candidate_records
            ],
            dtype=float,
        )
        for position, candidate in enumerate(candidate_records):
            identity = (
                str(candidate["division"]),
                pd.Timestamp(candidate["date"]).normalize(),
                str(candidate["home_team_market"]),
                str(candidate["away_team_market"]),
            )
            if identity in used_market_rows:
                similarities[position] = -np.inf
        best = int(np.argmax(similarities))
        if not np.isfinite(similarities[best]) or similarities[best] < minimum_similarity:
            continue

        selected = candidate_frame.iloc[best]
        identity = (
            str(selected["division"]),
            pd.Timestamp(selected["date"]).normalize(),
            str(selected["home_team_market"]),
            str(selected["away_team_market"]),
        )
        used_market_rows.add(identity)
        home_win = int(event_row["home_score"]) > int(event_row["away_score"])
        away_win = int(event_row["away_score"]) > int(event_row["home_score"])
        underdog_win = (
            home_win
            and float(selected["market_home_prob"])
            < float(selected["market_away_prob"])
        ) or (
            away_win
            and float(selected["market_away_prob"])
            < float(selected["market_home_prob"])
        )
        rows.append(
            {
                **event_row,
                **{name: selected[name] for name in _MARKET_FEATURES},
                "upset_jackpot_target": int(
                    bool(event_row["tail_target"]) and underdog_win
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(frame) < minimum_rows:
        raise RuntimeError(
            f"outcome-independent market join produced only {len(frame)} matches"
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
    return EventDataset(
        frame,
        tuple(column for column in frame.columns if column not in non_features),
    )
