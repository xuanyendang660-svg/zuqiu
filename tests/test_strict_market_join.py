from pathlib import Path

import pandas as pd

from football_v2.statsbomb_events import EventDataset
from football_v2.strict_market_join import join_market_events_without_score
from football_v2.wyscout_events import WyscoutIndexRecord


def test_strict_market_join_uses_names_not_final_score() -> None:
    date = pd.Timestamp("2017-08-12")
    frame = pd.DataFrame(
        [
            {
                "match_id": 1,
                "date": date,
                "competition_id_raw": 1,
                "season_id_raw": 201718,
                "home_team": "Arsenal",
                "away_team": "Leicester City",
                "home_score": 4,
                "away_score": 3,
                "tail_type": "shootout",
                "tail_target": 1,
            }
        ]
    )
    events = EventDataset(frame, tuple())
    records = [
        WyscoutIndexRecord(
            match_id=1,
            path=Path("unused.json"),
            date=date,
            source="matches_England.json",
            home_name="Arsenal",
            away_name="Leicester City",
            home_score=4,
            away_score=3,
        )
    ]
    market = pd.DataFrame(
        [
            {
                "division": "E0",
                "division_id": 1.0,
                "date": date,
                "home_team_market": "Arsenal",
                "away_team_market": "Leicester",
                "home_score": 0,
                "away_score": 0,
                "market_home_prob": 0.60,
                "market_draw_prob": 0.24,
                "market_away_prob": 0.16,
                "market_over25_prob": 0.55,
                "market_entropy": 0.94,
                "market_favorite_prob": 0.60,
                "market_underdog_prob": 0.16,
                "market_favorite_side": 1.0,
                "ah_line": -1.0,
            }
        ]
    )

    joined = join_market_events_without_score(
        events,
        records,
        market,
        minimum_rows=1,
    )

    assert len(joined.frame) == 1
    assert joined.frame.loc[0, "market_home_prob"] == 0.60
    assert joined.frame.loc[0, "upset_jackpot_target"] == 0
