import numpy as np
import pandas as pd

from football_v2.dataset import build_historical_dataset, select_market_columns


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2025-26",
                "date": "2025-08-01",
                "home_team": "A",
                "away_team": "B",
                "fthg": 2,
                "ftag": 0,
                "bet365_1x2_home_close": 1.80,
                "bet365_1x2_draw_close": 3.60,
                "bet365_1x2_away_close": 5.00,
                "bet365_over25_close": 1.95,
                "bet365_under25_close": 1.85,
                "ah_line": -0.75,
            },
            {
                "match_id": "m2",
                "season": "2025-26",
                "date": "2025-08-08",
                "home_team": "B",
                "away_team": "A",
                "fthg": 1,
                "ftag": 5,
                "bet365_1x2_home_close": 3.20,
                "bet365_1x2_draw_close": 3.50,
                "bet365_1x2_away_close": 2.10,
                "bet365_over25_close": 1.70,
                "bet365_under25_close": 2.10,
                "ah_line": 0.25,
            },
        ]
    )


def test_market_column_detection_and_strictly_pre_match_features() -> None:
    frame = _frame()
    columns = select_market_columns(frame)
    assert columns.home == "bet365_1x2_home_close"
    assert columns.over25 == "bet365_over25_close"

    dataset = build_historical_dataset(frame)
    first, second = dataset.frame.iloc[0], dataset.frame.iloc[1]

    assert np.isnan(first["home_gf10"])
    assert np.isnan(first["away_ga10"])
    assert second["home_gf10"] == 0.0
    assert second["home_ga10"] == 2.0
    assert second["away_gf10"] == 2.0
    assert second["away_ga10"] == 0.0
    assert np.isclose(
        second["market_home_prob"]
        + second["market_draw_prob"]
        + second["market_away_prob"],
        1.0,
    )
    assert second["target_home_goals"] == 1
    assert second["target_away_goals"] == 5
