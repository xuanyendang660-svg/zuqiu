import numpy as np

from football_v2.statsbomb_events import build_event_dataset, parse_statsbomb_match


def _match(match_id: int, date: str, home_score: int, away_score: int) -> dict:
    return {
        "match_id": match_id,
        "match_date": date,
        "competition": {"competition_id": 11},
        "season": {"season_id": 90},
        "home_team": {"home_team_id": 1, "home_team_name": "Home"},
        "away_team": {"away_team_id": 2, "away_team_name": "Away"},
        "home_score": home_score,
        "away_score": away_score,
    }


def _events(home_goal: bool = True) -> list[dict]:
    events = [
        {
            "type": {"name": "Starting XI"},
            "team": {"id": 1},
            "tactics": {
                "formation": 433,
                "lineup": [{"player": {"id": player}} for player in range(1, 12)],
            },
        },
        {
            "type": {"name": "Starting XI"},
            "team": {"id": 2},
            "tactics": {
                "formation": 442,
                "lineup": [{"player": {"id": player}} for player in range(21, 32)],
            },
        },
        {
            "type": {"name": "Pass"},
            "team": {"id": 1},
            "location": [70.0, 40.0],
            "pass": {"end_location": [105.0, 40.0]},
            "under_pressure": True,
        },
        {
            "type": {"name": "Pressure"},
            "team": {"id": 2},
            "location": [90.0, 40.0],
        },
        {
            "type": {"name": "Shot"},
            "team": {"id": 1},
            "minute": 12,
            "play_pattern": {"name": "From Counter"},
            "shot": {
                "statsbomb_xg": 0.35,
                "outcome": {"name": "Goal" if home_goal else "Saved"},
            },
        },
    ]
    return events


def test_statsbomb_parse_and_rolling_features_are_pre_match_only() -> None:
    first = parse_statsbomb_match(_match(1, "2020-01-01", 2, 0), _events())
    second = parse_statsbomb_match(_match(2, "2020-01-08", 1, 5), _events(False))
    dataset = build_event_dataset([first, second])

    first_row = dataset.frame.iloc[0]
    second_row = dataset.frame.iloc[1]
    assert np.isnan(first_row["home_xg_mean"])
    assert second_row["home_xg_mean"] == 0.35
    assert second_row["home_box_entries_mean"] == 1.0
    assert second_row["home_xi_continuity"] == 1.0
    assert second_row["home_rest_days"] == 7.0
    assert second_row["tail_target"] == 1
    assert second_row["tail_type"] == "away_blowout"
