import numpy as np
import pandas as pd

from football_v2.event_tail_model import EventTailConfig, walk_forward_event_tail_test
from football_v2.statsbomb_events import EventDataset


def _synthetic_event_dataset(rows: int = 900) -> EventDataset:
    rng = np.random.default_rng(41)
    records: list[dict[str, object]] = []
    for index in range(rows):
        tail = index % 18 == 0
        kind = (index // 18) % 3
        if tail and kind == 0:
            home, away, tail_type = 5, 1, "home_blowout"
        elif tail and kind == 1:
            home, away, tail_type = 1, 5, "away_blowout"
        elif tail:
            home, away, tail_type = 3, 3, "shootout"
        else:
            home, away, tail_type = (2, 1, "normal") if index % 2 else (1, 1, "normal")

        event_signal = 4.5 if tail else 0.2
        direction_signal = 1.0 if kind == 0 else -1.0 if kind == 1 else 0.0
        records.append(
            {
                "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=index),
                "match_id": index,
                "home_score": home,
                "away_score": away,
                "tail_target": int(tail),
                "tail_type": tail_type,
                "home_goals_mean": 1.4 + rng.normal(0, 0.25),
                "away_goals_mean": 1.1 + rng.normal(0, 0.25),
                "home_xg_mean": event_signal + max(direction_signal, 0) + rng.normal(0, 0.08),
                "away_xg_mean": event_signal + max(-direction_signal, 0) + rng.normal(0, 0.08),
                "home_collapse_conceded_mean": float(tail and kind == 1),
                "away_collapse_conceded_mean": float(tail and kind == 0),
                "diff_xg_mean": direction_signal + rng.normal(0, 0.05),
                "home_rest_days": 7.0,
                "away_rest_days": 7.0,
                "home_history": 10.0,
                "away_history": 10.0,
            }
        )
    frame = pd.DataFrame(records)
    feature_columns = tuple(
        column
        for column in frame.columns
        if column
        not in {"date", "match_id", "home_score", "away_score", "tail_target", "tail_type"}
    )
    return EventDataset(frame, feature_columns)


def test_event_features_create_high_lift_tail_alerts() -> None:
    dataset = _synthetic_event_dataset()
    report = walk_forward_event_tail_test(
        dataset,
        folds=2,
        initial_train_fraction=0.55,
        config=EventTailConfig(
            max_alert_coverage=0.10,
            minimum_alerts=4,
            minimum_lift=1.25,
            max_iter=120,
            min_samples_leaf=8,
        ),
    )
    assert report.event_alerts.alerts > 0
    assert report.event_alerts.lift is not None
    assert report.event_alerts.lift >= 5.0
    assert report.event_alerts.precision is not None
    assert report.event_alerts.precision >= 0.70
    assert report.event_alerts.lift > (
        report.result_only_alerts.lift or 0.0
    )
    assert report.event_alert_top5_accuracy is not None
