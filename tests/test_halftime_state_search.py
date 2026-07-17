from __future__ import annotations

import pandas as pd

from football_v2.halftime_state_search import run_halftime_state_search


def _rows(
    *,
    year: int,
    count: int,
    underdog_probability: float,
    final_scores: list[tuple[int, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        home_score, away_score = final_scores[index % len(final_scores)]
        rows.append(
            {
                "date": pd.Timestamp(year=year, month=8, day=1)
                + pd.Timedelta(days=index),
                "season": f"{year % 100:02d}{(year + 1) % 100:02d}",
                "division": "E0",
                "home_team": f"Home {year} {index}",
                "away_team": f"Away {year} {index}",
                "home_score": home_score,
                "away_score": away_score,
                "halftime_home": 1,
                "halftime_away": 0,
                "market_home_prob": underdog_probability,
                "market_draw_prob": 0.25,
                "market_away_prob": 1.0 - underdog_probability - 0.25,
            }
        )
    return rows


def test_nested_search_finds_calibrated_probability_state() -> None:
    rows: list[dict[str, object]] = []
    rows += _rows(
        year=2005,
        count=100,
        underdog_probability=0.28,
        final_scores=[(2, 0)] * 7 + [(1, 1)] * 3,
    )
    rows += _rows(
        year=2006,
        count=200,
        underdog_probability=0.34,
        final_scores=[(1, 1)] * 7 + [(2, 0)] * 3,
    )
    rows += _rows(
        year=2015,
        count=30,
        underdog_probability=0.28,
        final_scores=[(2, 0)] * 8 + [(1, 1)] * 2,
    )
    rows += _rows(
        year=2016,
        count=60,
        underdog_probability=0.34,
        final_scores=[(1, 1)] * 7 + [(2, 0)] * 3,
    )
    rows += _rows(
        year=2020,
        count=20,
        underdog_probability=0.28,
        final_scores=[(2, 0)] * 8 + [(1, 1)] * 2,
    )
    rows += _rows(
        year=2021,
        count=40,
        underdog_probability=0.34,
        final_scores=[(1, 1)] * 7 + [(2, 0)] * 3,
    )
    frame = pd.DataFrame(rows)

    report = run_halftime_state_search(
        frame,
        train_end_year=2011,
        calibration_end_year=2017,
    )

    exact = report.tasks["exact_score"]
    assert exact.selected_rules
    assert any(rule.prediction == "2-0" for rule in exact.selected_rules)
    assert exact.test.predictions >= 20
    assert exact.test.accuracy is not None
    assert exact.test.accuracy >= 0.70
    assert exact.test.baseline_accuracy is not None
    assert exact.test.accuracy > exact.test.baseline_accuracy
