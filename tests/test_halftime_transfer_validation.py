from __future__ import annotations

import pandas as pd

from football_v2.halftime_transfer_validation import (
    validate_halftime_rule_transfer,
)


def _rows(
    *,
    year: int,
    division: str,
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
                "division": division,
                "home_team": f"Home {division} {year} {index}",
                "away_team": f"Away {division} {year} {index}",
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


def test_transfer_filter_is_applied_before_shadow() -> None:
    discovery_rows: list[dict[str, object]] = []
    discovery_rows += _rows(
        year=2005,
        division="E0",
        count=100,
        underdog_probability=0.28,
        final_scores=[(2, 0)] * 7 + [(1, 1)] * 3,
    )
    discovery_rows += _rows(
        year=2006,
        division="D1",
        count=200,
        underdog_probability=0.34,
        final_scores=[(1, 1)] * 7 + [(2, 0)] * 3,
    )
    discovery_rows += _rows(
        year=2015,
        division="E0",
        count=30,
        underdog_probability=0.28,
        final_scores=[(2, 0)] * 8 + [(1, 1)] * 2,
    )
    discovery_rows += _rows(
        year=2016,
        division="D1",
        count=60,
        underdog_probability=0.34,
        final_scores=[(1, 1)] * 7 + [(2, 0)] * 3,
    )
    discovery_rows += _rows(
        year=2020,
        division="E0",
        count=20,
        underdog_probability=0.28,
        final_scores=[(2, 0)] * 8 + [(1, 1)] * 2,
    )

    transfer_rows: list[dict[str, object]] = []
    for division in ("E1", "D2"):
        transfer_rows += _rows(
            year=2019,
            division=division,
            count=30,
            underdog_probability=0.28,
            final_scores=[(2, 0)] * 8 + [(1, 1)] * 2,
        )
        transfer_rows += _rows(
            year=2020,
            division=division,
            count=30,
            underdog_probability=0.34,
            final_scores=[(1, 1)] * 7 + [(2, 0)] * 3,
        )

    shadow_rows: list[dict[str, object]] = []
    for division in ("N1", "P1"):
        shadow_rows += _rows(
            year=2023,
            division=division,
            count=30,
            underdog_probability=0.28,
            final_scores=[(2, 0)] * 7 + [(1, 1)] * 3,
        )

    report = validate_halftime_rule_transfer(
        pd.DataFrame(discovery_rows),
        pd.DataFrame(transfer_rows),
        pd.DataFrame(shadow_rows),
    )

    assert report.discovered_rules
    assert report.retained_rules
    assert report.transfer_combined.accuracy is not None
    assert report.transfer_combined.baseline_accuracy is not None
    assert report.transfer_combined.accuracy > report.transfer_combined.baseline_accuracy
    assert report.final_shadow.predictions == 60
    assert report.final_shadow.accuracy is not None
    assert report.final_shadow.accuracy >= 0.70
