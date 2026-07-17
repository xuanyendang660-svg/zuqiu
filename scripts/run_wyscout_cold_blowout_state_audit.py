from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from football_v2.event_tail_model import alert_metrics
from football_v2.live_cold_blowout_ranker import _role_targets
from football_v2.live_snapshots import build_live_snapshot_dataset
from football_v2.market_event_data import load_market_1718
from football_v2.strict_market_join import join_market_events_without_score
from football_v2.wyscout_events import (
    build_wyscout_event_dataset,
    load_wyscout_index,
    load_wyscout_league_matches,
)
from run_wyscout_cold_blowout_strict_audit import (
    _json_default,
    _normalise,
    _role_values,
    _wilson,
)


def _rule_metrics(alerts: np.ndarray, target: np.ndarray) -> dict[str, object]:
    metrics = asdict(alert_metrics(alerts, target))
    successes = int(np.logical_and(alerts, target == 1).sum())
    metrics["successes"] = successes
    metrics["precision_wilson_95"] = _wilson(successes, int(alerts.sum()))
    return metrics


def _oriented_state(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    underdog_goals, favorite_goals = _role_values(
        frame, "live_home_score", "live_away_score"
    )
    underdog_xg, favorite_xg = _role_values(frame, "live_home_xg", "live_away_xg")
    underdog_sot, favorite_sot = _role_values(
        frame, "live_home_shots_on_target", "live_away_shots_on_target"
    )
    underdog_shots, favorite_shots = _role_values(
        frame, "live_home_shots", "live_away_shots"
    )
    underdog_box, favorite_box = _role_values(
        frame, "live_home_box_entries", "live_away_box_entries"
    )
    underdog_red, favorite_red = _role_values(
        frame, "live_home_red_cards", "live_away_red_cards"
    )
    final_underdog, final_favorite = _role_values(frame, "home_score", "away_score")
    return {
        "underdog_goals": underdog_goals,
        "favorite_goals": favorite_goals,
        "lead": underdog_goals - favorite_goals,
        "underdog_xg": underdog_xg,
        "favorite_xg": favorite_xg,
        "xg_diff": underdog_xg - favorite_xg,
        "sot_diff": underdog_sot - favorite_sot,
        "shot_diff": underdog_shots - favorite_shots,
        "box_diff": underdog_box - favorite_box,
        "favorite_red_advantage": favorite_red - underdog_red,
        "final_underdog": final_underdog,
        "final_favorite": final_favorite,
    }


def _rules(value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    goals = value["underdog_goals"]
    lead = value["lead"]
    sustained = np.logical_or.reduce(
        [
            value["xg_diff"] >= 0,
            value["sot_diff"] >= 1,
            value["box_diff"] >= 2,
        ]
    )
    strong_sustained = np.logical_and.reduce(
        [
            value["xg_diff"] >= 0,
            value["sot_diff"] >= 0,
            value["shot_diff"] >= -1,
        ]
    )
    return {
        "score_2_0_or_better_unresolved": np.logical_and(goals >= 2, lead >= 2),
        "score_2_1_family": np.logical_and(goals >= 2, lead == 1),
        "underdog_two_goals_any_lead": np.logical_and(goals >= 2, lead >= 1),
        "lead_two_any_score": lead >= 2,
        "lead_and_sustained": np.logical_and(lead >= 1, sustained),
        "lead_and_strong_sustained": np.logical_and(lead >= 1, strong_sustained),
        "two_goals_and_sustained": np.logical_and.reduce(
            [goals >= 2, lead >= 1, sustained]
        ),
        "two_goals_strong_sustained": np.logical_and.reduce(
            [goals >= 2, lead >= 1, strong_sustained]
        ),
        "lead_with_favorite_red": np.logical_and(
            lead >= 1, value["favorite_red_advantage"] >= 1
        ),
        "one_goal_lead_pressure": np.logical_and.reduce(
            [goals == 1, lead == 1, value["xg_diff"] >= 0, value["sot_diff"] >= 0]
        ),
        "tied_but_dominant": np.logical_and.reduce(
            [lead == 0, value["xg_diff"] >= 0.5, value["sot_diff"] >= 2]
        ),
    }


def _score_distribution(
    alerts: np.ndarray,
    target: np.ndarray,
    value: dict[str, np.ndarray],
) -> dict[str, list[dict[str, object]]]:
    def rows(mask: np.ndarray) -> list[dict[str, object]]:
        scores = Counter(
            f"{int(underdog)}-{int(favorite)}"
            for underdog, favorite in zip(
                value["final_underdog"][mask],
                value["final_favorite"][mask],
                strict=True,
            )
        )
        total = sum(scores.values())
        return [
            {"oriented_score": score, "count": count, "rate": count / total}
            for score, count in scores.most_common(10)
        ]

    return {
        "all_alerts": rows(alerts),
        "true_targets": rows(np.logical_and(alerts, target == 1)),
    }


def _time_slices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(frame["date"])
    split = dates.quantile(0.5)
    return {
        "early_half": (dates <= split).to_numpy(dtype=bool),
        "late_half": (dates > split).to_numpy(dtype=bool),
    }


def _state_table(
    value: dict[str, np.ndarray], target: np.ndarray, league: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    states = sorted(
        set(
            zip(
                value["underdog_goals"].astype(int),
                value["favorite_goals"].astype(int),
                strict=True,
            )
        )
    )
    for underdog_goals, favorite_goals in states:
        mask = np.logical_and(
            value["underdog_goals"] == underdog_goals,
            value["favorite_goals"] == favorite_goals,
        )
        count = int(mask.sum())
        if count < 5:
            continue
        successes = int(np.logical_and(mask, target == 1).sum())
        league_rows = []
        for division in sorted(np.unique(league)):
            league_mask = np.logical_and(mask, league == division)
            league_count = int(league_mask.sum())
            if league_count == 0:
                continue
            league_successes = int(np.logical_and(league_mask, target == 1).sum())
            league_rows.append(
                {
                    "league": int(division),
                    "matches": league_count,
                    "successes": league_successes,
                    "precision": league_successes / league_count,
                }
            )
        rows.append(
            {
                "live_oriented_score": f"{underdog_goals}-{favorite_goals}",
                "matches": count,
                "successes": successes,
                "precision": successes / count,
                "precision_wilson_95": _wilson(successes, count),
                "leagues": league_rows,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["precision"]), -int(row["matches"])))


def _cutoff_report(frame: pd.DataFrame, cutoff: int) -> dict[str, object]:
    cutoff_frame = frame.loc[
        frame["snapshot_minute"].to_numpy(dtype=int) == cutoff
    ].reset_index(drop=True)
    value = _oriented_state(cutoff_frame)
    _, _, target = _role_targets(cutoff_frame)
    already_true = np.logical_and(value["underdog_goals"] >= 3, value["lead"] >= 2)
    eligible = np.logical_not(already_true)
    cutoff_frame = cutoff_frame.loc[eligible].reset_index(drop=True)
    value = _oriented_state(cutoff_frame)
    _, _, target = _role_targets(cutoff_frame)
    league = cutoff_frame["division_id"].to_numpy(dtype=int)
    rules = _rules(value)
    slices = _time_slices(cutoff_frame)

    rule_rows: dict[str, object] = {}
    for name, alerts in rules.items():
        by_league = {}
        for division in sorted(np.unique(league)):
            mask = league == division
            by_league[str(int(division))] = _rule_metrics(alerts[mask], target[mask])
        by_time = {
            label: _rule_metrics(alerts[mask], target[mask])
            for label, mask in slices.items()
        }
        rule_rows[name] = {
            "overall": _rule_metrics(alerts, target),
            "by_league": by_league,
            "by_time": by_time,
            "final_scores": _score_distribution(alerts, target, value),
        }

    return {
        "cutoff": cutoff,
        "matches_before_exclusion": int(len(eligible)),
        "already_true_excluded": int(already_true.sum()),
        "eligible_matches": int(eligible.sum()),
        "base_rate": float(np.mean(target)),
        "rules": rule_rows,
        "state_table": _state_table(value, target, league),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed live-state cold-blowout audit")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--market-cache", default=".cache/football-data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", default="artifacts/wyscout_cold_blowout_state_audit.json"
    )
    args = parser.parse_args()

    records = _normalise(load_wyscout_index(args.data_root))
    matches = load_wyscout_league_matches(args.data_root, workers=args.workers)
    event_dataset = build_wyscout_event_dataset(matches)
    market_event_dataset = join_market_events_without_score(
        event_dataset,
        records,
        load_market_1718(args.market_cache),
    )
    live_dataset = build_live_snapshot_dataset(
        args.data_root,
        market_event_dataset,
        records,
        cutoffs=(15, 30, 45, 60),
    )
    payload = {
        "dataset": {
            "matches": int(live_dataset.frame["match_id"].nunique()),
            "market_join_uses_final_score": False,
            "cutoffs": [15, 30, 45, 60],
        },
        "cutoffs": [
            _cutoff_report(live_dataset.frame, cutoff)
            for cutoff in (15, 30, 45, 60)
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
