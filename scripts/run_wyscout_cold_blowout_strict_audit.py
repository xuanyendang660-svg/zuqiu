from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import ceil, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from football_v2.event_tail_model import alert_metrics
from football_v2.live_cold_blowout_ranker import (
    _role_targets,
    leave_one_league_out_cold_blowout_ranker_test,
)
from football_v2.live_snapshots import LiveSnapshotDataset, build_live_snapshot_dataset
from football_v2.market_event_data import join_market_events, load_market_1718
from football_v2.wyscout_events import (
    WyscoutIndexRecord,
    build_wyscout_event_dataset,
    load_wyscout_index,
    load_wyscout_league_matches,
)


def _normalise(records: list[WyscoutIndexRecord]) -> list[WyscoutIndexRecord]:
    return [
        WyscoutIndexRecord(
            match_id=record.match_id,
            path=record.path,
            date=pd.Timestamp(record.date).normalize(),
            source=record.source,
            home_name=record.home_name,
            away_name=record.away_name,
            home_score=record.home_score,
            away_score=record.away_score,
        )
        for record in records
    ]


def _role_values(frame: pd.DataFrame, home_column: str, away_column: str) -> tuple[np.ndarray, np.ndarray]:
    underdog_home = (
        frame["market_home_prob"].to_numpy(dtype=float)
        < frame["market_away_prob"].to_numpy(dtype=float)
    )
    home = frame[home_column].to_numpy(dtype=float)
    away = frame[away_column].to_numpy(dtype=float)
    underdog = np.where(underdog_home, home, away)
    favorite = np.where(underdog_home, away, home)
    return underdog, favorite


def _wilson(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return {"low": max(0.0, center - half), "high": min(1.0, center + half)}


def _budget_mask_within_league(scores: np.ndarray, league: np.ndarray, budget: float) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    for value in sorted(np.unique(league)):
        indices = np.where(league == value)[0]
        count = max(1, int(ceil(len(indices) * budget)))
        order = indices[np.argsort(scores[indices])[::-1]]
        mask[order[:count]] = True
    return mask


def _simple_state_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    underdog_goals, favorite_goals = _role_values(
        frame, "live_home_score", "live_away_score"
    )
    underdog_xg, favorite_xg = _role_values(frame, "live_home_xg", "live_away_xg")
    underdog_sot, favorite_sot = _role_values(
        frame, "live_home_shots_on_target", "live_away_shots_on_target"
    )
    underdog_box, favorite_box = _role_values(
        frame, "live_home_box_entries", "live_away_box_entries"
    )
    underdog_red, favorite_red = _role_values(
        frame, "live_home_red_cards", "live_away_red_cards"
    )
    goal_margin = underdog_goals - favorite_goals
    score_only = 5.0 * goal_margin + 2.0 * underdog_goals
    event_state = (
        score_only
        + 1.25 * (underdog_xg - favorite_xg)
        + 0.35 * (underdog_sot - favorite_sot)
        + 0.05 * (underdog_box - favorite_box)
        + 1.5 * (favorite_red - underdog_red)
    )
    market_state = event_state + 2.0 * frame["market_underdog_prob"].to_numpy(dtype=float)
    return {
        "score_only": score_only,
        "event_state": event_state,
        "market_state": market_state,
    }


def _baseline_report(frame: pd.DataFrame, budgets: tuple[float, ...]) -> dict[str, object]:
    _, _, target = _role_targets(frame)
    league = frame["division_id"].to_numpy(dtype=int)
    output: dict[str, object] = {}
    for name, scores in _simple_state_scores(frame).items():
        budget_rows: dict[str, object] = {}
        for budget in budgets:
            alerts = _budget_mask_within_league(scores, league, budget)
            metrics = alert_metrics(alerts, target)
            successes = int(np.logical_and(alerts, target == 1).sum())
            row = asdict(metrics)
            row["precision_wilson_95"] = _wilson(successes, int(alerts.sum()))
            budget_rows[str(budget)] = row
        output[name] = budget_rows
    return output


def _strict_dataset(dataset: LiveSnapshotDataset, cutoff: int) -> tuple[LiveSnapshotDataset, dict[str, int]]:
    frame = dataset.frame.loc[
        dataset.frame["snapshot_minute"].to_numpy(dtype=int) == cutoff
    ].reset_index(drop=True)
    underdog_goals, favorite_goals = _role_values(
        frame, "live_home_score", "live_away_score"
    )
    already_resolved = np.logical_and(
        underdog_goals >= 3,
        underdog_goals - favorite_goals >= 2,
    )
    eligible = np.logical_not(already_resolved)
    strict_frame = frame.loc[eligible].reset_index(drop=True)
    return (
        LiveSnapshotDataset(strict_frame, dataset.feature_columns),
        {
            "original_matches": int(len(frame)),
            "already_resolved_excluded": int(already_resolved.sum()),
            "eligible_unresolved_matches": int(eligible.sum()),
        },
    )


def _json_default(value: object) -> object:
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict unresolved 30-minute cold-blowout audit"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--market-cache", default=".cache/football-data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", default="artifacts/wyscout_cold_blowout_strict_audit.json"
    )
    args = parser.parse_args()

    records = _normalise(load_wyscout_index(args.data_root))
    matches = load_wyscout_league_matches(args.data_root, workers=args.workers)
    event_dataset = build_wyscout_event_dataset(matches)
    market_event_dataset = join_market_events(
        event_dataset,
        records,
        load_market_1718(args.market_cache),
    )
    live_dataset = build_live_snapshot_dataset(
        args.data_root,
        market_event_dataset,
        records,
        cutoffs=(30,),
    )
    strict_dataset, eligibility = _strict_dataset(live_dataset, cutoff=30)
    budgets = (0.02, 0.04, 0.06)
    model_report = leave_one_league_out_cold_blowout_ranker_test(
        strict_dataset,
        cutoff=30,
        budgets=budgets,
    )
    payload = {
        "strict_unresolved_model": model_report.to_dict(),
        "simple_live_baselines": _baseline_report(strict_dataset.frame, budgets),
        "eligibility": eligibility,
        "audit_rules": {
            "cutoff": 30,
            "excluded_if_already_true": "underdog goals >= 3 and underdog lead >= 2",
            "selection_budget_applied_within_each_held_out_league": True,
            "true_cluster_only_metrics_are_diagnostic_not_deployable": True,
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
