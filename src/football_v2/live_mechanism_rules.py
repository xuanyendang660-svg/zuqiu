from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np

from .event_tail_model import AlertMetrics, alert_metrics
from .live_snapshots import LiveSnapshotDataset


@dataclass(frozen=True)
class RuleCutoffResult:
    cutoff: int
    matches: int
    selected_alerts: AlertMetrics
    fixed_rule_results: dict[str, AlertMetrics]


@dataclass(frozen=True)
class MechanismRuleReport:
    cutoffs: tuple[RuleCutoffResult, ...]
    fold_details: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoffs": [
                {
                    "cutoff": result.cutoff,
                    "matches": result.matches,
                    "selected_alerts": asdict(result.selected_alerts),
                    "fixed_rule_results": {
                        name: asdict(metrics)
                        for name, metrics in result.fixed_rule_results.items()
                    },
                }
                for result in self.cutoffs
            ],
            "fold_details": list(self.fold_details),
        }


def _oriented(frame: object) -> dict[str, np.ndarray]:
    home_underdog = (
        frame["market_home_prob"].to_numpy(dtype=float)
        < frame["market_away_prob"].to_numpy(dtype=float)
    )
    sign = np.where(home_underdog, 1.0, -1.0)

    def difference(home_name: str, away_name: str) -> np.ndarray:
        return sign * (
            frame[home_name].to_numpy(dtype=float)
            - frame[away_name].to_numpy(dtype=float)
        )

    def underdog_value(home_name: str, away_name: str) -> np.ndarray:
        return np.where(
            home_underdog,
            frame[home_name].to_numpy(dtype=float),
            frame[away_name].to_numpy(dtype=float),
        )

    def favorite_value(home_name: str, away_name: str) -> np.ndarray:
        return np.where(
            home_underdog,
            frame[away_name].to_numpy(dtype=float),
            frame[home_name].to_numpy(dtype=float),
        )

    return {
        "score_diff": difference("live_home_score", "live_away_score"),
        "xg_diff": difference("live_home_xg", "live_away_xg"),
        "sot_diff": difference(
            "live_home_shots_on_target", "live_away_shots_on_target"
        ),
        "shot_diff": difference("live_home_shots", "live_away_shots"),
        "box_diff": difference(
            "live_home_box_entries", "live_away_box_entries"
        ),
        "underdog_goals": underdog_value(
            "live_home_score", "live_away_score"
        ),
        "favorite_goals": favorite_value(
            "live_home_score", "live_away_score"
        ),
        "underdog_xg": underdog_value("live_home_xg", "live_away_xg"),
        "favorite_xg": favorite_value("live_home_xg", "live_away_xg"),
        "underdog_sot": underdog_value(
            "live_home_shots_on_target", "live_away_shots_on_target"
        ),
        "favorite_sot": favorite_value(
            "live_home_shots_on_target", "live_away_shots_on_target"
        ),
        "underdog_red": underdog_value(
            "live_home_red_cards", "live_away_red_cards"
        ),
        "favorite_red": favorite_value(
            "live_home_red_cards", "live_away_red_cards"
        ),
        "total_goals": frame["live_total_goals"].to_numpy(dtype=float),
        "recent_xg": frame["xg_last10"].to_numpy(dtype=float),
        "recent_goals": frame["goals_last10"].to_numpy(dtype=float),
    }


def _rules(frame: object) -> dict[str, np.ndarray]:
    value = _oriented(frame)
    return {
        "underdog_lead_two": value["score_diff"] >= 2,
        "underdog_two_goals_ahead": np.logical_and(
            value["underdog_goals"] >= 2,
            value["score_diff"] >= 1,
        ),
        "underdog_lead_pressure": np.logical_and.reduce(
            [
                value["score_diff"] >= 1,
                value["xg_diff"] >= 0,
                value["sot_diff"] >= 0,
                value["underdog_goals"] >= 1,
            ]
        ),
        "underdog_lead_favorite_red": np.logical_and(
            value["score_diff"] >= 1,
            value["favorite_red"] > value["underdog_red"],
        ),
        "underdog_lead_open_game": np.logical_and(
            value["score_diff"] >= 1,
            value["total_goals"] >= 3,
        ),
        "underdog_two_goals_resistance": np.logical_and.reduce(
            [
                value["underdog_goals"] >= 2,
                value["score_diff"] >= 1,
                np.logical_or(
                    value["underdog_xg"] >= 0.7 * value["favorite_xg"],
                    value["underdog_sot"] >= value["favorite_sot"],
                ),
            ]
        ),
        "underdog_lead_recent_chaos": np.logical_and.reduce(
            [
                value["score_diff"] >= 1,
                value["total_goals"] >= 2,
                np.logical_or(value["recent_xg"] >= 0.25, value["recent_goals"] >= 1),
            ]
        ),
        "underdog_tied_dominant": np.logical_and.reduce(
            [
                value["score_diff"] == 0,
                value["xg_diff"] >= 0.5,
                value["sot_diff"] >= 2,
                value["box_diff"] >= 1,
            ]
        ),
    }


def _wilson_lower(successes: int, total: int, z: float = 1.28) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return (centre - spread) / denominator


def _select_rule(
    rules: dict[str, np.ndarray],
    target: np.ndarray,
) -> str | None:
    base_rate = float(np.mean(target))
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for name, alerts in rules.items():
        count = int(alerts.sum())
        successes = int(np.logical_and(alerts, target == 1).sum())
        if count < 4 or successes < 2:
            continue
        precision = successes / count
        lower = _wilson_lower(successes, count)
        lift = precision / base_rate if base_rate > 0 else 0.0
        if lift < 3.0 or lower <= base_rate:
            continue
        candidates.append(((lower, precision, successes), name))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def leave_one_league_out_mechanism_test(
    dataset: LiveSnapshotDataset,
) -> MechanismRuleReport:
    frame = dataset.frame
    league = frame["division_id"].to_numpy(dtype=int)
    cutoffs = frame["snapshot_minute"].to_numpy(dtype=int)
    target = frame["upset_jackpot_target"].to_numpy(dtype=int)
    all_rules = _rules(frame)
    selected_by_cutoff: dict[int, list[np.ndarray]] = {
        cutoff: [] for cutoff in sorted(np.unique(cutoffs))
    }
    targets_by_cutoff: dict[int, list[np.ndarray]] = {
        cutoff: [] for cutoff in selected_by_cutoff
    }
    fold_details: list[dict[str, object]] = []

    for held_out in sorted(np.unique(league)):
        train_league = league != held_out
        test_league = league == held_out
        fold_entry: dict[str, object] = {"held_out_league": int(held_out), "cutoffs": {}}
        for cutoff in selected_by_cutoff:
            train_mask = np.logical_and(train_league, cutoffs == cutoff)
            test_mask = np.logical_and(test_league, cutoffs == cutoff)
            train_rules = {name: alerts[train_mask] for name, alerts in all_rules.items()}
            selected = _select_rule(train_rules, target[train_mask])
            alerts = (
                all_rules[selected][test_mask]
                if selected is not None
                else np.zeros(int(test_mask.sum()), dtype=bool)
            )
            selected_by_cutoff[cutoff].append(alerts)
            targets_by_cutoff[cutoff].append(target[test_mask])
            fold_entry["cutoffs"][str(cutoff)] = {
                "selected_rule": selected,
                "test": asdict(alert_metrics(alerts, target[test_mask])),
            }
        fold_details.append(fold_entry)

    results: list[RuleCutoffResult] = []
    for cutoff in selected_by_cutoff:
        cutoff_mask = cutoffs == cutoff
        cutoff_target = target[cutoff_mask]
        fixed = {
            name: alert_metrics(alerts[cutoff_mask], cutoff_target)
            for name, alerts in all_rules.items()
        }
        selected_alerts = np.concatenate(selected_by_cutoff[cutoff])
        selected_target = np.concatenate(targets_by_cutoff[cutoff])
        results.append(
            RuleCutoffResult(
                cutoff=cutoff,
                matches=len(selected_target),
                selected_alerts=alert_metrics(selected_alerts, selected_target),
                fixed_rule_results=fixed,
            )
        )
    return MechanismRuleReport(tuple(results), tuple(fold_details))
