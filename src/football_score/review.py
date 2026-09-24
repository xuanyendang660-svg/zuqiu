"""Read-only weekly audit of immutable freezes and verified 90-minute results."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .workflow import _axis, _parse_utc


def _unique(records, name):
    out = {}
    for record in records:
        key = record.get("match_id")
        if not key or key in out:
            raise ValueError(f"DATA_BLOCKED: duplicate or missing {name} match_id: {key}")
        out[key] = record
    return out


def review(freezes: list[dict], results: list[dict]):
    forecast = _unique(freezes, "freeze")
    settled = _unique(results, "result")
    per_match, patterns = [], defaultdict(list)
    for match_id, old in forecast.items():
        item = settled.get(match_id)
        if item is None:
            per_match.append({"match_id": match_id, "status": "RESULT_MISSING"})
            continue
        try:
            freeze_at = _parse_utc(old["frozen_at"])
            kickoff = _parse_utc(old["kickoff_utc"])
            if freeze_at >= kickoff:
                raise ValueError("freeze at or after kickoff")
            if item.get("status") != "FT_90":
                raise ValueError("result is not settled for 90-minute regulation time")
            pred = tuple(int(v) for v in old["score"])
            actual = tuple(int(v) for v in item["score"])
            if len(pred) != 2 or len(actual) != 2 or min(*pred, *actual) < 0:
                raise ValueError("invalid score pair")
            if old.get("match_id") != item.get("match_id"):
                raise ValueError("mismatched match ID")
            if old.get("home_team") != item.get("home_team") or old.get("away_team") != item.get("away_team"):
                raise ValueError("home/away team mismatch")
            if "settled_at" in item and _parse_utc(item["settled_at"]) < kickoff:
                raise ValueError("result settled before kickoff")
        except (ValueError, KeyError, TypeError) as exc:
            per_match.append({"match_id": match_id, "status": f"DATA_BLOCKED: {exc}"})
            continue
        p, a = _axis(*pred), _axis(*actual)
        axes = {k: p[k] == a[k] for k in ("direction", "total_band", "btts")}
        axes["margin"] = p["margin"] == a["margin"] if axes["direction"] else None
        axes["loser_or_draw_goals"] = ((p["loser_goals"] == a["loser_goals"]
                                          if a["direction"] != "平" else
                                          p["draw_goals"] == a["draw_goals"])
                                         if axes["direction"] else None)
        predicted_state = old.get("stop_state")
        actual_state = item.get("stop_state")
        axes["stop_state"] = (predicted_state == actual_state if predicted_state and actual_state
                              and predicted_state != "UNKNOWN_PREMATCH" else None)
        failure = []
        if not axes["direction"]: failure.append("direction")
        if axes["margin"] is False: failure.append("margin")
        if axes["loser_or_draw_goals"] is False: failure.append("loser_or_draw_goals")
        if not axes["total_band"]: failure.append("total_band")
        if axes["stop_state"] is False: failure.append("stop_state")
        for axis in failure:
            patterns[axis].append({"id": match_id, "date": kickoff.date().isoformat(),
                                   "market_status": old.get("market_status", "UNKNOWN")})
        distribution = old.get("distribution", {})
        actual_probability = distribution.get(f"{actual[0]}-{actual[1]}")
        per_match.append({"match_id": match_id, "status": "REVIEWED",
                          "predicted_score": list(pred), "actual_score": list(actual),
                          "exact_hit": pred == actual, "predicted_axes": p, "actual_axes": a,
                          "axes_hit": axes, "actual_score_probability": actual_probability,
                          "market_status_at_freeze": old.get("market_status", "UNKNOWN"),
                          "forecast_stop_state": predicted_state, "actual_stop_state": actual_state,
                          "event_evidence": item.get("event_evidence", []),
                          "attribution": "UNDETERMINED: score errors alone do not identify a cause"})
    names = {
        "direction": "检验方向分布及球队强度特征；先看同条件的胜平负校准。",
        "margin": "检验胜差分布的条件校准及低分尾部。",
        "loser_or_draw_goals": "检验败方进球或平局双方进球条件分布。",
        "total_band": "检验负二项总进球离散度与总球带校准。",
        "stop_state": "核对赛前状态证据与赛后事件时间线，不用终场比分代替时间线。",
    }
    hypotheses = []
    for axis, examples in sorted(patterns.items(), key=lambda x: -len(x[1])):
        days = {e["date"] for e in examples}
        if len(examples) < 3 or len(days) < 2:
            continue
        hypotheses.append({"module": axis, "matches": [e["id"] for e in examples],
                           "error_count": len(examples), "independent_dates": len(days),
                           "status": "VALIDATE_FIRST",
                           "next_week_test": names[axis] + " 前瞻冻结并按比赛日期与原版成对比较，保留反例。"})
    blocked = Counter(item["status"] for item in per_match if item["status"] != "REVIEWED")
    return {"status": "AUDIT", "match_count": len(forecast),
            "reviewed_count": sum(x["status"] == "REVIEWED" for x in per_match),
            "blocked": dict(blocked), "matches": per_match,
            "recurring_errors": hypotheses,
            "next_week_validation": hypotheses[:3],
            "interpretation": ("Repeated axis errors are hypotheses, not proven model faults. "
                               "Check source and event evidence before changing only implicated modules; "
                               "no global reversal from batch direction or totals."),
            "unmatched_results": sorted(set(settled) - set(forecast))}


def review_files(frozen_path: str, results_path: str):
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    actual = json.loads(Path(results_path).read_text(encoding="utf-8"))
    return review(frozen["predictions"], actual["results"])
