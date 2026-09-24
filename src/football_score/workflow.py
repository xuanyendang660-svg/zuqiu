"""Chronological training, as-of market gates, paired evaluation and predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from .model import (
    dc_distribution,
    fit_dc,
    fit_joint,
    fuse,
    joint_distribution,
    market_distribution,
    top_score,
)


def load_history(path: str | Path) -> tuple[list[dict], str]:
    payload = Path(path).read_bytes()
    raw = list(csv.DictReader(payload.decode("utf-8-sig").splitlines()))
    if not raw:
        raise ValueError("DATA_BLOCKED: historical data is empty")
    rows, seen = [], set()
    for record in raw:
        if not record.get("match_id"):
            raise ValueError("DATA_BLOCKED: match_id is missing")
        if record["match_id"] in seen:
            raise ValueError(f"DATA_BLOCKED: duplicate match_id {record['match_id']}")
        seen.add(record["match_id"])
        try:
            date.fromisoformat(record["date"])
            home, away = int(record["actual_home_goals"]), int(record["actual_away_goals"])
        except (ValueError, KeyError) as exc:
            raise ValueError(f"DATA_BLOCKED: invalid date or score: {record['match_id']}") from exc
        if min(home, away) < 0 or not all(record.get(k) for k in
                                          ("league_code", "home_team", "away_team")):
            raise ValueError(f"DATA_BLOCKED: invalid identity or score: {record['match_id']}")
        rows.append({k: record[k] for k in ("match_id", "date", "league_code",
                     "home_team", "away_team", "actual_home_goals", "actual_away_goals")})
    rows.sort(key=lambda r: (r["date"], r["match_id"]))
    _rebuild_prior(rows)
    return rows, hashlib.sha256(payload).hexdigest()


def _rebuild_prior(rows: list[dict]):
    """Recompute all score-derived features before updating *any* match on a date."""
    league = defaultdict(lambda: [0., 0., 0])
    teams = defaultdict(lambda: deque(maxlen=8))
    elo = defaultdict(lambda: 1500.)
    by_date = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)
    for day in sorted(by_date):
        group = by_date[day]
        for r in group:
            code = r["league_code"]
            sum_h, sum_a, n = league[code]
            lh = (sum_h + 20 * 1.40) / (n + 20)
            la = (sum_a + 20 * 1.25) / (n + 20)
            hk = (code, r["home_team"])
            ak = (code, r["away_team"])
            hs, ass = teams[hk], teams[ak]
            r.update(league_home_goals_pre=lh, league_away_goals_pre=la,
                     league_avg_goals_pre=lh + la,
                     home_recent8_gf=sum(v[0] for v in hs) / len(hs) if hs else lh,
                     home_recent8_ga=sum(v[1] for v in hs) / len(hs) if hs else la,
                     away_recent8_gf=sum(v[0] for v in ass) / len(ass) if ass else la,
                     away_recent8_ga=sum(v[1] for v in ass) / len(ass) if ass else lh,
                     home_elo_pre=elo[hk], away_elo_pre=elo[ak],
                     prior_home_count=len(hs), prior_away_count=len(ass))
        for r in group:
            code = r["league_code"]
            h, a = int(r["actual_home_goals"]), int(r["actual_away_goals"])
            hk = (code, r["home_team"])
            ak = (code, r["away_team"])
            expected = 1 / (1 + 10 ** (-(elo[hk] + 65 - elo[ak]) / 400))
            change = 20 * ((1 if h > a else 0 if h < a else .5) - expected)
            elo[hk] += change
            elo[ak] -= change
            teams[hk].append((h, a))
            teams[ak].append((a, h))
            league[code][0] += h
            league[code][1] += a
            league[code][2] += 1


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return dt.astimezone(UTC)


def market_asof(row: dict, quotes: dict | None):
    if quotes is None:
        return None, "MARKET_MISSING"
    try:
        if quotes["match_id"] != row["match_id"]:
            raise ValueError("match identity mismatch")
        kickoff = _parse_utc(quotes["kickoff_utc"])
        observed = _parse_utc(quotes["observed_at"])
        published = _parse_utc(quotes["published_at"])
        if row.get("kickoff_utc") and kickoff != _parse_utc(row["kickoff_utc"]):
            raise ValueError("kickoff timestamp mismatch")
        if abs((kickoff.date() - date.fromisoformat(row["date"])).days) > 1:
            raise ValueError("kickoff date mismatch")
        if max(observed, published) >= kickoff:
            raise ValueError("quote recorded or published at/after kickoff")
        if observed < published:
            raise ValueError("observation predates publication")
        if "source" not in quotes:
            raise ValueError("source is missing")
        return quotes, "ASOF_VERIFIED"
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"MARKET_BLOCKED: {exc}"


def read_quotes(path: str | Path | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("quotes must be a JSON list of timestamped snapshots")
    out = {}
    for quote in data:
        key = quote["match_id"]
        if key in out:
            raise ValueError(f"duplicate quote match_id: {key}; choose the snapshot explicitly")
        out[key] = quote
    return out


def _axis(h: int, a: int):
    direction = "主胜" if h > a else "客胜" if a > h else "平"
    band = "0–1" if h + a <= 1 else "5+" if h + a >= 5 else str(h + a)
    return {"direction": direction, "margin": abs(h - a),
            "loser_goals": min(h, a) if h != a else None,
            "draw_goals": h if h == a else None, "total_band": band,
            "btts": bool(h and a)}


def _metrics(rows: list[dict], probs: list[dict]):
    n = len(rows)
    if n == 0:
        return None
    hits, logs, direction, margin, loser, band = 0, [], 0, 0, 0, 0
    brier_dir, brier_band = 0., 0.
    for r, q in zip(rows, probs, strict=True):
        actual = int(r["actual_home_goals"]), int(r["actual_away_goals"])
        pred = top_score(q)
        x, y = _axis(*pred), _axis(*actual)
        hits += pred == actual
        logs.append(-math.log(max(q.get(actual, 0), 1e-15)))
        direction += x["direction"] == y["direction"]
        if x["direction"] == y["direction"]:
            margin += x["margin"] == y["margin"]
            loser += (x["loser_goals"] == y["loser_goals"] if y["direction"] != "平"
                      else x["draw_goals"] == y["draw_goals"])
        band += x["total_band"] == y["total_band"]
        brier_dir += (sum((sum(v for (h, a), v in q.items()
                                if _axis(h, a)["direction"] == key) - (key == y["direction"])) ** 2
                          for key in ("主胜", "平", "客胜")) / 3)
        brier_band += (sum((sum(v for (h, a), v in q.items()
                                 if _axis(h, a)["total_band"] == key) - (key == y["total_band"])) ** 2
                           for key in ("0–1", "2", "3", "4", "5+")) / 5)
    return {"n": n, "exact_hit_rate": hits / n, "mean_log_loss": sum(logs) / n,
            "direction_hit_rate": direction / n,
            "margin_hit_rate_given_correct_direction": margin / direction if direction else None,
            "loser_or_draw_goals_hit_rate_given_correct_direction": loser / direction if direction else None,
            "axis_conditional_sample_count": direction, "total_band_hit_rate": band / n,
            "direction_brier": brier_dir / n, "total_band_brier": brier_band / n}


def _calibration(rows: list[dict], probs: list[dict]):
    buckets = {axis: {label: {"predicted": 0., "observed": 0}
                      for label in labels} for axis, labels in
               (("direction", ("主胜", "平", "客胜")),
                ("total_band", ("0–1", "2", "3", "4", "5+")))}
    for row, q in zip(rows, probs, strict=True):
        actual = _axis(int(row["actual_home_goals"]), int(row["actual_away_goals"]))
        for axis, values in buckets.items():
            values[actual[axis]]["observed"] += 1
        for (h, a), prob in q.items():
            axes = _axis(h, a)
            for axis, values in buckets.items():
                values[axes[axis]]["predicted"] += prob
    return {axis: {label: {k: v / len(rows) for k, v in counts.items()}
                   for label, counts in values.items()} for axis, values in buckets.items()}


def _paired_by_day(rows: list[dict], qa: list[dict], qb: list[dict], draws=400):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        score = int(row["actual_home_goals"]), int(row["actual_away_goals"])
        diff_log = (math.log(max(qa[i].get(score, 0), 1e-15)) -
                    math.log(max(qb[i].get(score, 0), 1e-15)))
        diff_hit = int(top_score(qa[i]) == score) - int(top_score(qb[i]) == score)
        groups[row["date"]].append((diff_log, diff_hit))
    keys = list(groups)
    rng = random.Random(11)
    log_diffs, hit_diffs = [], []
    for _ in range(draws):
        values = [x for day in rng.choices(keys, k=len(keys)) for x in groups[day]]
        logs = hits = 0.
        for dl, dh in values:
            logs += dl
            hits += dh
        log_diffs.append(logs / len(values))
        hit_diffs.append(hits / len(values))
    return {"log_loss_improvement_95pct_interval": [float(x) for x in np.quantile(log_diffs, [.025, .975])],
            "exact_hit_improvement_95pct_interval": [float(x) for x in np.quantile(hit_diffs, [.025, .975])],
            "bootstrap_unit": "calendar_date", "draws": draws}


def _make_folds(rows: list[dict]):
    days = sorted({r["date"] for r in rows})
    for frac in (.55, .70, .85):
        start = days[int(len(days) * frac)]
        end = days[min(len(days) - 1, int(len(days) * (frac + .10)))]
        past = [r for r in rows if r["date"] < start]
        future = [r for r in rows if start <= r["date"] < end]
        if len(past) >= 100 and future:
            yield past, future


def train_and_backtest(rows: list[dict], source_sha: str, holdout_start: str,
                       quotes: dict | None = None):
    date.fromisoformat(holdout_start)
    past = [r for r in rows if r["date"] < holdout_start]
    hold = [r for r in rows if r["date"] >= holdout_start]
    if len(past) < 100 or len(hold) < 30:
        raise ValueError("DATA_BLOCKED: need >=100 past matches and >=30 later holdout matches")
    quotes = quotes or {}
    valid = {r["match_id"]: market_asof(r, quotes.get(r["match_id"]))
             for r in rows}
    market_count = sum(valid[r["match_id"]][0] is not None for r in past)
    weight = 1.0
    oof_info = {"verified_market_matches": market_count, "folds": 0,
                "selected_weight": 1.0, "reason": "no verified as-of quotes"}
    if market_count >= 30:
        candidates = [0., .25, .5, .75, 1.]
        sums = {w: 0. for w in candidates}
        n = 0
        folds = 0
        for earlier, later in _make_folds(past):
            available = [r for r in later if valid[r["match_id"]][0] is not None]
            if not available:
                continue
            theta = fit_joint(earlier)
            folds += 1
            for r in available:
                qf = joint_distribution(r, theta)
                qm, _ = market_distribution(valid[r["match_id"]][0])
                score = int(r["actual_home_goals"]), int(r["actual_away_goals"])
                for w in candidates:
                    q = w * qf.get(score, 0) + (1 - w) * qm.get(score, 0)
                    sums[w] -= math.log(max(q, 1e-15))
                n += 1
        if n >= 30:
            weight = min(candidates, key=lambda w: (sums[w] / n, -w))
            oof_info = {"verified_market_matches": market_count, "folds": folds,
                        "oof_matches": n, "selected_weight": weight,
                        "mean_log_loss_by_weight": {str(w): sums[w] / n for w in candidates}}
        else:
            oof_info["reason"] = "fewer than 30 verified out-of-fold quotes"
    theta = fit_joint(past)
    dc_theta = fit_dc(past)
    q_joint, q_dc, q_fused = [], [], []
    markets, blocked = 0, defaultdict(int)
    for r in hold:
        qf = joint_distribution(r, theta)
        qd = dc_distribution(r, dc_theta)
        quote, status = valid[r["match_id"]]
        qm = None
        if quote is not None:
            qm, _ = market_distribution(quote)
            markets += 1
        else:
            blocked[status] += 1
        q_joint.append(qf)
        q_dc.append(qd)
        q_fused.append(fuse(qf, qm, weight))
    by_league = {}
    for league in sorted({r["league_code"] for r in hold}):
        ids = [i for i, r in enumerate(hold) if r["league_code"] == league]
        by_league[league] = {"joint": _metrics([hold[i] for i in ids],
                                                 [q_joint[i] for i in ids]),
                              "dc": _metrics([hold[i] for i in ids],
                                               [q_dc[i] for i in ids])}
    artifact = {"schema": "football-score-joint-v1", "trained_through": past[-1]["date"],
                "holdout_start": holdout_start, "source_sha256": source_sha,
                "joint_theta": theta, "dc_theta": dc_theta, "football_weight": weight,
                "market_tuning": oof_info, "train_count": len(past),
                "data_scope": "90-minute scores; prior scores rebuilt in calendar-day groups"}
    report = {"status": "RESEARCH_ONLY", "source_sha256": source_sha,
              "sample_count": len(rows),
              "train_count": len(past), "holdout_count": len(hold),
              "holdout_range": [hold[0]["date"], hold[-1]["date"]],
              "joint": _metrics(hold, q_joint), "dc": _metrics(hold, q_dc),
              "fused": _metrics(hold, q_fused),
              "joint_vs_dc": _paired_by_day(hold, q_joint, q_dc),
              "fused_vs_dc": _paired_by_day(hold, q_fused, q_dc),
              "calibration": {"joint": _calibration(hold, q_joint),
                              "dc": _calibration(hold, q_dc)},
              "by_league": by_league,
              "market_holdout_asof_verified": markets, "market_blocked": dict(blocked),
              "limitations": ["Dataset quote timestamps are absent; embedded prices were excluded.",
                              "Score-derived features are rebuilt by calendar day.",
                              "True prospective freeze and event timeline are not provided."]}
    return artifact, report


def predict(row: dict, artifact: dict):
    if artifact.get("schema") != "football-score-joint-v1":
        raise ValueError("MODEL_UNAVAILABLE: train the joint model first")
    if not all(row.get(k) for k in ("match_id", "home_team", "away_team", "kickoff_utc")):
        raise ValueError("DATA_BLOCKED: match identity, teams and kickoff_utc are required")
    kickoff = _parse_utc(row["kickoff_utc"])
    if kickoff <= datetime.now(UTC):
        raise ValueError("DATA_BLOCKED: freeze must precede kickoff")
    if not row.get("feature_source") or not all(row.get(k) for k in
                                                 ("features_published_at", "features_observed_at")):
        raise ValueError("DATA_BLOCKED: feature source, publication and observation times required")
    published = _parse_utc(row["features_published_at"])
    observed = _parse_utc(row["features_observed_at"])
    if not published <= observed < kickoff:
        raise ValueError("DATA_BLOCKED: feature timestamps outside pre-kickoff window")
    if artifact["trained_through"] >= kickoff.date().isoformat():
        raise ValueError("MODEL_UNAVAILABLE: model trained on/after target match date")
    qf = joint_distribution(row, artifact["joint_theta"])
    quote, market_status = market_asof({"match_id": row["match_id"],
                                       "date": kickoff.date().isoformat(),
                                       "kickoff_utc": row["kickoff_utc"]}, row.get("market"))
    market_info = None
    qm = None
    if quote:
        qm, market_info = market_distribution(quote)
    q = fuse(qf, qm, artifact["football_weight"])
    score = top_score(q)
    stop_state = row.get("stop_state", "UNKNOWN_PREMATCH")
    if stop_state != "UNKNOWN_PREMATCH":
        if not row.get("stop_state_evidence") or not row.get("stop_state_published_at"):
            raise ValueError("DATA_BLOCKED: named stop state requires pre-match evidence")
        if _parse_utc(row["stop_state_published_at"]) >= kickoff:
            raise ValueError("DATA_BLOCKED: stop state evidence was published after kickoff")
    return {"match_id": row["match_id"], "home_team": row["home_team"],
            "away_team": row["away_team"], "kickoff_utc": row["kickoff_utc"],
            "model_source_sha256": artifact["source_sha256"],
            "score": list(score), "score_probability": q[score],
            "axes": _axis(*score), "stop_state": stop_state,
            "stop_state_evidence": row.get("stop_state_evidence"),
            "feature_source": row["feature_source"],
            "features_published_at": row["features_published_at"],
            "features_observed_at": row["features_observed_at"],
            "market_status": market_status, "market_diagnostics": market_info,
            "distribution": {f"{h}-{a}": v for (h, a), v in q.items()},
            "frozen_at": datetime.now(UTC).isoformat()}


def write_new(path: str | Path, data: dict):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
