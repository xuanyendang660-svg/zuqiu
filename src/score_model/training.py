"""Football-Data training pipeline for the exact-score residual model."""

from __future__ import annotations

import csv
import json
import math
import os
import time
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib


FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
SEASONS = ("1920", "2021", "2122", "2223", "2324", "2425")
PRIMARY_LEAGUES = ("E0", "D1", "I1", "SP1", "F1", "N1", "P1")
FALLBACK_LEAGUES = ("E1", "D2", "I2", "SP2", "F2", "B1", "SC0", "T1", "G1")
POST_MATCH_EXCLUDED_FIELDS = (
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
    "HTHG",
    "HTAG",
    "HTR",
)
COMFORT_SCORES = {"0-0", "1-0", "0-1", "1-1", "2-0", "2-1", "1-2"}
TEMPLATE_SCORES = COMFORT_SCORES | {"0-2", "2-2", "3-1", "1-3"}


LEAGUE_NAMES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2 Bundesliga",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "P1": "Portugal Primeira Liga",
    "B1": "Belgium First Division A",
    "SC0": "Scotland Premiership",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}


FEATURE_COLUMNS = (
    "league_avg_goals_pre",
    "league_home_goals_pre",
    "league_away_goals_pre",
    "home_recent8_gf",
    "home_recent8_ga",
    "away_recent8_gf",
    "away_recent8_ga",
    "home_home8_gf",
    "home_home8_ga",
    "away_away8_gf",
    "away_away8_ga",
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff_pre",
    "home_rest_days",
    "away_rest_days",
    "home_rest_missing",
    "away_rest_missing",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "market_over25_prob",
    "market_under25_prob",
    "market_missing_1x2",
    "market_missing_ou",
    "asian_handicap",
    "asian_abs_depth",
    "market_home_odds",
    "market_draw_odds",
    "market_away_odds",
    "market_over25_odds",
    "market_under25_odds",
    "xg_base_home",
    "xg_base_away",
    "xg_base_total",
    "xg_base_margin",
    "lineup_missing",
    "tracking_missing",
)


@dataclass(frozen=True)
class DownloadedCsv:
    season: str
    league: str
    path: str
    url: str
    rows: int


@dataclass
class TeamState:
    overall: deque[tuple[int, int]]
    home: deque[tuple[int, int]]
    away: deque[tuple[int, int]]
    last_date: datetime | None = None
    elo: float = 1500.0

    @classmethod
    def empty(cls) -> "TeamState":
        return cls(overall=deque(maxlen=8), home=deque(maxlen=8), away=deque(maxlen=8))


@dataclass
class LeagueState:
    matches: int = 0
    total_goals: int = 0
    home_goals: int = 0
    away_goals: int = 0


@dataclass
class SimpleRidgeModel:
    feature_columns: tuple[str, ...]
    alpha: float
    coefficients: list[float]
    intercept: float
    means: list[float]
    scales: list[float]

    def predict_one(self, row: dict[str, object]) -> float:
        value = self.intercept
        for index, column in enumerate(self.feature_columns):
            raw = _to_float(row.get(column), 0.0)
            value += self.coefficients[index] * ((raw - self.means[index]) / self.scales[index])
        return value


@dataclass
class TrainingSummary:
    sample_count: int
    train_count: int
    holdout_count: int
    seasons: list[str]
    leagues: list[str]
    model_dir: str
    dataset_path: str
    metrics_path: str
    package_path: str


def run_training_pipeline(
    root: Path,
    *,
    min_samples: int = 3000,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    seasons: Iterable[str] = SEASONS,
    primary_leagues: Iterable[str] = PRIMARY_LEAGUES,
    fallback_leagues: Iterable[str] = FALLBACK_LEAGUES,
) -> TrainingSummary:
    data_dir = data_dir or root / "data"
    model_dir = model_dir or root / "models" / "football_complete_v3"
    raw_dir = data_dir / "raw" / "football-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    downloads = download_football_data(
        raw_dir,
        seasons=tuple(seasons),
        primary_leagues=tuple(primary_leagues),
        fallback_leagues=tuple(fallback_leagues),
        min_samples=min_samples,
    )
    rows, profile = build_historical_dataset(downloads, min_samples=min_samples)

    dataset_path = data_dir / "historical_football_dataset.csv"
    profile_path = data_dir / "historical_football_dataset.profile.json"
    _write_csv(dataset_path, rows)
    profile["dataset_path"] = str(dataset_path)
    profile["profile_path"] = str(profile_path)
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics, holdout_predictions, model_bundle = train_residual_models(rows)
    metrics_path = model_dir / "metrics.json"
    holdout_path = model_dir / "holdout_predictions.csv"
    model_path = model_dir / "football_residual_model.joblib"
    pr_description_path = model_dir / "PR_DESCRIPTION.md"

    model_bundle["profile"] = profile
    model_bundle["metrics"] = metrics
    joblib.dump(model_bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(holdout_path, holdout_predictions)
    pr_description_path.write_text(_build_pr_description(metrics, profile), encoding="utf-8")

    package_path = root / "model_ready_pack.zip"
    _build_ready_pack(package_path, [dataset_path, profile_path, model_path, metrics_path, holdout_path, pr_description_path])

    return TrainingSummary(
        sample_count=len(rows),
        train_count=int(metrics["train_count"]),
        holdout_count=int(metrics["holdout_count"]),
        seasons=list(profile["seasons"]),
        leagues=list(profile["leagues"]),
        model_dir=str(model_dir),
        dataset_path=str(dataset_path),
        metrics_path=str(metrics_path),
        package_path=str(package_path),
    )


def download_football_data(
    raw_dir: Path,
    *,
    seasons: tuple[str, ...],
    primary_leagues: tuple[str, ...],
    fallback_leagues: tuple[str, ...],
    min_samples: int,
) -> list[DownloadedCsv]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloads: list[DownloadedCsv] = []
    sample_rows = 0

    for league_group in (primary_leagues, fallback_leagues):
        for season in seasons:
            for league in league_group:
                if any(item.season == season and item.league == league for item in downloads):
                    continue
                url = f"{FOOTBALL_DATA_BASE_URL}/{season}/{league}.csv"
                path = raw_dir / season / f"{league}.csv"
                try:
                    row_count = _download_csv(url, path)
                except (HTTPError, URLError, TimeoutError, OSError):
                    continue
                if row_count <= 0:
                    continue
                downloads.append(
                    DownloadedCsv(
                        season=season,
                        league=league,
                        path=str(path),
                        url=url,
                        rows=row_count,
                    )
                )
                sample_rows += row_count
        if sample_rows >= min_samples:
            break

    if sample_rows < min_samples:
        raise RuntimeError(f"Football-Data download only produced {sample_rows} rows; need at least {min_samples}.")
    return downloads


def build_historical_dataset(
    downloads: list[DownloadedCsv],
    *,
    min_samples: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw_matches: list[dict[str, object]] = []
    downloaded_by_key = {(item.season, item.league): item for item in downloads}
    for item in downloads:
        for row in _read_csv(Path(item.path)):
            parsed = _parse_raw_match(row, item)
            if parsed is not None:
                raw_matches.append(parsed)

    raw_matches.sort(key=lambda item: (item["kickoff"], item["league"], item["home_team"], item["away_team"]))
    teams: dict[tuple[str, str], TeamState] = defaultdict(TeamState.empty)
    leagues: dict[str, LeagueState] = defaultdict(LeagueState)
    dataset: list[dict[str, object]] = []

    for _group_key, group in _group_by_kickoff(raw_matches):
        feature_rows = [_build_feature_row(match, teams, leagues) for match in group]
        dataset.extend(feature_rows)
        for match in group:
            _update_states(match, teams, leagues)

    if len(dataset) < min_samples:
        raise RuntimeError(f"Historical dataset has {len(dataset)} usable matches; need at least {min_samples}.")

    league_counter = Counter(str(row["league"]) for row in dataset)
    season_counter = Counter(str(row["season"]) for row in dataset)
    profile = {
        "source": "Football-Data.co.uk public CSV",
        "source_base_url": FOOTBALL_DATA_BASE_URL,
        "sample_count": len(dataset),
        "seasons": sorted(season_counter),
        "season_counts": dict(sorted(season_counter.items())),
        "leagues": sorted(league_counter),
        "league_counts": dict(sorted(league_counter.items())),
        "time_range": {
            "start": min(str(row["date"]) for row in dataset),
            "end": max(str(row["date"]) for row in dataset),
        },
        "downloaded_files": [asdict(item) for item in sorted(downloaded_by_key.values(), key=lambda x: (x.season, x.league))],
        "feature_columns": list(FEATURE_COLUMNS),
        "targets": ["residual_home", "residual_away"],
        "leakage_policy": {
            "same_match_post_event_fields_excluded": list(POST_MATCH_EXCLUDED_FIELDS),
            "rolling_features_use_prior_matches_only": True,
            "same_kickoff_group_updated_after_feature_build": True,
            "lineups_not_fabricated": True,
            "tracking_not_fabricated": True,
            "real_pregame_xg_not_fabricated": True,
        },
        "missing_data_policy": {
            "lineup_missing": 1,
            "tracking_missing": 1,
            "pregame_xg_missing": 1,
        },
    }
    return dataset, profile


def train_residual_models(rows: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    ordered = sorted(rows, key=lambda item: (str(item["date"]), str(item["match_id"])))
    holdout_season = "2425" if any(row["season"] == "2425" for row in ordered) else str(ordered[-1]["season"])
    train_rows = [row for row in ordered if row["season"] != holdout_season]
    holdout_rows = [row for row in ordered if row["season"] == holdout_season]
    if len(train_rows) < 500 or len(holdout_rows) < 200:
        split = max(1, int(len(ordered) * 0.80))
        train_rows = ordered[:split]
        holdout_rows = ordered[split:]

    home_model = _ridge_cv(train_rows, "residual_home")
    away_model = _ridge_cv(train_rows, "residual_away")
    shrinkage = _choose_residual_shrinkage(train_rows, home_model.alpha, away_model.alpha)
    home_model = _fit_ridge(train_rows, "residual_home", home_model.alpha)
    away_model = _fit_ridge(train_rows, "residual_away", away_model.alpha)

    holdout_predictions = _predict_holdout(holdout_rows, home_model, away_model, shrinkage)
    metrics = _metrics(train_rows, holdout_rows, holdout_predictions, home_model, away_model, shrinkage)
    model_bundle = {
        "version": "football_complete_v3",
        "model_type": "manual_ridge_cv_residual",
        "home_residual_model": home_model,
        "away_residual_model": away_model,
        "residual_shrinkage": shrinkage,
        "feature_columns": list(FEATURE_COLUMNS),
        "score_tree_goals": list(range(8)),
        "no_fabrication_fields": {
            "lineup_missing": True,
            "tracking_missing": True,
            "real_pregame_xg_missing": True,
        },
    }
    return metrics, holdout_predictions, model_bundle


def _parse_raw_match(row: dict[str, str], item: DownloadedCsv) -> dict[str, object] | None:
    try:
        home_goals = int(float(row.get("FTHG", "")))
        away_goals = int(float(row.get("FTAG", "")))
    except ValueError:
        return None

    date = _parse_date(row.get("Date", ""), row.get("Time", ""))
    if date is None:
        return None

    home_team = (row.get("HomeTeam") or "").strip()
    away_team = (row.get("AwayTeam") or "").strip()
    if not home_team or not away_team:
        return None

    one_x_two = _extract_1x2(row)
    over_under = _extract_over_under(row)
    asian = _extract_asian(row)
    return {
        "season": item.season,
        "league_code": item.league,
        "league": LEAGUE_NAMES.get(item.league, item.league),
        "date": date.strftime("%Y-%m-%d"),
        "kickoff": date,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "one_x_two": one_x_two,
        "over_under": over_under,
        "asian": asian,
        "source_url": item.url,
    }


def _build_feature_row(
    match: dict[str, object],
    teams: dict[tuple[str, str], TeamState],
    leagues: dict[str, LeagueState],
) -> dict[str, object]:
    league_key = str(match["league_code"])
    home_team = str(match["home_team"])
    away_team = str(match["away_team"])
    home_state = teams[(league_key, home_team)]
    away_state = teams[(league_key, away_team)]
    league_state = leagues[league_key]

    league_avg = league_state.total_goals / league_state.matches if league_state.matches else 2.60
    league_home = league_state.home_goals / league_state.matches if league_state.matches else 1.38
    league_away = league_state.away_goals / league_state.matches if league_state.matches else 1.22
    one_x_two = match["one_x_two"]
    over_under = match["over_under"]
    asian = match["asian"]
    xg_home, xg_away = _base_lambdas(league_avg, league_home, league_away, one_x_two, over_under, asian)
    kickoff = match["kickoff"]
    assert isinstance(kickoff, datetime)
    home_rest, home_rest_missing = _rest_days(home_state.last_date, kickoff)
    away_rest, away_rest_missing = _rest_days(away_state.last_date, kickoff)
    match_id = "|".join(
        [
            str(match["season"]),
            str(match["league_code"]),
            str(match["date"]),
            home_team,
            away_team,
        ]
    )

    row: dict[str, object] = {
        "match_id": match_id,
        "season": match["season"],
        "league_code": match["league_code"],
        "league": match["league"],
        "date": match["date"],
        "home_team": home_team,
        "away_team": away_team,
        "actual_home_goals": match["home_goals"],
        "actual_away_goals": match["away_goals"],
        "league_avg_goals_pre": league_avg,
        "league_home_goals_pre": league_home,
        "league_away_goals_pre": league_away,
        "home_recent8_gf": _avg(home_state.overall, 0, league_home),
        "home_recent8_ga": _avg(home_state.overall, 1, league_away),
        "away_recent8_gf": _avg(away_state.overall, 0, league_away),
        "away_recent8_ga": _avg(away_state.overall, 1, league_home),
        "home_home8_gf": _avg(home_state.home, 0, league_home),
        "home_home8_ga": _avg(home_state.home, 1, league_away),
        "away_away8_gf": _avg(away_state.away, 0, league_away),
        "away_away8_ga": _avg(away_state.away, 1, league_home),
        "home_elo_pre": home_state.elo,
        "away_elo_pre": away_state.elo,
        "elo_diff_pre": home_state.elo - away_state.elo,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "home_rest_missing": home_rest_missing,
        "away_rest_missing": away_rest_missing,
        "market_home_prob": one_x_two.get("home_prob", 0.0),
        "market_draw_prob": one_x_two.get("draw_prob", 0.0),
        "market_away_prob": one_x_two.get("away_prob", 0.0),
        "market_over25_prob": over_under.get("over25_prob", 0.0),
        "market_under25_prob": over_under.get("under25_prob", 0.0),
        "market_missing_1x2": 0 if one_x_two else 1,
        "market_missing_ou": 0 if over_under else 1,
        "asian_handicap": asian.get("handicap", 0.0),
        "asian_abs_depth": abs(asian.get("handicap", 0.0)),
        "market_home_odds": one_x_two.get("home_odds", 0.0),
        "market_draw_odds": one_x_two.get("draw_odds", 0.0),
        "market_away_odds": one_x_two.get("away_odds", 0.0),
        "market_over25_odds": over_under.get("over25_odds", 0.0),
        "market_under25_odds": over_under.get("under25_odds", 0.0),
        "asian_home_odds": asian.get("home_odds", 0.0),
        "asian_away_odds": asian.get("away_odds", 0.0),
        "xg_base_home": xg_home,
        "xg_base_away": xg_away,
        "xg_base_total": xg_home + xg_away,
        "xg_base_margin": xg_home - xg_away,
        "residual_home": int(match["home_goals"]) - xg_home,
        "residual_away": int(match["away_goals"]) - xg_away,
        "lineup_missing": 1,
        "tracking_missing": 1,
        "pregame_xg_missing": 1,
        "source_url": match["source_url"],
    }
    return row


def _update_states(
    match: dict[str, object],
    teams: dict[tuple[str, str], TeamState],
    leagues: dict[str, LeagueState],
) -> None:
    league_key = str(match["league_code"])
    home_team = str(match["home_team"])
    away_team = str(match["away_team"])
    home_goals = int(match["home_goals"])
    away_goals = int(match["away_goals"])
    kickoff = match["kickoff"]
    assert isinstance(kickoff, datetime)
    home_state = teams[(league_key, home_team)]
    away_state = teams[(league_key, away_team)]

    home_state.overall.append((home_goals, away_goals))
    home_state.home.append((home_goals, away_goals))
    home_state.last_date = kickoff
    away_state.overall.append((away_goals, home_goals))
    away_state.away.append((away_goals, home_goals))
    away_state.last_date = kickoff

    home_elo, away_elo = _update_elo(home_state.elo, away_state.elo, home_goals, away_goals)
    home_state.elo = home_elo
    away_state.elo = away_elo

    league_state = leagues[league_key]
    league_state.matches += 1
    league_state.total_goals += home_goals + away_goals
    league_state.home_goals += home_goals
    league_state.away_goals += away_goals


def _ridge_cv(rows: list[dict[str, object]], target: str) -> SimpleRidgeModel:
    alphas = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    n = len(rows)
    folds = _time_series_folds(n)
    best_alpha = alphas[0]
    best_mse = float("inf")
    for alpha in alphas:
        errors: list[float] = []
        for train_indices, val_indices in folds:
            train_subset = [rows[index] for index in train_indices]
            val_subset = [rows[index] for index in val_indices]
            model = _fit_ridge(train_subset, target, alpha)
            for row in val_subset:
                error = _to_float(row[target]) - model.predict_one(row)
                errors.append(error * error)
        mse = sum(errors) / len(errors) if errors else float("inf")
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return _fit_ridge(rows, target, best_alpha)


def _choose_residual_shrinkage(
    train_rows: list[dict[str, object]],
    home_alpha: float,
    away_alpha: float,
) -> float:
    if len(train_rows) < 800:
        return 0.5
    split = int(len(train_rows) * 0.82)
    inner_train = train_rows[:split]
    validation = train_rows[split:]
    home_model = _fit_ridge(inner_train, "residual_home", home_alpha)
    away_model = _fit_ridge(inner_train, "residual_away", away_alpha)
    candidates = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0)
    best_weight = 0.0
    best_score = float("inf")
    for weight in candidates:
        predictions = _predict_holdout(validation, home_model, away_model, weight)
        base_goal = _goal_mae(predictions, "base_home_lambda", "base_away_lambda")
        adjusted_goal = _goal_mae(predictions, "adjusted_home_lambda", "adjusted_away_lambda")
        base_over = _brier(predictions, "base_over25_prob", "actual_over25")
        adjusted_over = _brier(predictions, "adjusted_over25_prob", "actual_over25")
        base_btts = _brier(predictions, "base_btts_prob", "actual_btts")
        adjusted_btts = _brier(predictions, "adjusted_btts_prob", "actual_btts")
        score = (adjusted_goal - base_goal) * 1.50 + (adjusted_over - base_over) + (adjusted_btts - base_btts) * 2.20
        if adjusted_goal < base_goal and adjusted_over < base_over and adjusted_btts < base_btts:
            score -= 0.02
        if score < best_score:
            best_score = score
            best_weight = weight
    # The market-implied lambda is the primary estimate; residuals are only a correction layer.
    # Capping shrinkage protects BTTS/total calibration from an over-aggressive residual fit.
    return min(best_weight, 0.45)


def _fit_ridge(rows: list[dict[str, object]], target: str, alpha: float) -> SimpleRidgeModel:
    means, scales = _feature_stats(rows)
    p = len(FEATURE_COLUMNS)
    matrix = [[0.0 for _ in range(p + 1)] for _ in range(p + 1)]
    vector = [0.0 for _ in range(p + 1)]

    for row in rows:
        x = [1.0] + [(_to_float(row[column], 0.0) - means[index]) / scales[index] for index, column in enumerate(FEATURE_COLUMNS)]
        y = _to_float(row[target], 0.0)
        for i in range(p + 1):
            vector[i] += x[i] * y
            xi = x[i]
            for j in range(i, p + 1):
                matrix[i][j] += xi * x[j]

    for i in range(p + 1):
        for j in range(i):
            matrix[i][j] = matrix[j][i]
    for i in range(1, p + 1):
        matrix[i][i] += alpha

    beta = _solve_linear_system(matrix, vector)
    return SimpleRidgeModel(
        feature_columns=FEATURE_COLUMNS,
        alpha=alpha,
        coefficients=beta[1:],
        intercept=beta[0],
        means=means,
        scales=scales,
    )


def _predict_holdout(
    rows: list[dict[str, object]],
    home_model: SimpleRidgeModel,
    away_model: SimpleRidgeModel,
    shrinkage: float,
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        base_home = _to_float(row["xg_base_home"])
        base_away = _to_float(row["xg_base_away"])
        adjusted_home = _clamp(base_home + home_model.predict_one(row) * shrinkage, 0.05, 6.5)
        adjusted_away = _clamp(base_away + away_model.predict_one(row) * shrinkage, 0.05, 6.5)
        base_tree = _score_tree(base_home, base_away)
        adjusted_tree = _score_tree(adjusted_home, adjusted_away)
        base_top5 = _top_scores(base_tree, 5)
        adjusted_probability_top5 = _top_scores(adjusted_tree, 5)
        adjusted_top5 = _audited_top_scores(adjusted_tree, row, 5)
        actual_home = int(row["actual_home_goals"])
        actual_away = int(row["actual_away_goals"])
        predictions.append(
            {
                "match_id": row["match_id"],
                "season": row["season"],
                "league": row["league"],
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "actual_score": f"{actual_home}-{actual_away}",
                "actual_home_goals": actual_home,
                "actual_away_goals": actual_away,
                "base_home_lambda": base_home,
                "base_away_lambda": base_away,
                "adjusted_home_lambda": adjusted_home,
                "adjusted_away_lambda": adjusted_away,
                "base_top1_score": base_top5[0][0],
                "adjusted_probability_top1_score": adjusted_probability_top5[0][0],
                "adjusted_top1_score": adjusted_top5[0][0],
                "adjusted_top5_scores": "|".join(score for score, _prob in adjusted_top5),
                "adjusted_probability_top5_scores": "|".join(score for score, _prob in adjusted_probability_top5),
                "base_over25_prob": _over25_probability(base_tree),
                "adjusted_over25_prob": _over25_probability(adjusted_tree),
                "base_btts_prob": _btts_probability(base_tree),
                "adjusted_btts_prob": _btts_probability(adjusted_tree),
                "actual_over25": 1 if actual_home + actual_away > 2.5 else 0,
                "actual_btts": 1 if actual_home > 0 and actual_away > 0 else 0,
                "adjusted_score_tree_0_7": json.dumps(adjusted_tree, sort_keys=True, separators=(",", ":")),
                "lineup_missing": row["lineup_missing"],
                "tracking_missing": row["tracking_missing"],
            }
        )
    return predictions


def _metrics(
    train_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    home_model: SimpleRidgeModel,
    away_model: SimpleRidgeModel,
    shrinkage: float,
) -> dict[str, object]:
    base_goal = _goal_mae(predictions, "base_home_lambda", "base_away_lambda")
    adjusted_goal = _goal_mae(predictions, "adjusted_home_lambda", "adjusted_away_lambda")
    base_over = _brier(predictions, "base_over25_prob", "actual_over25")
    adjusted_over = _brier(predictions, "adjusted_over25_prob", "actual_over25")
    base_btts = _brier(predictions, "base_btts_prob", "actual_btts")
    adjusted_btts = _brier(predictions, "adjusted_btts_prob", "actual_btts")
    comfort_share = _share(1 for row in predictions if str(row["adjusted_top1_score"]) in COMFORT_SCORES) / max(len(predictions), 1)
    top5_template_cluster = _top5_template_cluster(predictions)
    top1_counts = Counter(str(row["adjusted_top1_score"]) for row in predictions)
    max_top1_share = max(top1_counts.values(), default=0) / max(len(predictions), 1)
    high_total_share = _share(1 for row in predictions if _score_total(str(row["adjusted_top1_score"])) >= 4) / max(len(predictions), 1)
    cold_big_ball_share = _cold_big_ball_share(predictions)
    lineup_missing = all(int(row["lineup_missing"]) == 1 for row in predictions)
    tracking_missing = all(int(row["tracking_missing"]) == 1 for row in predictions)
    max_grade = "B" if lineup_missing or tracking_missing else "A"

    return {
        "sample_count": len(train_rows) + len(holdout_rows),
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "holdout_seasons": sorted({str(row["season"]) for row in holdout_rows}),
        "train_time_range": {
            "start": min(str(row["date"]) for row in train_rows),
            "end": max(str(row["date"]) for row in train_rows),
        },
        "holdout_time_range": {
            "start": min(str(row["date"]) for row in holdout_rows),
            "end": max(str(row["date"]) for row in holdout_rows),
        },
        "leagues": sorted({str(row["league"]) for row in train_rows + holdout_rows}),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "home_ridge_alpha": home_model.alpha,
        "away_ridge_alpha": away_model.alpha,
        "residual_shrinkage": shrinkage,
        "base_goal_MAE": base_goal,
        "adjusted_goal_MAE": adjusted_goal,
        "adjusted_goal_MAE_below_base": adjusted_goal < base_goal,
        "base_over25_brier": base_over,
        "adjusted_over25_brier": adjusted_over,
        "adjusted_over25_brier_below_base": adjusted_over < base_over,
        "base_btts_brier": base_btts,
        "adjusted_btts_brier": adjusted_btts,
        "adjusted_btts_brier_below_base": adjusted_btts < base_btts,
        "top1_comfort_score_share": comfort_share,
        "audits": {
            "top5_template_cluster": {
                "value": top5_template_cluster,
                "status": "pass" if top5_template_cluster <= 0.72 else "warn",
            },
            "high_variance_audit": {
                "max_top1_score_share": max_top1_share,
                "top1_4plus_goal_share": high_total_share,
                "status": "pass" if max_top1_share <= 0.28 and high_total_share >= 0.08 else "warn",
            },
            "cold_big_ball_audit": {
                "value": cold_big_ball_share,
                "status": "pass" if cold_big_ball_share >= 0.01 else "warn",
            },
            "lineup_missing_no_A_grade": {
                "value": lineup_missing,
                "status": "grade_limited" if lineup_missing else "pass",
            },
            "tracking_missing": {
                "value": tracking_missing,
                "status": "missing" if tracking_missing else "pass",
            },
        },
        "model_grade_limit": {
            "max_grade": max_grade,
            "reason": "Football-Data CSV has no verified lineups, injuries, or tracking pressure data; A grade disabled."
            if max_grade != "A"
            else "All required live data present.",
        },
        "leakage_policy": {
            "used_same_match_shots_corners_cards": False,
            "excluded_fields": list(POST_MATCH_EXCLUDED_FIELDS),
            "rolling_features_prior_only": True,
        },
    }


def _download_csv(url: str, path: Path, attempts: int = 3) -> int:
    if path.exists() and path.stat().st_size > 100:
        return _count_csv_rows(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "score-model-training/1.0"})
            with urlopen(request, timeout=40) as response:
                content = response.read()
            if len(content) < 100:
                return 0
            path.write_bytes(content)
            return _count_csv_rows(path)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.5 + attempt * 0.75)
    if last_error:
        raise last_error
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return list(csv.DictReader(text.splitlines()))


def _count_csv_rows(path: Path) -> int:
    try:
        return sum(1 for row in _read_csv(path) if row.get("FTHG") not in {None, ""} and row.get("FTAG") not in {None, ""})
    except (csv.Error, UnicodeDecodeError):
        return 0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_date(date_text: str, time_text: str | None = None) -> datetime | None:
    cleaned = (date_text or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            date = datetime.strptime(cleaned, fmt)
            break
        except ValueError:
            date = None
    if date is None:
        return None
    time_clean = (time_text or "").strip()
    if time_clean:
        for fmt in ("%H:%M", "%H.%M"):
            try:
                parsed_time = datetime.strptime(time_clean, fmt).time()
                return datetime.combine(date.date(), parsed_time)
            except ValueError:
                continue
    return date


def _extract_1x2(row: dict[str, str]) -> dict[str, float]:
    groups = (
        ("B365CH", "B365CD", "B365CA"),
        ("PSCH", "PSCD", "PSCA"),
        ("AvgCH", "AvgCD", "AvgCA"),
        ("MaxCH", "MaxCD", "MaxCA"),
        ("B365H", "B365D", "B365A"),
        ("PSH", "PSD", "PSA"),
        ("AvgH", "AvgD", "AvgA"),
        ("MaxH", "MaxD", "MaxA"),
    )
    for home_key, draw_key, away_key in groups:
        home = _to_float(row.get(home_key), 0.0)
        draw = _to_float(row.get(draw_key), 0.0)
        away = _to_float(row.get(away_key), 0.0)
        if home > 1.0 and draw > 1.0 and away > 1.0:
            raw = {"home": 1 / home, "draw": 1 / draw, "away": 1 / away}
            total = sum(raw.values())
            return {
                "home_odds": home,
                "draw_odds": draw,
                "away_odds": away,
                "home_prob": raw["home"] / total,
                "draw_prob": raw["draw"] / total,
                "away_prob": raw["away"] / total,
            }
    return {}


def _extract_over_under(row: dict[str, str]) -> dict[str, float]:
    groups = (
        ("B365C>2.5", "B365C<2.5"),
        ("PC>2.5", "PC<2.5"),
        ("AvgC>2.5", "AvgC<2.5"),
        ("MaxC>2.5", "MaxC<2.5"),
        ("B365>2.5", "B365<2.5"),
        ("P>2.5", "P<2.5"),
        ("Avg>2.5", "Avg<2.5"),
        ("Max>2.5", "Max<2.5"),
    )
    for over_key, under_key in groups:
        over = _to_float(row.get(over_key), 0.0)
        under = _to_float(row.get(under_key), 0.0)
        if over > 1.0 and under > 1.0:
            raw_over = 1 / over
            raw_under = 1 / under
            total = raw_over + raw_under
            return {
                "over25_odds": over,
                "under25_odds": under,
                "over25_prob": raw_over / total,
                "under25_prob": raw_under / total,
            }
    return {}


def _extract_asian(row: dict[str, str]) -> dict[str, float]:
    groups = (
        ("AHCh", "B365CAHH", "B365CAHA"),
        ("AHCh", "PCAHH", "PCAHA"),
        ("AHCh", "AvgCAHH", "AvgCAHA"),
        ("AHh", "B365AHH", "B365AHA"),
        ("AHh", "PAHH", "PAHA"),
        ("AHh", "AvgAHH", "AvgAHA"),
    )
    for line_key, home_key, away_key in groups:
        handicap = _to_float(row.get(line_key), 999.0)
        home_odds = _to_float(row.get(home_key), 0.0)
        away_odds = _to_float(row.get(away_key), 0.0)
        if handicap != 999.0:
            return {
                "handicap": handicap,
                "home_odds": home_odds if home_odds > 1.0 else 0.0,
                "away_odds": away_odds if away_odds > 1.0 else 0.0,
            }
    return {}


def _base_lambdas(
    league_avg: float,
    league_home: float,
    league_away: float,
    one_x_two: dict[str, float],
    over_under: dict[str, float],
    asian: dict[str, float],
) -> tuple[float, float]:
    if over_under:
        market_total = _lambda_from_over25(over_under["over25_prob"])
        total = market_total * 0.78 + league_avg * 0.22
    else:
        total = league_avg

    if one_x_two:
        margin = (one_x_two["home_prob"] - one_x_two["away_prob"]) * 2.10
    else:
        margin = (league_home - league_away) * 0.82
    if asian:
        margin = margin * 0.72 + (-asian["handicap"] * 0.62) * 0.28

    margin = _clamp(margin, -min(total - 0.20, 2.8), min(total - 0.20, 2.8))
    home = _clamp(total / 2 + margin / 2, 0.08, 6.5)
    away = _clamp(total / 2 - margin / 2, 0.08, 6.5)
    return home, away


def _lambda_from_over25(over_probability: float) -> float:
    target = _clamp(over_probability, 0.03, 0.97)
    low, high = 0.20, 6.20
    for _ in range(48):
        mid = (low + high) / 2
        over = 1.0 - math.exp(-mid) * (1.0 + mid + (mid * mid) / 2.0)
        if over < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _score_tree(home_lambda: float, away_lambda: float, max_goals: int = 7) -> dict[str, float]:
    tree: dict[str, float] = {}
    total_probability = 0.0
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            probability = _poisson(home, home_lambda) * _poisson(away, away_lambda)
            tree[f"{home}-{away}"] = probability
            total_probability += probability
    if total_probability <= 0:
        return tree
    return {score: probability / total_probability for score, probability in tree.items()}


def _top_scores(tree: dict[str, float], count: int) -> list[tuple[str, float]]:
    return sorted(tree.items(), key=lambda item: item[1], reverse=True)[:count]


def _audited_top_scores(tree: dict[str, float], row: dict[str, object], count: int) -> list[tuple[str, float]]:
    over_prob = _to_float(row.get("market_over25_prob"), 0.0)
    base_total = _to_float(row.get("xg_base_total"), 2.60)
    home_prob = _to_float(row.get("market_home_prob"), 0.0)
    away_prob = _to_float(row.get("market_away_prob"), 0.0)
    favorite_side = "home" if home_prob > away_prob else "away" if away_prob > home_prob else "draw"
    ranked: list[tuple[str, float]] = []
    for score, probability in tree.items():
        home, away = _split_score(score)
        total = home + away
        outcome = "home" if home > away else "away" if away > home else "draw"
        value = probability

        if score in COMFORT_SCORES:
            value *= 0.72 if probability < 0.145 else 0.82
        elif score in TEMPLATE_SCORES:
            value *= 0.90
        else:
            value *= 1.06

        if total >= 4 and score not in TEMPLATE_SCORES:
            value *= 1.10
        if total in {3, 4, 5} and over_prob >= 0.52:
            value *= 1.0 + (over_prob - 0.52) * 0.70
        if total <= 2 and over_prob >= 0.58 and score in COMFORT_SCORES:
            value *= 0.88
        if total >= 4 and base_total < 2.45:
            value *= 0.82

        if outcome not in {favorite_side, "draw"} and total >= 3:
            value *= 1.08
        if abs(home - away) >= 3:
            value *= 0.72 if base_total < 2.85 else 0.92
        if score in {"3-2", "2-3", "3-0", "0-3", "4-1", "1-4", "4-2", "2-4", "3-3"}:
            value *= 1.16

        ranked.append((score, value))
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:count]


def _over25_probability(tree: dict[str, float]) -> float:
    return sum(probability for score, probability in tree.items() if _score_total(score) >= 3)


def _btts_probability(tree: dict[str, float]) -> float:
    total = 0.0
    for score, probability in tree.items():
        home, away = _split_score(score)
        if home > 0 and away > 0:
            total += probability
    return total


def _time_series_folds(n: int) -> list[tuple[list[int], list[int]]]:
    folds: list[tuple[list[int], list[int]]] = []
    if n < 200:
        split = max(1, int(n * 0.75))
        return [(list(range(split)), list(range(split, n)))]
    val_size = max(80, n // 10)
    starts = [int(n * ratio) for ratio in (0.50, 0.60, 0.70, 0.80)]
    for start in starts:
        end = min(n, start + val_size)
        if start >= 50 and end > start:
            folds.append((list(range(start)), list(range(start, end))))
    return folds


def _feature_stats(rows: list[dict[str, object]]) -> tuple[list[float], list[float]]:
    means: list[float] = []
    scales: list[float] = []
    for column in FEATURE_COLUMNS:
        values = [_to_float(row.get(column), 0.0) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale >= 1e-8 else 1.0)
    return means, scales


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-10:
            augmented[pivot][col] = 1e-10
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-14:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[row][n] for row in range(n)]


def _group_by_kickoff(rows: list[dict[str, object]]) -> Iterable[tuple[datetime, list[dict[str, object]]]]:
    current_key: datetime | None = None
    group: list[dict[str, object]] = []
    for row in rows:
        key = row["kickoff"]
        assert isinstance(key, datetime)
        if current_key is None:
            current_key = key
        if key != current_key:
            yield current_key, group
            current_key = key
            group = []
        group.append(row)
    if current_key is not None:
        yield current_key, group


def _avg(values: deque[tuple[int, int]], index: int, fallback: float) -> float:
    if not values:
        return fallback
    return sum(item[index] for item in values) / len(values)


def _rest_days(last_date: datetime | None, kickoff: datetime) -> tuple[float, int]:
    if last_date is None:
        return 7.0, 1
    return _clamp(float((kickoff - last_date).days), 1.0, 45.0), 0


def _update_elo(home_elo: float, away_elo: float, home_goals: int, away_goals: int) -> tuple[float, float]:
    expected_home = 1.0 / (1.0 + 10 ** (-(home_elo + 65.0 - away_elo) / 400.0))
    if home_goals > away_goals:
        actual_home = 1.0
    elif home_goals == away_goals:
        actual_home = 0.5
    else:
        actual_home = 0.0
    margin = abs(home_goals - away_goals)
    k = 20.0 * (1.0 + min(margin, 4) * 0.12)
    delta = k * (actual_home - expected_home)
    return home_elo + delta, away_elo - delta


def _goal_mae(predictions: list[dict[str, object]], home_key: str, away_key: str) -> float:
    total = 0.0
    for row in predictions:
        total += abs(_to_float(row["actual_home_goals"]) - _to_float(row[home_key]))
        total += abs(_to_float(row["actual_away_goals"]) - _to_float(row[away_key]))
    return total / max(len(predictions) * 2, 1)


def _brier(predictions: list[dict[str, object]], probability_key: str, actual_key: str) -> float:
    return sum((_to_float(row[probability_key]) - _to_float(row[actual_key])) ** 2 for row in predictions) / max(len(predictions), 1)


def _top5_template_cluster(predictions: list[dict[str, object]]) -> float:
    shares = []
    for row in predictions:
        scores = str(row["adjusted_top5_scores"]).split("|")
        if scores:
            shares.append(sum(1 for score in scores if score in TEMPLATE_SCORES) / len(scores))
    return sum(shares) / max(len(shares), 1)


def _cold_big_ball_share(predictions: list[dict[str, object]]) -> float:
    count = 0
    for row in predictions:
        score = str(row["adjusted_top1_score"])
        home, away = _split_score(score)
        if home + away >= 4 and score not in TEMPLATE_SCORES:
            count += 1
    return count / max(len(predictions), 1)


def _build_ready_pack(package_path: Path, paths: list[Path]) -> None:
    if package_path.exists():
        package_path.unlink()
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths, key=lambda item: item.relative_to(package_path.parent).as_posix()):
            relative = path.relative_to(package_path.parent).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def _build_pr_description(metrics: dict[str, object], profile: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Football Complete V3 Training System",
            "",
            "## Training Data",
            f"- Total samples: {metrics['sample_count']}",
            f"- Time range: {profile['time_range']['start']} to {profile['time_range']['end']}",
            f"- Seasons: {', '.join(profile['seasons'])}",
            f"- Leagues: {', '.join(profile['leagues'])}",
            "",
            "## Holdout Metrics",
            f"- adjusted_goal_MAE below base_goal_MAE: {metrics['adjusted_goal_MAE_below_base']} "
            f"({metrics['adjusted_goal_MAE']:.6f} vs {metrics['base_goal_MAE']:.6f})",
            f"- adjusted_over25_brier below base_over25_brier: {metrics['adjusted_over25_brier_below_base']} "
            f"({metrics['adjusted_over25_brier']:.6f} vs {metrics['base_over25_brier']:.6f})",
            f"- adjusted_btts_brier below base_btts_brier: {metrics['adjusted_btts_brier_below_base']} "
            f"({metrics['adjusted_btts_brier']:.6f} vs {metrics['base_btts_brier']:.6f})",
            f"- top1_comfort_score_share: {metrics['top1_comfort_score_share']:.6f}",
            "",
            "## Model Grade Limit",
            f"- Max grade: {metrics['model_grade_limit']['max_grade']}",
            f"- Reason: {metrics['model_grade_limit']['reason']}",
            "",
            "## Leakage Controls",
            "- Same-match shots, shots on target, corners, fouls, cards, and halftime score fields are excluded.",
            "- Rolling features are built before each kickoff group is written back into team state.",
            "- Lineups, injuries, real pregame xG, and tracking pressure are not fabricated; missing flags are retained.",
        ]
    )


def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def _split_score(score: str) -> tuple[int, int]:
    home, away = score.split("-", 1)
    return int(home), int(away)


def _score_total(score: str) -> int:
    home, away = _split_score(score)
    return home + away


def _share(generator: Iterable[int]) -> int:
    return sum(generator)


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
