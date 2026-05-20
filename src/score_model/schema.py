"""Data contracts for the exact-score model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MarketInput:
    """Market view used for calibration and wrong-price detection."""

    exact_score_odds: dict[str, float] = field(default_factory=dict)
    one_x_two_odds: dict[str, float] = field(default_factory=dict)
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None
    asian_handicap: float | None = None
    total_goals_line: float | None = None
    home_money_heat: float = 0.0
    draw_money_heat: float = 0.0
    away_money_heat: float = 0.0
    favorite_money_heat: float = 0.0
    over_money_heat: float = 0.0
    under_money_heat: float = 0.0
    bookmaker_trap_risk: float = 0.0
    confidence: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MarketInput":
        data = data or {}
        return cls(
            exact_score_odds={str(k): float(v) for k, v in data.get("exact_score_odds", {}).items()},
            one_x_two_odds={str(k): float(v) for k, v in data.get("one_x_two_odds", {}).items()},
            expected_home_goals=_optional_float(data.get("expected_home_goals")),
            expected_away_goals=_optional_float(data.get("expected_away_goals")),
            asian_handicap=_optional_float(data.get("asian_handicap")),
            total_goals_line=_optional_float(data.get("total_goals_line")),
            home_money_heat=float(data.get("home_money_heat", 0.0)),
            draw_money_heat=float(data.get("draw_money_heat", 0.0)),
            away_money_heat=float(data.get("away_money_heat", 0.0)),
            favorite_money_heat=float(data.get("favorite_money_heat", 0.0)),
            over_money_heat=float(data.get("over_money_heat", 0.0)),
            under_money_heat=float(data.get("under_money_heat", 0.0)),
            bookmaker_trap_risk=float(data.get("bookmaker_trap_risk", 0.0)),
            confidence=float(data.get("confidence", 0.5)),
        )


@dataclass(frozen=True)
class MatchInput:
    """Single-match features.

    Most numeric features are intentionally scaled around neutral values:
    strengths near 1.0, contextual edges near -1.0..1.0, and penalties near 0.0..1.0.
    """

    match_id: str
    league: str
    home_team: str
    away_team: str
    league_avg_goals: float = 2.62
    league_home_advantage: float = 0.17
    league_draw_bias: float = 0.0
    league_volatility: float = 0.5
    home_attack: float = 1.0
    away_attack: float = 1.0
    home_defense: float = 1.0
    away_defense: float = 1.0
    recent_home_xg: float | None = None
    recent_away_xg: float | None = None
    recent_home_xga: float | None = None
    recent_away_xga: float | None = None
    home_form: float = 0.0
    away_form: float = 0.0
    home_absence_impact: float = 0.0
    away_absence_impact: float = 0.0
    home_rotation_risk: float = 0.0
    away_rotation_risk: float = 0.0
    home_rest_edge: float = 0.0
    away_rest_edge: float = 0.0
    home_motivation: float = 0.0
    away_motivation: float = 0.0
    home_rank: int | None = None
    away_rank: int | None = None
    home_points: int | None = None
    away_points: int | None = None
    home_goal_difference: int | None = None
    away_goal_difference: int | None = None
    home_table_pressure: float = 0.0
    away_table_pressure: float = 0.0
    home_survival_pressure: float = 0.0
    away_survival_pressure: float = 0.0
    home_big_win_need: float = 0.0
    away_big_win_need: float = 0.0
    home_draw_sufficient: bool = False
    away_draw_sufficient: bool = False
    home_settled: bool = False
    away_settled: bool = False
    is_final_round: bool = False
    endgame_chaos: float = 0.0
    home_celebration_risk: float = 0.0
    away_celebration_risk: float = 0.0
    home_collapse_risk: float = 0.0
    away_collapse_risk: float = 0.0
    weather_goal_drag: float = 0.0
    referee_goal_bias: float = 0.0
    market: MarketInput = field(default_factory=MarketInput)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchInput":
        return cls(
            match_id=str(data["match_id"]),
            league=str(data["league"]),
            home_team=str(data["home_team"]),
            away_team=str(data["away_team"]),
            league_avg_goals=float(data.get("league_avg_goals", 2.62)),
            league_home_advantage=float(data.get("league_home_advantage", 0.17)),
            league_draw_bias=float(data.get("league_draw_bias", 0.0)),
            league_volatility=float(data.get("league_volatility", 0.5)),
            home_attack=float(data.get("home_attack", 1.0)),
            away_attack=float(data.get("away_attack", 1.0)),
            home_defense=float(data.get("home_defense", 1.0)),
            away_defense=float(data.get("away_defense", 1.0)),
            recent_home_xg=_optional_float(data.get("recent_home_xg")),
            recent_away_xg=_optional_float(data.get("recent_away_xg")),
            recent_home_xga=_optional_float(data.get("recent_home_xga")),
            recent_away_xga=_optional_float(data.get("recent_away_xga")),
            home_form=float(data.get("home_form", 0.0)),
            away_form=float(data.get("away_form", 0.0)),
            home_absence_impact=float(data.get("home_absence_impact", 0.0)),
            away_absence_impact=float(data.get("away_absence_impact", 0.0)),
            home_rotation_risk=float(data.get("home_rotation_risk", 0.0)),
            away_rotation_risk=float(data.get("away_rotation_risk", 0.0)),
            home_rest_edge=float(data.get("home_rest_edge", 0.0)),
            away_rest_edge=float(data.get("away_rest_edge", 0.0)),
            home_motivation=float(data.get("home_motivation", 0.0)),
            away_motivation=float(data.get("away_motivation", 0.0)),
            home_rank=_optional_int(data.get("home_rank")),
            away_rank=_optional_int(data.get("away_rank")),
            home_points=_optional_int(data.get("home_points")),
            away_points=_optional_int(data.get("away_points")),
            home_goal_difference=_optional_int(data.get("home_goal_difference")),
            away_goal_difference=_optional_int(data.get("away_goal_difference")),
            home_table_pressure=float(data.get("home_table_pressure", 0.0)),
            away_table_pressure=float(data.get("away_table_pressure", 0.0)),
            home_survival_pressure=float(data.get("home_survival_pressure", 0.0)),
            away_survival_pressure=float(data.get("away_survival_pressure", 0.0)),
            home_big_win_need=float(data.get("home_big_win_need", 0.0)),
            away_big_win_need=float(data.get("away_big_win_need", 0.0)),
            home_draw_sufficient=_bool(data.get("home_draw_sufficient", False)),
            away_draw_sufficient=_bool(data.get("away_draw_sufficient", False)),
            home_settled=_bool(data.get("home_settled", False)),
            away_settled=_bool(data.get("away_settled", False)),
            is_final_round=_bool(data.get("is_final_round", False)),
            endgame_chaos=float(data.get("endgame_chaos", 0.0)),
            home_celebration_risk=float(data.get("home_celebration_risk", 0.0)),
            away_celebration_risk=float(data.get("away_celebration_risk", 0.0)),
            home_collapse_risk=float(data.get("home_collapse_risk", 0.0)),
            away_collapse_risk=float(data.get("away_collapse_risk", 0.0)),
            weather_goal_drag=float(data.get("weather_goal_drag", 0.0)),
            referee_goal_bias=float(data.get("referee_goal_bias", 0.0)),
            market=MarketInput.from_dict(data.get("market")),
        )


@dataclass(frozen=True)
class ScoreCandidate:
    score: str
    home_goals: int
    away_goals: int
    total_goals: int
    real_probability: float
    market_probability: float
    edge_ratio: float
    edge_gap: float
    structure_score: float
    cold_bonus: float
    conflict_penalty: float
    final_value: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FinalPick:
    match_id: str
    league: str
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    score: str
    confidence_band: str
    candidate: ScoreCandidate
    layer_winners: tuple[ScoreCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "expected_goals": {
                "home": round(self.expected_home_goals, 3),
                "away": round(self.expected_away_goals, 3),
            },
            "final_score": self.score,
            "confidence_band": self.confidence_band,
            "selected_candidate": _candidate_to_dict(self.candidate),
            "total_goal_layer_winners": [_candidate_to_dict(item) for item in self.layer_winners],
        }


def _candidate_to_dict(candidate: ScoreCandidate) -> dict[str, Any]:
    return {
        "score": candidate.score,
        "total_goals": candidate.total_goals,
        "real_probability": round(candidate.real_probability, 5),
        "market_probability": round(candidate.market_probability, 5),
        "edge_ratio": round(candidate.edge_ratio, 3),
        "edge_gap": round(candidate.edge_gap, 5),
        "structure_score": round(candidate.structure_score, 3),
        "cold_bonus": round(candidate.cold_bonus, 3),
        "conflict_penalty": round(candidate.conflict_penalty, 3),
        "final_value": round(candidate.final_value, 6),
        "reasons": list(candidate.reasons),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
