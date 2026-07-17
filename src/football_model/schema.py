from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any


class SourceGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class EvidenceType(StrEnum):
    OFFICIAL_LINEUP = "official_lineup"
    OFFICIAL_INJURY = "official_injury"
    MARKET_CROSS_LINE = "market_cross_line"
    COMPETITION_RULE = "competition_rule"
    WEATHER_OR_VENUE = "weather_or_venue"
    USER_FEEDBACK = "user_feedback"
    NARRATIVE = "narrative"


HARD_EVIDENCE_TYPES = {
    EvidenceType.OFFICIAL_LINEUP,
    EvidenceType.OFFICIAL_INJURY,
    EvidenceType.MARKET_CROSS_LINE,
    EvidenceType.COMPETITION_RULE,
    EvidenceType.WEATHER_OR_VENUE,
}


@dataclass(frozen=True)
class Evidence:
    kind: EvidenceType
    description: str
    source: str
    grade: SourceGrade
    observed_at: str

    @property
    def is_hard(self) -> bool:
        return self.kind in HARD_EVIDENCE_TYPES and self.grade in {SourceGrade.A, SourceGrade.B}


@dataclass(frozen=True)
class PredictionSnapshot:
    match_id: str
    home_team: str
    away_team: str
    frozen_at: str
    exact_score: tuple[int, int]
    probabilities: dict[str, float]
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    model_version: str = "0.1.0"

    @classmethod
    def now(
        cls,
        *,
        match_id: str,
        home_team: str,
        away_team: str,
        exact_score: tuple[int, int],
        probabilities: dict[str, float],
        evidence: tuple[Evidence, ...] = (),
        model_version: str = "0.1.0",
    ) -> "PredictionSnapshot":
        return cls(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            frozen_at=datetime.now(timezone.utc).isoformat(),
            exact_score=exact_score,
            probabilities=probabilities,
            evidence=evidence,
            model_version=model_version,
        )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return sha256(payload.encode("utf-8")).hexdigest()

    def can_be_revised_by(self, new_evidence: list[Evidence]) -> bool:
        """Only hard, newly observed evidence may unlock a frozen prediction."""
        existing = {(e.kind, e.description, e.observed_at) for e in self.evidence}
        return any(
            e.is_hard and (e.kind, e.description, e.observed_at) not in existing
            for e in new_evidence
        )


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Unsupported type: {type(value)!r}")
