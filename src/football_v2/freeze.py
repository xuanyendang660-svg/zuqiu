from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
import json


class EvidenceKind(StrEnum):
    OFFICIAL_LINEUP = "official_lineup"
    OFFICIAL_INJURY = "official_injury"
    MARKET_CROSS_LINE = "market_cross_line"
    COMPETITION_RULE = "competition_rule"
    WEATHER_OR_VENUE = "weather_or_venue"
    USER_FEEDBACK = "user_feedback"
    NARRATIVE = "narrative"


_HARD_EVIDENCE = {
    EvidenceKind.OFFICIAL_LINEUP,
    EvidenceKind.OFFICIAL_INJURY,
    EvidenceKind.MARKET_CROSS_LINE,
    EvidenceKind.COMPETITION_RULE,
    EvidenceKind.WEATHER_OR_VENUE,
}


@dataclass(frozen=True)
class Evidence:
    kind: EvidenceKind
    description: str
    observed_at: str
    source: str

    @property
    def is_hard(self) -> bool:
        return self.kind in _HARD_EVIDENCE


@dataclass(frozen=True)
class FrozenPrediction:
    match_id: str
    score: tuple[int, int]
    model_version: str
    frozen_at: str
    feature_digest: str
    evidence: tuple[Evidence, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        match_id: str,
        score: tuple[int, int],
        model_version: str,
        frozen_at: str,
        feature_payload: dict[str, object],
        evidence: tuple[Evidence, ...] = (),
    ) -> "FrozenPrediction":
        payload = json.dumps(feature_payload, sort_keys=True, ensure_ascii=False)
        return cls(
            match_id=match_id,
            score=score,
            model_version=model_version,
            frozen_at=frozen_at,
            feature_digest=sha256(payload.encode("utf-8")).hexdigest(),
            evidence=evidence,
        )

    def can_revise(self, new_evidence: tuple[Evidence, ...]) -> bool:
        existing = {
            (item.kind, item.description, item.observed_at, item.source)
            for item in self.evidence
        }
        return any(
            item.is_hard
            and (item.kind, item.description, item.observed_at, item.source) not in existing
            for item in new_evidence
        )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, default=str)
        return sha256(payload.encode("utf-8")).hexdigest()
