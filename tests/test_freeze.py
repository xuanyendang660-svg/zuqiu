from football_v2.freeze import Evidence, EvidenceKind, FrozenPrediction


def _snapshot() -> FrozenPrediction:
    return FrozenPrediction.create(
        match_id="home-away",
        score=(1, 5),
        model_version="0.2.0",
        frozen_at="2026-07-17T12:00:00Z",
        feature_payload={"market_home": 0.64, "away_transition": 0.91},
    )


def test_user_feedback_cannot_change_frozen_prediction() -> None:
    feedback = Evidence(
        kind=EvidenceKind.USER_FEEDBACK,
        description="this looks too extreme",
        observed_at="2026-07-17T12:10:00Z",
        source="chat",
    )
    assert not _snapshot().can_revise((feedback,))


def test_official_lineup_can_unlock_prediction() -> None:
    lineup = Evidence(
        kind=EvidenceKind.OFFICIAL_LINEUP,
        description="starting goalkeeper withdrawn",
        observed_at="2026-07-17T12:10:00Z",
        source="club official",
    )
    assert _snapshot().can_revise((lineup,))
