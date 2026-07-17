from football_model.schema import Evidence, EvidenceType, PredictionSnapshot, SourceGrade


def test_user_feedback_cannot_unlock_frozen_prediction() -> None:
    snapshot = PredictionSnapshot.now(
        match_id="a-b",
        home_team="A",
        away_team="B",
        exact_score=(1, 2),
        probabilities={"home": 0.25, "draw": 0.25, "away": 0.50},
    )
    feedback = Evidence(
        kind=EvidenceType.USER_FEEDBACK,
        description="too many favourites",
        source="chat",
        grade=SourceGrade.C,
        observed_at="2026-07-17T00:00:00Z",
    )
    assert not snapshot.can_be_revised_by([feedback])


def test_official_lineup_can_unlock_frozen_prediction() -> None:
    snapshot = PredictionSnapshot.now(
        match_id="a-b",
        home_team="A",
        away_team="B",
        exact_score=(1, 2),
        probabilities={"home": 0.25, "draw": 0.25, "away": 0.50},
    )
    lineup = Evidence(
        kind=EvidenceType.OFFICIAL_LINEUP,
        description="away starting goalkeeper ruled out",
        source="club official",
        grade=SourceGrade.A,
        observed_at="2026-07-17T00:30:00Z",
    )
    assert snapshot.can_be_revised_by([lineup])
