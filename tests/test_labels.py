import pytest

from football_v2.labels import ScoreArchetype, classify_score


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        ((0, 0), ScoreArchetype.LOW_EVENT),
        ((1, 0), ScoreArchetype.LOW_EVENT),
        ((2, 1), ScoreArchetype.NORMAL),
        ((3, 2), ScoreArchetype.SHOOTOUT),
        ((3, 0), ScoreArchetype.HOME_CONTROL),
        ((0, 3), ScoreArchetype.AWAY_CONTROL),
        ((6, 1), ScoreArchetype.HOME_BLOWOUT),
        ((1, 5), ScoreArchetype.AWAY_BLOWOUT),
    ],
)
def test_score_archetypes(score: tuple[int, int], expected: ScoreArchetype) -> None:
    assert classify_score(*score) is expected
