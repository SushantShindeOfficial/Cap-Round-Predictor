"""University fuzzy-match tests."""

from cap_predictor.utils.university_match import is_hu_affiliated, score_university


def test_mumbai_aliases_match():
    assert score_university("Mumbai University", "University of Mumbai") >= 75
    assert is_hu_affiliated("Mumbai University", "University of Mumbai")


def test_autonomous_not_mumbai():
    assert score_university("Mumbai University", "Autonomous Institute") < 75


def test_pune_alias():
    assert is_hu_affiliated("SPPU", "Savitribai Phule Pune University")
