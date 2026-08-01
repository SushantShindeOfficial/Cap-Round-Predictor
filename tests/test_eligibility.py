"""Eligibility helper tests."""

from cap_predictor.models import StudentProfile
from cap_predictor.utils.eligibility import eligibility_notes


def test_tfws_note():
    notes = eligibility_notes(
        StudentProfile(
            percentile=90,
            category="TFWS",
            gender="M",
            home_university="MU",
            seat_type="STATE",
            target_branch="CSE",
        )
    )
    assert any("TFWS" in n for n in notes)


def test_pwd_note():
    notes = eligibility_notes(
        StudentProfile(
            percentile=90,
            category="PWDOPENS",
            gender="M",
            home_university="MU",
            seat_type="STATE",
            target_branch="CSE",
        )
    )
    assert any("PWD" in n for n in notes)
