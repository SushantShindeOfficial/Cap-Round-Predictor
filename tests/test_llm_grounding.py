"""LLM grounding invent-check tests."""

from cap_predictor.llm.grounded import (
    invented_institute_codes,
    strip_invented_sentences,
    template_narrative,
)
from cap_predictor.models import (
    Band,
    ChanceLabel,
    CollegePrediction,
    PredictionResult,
    StudentProfile,
)


def _result() -> PredictionResult:
    college = CollegePrediction(
        institute_code=3012,
        institute_name="VJTI",
        branch_code="301224510",
        branch_name="Computer Engineering",
        band=Band.SAFE,
        chance_label=ChanceLabel.HIGH,
        margin=4.0,
        primary_closing_percentile=88.0,
        primary_year=2025,
        primary_round=1,
        primary_category_code="GOPENS",
        rule_reason="ok",
    )
    return PredictionResult(
        profile=StudentProfile(
            percentile=92.5,
            category="GOPENS",
            gender="M",
            home_university="Mumbai University",
            seat_type="ALL",
            target_branch="Computer Engineering",
        ),
        safe=[college],
        preference_order=[college],
        eligible=[college],
    )


def test_invented_codes_detected():
    result = _result()
    text = "Consider 3012 and also try 9999 which is great."
    assert invented_institute_codes(text, result) == ["9999"]


def test_years_not_flagged_as_institute_codes():
    result = _result()
    text = "Based on 2025 round data for institute 3012."
    assert invented_institute_codes(text, result) == []


def test_strip_invented_sentences():
    text = "Prefer 3012 first. Avoid 8888 entirely. Keep going."
    cleaned = strip_invented_sentences(text, ["8888"])
    assert "8888" not in cleaned
    assert "3012" in cleaned


def test_template_narrative_mentions_codes():
    text = template_narrative(_result())
    assert "3012" in text
    assert "Safe:" in text or "safe" in text.lower()
