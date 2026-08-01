"""CSV export and filter helpers."""

from cap_predictor.models import Band, ChanceLabel, CollegePrediction, StudentProfile
from cap_predictor.predictor.engine import evidence_confidence_from_trend
from cap_predictor.models import ClosingTrendPoint
from cap_predictor.report.csv_export import filter_predictions, result_to_csv_string
from cap_predictor.models import PredictionResult


def _mk(code: int, band: Band, univ: str, hu: float) -> CollegePrediction:
    return CollegePrediction(
        institute_code=code,
        institute_name=f"College {code}",
        university=univ,
        branch_code=str(code),
        branch_name="CSE",
        band=band,
        chance_label=ChanceLabel.MEDIUM,
        margin=1.0,
        primary_closing_percentile=90.0,
        primary_year=2025,
        primary_round=1,
        primary_category_code="GOPENS",
        rule_reason="t",
        hu_match_score=hu,
        evidence_confidence="medium",
    )


def test_filter_by_university_and_band():
    items = [
        _mk(1, Band.SAFE, "Mumbai University", 90),
        _mk(2, Band.DREAM, "Pune University", 40),
    ]
    filtered = filter_predictions(items, university_contains="mumbai", bands={"safe"})
    assert len(filtered) == 1
    assert filtered[0].institute_code == 1


def test_csv_export_contains_bands():
    result = PredictionResult(
        profile=StudentProfile(
            percentile=90,
            category="GOPENS",
            gender="M",
            home_university="Mumbai University",
            seat_type="ALL",
            target_branch="CSE",
        ),
        safe=[_mk(1, Band.SAFE, "MU", 80)],
        target=[_mk(2, Band.TARGET, "MU", 70)],
        dream=[],
    )
    csv_text = result_to_csv_string(result)
    assert "College 1" in csv_text
    assert "safe" in csv_text
    assert "evidence_confidence" in csv_text


def test_evidence_confidence_levels():
    assert evidence_confidence_from_trend([]) == "low"
    assert (
        evidence_confidence_from_trend(
            [ClosingTrendPoint(year=2025, round=1, closing_percentile=90, category_code="GOPENS")]
        )
        == "low"
    )
    assert (
        evidence_confidence_from_trend(
            [
                ClosingTrendPoint(year=2025, round=1, closing_percentile=90, category_code="GOPENS"),
                ClosingTrendPoint(year=2025, round=4, closing_percentile=91, category_code="GOPENS"),
            ]
        )
        == "medium"
    )
    assert (
        evidence_confidence_from_trend(
            [
                ClosingTrendPoint(year=2024, round=1, closing_percentile=89, category_code="GOPENS"),
                ClosingTrendPoint(year=2025, round=1, closing_percentile=90, category_code="GOPENS"),
            ]
        )
        == "high"
    )
