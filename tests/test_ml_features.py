"""ML feature export tests."""

from cap_predictor.ml.features import build_feature_frame
from cap_predictor.models import (
    Band,
    ChanceLabel,
    ClosingTrendPoint,
    CollegePrediction,
    PredictionResult,
    StudentProfile,
)


def test_feature_frame_columns():
    college = CollegePrediction(
        institute_code=1002,
        institute_name="GCOE",
        university="Autonomous Institute",
        branch_code="100224210",
        branch_name="CSE",
        band=Band.TARGET,
        chance_label=ChanceLabel.MEDIUM,
        margin=1.5,
        primary_closing_percentile=91.0,
        primary_year=2025,
        primary_round=4,
        primary_category_code="GOPENS",
        rule_reason="t",
        closing_trend=[
            ClosingTrendPoint(year=2024, round=1, closing_percentile=90.0, category_code="GOPENS"),
            ClosingTrendPoint(year=2025, round=4, closing_percentile=91.0, category_code="GOPENS"),
        ],
        evidence_confidence="high",
    )
    result = PredictionResult(
        profile=StudentProfile(
            percentile=92.5,
            category="GOPENS",
            gender="M",
            home_university="Mumbai University",
            seat_type="ALL",
            target_branch="CSE",
        ),
        preference_order=[college],
        target=[college],
    )
    df = build_feature_frame(result)
    assert len(df) == 1
    assert df.iloc[0]["margin"] == 1.5
    assert df.iloc[0]["yoy_delta"] == 1.0
    assert "student_percentile" in df.columns
