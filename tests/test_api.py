"""FastAPI health and predict endpoint tests."""

from unittest.mock import patch

import pandas as pd
import pytest

from cap_predictor.models import (
    Band,
    ChanceLabel,
    CollegePrediction,
    PredictionResult,
    StudentProfile,
)


@pytest.fixture
def client():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cap_predictor.api import app

    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_mocked(client, tmp_path):
    html = tmp_path / "report.html"
    html.write_text("<html></html>", encoding="utf-8")
    csv = tmp_path / "report.csv"
    csv.write_text("a,b\n", encoding="utf-8")

    college = CollegePrediction(
        institute_code=3012,
        institute_name="VJTI Mumbai",
        university="Mumbai University",
        branch_code="301224510",
        branch_name="Computer Engineering",
        band=Band.SAFE,
        chance_label=ChanceLabel.HIGH,
        margin=5.0,
        primary_closing_percentile=90.0,
        primary_year=2025,
        primary_round=1,
        primary_category_code="GOPENS",
        evidence_confidence="medium",
        rule_reason="test",
    )
    result = PredictionResult(
        profile=StudentProfile(
            percentile=95.0,
            category="GOPENS",
            gender="M",
            home_university="Mumbai University",
            seat_type="ALL",
            target_branch="Computer Engineering",
        ),
        safe=[college],
        target=[],
        dream=[],
        unlikely=[],
        preference_order=[college],
        matched_cutoff_rows=1,
        notes=["ok"],
        narrative="Test narrative",
    )

    with (
        patch("cap_predictor.api.load_all_cutoffs", return_value=pd.DataFrame()),
        patch(
            "cap_predictor.api.run_prediction_report",
            return_value=(result, html, None, csv, None, None),
        ),
    ):
        resp = client.post(
            "/predict",
            json={
                "percentile": 95.0,
                "category": "GOPENS",
                "gender": "M",
                "home_university": "Mumbai University",
                "seat_type": "ALL",
                "target_branch": "Computer Engineering",
                "use_llm": False,
                "use_ml": False,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["counts"]["safe"] == 1
    assert body["counts"]["preferences"] == 1
    assert body["preference_order"][0]["institute_code"] == 3012
    assert body["narrative"] == "Test narrative"
