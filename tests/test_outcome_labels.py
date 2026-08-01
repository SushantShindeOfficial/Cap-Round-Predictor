"""Outcome label loader and feature join tests."""

from pathlib import Path

import pandas as pd
import pytest

from cap_predictor.ml.outcome_labels import (
    labels_to_training_features,
    load_outcome_labels,
    write_example_labels,
)


def test_write_and_load_example_labels(tmp_path: Path):
    path = tmp_path / "allotments.csv"
    write_example_labels(path)
    df = load_outcome_labels(path)
    assert len(df) == 2
    assert set(df.columns) >= {
        "student_percentile",
        "institute_code",
        "branch_code",
        "category_code",
        "year",
        "round",
        "allotted",
    }
    assert bool(df.iloc[0]["allotted"]) is True
    assert bool(df.iloc[1]["allotted"]) is False


def test_load_labels_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_outcome_labels(tmp_path / "missing.csv")


def test_load_labels_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"student_percentile": [90.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_outcome_labels(path)


def test_labels_to_training_features():
    labels = pd.DataFrame(
        [
            {
                "student_percentile": 96.5,
                "institute_code": 3012,
                "branch_code": "301224510",
                "category_code": "GOPENH",
                "year": 2024,
                "round": 1,
                "allotted": True,
                "band": "safe",
                "home_university": "Mumbai University",
                "gender": "M",
                "seat_type": "HU",
            }
        ]
    )
    cutoffs = pd.DataFrame(
        [
            {
                "year": 2024,
                "round": 1,
                "institute_code": 3012,
                "institute_name": "VJTI",
                "university": "Mumbai University",
                "branch_code": "301224510",
                "branch_name": "Computer Engineering",
                "category_code": "GOPENH",
                "stage": "I",
                "closing_percentile": 95.2,
                "seat_type_section": "HU_TO_HU",
            }
        ]
    )
    features = labels_to_training_features(labels, cutoffs)
    assert len(features) == 1
    assert features.iloc[0]["band"] == "safe"
    assert features.iloc[0]["margin"] == pytest.approx(1.3, abs=1e-6)
