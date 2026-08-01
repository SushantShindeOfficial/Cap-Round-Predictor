"""Unit tests for seat-matrix mapping, overlay, and page parser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cap_predictor.data.college_master import (
    college_tier_score,
    seed_college_master_from_seat_matrix,
)
from cap_predictor.models import Band, ChanceLabel, CollegePrediction
from cap_predictor.parser.seat_matrix_parser import parse_seat_matrix_page_text
from cap_predictor.predictor.seat_overlay import (
    enrich_predictions_with_seats,
    filter_zero_seat_predictions,
    lookup_seats,
)
from cap_predictor.utils.seat_matrix_map import (
    course_code_from_cutoff_branch,
    map_cutoff_category_to_seat_cell,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_seat_matrix_page.txt"


def test_map_gopenh_gopenso():
    h = map_cutoff_category_to_seat_cell("GOPENH")
    assert h is not None
    assert h.quota == "HU" and h.category_code == "OPEN" and h.gender == "G"

    s = map_cutoff_category_to_seat_cell("GOPENS")
    assert s is not None
    assert s.quota == "STATE" and s.category_code == "OPEN" and s.gender == "G"

    o = map_cutoff_category_to_seat_cell("LOPENO")
    assert o is not None
    assert o.quota == "OHU" and o.category_code == "OPEN" and o.gender == "L"


def test_course_code_from_cutoff_branch():
    assert course_code_from_cutoff_branch("301224510", 3012) == "2451"
    assert course_code_from_cutoff_branch("0100224210", 1002) == "2421"


def test_parse_seat_matrix_page_fixture():
    text = FIXTURE.read_text(encoding="utf-8")
    rows = parse_seat_matrix_page_text(
        text, year=2025, source_pdf="fixture.pdf", page_number=1
    )
    assert rows
    assert any(r["quota"] == "HU" and r["category_code"] == "OPEN" and r["gender"] == "G" for r in rows)
    assert any(r["quota"] == "OHU" for r in rows)
    # At least one OPEN-G HU row with seats
    hu_open_g = [
        r
        for r in rows
        if r["quota"] == "HU" and r["category_code"] == "OPEN" and r["gender"] == "G"
    ]
    assert hu_open_g
    assert hu_open_g[0]["institute_code"] > 0
    assert hu_open_g[0]["course_code"]
    assert hu_open_g[0]["seats"] == 13

def test_lookup_and_zero_seat_filter():
    seat_df = pd.DataFrame(
        [
            {
                "year": 2025,
                "institute_code": 3012,
                "course_code": "2451",
                "quota": "STATE",
                "category_code": "OPEN",
                "gender": "G",
                "seats": 12,
                "si": 60,
            },
            {
                "year": 2025,
                "institute_code": 3012,
                "course_code": "2451",
                "quota": "HU",
                "category_code": "OPEN",
                "gender": "G",
                "seats": 0,
                "si": 60,
            },
        ]
    )
    seats, year, quota, si = lookup_seats(
        seat_df,
        institute_code=3012,
        branch_code="301224510",
        category_code="GOPENS",
        year=2025,
    )
    assert seats == 12 and year == 2025 and quota == "STATE" and si == 60

    preds = [
        CollegePrediction(
            institute_code=3012,
            institute_name="Test",
            branch_code="301224510",
            branch_name="Computer Engineering",
            band=Band.SAFE,
            chance_label=ChanceLabel.HIGH,
            margin=5.0,
            primary_closing_percentile=87.0,
            primary_year=2025,
            primary_round=1,
            primary_category_code="GOPENH",
            rule_reason="test",
        ),
        CollegePrediction(
            institute_code=3012,
            institute_name="Test",
            branch_code="301224510",
            branch_name="Computer Engineering",
            band=Band.TARGET,
            chance_label=ChanceLabel.MEDIUM,
            margin=1.0,
            primary_closing_percentile=91.0,
            primary_year=2025,
            primary_round=1,
            primary_category_code="GOPENS",
            rule_reason="test",
        ),
    ]
    enriched, stats = enrich_predictions_with_seats(preds, seat_df)
    assert stats["matched"] == 2
    assert enriched[0].available_seats == 0
    assert enriched[1].available_seats == 12
    assert enriched[1].final_score is not None
    keep, excluded = filter_zero_seat_predictions(enriched)
    assert len(excluded) == 1 and len(keep) == 1
    assert keep[0].available_seats == 12


def test_seed_college_master(tmp_path, monkeypatch):
    from cap_predictor.data import college_master as cm

    monkeypatch.setattr(cm, "data_masters_dir", lambda: tmp_path)
    seat_df = pd.DataFrame(
        [
            {
                "institute_code": 1002,
                "institute_name": "Govt College",
                "status": "Government Autonomous",
            },
            {
                "institute_code": 3012,
                "institute_name": "Private College",
                "status": "Un-Aided",
            },
        ]
    )
    path = seed_college_master_from_seat_matrix(seat_df)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 2
    gov = loaded[loaded["institute_code"] == 1002].iloc[0]
    assert bool(gov["government"]) is True
    assert college_tier_score(gov.to_dict()) > college_tier_score(
        loaded[loaded["institute_code"] == 3012].iloc[0].to_dict()
    )
