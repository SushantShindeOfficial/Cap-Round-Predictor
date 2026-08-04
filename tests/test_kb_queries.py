"""Tests for KB query tools (DuckDB cutoffs + seats)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cap_predictor.data.duckdb_store import (
    sync_cutoffs_to_duckdb,
    sync_seat_matrix_to_duckdb,
)
from cap_predictor.kb.ask import heuristic_route
from cap_predictor.kb.queries import (
    query_cutoffs,
    query_seats,
    query_seats_with_cutoffs,
    run_tool,
)


def _cutoffs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": 2025,
                "round": 1,
                "institute_code": 3012,
                "institute_name": "VJTI",
                "university": "Mumbai University",
                "branch_code": "301224510",
                "branch_name": "Computer Engineering",
                "seat_type_section": "STATE_LEVEL",
                "category_code": "GOPENS",
                "closing_percentile": 99.0,
                "closing_merit": 500,
            },
            {
                "year": 2025,
                "round": 1,
                "institute_code": 2008,
                "institute_name": "Lower Tier",
                "university": "SGBAU",
                "branch_code": "200824510",
                "branch_name": "Computer Engineering",
                "seat_type_section": "STATE_LEVEL",
                "category_code": "GOPENS",
                "closing_percentile": 68.5,
                "closing_merit": 45000,
            },
            {
                "year": 2025,
                "round": 1,
                "institute_code": 2137,
                "institute_name": "NT College",
                "university": "SRTMU",
                "branch_code": "213724510",
                "branch_name": "Computer Engineering",
                "seat_type_section": "HU_TO_HU",
                "category_code": "LNTD",
                "closing_percentile": 55.0,
                "closing_merit": 52000,
            },
        ]
    )


def _seats_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": 2025,
                "institute_code": 2137,
                "institute_name": "NT College",
                "choice_code": "0213724510",
                "course_code": "2451",
                "course_name": "Computer Engineering",
                "quota": "HU",
                "category_code": "NTD",
                "gender": "L",
                "seats": 2,
                "si": 60,
            },
            {
                "year": 2025,
                "institute_code": 3012,
                "institute_name": "VJTI",
                "choice_code": "0301224510",
                "course_code": "2451",
                "course_name": "Computer Engineering",
                "quota": "STATE",
                "category_code": "OPEN",
                "gender": "G",
                "seats": 10,
                "si": 60,
            },
        ]
    )


def test_query_cutoffs_percentile_and_merit(tmp_path: Path):
    db = tmp_path / "kb.duckdb"
    sync_cutoffs_to_duckdb(_cutoffs_df(), db_path=db)
    sync_seat_matrix_to_duckdb(_seats_df(), db_path=db)

    low = query_cutoffs(max_closing_percentile=71.0, db_path=db)
    assert low["row_count"] >= 1
    assert all(r["closing_percentile"] <= 71.0 for r in low["rows"])

    merit = query_cutoffs(min_closing_merit=40000, max_closing_merit=60000, db_path=db)
    codes = {r["institute_code"] for r in merit["rows"]}
    assert 2008 in codes or 2137 in codes
    assert 3012 not in codes


def test_query_seats_lntd_maps_to_ntd_l(tmp_path: Path):
    db = tmp_path / "kb.duckdb"
    sync_cutoffs_to_duckdb(_cutoffs_df(), db_path=db)
    sync_seat_matrix_to_duckdb(_seats_df(), db_path=db)

    seats = query_seats(category_code="LNTD", min_seats=1, db_path=db)
    assert seats["row_count"] >= 1
    assert seats["rows"][0]["category_code"] == "NTD"
    assert seats["rows"][0]["gender"] == "L"


def test_query_seats_with_cutoffs(tmp_path: Path):
    db = tmp_path / "kb.duckdb"
    sync_cutoffs_to_duckdb(_cutoffs_df(), db_path=db)
    sync_seat_matrix_to_duckdb(_seats_df(), db_path=db)

    joined = query_seats_with_cutoffs(category_code="LNTD", year=2025, db_path=db)
    assert joined["row_count"] >= 1
    item = joined["rows"][0]
    assert item["seat"]["institute_code"] == 2137
    assert item["cutoff_count"] >= 1


def test_run_tool_and_heuristic():
    tool, args = heuristic_route("colleges with cutoff below 71%")
    assert tool == "query_cutoffs"
    assert args.get("max_closing_percentile") == 71.0

    tool2, args2 = heuristic_route("which colleges have seats for LNTD and what was their cutoff?")
    assert tool2 == "query_seats_with_cutoffs"
    assert args2.get("category_code") == "LNTD"

    empty = run_tool("nope", {})
    assert empty["row_count"] == 0
