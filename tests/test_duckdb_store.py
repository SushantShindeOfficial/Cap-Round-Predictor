"""DuckDB store tests."""

import pandas as pd
import pytest

from cap_predictor.data.duckdb_store import (
    duckdb_available,
    list_distinct,
    load_cutoffs_from_duckdb,
    sync_cutoffs_to_duckdb,
)


pytestmark = pytest.mark.skipif(not duckdb_available(), reason="duckdb not installed")


def test_sync_and_query(tmp_path):
    df = pd.DataFrame(
        [
            {
                "year": 2025,
                "round": 1,
                "institute_code": 1002,
                "institute_name": "GCOE",
                "university": "Autonomous Institute",
                "branch_code": "100224210",
                "branch_name": "Computer Science and Engineering",
                "seat_type_section": "STATE_LEVEL",
                "category_code": "GOPENS",
                "gender_flag": "G",
                "reservation": "OPEN",
                "quota_suffix": "S",
                "stage": "I",
                "closing_merit": 100,
                "closing_percentile": 99.0,
                "source_pdf": "t.pdf",
                "page_number": 1,
            }
        ]
    )
    db = tmp_path / "t.duckdb"
    sync_cutoffs_to_duckdb(df, db_path=db)
    out = load_cutoffs_from_duckdb(db_path=db, year=2025, category_code="GOPENS")
    assert len(out) == 1
    assert list_distinct("branch_name", db_path=db) == ["Computer Science and Engineering"]
