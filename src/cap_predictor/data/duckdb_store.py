"""DuckDB-backed cutoff store for fast filtered queries."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cap_predictor.utils.paths import data_processed_dir

logger = logging.getLogger(__name__)

TABLE_NAME = "cutoffs"


def duckdb_path() -> Path:
    return data_processed_dir() / "cutoffs.duckdb"


def _connect(path: Path | None = None):
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "duckdb is required. Install with: pip install duckdb"
        ) from exc
    return duckdb.connect(str(path or duckdb_path()))


def sync_cutoffs_to_duckdb(df: pd.DataFrame, *, db_path: Path | None = None) -> Path:
    """Replace the cutoffs table from a DataFrame and return DB path."""
    path = db_path or duckdb_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(path)
    try:
        con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        con.register("_cutoffs_df", df)
        con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM _cutoffs_df")
        con.unregister("_cutoffs_df")
        n = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        logger.info("Synced %s rows into DuckDB %s", n, path)
    finally:
        con.close()
    return path


def load_cutoffs_from_duckdb(
    *,
    db_path: Path | None = None,
    year: int | None = None,
    round_num: int | None = None,
    branch_name: str | None = None,
    category_code: str | None = None,
) -> pd.DataFrame:
    path = db_path or duckdb_path()
    if not path.exists():
        raise FileNotFoundError(
            f"DuckDB store not found at {path}. Run parse or build-db first."
        )
    clauses: list[str] = []
    params: list = []
    if year is not None:
        clauses.append("year = ?")
        params.append(year)
    if round_num is not None:
        clauses.append("round = ?")
        params.append(round_num)
    if branch_name:
        clauses.append("lower(branch_name) LIKE ?")
        params.append(f"%{branch_name.lower()}%")
    if category_code:
        clauses.append("category_code = ?")
        params.append(category_code.upper())

    sql = f"SELECT * FROM {TABLE_NAME}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    con = _connect(path)
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


def list_distinct(column: str, *, db_path: Path | None = None) -> list[str]:
    path = db_path or duckdb_path()
    if not path.exists():
        return []
    if column not in {"branch_name", "university", "category_code", "institute_name"}:
        raise ValueError(f"Unsupported column: {column}")
    con = _connect(path)
    try:
        rows = con.execute(
            f"SELECT DISTINCT {column} AS v FROM {TABLE_NAME} "
            f"WHERE {column} IS NOT NULL AND cast({column} AS VARCHAR) <> '' "
            f"ORDER BY v"
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        con.close()


def duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False
