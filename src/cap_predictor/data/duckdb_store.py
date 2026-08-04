"""DuckDB-backed cutoff + seat-matrix store for fast filtered queries."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cap_predictor.utils.paths import data_processed_dir

logger = logging.getLogger(__name__)

TABLE_CUTOFFS = "cutoffs"
TABLE_SEATS = "seat_matrix"
# Back-compat alias
TABLE_NAME = TABLE_CUTOFFS


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
        con.execute(f"DROP TABLE IF EXISTS {TABLE_CUTOFFS}")
        con.register("_cutoffs_df", df)
        con.execute(f"CREATE TABLE {TABLE_CUTOFFS} AS SELECT * FROM _cutoffs_df")
        con.unregister("_cutoffs_df")
        n = con.execute(f"SELECT COUNT(*) FROM {TABLE_CUTOFFS}").fetchone()[0]
        logger.info("Synced %s cutoff rows into DuckDB %s", n, path)
    finally:
        con.close()
    return path


def sync_seat_matrix_to_duckdb(
    df: pd.DataFrame | None = None,
    *,
    db_path: Path | None = None,
) -> Path:
    """Replace seat_matrix table. Loads parquet if df is None."""
    path = db_path or duckdb_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        parquet = data_processed_dir() / "seat_matrix_all.parquet"
        if parquet.exists():
            df = pd.read_parquet(parquet)
        else:
            df = pd.DataFrame()

    con = _connect(path)
    try:
        con.execute(f"DROP TABLE IF EXISTS {TABLE_SEATS}")
        if df is None or df.empty:
            con.execute(
                f"""
                CREATE TABLE {TABLE_SEATS} (
                    year INTEGER,
                    institute_code INTEGER,
                    institute_name VARCHAR,
                    status VARCHAR,
                    choice_code VARCHAR,
                    course_code VARCHAR,
                    course_name VARCHAR,
                    quota VARCHAR,
                    category_code VARCHAR,
                    gender VARCHAR,
                    seats INTEGER,
                    si INTEGER,
                    cap_seats INTEGER,
                    source_pdf VARCHAR,
                    page_number INTEGER
                )
                """
            )
            logger.info("Created empty seat_matrix table in %s", path)
        else:
            con.register("_seats_df", df)
            con.execute(f"CREATE TABLE {TABLE_SEATS} AS SELECT * FROM _seats_df")
            con.unregister("_seats_df")
            n = con.execute(f"SELECT COUNT(*) FROM {TABLE_SEATS}").fetchone()[0]
            logger.info("Synced %s seat-matrix rows into DuckDB %s", n, path)
    finally:
        con.close()
    return path


def sync_knowledge_base_to_duckdb(
    *,
    cutoffs: pd.DataFrame | None = None,
    seat_matrix: pd.DataFrame | None = None,
    db_path: Path | None = None,
    force_reparse_cutoffs: bool = False,
) -> Path:
    """Sync cutoffs (+ optional seat matrix) into one DuckDB file."""
    if cutoffs is None:
        from cap_predictor.parser.normalizer import load_all_cutoffs

        cutoffs = load_all_cutoffs(force_reparse=force_reparse_cutoffs)
    path = sync_cutoffs_to_duckdb(cutoffs, db_path=db_path)
    sync_seat_matrix_to_duckdb(seat_matrix, db_path=path)
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

    sql = f"SELECT * FROM {TABLE_CUTOFFS}"
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
            f"SELECT DISTINCT {column} AS v FROM {TABLE_CUTOFFS} "
            f"WHERE {column} IS NOT NULL AND cast({column} AS VARCHAR) <> '' "
            f"ORDER BY v"
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        con.close()


def table_exists(table: str, *, db_path: Path | None = None) -> bool:
    path = db_path or duckdb_path()
    if not path.exists():
        return False
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = ?",
            [table],
        ).fetchone()
        return bool(rows and rows[0])
    finally:
        con.close()


def duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False
