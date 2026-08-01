"""Normalize CutoffRecord lists into Pandas DataFrames and cache parquet."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from cap_predictor.models import CutoffRecord
from cap_predictor.parser.pdf_parser import parse_pdf
from cap_predictor.utils.paths import data_processed_dir, default_pdf_paths

logger = logging.getLogger(__name__)

COLUMNS = [
    "year",
    "round",
    "institute_code",
    "institute_name",
    "university",
    "branch_code",
    "branch_name",
    "seat_type_section",
    "category_code",
    "gender_flag",
    "reservation",
    "quota_suffix",
    "stage",
    "closing_merit",
    "closing_percentile",
    "source_pdf",
    "page_number",
]


def records_to_dataframe(records: list[CutoffRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame([r.model_dump() for r in records])
    return df.reindex(columns=COLUMNS)


def cache_path_for_pdf(pdf_path: Path) -> Path:
    stem = pdf_path.stem.replace(" ", "_")
    return data_processed_dir() / f"{stem}.parquet"


def _pdf_fingerprint(pdf_path: Path) -> dict:
    stat = pdf_path.stat()
    return {
        "source_pdf": pdf_path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_is_fresh(pdf_path: Path, parquet_path: Path) -> bool:
    meta_path = parquet_path.with_suffix(".meta.json")
    if not parquet_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    fp = _pdf_fingerprint(pdf_path)
    return (
        meta.get("size_bytes") == fp["size_bytes"]
        and meta.get("mtime_ns") == fp["mtime_ns"]
        and meta.get("source_pdf") == fp["source_pdf"]
    )


def parse_and_cache(
    pdf_path: Path,
    *,
    force: bool = False,
    max_pages: int | None = None,
) -> pd.DataFrame:
    out = cache_path_for_pdf(pdf_path)
    if max_pages is None and not force and _cache_is_fresh(pdf_path, out):
        logger.info("Loading cached cutoffs from %s", out)
        return pd.read_parquet(out)

    if out.exists() and not force and max_pages is None:
        logger.info("Cache stale for %s — re-parsing", pdf_path.name)

    records, stats = parse_pdf(pdf_path, max_pages=max_pages)
    df = records_to_dataframe(records)
    if max_pages is None:
        df.to_parquet(out, index=False)
        meta = {
            **_pdf_fingerprint(pdf_path),
            "rows": stats.rows,
            "institutes": len(stats.institutes),
            "branches": len(stats.branches),
            "pages": stats.pages,
        }
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Wrote %s (%s rows)", out, len(df))
    return df


def load_all_cutoffs(
    pdf_paths: list[Path] | None = None,
    *,
    force_reparse: bool = False,
    max_pages: int | None = None,
) -> pd.DataFrame:
    paths = pdf_paths or default_pdf_paths()
    if not paths:
        raise FileNotFoundError(
            "No CAP cutoff PDFs found in Knowledge Base/. "
            "Expected files like 2024ENGG_CAP1_CutOff.pdf"
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        if "CutOff" not in path.name and "CUTOFF" not in path.name.upper():
            continue
        frames.append(parse_and_cache(path, force=force_reparse, max_pages=max_pages))

    if not frames:
        raise FileNotFoundError("No CutOff PDFs parsed.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=[
            "year",
            "round",
            "institute_code",
            "branch_code",
            "seat_type_section",
            "category_code",
            "stage",
            "closing_merit",
            "closing_percentile",
        ]
    )
    combined = data_processed_dir() / "cutoffs_all.parquet"
    if max_pages is None:
        df.to_parquet(combined, index=False)
        try:
            from cap_predictor.data.duckdb_store import duckdb_available, sync_cutoffs_to_duckdb

            if duckdb_available():
                sync_cutoffs_to_duckdb(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DuckDB sync skipped: %s", exc)
    logger.info("Combined cutoff dataset: %s rows from %s PDFs", len(df), len(frames))
    return df
