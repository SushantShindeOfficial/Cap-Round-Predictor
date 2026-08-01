"""Load/cache seat matrix parquets with SHA256 invalidation."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from cap_predictor.parser.seat_matrix_parser import parse_seat_matrix_pdf
from cap_predictor.utils.paths import data_processed_dir, default_seat_matrix_pdf_paths

logger = logging.getLogger(__name__)

SEAT_COLUMNS = [
    "year",
    "institute_code",
    "institute_name",
    "status",
    "choice_code",
    "course_code",
    "course_name",
    "quota",
    "category_code",
    "gender",
    "seats",
    "si",
    "cap_seats",
    "ms_seats",
    "minority_seats",
    "all_india",
    "institute_seats",
    "orphan",
    "source_pdf",
    "page_number",
]


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _fingerprint(pdf_path: Path) -> dict:
    stat = pdf_path.stat()
    return {
        "source_pdf": pdf_path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(pdf_path),
    }


def cache_path_for_seat_matrix(pdf_path: Path) -> Path:
    return data_processed_dir() / f"seat_matrix_{pdf_path.stem}.parquet"


def _cache_fresh(pdf_path: Path, parquet_path: Path) -> bool:
    meta_path = parquet_path.with_suffix(".meta.json")
    if not parquet_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    fp = _fingerprint(pdf_path)
    if meta.get("sha256"):
        return meta.get("sha256") == fp["sha256"] and meta.get("source_pdf") == fp["source_pdf"]
    return (
        meta.get("size_bytes") == fp["size_bytes"]
        and meta.get("mtime_ns") == fp["mtime_ns"]
        and meta.get("source_pdf") == fp["source_pdf"]
    )


def parse_and_cache_seat_matrix(
    pdf_path: Path,
    *,
    force: bool = False,
    max_pages: int | None = None,
) -> pd.DataFrame:
    out = cache_path_for_seat_matrix(pdf_path)
    if max_pages is None and not force and _cache_fresh(pdf_path, out):
        logger.info("Loading cached seat matrix from %s", out)
        return pd.read_parquet(out)

    rows = parse_seat_matrix_pdf(pdf_path, max_pages=max_pages)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=SEAT_COLUMNS)
    else:
        df = df.reindex(columns=SEAT_COLUMNS)

    if max_pages is None:
        df.to_parquet(out, index=False)
        meta = {**_fingerprint(pdf_path), "rows": len(df)}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Wrote %s (%s rows)", out, len(df))
    return df


def load_all_seat_matrix(
    pdf_paths: list[Path] | None = None,
    *,
    force_reparse: bool = False,
    max_pages: int | None = None,
) -> pd.DataFrame:
    paths = pdf_paths if pdf_paths is not None else default_seat_matrix_pdf_paths()
    if not paths:
        logger.warning("No seat matrix PDFs found under Knowledge Base/Seat Matrix/")
        return pd.DataFrame(columns=SEAT_COLUMNS)

    frames = [
        parse_and_cache_seat_matrix(p, force=force_reparse, max_pages=max_pages) for p in paths
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SEAT_COLUMNS)
    if max_pages is None and not df.empty:
        combined = data_processed_dir() / "seat_matrix_all.parquet"
        df.to_parquet(combined, index=False)
        logger.info("Combined seat matrix: %s rows", len(df))
    return df
