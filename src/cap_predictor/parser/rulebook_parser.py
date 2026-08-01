"""Parse CET Information Brochure PDF into searchable text chunks."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from cap_predictor.utils.paths import (
    data_processed_dir,
    default_rulebook_pdf_paths,
)

logger = logging.getLogger(__name__)

HEADER_RE = re.compile(
    r"^Information Brochure for Admission to Under Graduate and Post Graduate "
    r"Technical Courses.*?$",
    re.I | re.M,
)
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$", re.M)


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def clean_page_text(text: str) -> str:
    text = HEADER_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_rulebook_pdf(pdf_path: Path | str, *, max_pages: int | None = None) -> list[dict]:
    """Return list of {page, text, source_pdf} chunks (one per page with content)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    chunks: list[dict] = []
    try:
        limit = len(doc) if max_pages is None else min(max_pages, len(doc))
        for i in range(limit):
            raw = doc[i].get_text("text") or ""
            text = clean_page_text(raw)
            if len(text) < 80:
                continue
            chunks.append(
                {
                    "page": i + 1,
                    "text": text,
                    "source_pdf": pdf_path.name,
                }
            )
    finally:
        doc.close()
    logger.info("Parsed %s rulebook pages from %s", len(chunks), pdf_path.name)
    return chunks


def chunks_cache_path() -> Path:
    return data_processed_dir() / "rulebook_chunks.json"


def parse_and_cache_rulebook(
    pdf_paths: list[Path] | None = None,
    *,
    force: bool = False,
    max_pages: int | None = None,
) -> list[dict]:
    paths = pdf_paths if pdf_paths is not None else default_rulebook_pdf_paths()
    out = chunks_cache_path()
    meta_path = out.with_suffix(".meta.json")

    if not paths:
        logger.warning("No rulebook PDFs under Knowledge Base/rulebook/")
        return []

    fingerprints = [
        {"source_pdf": p.name, "sha256": _sha256_file(p), "size_bytes": p.stat().st_size}
        for p in paths
    ]
    if (
        max_pages is None
        and not force
        and out.exists()
        and meta_path.exists()
    ):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprints") == fingerprints:
                return json.loads(out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    chunks: list[dict] = []
    for p in paths:
        chunks.extend(parse_rulebook_pdf(p, max_pages=max_pages))

    if max_pages is None:
        out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        meta_path.write_text(
            json.dumps({"fingerprints": fingerprints, "chunks": len(chunks)}, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote %s (%s chunks)", out, len(chunks))
    return chunks
