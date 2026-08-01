"""Parser quality validation reports."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from cap_predictor.parser.pdf_parser import parse_pdf
from cap_predictor.utils.paths import data_processed_dir, default_pdf_paths, reports_dir

logger = logging.getLogger(__name__)


def validate_pdfs(
    pdf_paths: list[Path] | None = None,
    *,
    max_pages: int | None = None,
) -> dict:
    paths = pdf_paths or default_pdf_paths()
    paths = [p for p in paths if "CutOff" in p.name or "CUTOFF" in p.name.upper()]
    reports = []
    totals = {
        "pdfs": 0,
        "pages": 0,
        "rows": 0,
        "pages_without_rows": 0,
        "unknown_token_count": 0,
    }

    for path in paths:
        logger.info("Validating %s", path.name)
        _, stats = parse_pdf(path, max_pages=max_pages)
        item = {"source_pdf": path.name, **stats.to_dict()}
        # expected rough density: CET cutoffs usually yield many rows/page
        item["rows_per_page"] = round(stats.rows / stats.pages, 2) if stats.pages else 0.0
        item["quality_flag"] = (
            "ok"
            if item["empty_page_pct"] < 40 and item["rows_per_page"] >= 3
            else "review"
        )
        reports.append(item)
        totals["pdfs"] += 1
        totals["pages"] += stats.pages
        totals["rows"] += stats.rows
        totals["pages_without_rows"] += stats.pages_without_rows
        totals["unknown_token_count"] += sum(stats.unknown_tokens.values())

    totals["empty_page_pct"] = (
        round(100.0 * totals["pages_without_rows"] / totals["pages"], 2)
        if totals["pages"]
        else 0.0
    )
    totals["rows_per_page"] = (
        round(totals["rows"] / totals["pages"], 2) if totals["pages"] else 0.0
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "totals": totals,
        "pdfs": reports,
    }
    return payload


def write_validation_report(
    payload: dict,
    out_path: Path | str | None = None,
) -> Path:
    out = Path(out_path) if out_path else reports_dir() / "parser_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # also keep a copy under data/processed
    mirror = data_processed_dir() / "parser_validation.json"
    mirror.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote validation report %s", out)
    return out
