"""Parse CET Cell Seat Matrix PDFs into category-level seat rows."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

INST_RE = re.compile(r"^0*(\d{4,5})\s*-\s*(.+)$")
CHOICE_RE = re.compile(r"^(\d{10}[A-Za-z]*)$")
CAP_SEATS_RE = re.compile(r"CAP Seats\s*:\s*(\d+)", re.I)
EWS_RE = re.compile(r"Economically Weaker Section \(EWS\) Seats\s*:\s*(\d+)", re.I)
TFWS_RE = re.compile(
    r"Tution Fee Waiver Scheme Choice Code:\s*(\S+)\s*:.*?Seats\s*:\s*(\d+)",
    re.I | re.S,
)
YEAR_RE = re.compile(r"(20\d{2})")
INT_RE = re.compile(r"^\d+$")

# Category order in the G/L grid (before Total)
GRID_CATEGORIES = (
    "OPEN",
    "SC",
    "ST",
    "VJDT",
    "NTB",
    "NTC",
    "NTD",
    "OBC",
    "SEBC",
)

# PWD/DEF row without G/L split: 9 cats + Total
PWD_DEF_CATEGORIES = GRID_CATEGORIES  # 9 values + total as 10th


def infer_year_from_path(pdf_path: Path, sample_text: str = "") -> int:
    m = YEAR_RE.search(pdf_path.name)
    if m:
        return int(m.group(1))
    m = YEAR_RE.search(sample_text)
    return int(m.group(1)) if m else 2025


def _ints_from_tokens(tokens: list[str]) -> list[int]:
    return [int(t) for t in tokens if INT_RE.match(t)]


def parse_seat_matrix_page_text(
    text: str,
    *,
    year: int,
    source_pdf: str,
    page_number: int,
) -> list[dict]:
    """Parse one seat-matrix page into category-level seat dicts."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    institute_code: int | None = None
    institute_name: str | None = None
    status = ""
    choice_code: str | None = None
    course_name: str | None = None
    si = 0
    ms_seats = 0
    minority_seats = 0
    all_india = 0
    institute_seats = 0
    orphan = 0
    cap_seats = 0

    for ln in lines[:12]:
        m = CAP_SEATS_RE.search(ln)
        if m:
            cap_seats = int(m.group(1))
        m = INST_RE.match(ln)
        if m and institute_code is None:
            institute_code = int(m.group(1))
            institute_name = m.group(2).strip()
        elif institute_code is not None and not status and (
            "Autonomous" in ln or "Aided" in ln or "Un-Aided" in ln or "Government" in ln
        ):
            if not INST_RE.match(ln) and "CAP Seats" not in ln:
                status = ln

    # Choice code appears after summary headers (after Orphan label)
    for i, ln in enumerate(lines):
        compact = ln.replace(" ", "")
        cm = CHOICE_RE.match(compact)
        if not cm:
            continue
        choice_code = cm.group(1)
        # next line = course name
        if i + 1 < len(lines) and not INT_RE.match(lines[i + 1]):
            course_name = lines[i + 1]
            start = i + 2
        else:
            start = i + 1
        nums: list[int] = []
        j = start
        while j < len(lines) and len(nums) < 6:
            if INT_RE.match(lines[j]):
                nums.append(int(lines[j]))
            j += 1
        if len(nums) >= 6:
            si, ms_seats, minority_seats, all_india, institute_seats, orphan = nums[:6]
        break

    if institute_code is None or choice_code is None:
        return []

    # Prefer institute from choice prefix (5 digits)
    if len(choice_code) >= 5 and choice_code[:5].isdigit():
        institute_code = int(choice_code[:5])
    course_code = choice_code[5:9] if len(choice_code) >= 9 else ""

    rows: list[dict] = []

    def add_row(
        quota: str,
        category_code: str,
        gender: str,
        seats: int,
    ) -> None:
        rows.append(
            {
                "year": year,
                "institute_code": institute_code,
                "institute_name": institute_name,
                "status": status,
                "choice_code": choice_code,
                "course_code": course_code,
                "course_name": course_name,
                "quota": quota,
                "category_code": category_code,
                "gender": gender,
                "seats": int(seats),
                "si": si,
                "cap_seats": cap_seats,
                "ms_seats": ms_seats,
                "minority_seats": minority_seats,
                "all_india": all_india,
                "institute_seats": institute_seats,
                "orphan": orphan,
                "source_pdf": source_pdf,
                "page_number": page_number,
            }
        )

    # Scan for quota grid rows
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln in {"State Level", "HU", "OHU"}:
            quota = {"State Level": "STATE", "HU": "HU", "OHU": "OHU"}[ln]
            # gather following ints until 19 collected
            nums = []
            k = i + 1
            while k < len(lines) and len(nums) < 19:
                if INT_RE.match(lines[k]):
                    nums.append(int(lines[k]))
                elif lines[k] in {"PWD", "DEF", "State Level", "HU", "OHU"}:
                    break
                k += 1
            if len(nums) >= 19:
                for idx, cat in enumerate(GRID_CATEGORIES):
                    add_row(quota, cat, "G", nums[idx * 2])
                    add_row(quota, cat, "L", nums[idx * 2 + 1])
                add_row(quota, "TOTAL", "ALL", nums[18])
            i = k
            continue

        if ln in {"PWD", "DEF"}:
            quota = ln
            nums = []
            k = i + 1
            while k < len(lines) and len(nums) < 10:
                if INT_RE.match(lines[k]):
                    nums.append(int(lines[k]))
                elif "Common Reserved" in lines[k] or lines[k] in {"PWD", "DEF"}:
                    break
                elif lines[k].startswith("Economically"):
                    break
                k += 1
            if len(nums) >= 10:
                for idx, cat in enumerate(PWD_DEF_CATEGORIES):
                    add_row(quota, cat, "ALL", nums[idx])
                add_row(quota, "TOTAL", "ALL", nums[9])
            i = k
            continue

        m = EWS_RE.search(ln)
        if m:
            add_row("EWS", "EWS", "ALL", int(m.group(1)))
            i += 1
            continue

        if "Tution Fee Waiver" in ln or "Tuition Fee Waiver" in ln:
            # seats may be on same line
            m2 = re.search(r"Seats\s*:\s*(\d+)", ln, re.I)
            if m2:
                add_row("TFWS", "TFWS", "ALL", int(m2.group(1)))
            i += 1
            continue

        if orphan and ln == "Orphan":
            pass
        i += 1

    if orphan:
        add_row("ORPHAN", "ORPHAN", "ALL", orphan)

    return rows


def parse_seat_matrix_pdf(
    pdf_path: Path | str,
    *,
    max_pages: int | None = None,
) -> list[dict]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    rows: list[dict] = []
    try:
        n = len(doc)
        limit = n if max_pages is None else min(max_pages, n)
        sample = doc[0].get_text("text") if n else ""
        year = infer_year_from_path(pdf_path, sample)
        logger.info("Parsing seat matrix %s year=%s pages=%s", pdf_path.name, year, limit)
        for page_idx in range(limit):
            text = doc[page_idx].get_text("text")
            page_rows = parse_seat_matrix_page_text(
                text,
                year=year,
                source_pdf=pdf_path.name,
                page_number=page_idx + 1,
            )
            rows.extend(page_rows)
            if (page_idx + 1) % 200 == 0:
                logger.info("  seat matrix progress %s/%s pages (%s rows)", page_idx + 1, limit, len(rows))
    finally:
        doc.close()
    logger.info("Parsed %s seat rows from %s", len(rows), pdf_path.name)
    return rows
