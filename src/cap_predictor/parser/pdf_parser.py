"""Parse Maharashtra CET CAP cutoff PDFs into CutoffRecord rows."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from cap_predictor.models import CutoffRecord
from cap_predictor.utils.category_map import (
    SEAT_TYPE_SECTIONS,
    is_plausible_category_token,
    match_section_header,
    parse_category_code,
    repair_category_headers,
)

logger = logging.getLogger(__name__)

INSTITUTE_RE = re.compile(r"^0*(\d{4})\s*-\s*(.+)$")
BRANCH_RE = re.compile(r"^0*(\d{9})\s*-\s*(.+)$")
PERCENTILE_RE = re.compile(r"^\((\d+\.\d+)\)$")
MERIT_RE = re.compile(r"^(\d{1,6})$")
YEAR_RE = re.compile(r"(20\d{2})")

X_MATCH_TOLERANCE = 28.0


@dataclass
class Span:
    x: float
    y: float
    text: str


@dataclass
class ParseStats:
    pages: int = 0
    institutes: set[int] = field(default_factory=set)
    branches: set[str] = field(default_factory=set)
    rows: int = 0
    pages_with_rows: int = 0
    pages_without_rows: int = 0
    unknown_tokens: dict[str, int] = field(default_factory=dict)
    quarantined_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        empty_pct = (
            round(100.0 * self.pages_without_rows / self.pages, 2) if self.pages else 0.0
        )
        return {
            "pages": self.pages,
            "rows": self.rows,
            "institutes": len(self.institutes),
            "branches": len(self.branches),
            "pages_with_rows": self.pages_with_rows,
            "pages_without_rows": self.pages_without_rows,
            "empty_page_pct": empty_pct,
            "unknown_token_count": sum(self.unknown_tokens.values()),
            "top_unknown_tokens": dict(
                sorted(self.unknown_tokens.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
            ),
            "quarantined_pages": self.quarantined_pages[:50],
        }


def infer_year_round(pdf_path: Path, page_text: str = "") -> tuple[int, int]:
    name = pdf_path.name
    year_match = YEAR_RE.search(name) or YEAR_RE.search(page_text)
    year = int(year_match.group(1)) if year_match else 2024

    round_num = 1
    m = re.search(r"CAP\s*([1-4])", name, re.IGNORECASE)
    if m:
        round_num = int(m.group(1))
    elif re.search(r"CAP\s*II\b", name, re.IGNORECASE) or re.search(
        r"CAP Round II", page_text, re.IGNORECASE
    ):
        round_num = 2
    elif re.search(r"CAP\s*III\b", name, re.IGNORECASE):
        round_num = 3
    elif re.search(r"CAP\s*I\b", name, re.IGNORECASE) or "CAP1" in name.upper():
        round_num = 1
    return year, round_num


def _page_spans(page: fitz.Page) -> list[Span]:
    spans: list[Span] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                x0, y0, _, _ = s["bbox"]
                spans.append(Span(x=float(x0), y=float(y0), text=text))
    spans.sort(key=lambda sp: (sp.y, sp.x))
    return spans


def _cluster_by_y(spans: list[Span], y_tol: float = 3.0) -> list[list[Span]]:
    if not spans:
        return []
    rows: list[list[Span]] = [[spans[0]]]
    for sp in spans[1:]:
        if abs(sp.y - rows[-1][0].y) <= y_tol:
            rows[-1].append(sp)
        else:
            rows.append([sp])
    for row in rows:
        row.sort(key=lambda s: s.x)
    return rows


def _join_row(row: list[Span]) -> str:
    return " ".join(s.text for s in row)


def _nearest_category(x: float, categories: list[tuple[float, str]]) -> str | None:
    if not categories:
        return None
    best = min(categories, key=lambda c: abs(c[0] - x))
    if abs(best[0] - x) <= X_MATCH_TOLERANCE:
        return best[1]
    return None


def _extract_university(status_text: str) -> str | None:
    m = re.search(r"Home University\s*:\s*(.+)$", status_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_pdf(
    pdf_path: Path | str,
    *,
    max_pages: int | None = None,
) -> tuple[list[CutoffRecord], ParseStats]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    stats = ParseStats()
    records: list[CutoffRecord] = []

    doc = fitz.open(pdf_path)
    try:
        n_pages = len(doc)
        stats.pages = n_pages if max_pages is None else min(max_pages, n_pages)
        sample_text = doc[0].get_text() if n_pages else ""
        year, round_num = infer_year_round(pdf_path, sample_text)
        logger.info(
            "Parsing %s (year=%s round=%s pages=%s)",
            pdf_path.name,
            year,
            round_num,
            stats.pages,
        )

        institute_code: int | None = None
        institute_name: str | None = None
        branch_code: str | None = None
        branch_name: str | None = None
        university: str | None = None
        seat_section: str | None = None
        categories: list[tuple[float, str]] = []
        header_buffer: list[Span] = []
        current_stage: str | None = None
        stage_values: dict[str, dict[str, tuple[int | None, float | None]]] = {}

        def flush_records(page_number: int) -> None:
            nonlocal stage_values
            if not (
                institute_code
                and institute_name
                and branch_code
                and branch_name
                and seat_section
            ):
                stage_values = {}
                return
            for stage, mapping in stage_values.items():
                for cat, (merit, pct) in mapping.items():
                    if merit is None and pct is None:
                        continue
                    parts = parse_category_code(cat)
                    rec = CutoffRecord(
                        year=year,
                        round=round_num,
                        institute_code=institute_code,
                        institute_name=institute_name,
                        university=university,
                        branch_code=branch_code,
                        branch_name=branch_name,
                        seat_type_section=seat_section,
                        category_code=cat,
                        gender_flag=parts.get("gender_flag"),  # type: ignore[arg-type]
                        reservation=parts.get("reservation"),  # type: ignore[arg-type]
                        quota_suffix=parts.get("quota_suffix"),  # type: ignore[arg-type]
                        stage=stage,  # type: ignore[arg-type]
                        closing_merit=merit,
                        closing_percentile=pct,
                        source_pdf=pdf_path.name,
                        page_number=page_number,
                    )
                    records.append(rec)
                    stats.rows += 1
                    stats.institutes.add(institute_code)
                    stats.branches.add(branch_code)
            stage_values = {}

        def commit_header_buffer() -> None:
            nonlocal categories, header_buffer
            if not header_buffer:
                return
            texts = [s.text for s in header_buffer]
            repaired = repair_category_headers(texts)
            cats: list[tuple[float, str]] = []
            buf = list(header_buffer)
            for code in repaired:
                if not buf:
                    break
                used: list[Span] = []
                acc = ""
                while buf and acc != code:
                    used.append(buf.pop(0))
                    acc = "".join(s.text.strip().upper() for s in used)
                    if len(acc) > len(code) + 2:
                        break
                cats.append((used[0].x if used else 0.0, code))
            if cats:
                categories = cats
            header_buffer = []

        for page_index in range(stats.pages):
            page = doc[page_index]
            page_number = page_index + 1
            spans = _page_spans(page)
            rows = _cluster_by_y(spans)
            rows_before = stats.rows

            for row in rows:
                joined = _join_row(row)
                if not joined.strip():
                    continue

                if joined.startswith("Legends:") or "Cut Off Indicates" in joined:
                    continue
                if joined in {"D", "i", "r"} or joined == "Government of Maharashtra":
                    continue
                if "State Common Entrance Test Cell" in joined:
                    continue
                if "Cut Off List for Maharashtra" in joined:
                    continue
                if "Degree Courses In Engineering" in joined:
                    continue

                inst_m = INSTITUTE_RE.match(joined.strip())
                if inst_m and len(inst_m.group(1)) == 4:
                    flush_records(page_number)
                    institute_code = int(inst_m.group(1))
                    institute_name = inst_m.group(2).strip()
                    branch_code = None
                    branch_name = None
                    university = None
                    seat_section = None
                    categories = []
                    header_buffer = []
                    current_stage = None
                    stage_values = {}
                    continue

                br_m = BRANCH_RE.match(joined.strip())
                if br_m:
                    flush_records(page_number)
                    branch_code = br_m.group(1)
                    branch_name = br_m.group(2).strip()
                    seat_section = None
                    categories = []
                    header_buffer = []
                    current_stage = None
                    stage_values = {}
                    continue

                # Seat section headers first — they contain the words "Home University".
                joined_norm = " ".join(joined.split())
                section_key = None
                for header, key in SEAT_TYPE_SECTIONS.items():
                    if joined_norm == header or joined_norm.startswith(header):
                        section_key = key
                        break
                if section_key is None:
                    section_key = match_section_header(joined_norm)

                if section_key:
                    flush_records(page_number)
                    seat_section = section_key
                    categories = []
                    header_buffer = []
                    current_stage = None
                    stage_values = {}
                    continue

                if joined_norm.startswith("Status:") or (
                    "Home University" in joined_norm and "Seats Allotted" not in joined_norm
                ):
                    university = _extract_university(joined_norm) or university
                    continue

                if (
                    len(row) == 1
                    and row[0].text.strip() == "Stage"
                    and row[0].x < 80
                ):
                    commit_header_buffer()
                    flush_records(page_number)
                    current_stage = None
                    continue

                stage_token = None
                merit_spans: list[Span] = []
                for sp in row:
                    t = sp.text.strip()
                    if t in {"I", "II", "III"} and sp.x < 70:
                        stage_token = t
                    elif MERIT_RE.match(t):
                        merit_spans.append(sp)

                if stage_token:
                    commit_header_buffer()
                    current_stage = stage_token
                    stage_values.setdefault(current_stage, {})
                    if categories and merit_spans:
                        for sp in merit_spans:
                            cat = _nearest_category(sp.x, categories)
                            if not cat:
                                continue
                            merit = int(sp.text)
                            prev = stage_values[current_stage].get(cat, (None, None))
                            stage_values[current_stage][cat] = (merit, prev[1])
                    continue

                pct_spans = [sp for sp in row if PERCENTILE_RE.match(sp.text.strip())]
                if pct_spans and current_stage and categories:
                    for sp in pct_spans:
                        cat = _nearest_category(sp.x, categories)
                        if not cat:
                            continue
                        m = PERCENTILE_RE.match(sp.text.strip())
                        if not m:
                            continue
                        pct = float(m.group(1))
                        prev = stage_values.setdefault(current_stage, {}).get(
                            cat, (None, None)
                        )
                        stage_values[current_stage][cat] = (prev[0], pct)
                    continue

                if (
                    current_stage
                    and categories
                    and any(MERIT_RE.match(sp.text.strip()) for sp in row)
                    and all(
                        MERIT_RE.match(sp.text.strip())
                        or sp.text.strip() in {"I", "II", "III"}
                        for sp in row
                    )
                ):
                    for sp in row:
                        if not MERIT_RE.match(sp.text.strip()):
                            continue
                        cat = _nearest_category(sp.x, categories)
                        if not cat:
                            continue
                        merit = int(sp.text)
                        prev = stage_values.setdefault(current_stage, {}).get(
                            cat, (None, None)
                        )
                        stage_values[current_stage][cat] = (merit, prev[1])
                    continue

                tokens = [sp for sp in row if is_plausible_category_token(sp.text)]
                if tokens and all(
                    is_plausible_category_token(sp.text)
                    and sp.text.strip().upper() not in {"STAGE", "I", "II", "III"}
                    for sp in tokens
                ):
                    if stage_values:
                        flush_records(page_number)
                    header_buffer.extend(tokens)
                    continue

                # Track uppercase-looking tokens that look like broken category headers
                for sp in row:
                    tok = sp.text.strip().upper()
                    if (
                        3 <= len(tok) <= 16
                        and tok.isalpha()
                        and tok not in {"STAGE", "STATUS", "GOVERNMENT", "UNIVERSITY"}
                        and not is_plausible_category_token(tok)
                        and not MERIT_RE.match(tok)
                    ):
                        stats.unknown_tokens[tok] = stats.unknown_tokens.get(tok, 0) + 1

            commit_header_buffer()
            if stage_values:
                flush_records(page_number)

            if stats.rows > rows_before:
                stats.pages_with_rows += 1
            else:
                stats.pages_without_rows += 1
                stats.quarantined_pages.append(page_number)

        flush_records(stats.pages)

    finally:
        doc.close()

    logger.info(
        "Parsed %s: rows=%s institutes=%s branches=%s empty_pages=%s/%s",
        pdf_path.name,
        stats.rows,
        len(stats.institutes),
        len(stats.branches),
        stats.pages_without_rows,
        stats.pages,
    )
    return records, stats


def parse_pdf_text_fixture(
    text: str,
    *,
    year: int = 2024,
    round_num: int = 1,
    source_pdf: str = "fixture.pdf",
) -> list[CutoffRecord]:
    """Line-based parser for unit-test fixtures (no coordinates)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    records: list[CutoffRecord] = []
    institute_code = institute_name = None
    branch_code = branch_name = None
    university = None
    seat_section = None
    categories: list[str] = []
    stage: str | None = None
    page_number = 1

    i = 0
    while i < len(lines):
        line = lines[i]
        inst_m = INSTITUTE_RE.match(line)
        if inst_m and len(inst_m.group(1)) == 4:
            institute_code = int(inst_m.group(1))
            institute_name = inst_m.group(2).strip()
            branch_code = None
            branch_name = None
            university = None
            seat_section = None
            categories = []
            stage = None
            i += 1
            continue
        br_m = BRANCH_RE.match(line)
        if br_m:
            branch_code = br_m.group(1)
            branch_name = br_m.group(2).strip()
            seat_section = None
            categories = []
            stage = None
            i += 1
            continue
        if line.startswith("Status:") or (
            "Home University :" in line and "Seats Allotted" not in line
        ):
            university = _extract_university(line) or university
            i += 1
            continue
        section = match_section_header(line)
        if section or line in SEAT_TYPE_SECTIONS:
            seat_section = section or SEAT_TYPE_SECTIONS.get(line, line)
            categories = []
            stage = None
            i += 1
            continue
        if line == "Stage":
            stage = None
            categories = []
            i += 1
            continue
        if line in {"I", "II", "III"}:
            stage = line
            i += 1
            values: list[tuple[int | None, float | None]] = []
            while i < len(lines):
                if MERIT_RE.match(lines[i]):
                    merit = int(lines[i])
                    pct = None
                    if i + 1 < len(lines) and PERCENTILE_RE.match(lines[i + 1]):
                        pct = float(PERCENTILE_RE.match(lines[i + 1]).group(1))  # type: ignore[union-attr]
                        i += 2
                    else:
                        i += 1
                    values.append((merit, pct))
                    continue
                break
            for cat, (merit, pct) in zip(categories, values):
                if not (
                    institute_code
                    and institute_name
                    and branch_code
                    and branch_name
                    and seat_section
                ):
                    continue
                parts = parse_category_code(cat)
                records.append(
                    CutoffRecord(
                        year=year,
                        round=round_num,
                        institute_code=institute_code,
                        institute_name=institute_name,
                        university=university,
                        branch_code=branch_code,
                        branch_name=branch_name,
                        seat_type_section=seat_section,
                        category_code=cat,
                        gender_flag=parts.get("gender_flag"),  # type: ignore[arg-type]
                        reservation=parts.get("reservation"),  # type: ignore[arg-type]
                        quota_suffix=parts.get("quota_suffix"),  # type: ignore[arg-type]
                        stage=stage,  # type: ignore[arg-type]
                        closing_merit=merit,
                        closing_percentile=pct,
                        source_pdf=source_pdf,
                        page_number=page_number,
                    )
                )
            continue

        if is_plausible_category_token(line) and line not in {"Stage", "I", "II"}:
            buf = [line]
            j = i + 1
            while (
                j < len(lines)
                and is_plausible_category_token(lines[j])
                and lines[j] not in {"I", "II", "III", "Stage"}
            ):
                buf.append(lines[j])
                j += 1
            categories = repair_category_headers(buf)
            i = j
            continue
        i += 1

    return records
