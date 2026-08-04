"""Knowledge-base query tools over DuckDB cutoffs + seat matrix (+ rulebook)."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from cap_predictor.data.duckdb_store import (
    TABLE_CUTOFFS,
    TABLE_SEATS,
    _connect,
    duckdb_path,
    table_exists,
)
from cap_predictor.utils.seat_matrix_map import (
    course_code_from_cutoff_branch,
    map_cutoff_category_to_seat_cell,
)

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Historical CAP data from parsed PDFs only — not an admission guarantee. "
    "Verify with official CET Cell allotment."
)
DEFAULT_LIMIT = 100


def _json_rows(df: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    work = df.head(limit).copy()
    # Convert numpy / NaN to JSON-friendly values
    records = work.where(pd.notna(work), None).to_dict(orient="records")
    out: list[dict[str, Any]] = []
    for rec in records:
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:  # noqa: BLE001
                    pass
            clean[str(k)] = v
        out.append(clean)
    return out


def _result(rows: list[dict[str, Any]], *, note: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "row_count": len(rows),
        "rows": rows,
        "disclaimer": DISCLAIMER,
    }
    if note:
        payload["note"] = note
    return payload


def query_cutoffs(
    *,
    max_closing_percentile: float | None = None,
    min_closing_percentile: float | None = None,
    min_closing_merit: int | None = None,
    max_closing_merit: int | None = None,
    category_code: str | None = None,
    branch_name: str | None = None,
    branch_names: list[str] | None = None,
    year: int | None = None,
    round_num: int | None = None,
    institute_code: int | None = None,
    university_contains: str | None = None,
    limit: int = DEFAULT_LIMIT,
    db_path: Any = None,
) -> dict[str, Any]:
    """Filter cutoff rows (percentile / merit / category / branch / year / round)."""
    path = db_path or duckdb_path()
    if not path.exists() or not table_exists(TABLE_CUTOFFS, db_path=path):
        return _result([], note="DuckDB cutoffs table missing. Run: python -m cap_predictor.cli build-db")

    clauses: list[str] = []
    params: list[Any] = []
    if max_closing_percentile is not None:
        clauses.append("closing_percentile IS NOT NULL AND closing_percentile <= ?")
        params.append(float(max_closing_percentile))
    if min_closing_percentile is not None:
        clauses.append("closing_percentile IS NOT NULL AND closing_percentile >= ?")
        params.append(float(min_closing_percentile))
    if min_closing_merit is not None:
        clauses.append("closing_merit IS NOT NULL AND closing_merit >= ?")
        params.append(int(min_closing_merit))
    if max_closing_merit is not None:
        clauses.append("closing_merit IS NOT NULL AND closing_merit <= ?")
        params.append(int(max_closing_merit))
    if category_code:
        clauses.append("upper(category_code) = ?")
        params.append(category_code.strip().upper())
    names = [b.strip() for b in (branch_names or []) if b and str(b).strip()]
    if branch_name and branch_name.strip():
        names.append(branch_name.strip())
    if names:
        ors = []
        for n in names:
            ors.append("lower(branch_name) = ?")
            params.append(n.lower())
        clauses.append("(" + " OR ".join(ors) + ")")
    if year is not None:
        clauses.append("year = ?")
        params.append(int(year))
    if round_num is not None:
        clauses.append("round = ?")
        params.append(int(round_num))
    if institute_code is not None:
        clauses.append("institute_code = ?")
        params.append(int(institute_code))
    if university_contains and university_contains.strip():
        clauses.append(
            "(lower(coalesce(university,'')) LIKE ? OR lower(institute_name) LIKE ?)"
        )
        needle = f"%{university_contains.strip().lower()}%"
        params.extend([needle, needle])

    cols = (
        "year, round, institute_code, institute_name, university, branch_code, "
        "branch_name, category_code, seat_type_section, closing_percentile, closing_merit"
    )
    sql = f"SELECT {cols} FROM {TABLE_CUTOFFS}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY closing_percentile ASC NULLS LAST, institute_code LIMIT ?"
    params.append(int(limit))

    con = _connect(path)
    try:
        df = con.execute(sql, params).df()
    finally:
        con.close()
    return _result(_json_rows(df, limit=limit))


def query_seats(
    *,
    category_code: str | None = None,
    gender: str | None = None,
    quota: str | None = None,
    course_code: str | None = None,
    course_name_contains: str | None = None,
    institute_code: int | None = None,
    year: int | None = None,
    min_seats: int | None = None,
    limit: int = DEFAULT_LIMIT,
    db_path: Any = None,
) -> dict[str, Any]:
    """Filter seat-matrix inventory rows."""
    path = db_path or duckdb_path()
    if not path.exists() or not table_exists(TABLE_SEATS, db_path=path):
        return _result(
            [],
            note="DuckDB seat_matrix missing. Parse seat matrix then run build-db.",
        )

    clauses: list[str] = []
    params: list[Any] = []
    if category_code:
        # Allow CAP codes like LNTD → map to matrix cell when possible
        raw = category_code.strip().upper()
        cell = map_cutoff_category_to_seat_cell(raw)
        if cell is not None:
            clauses.append("upper(category_code) = ?")
            params.append(cell.category_code.upper())
            # Only force quota when CAP code has H/O/S suffix
            has_suffix = raw.endswith(("H", "O", "S")) and not raw.endswith("WS")
            if quota:
                q = quota.strip().upper().replace("STATE LEVEL", "STATE")
                if q == "STATE_LEVEL":
                    q = "STATE"
                clauses.append("upper(quota) = ?")
                params.append(q)
            elif has_suffix:
                clauses.append("upper(quota) = ?")
                params.append(cell.quota.upper())
            if not gender and cell.gender in {"G", "L"}:
                clauses.append("upper(gender) = ?")
                params.append(cell.gender.upper())
        else:
            clauses.append("upper(category_code) = ?")
            params.append(raw)
    if gender:
        clauses.append("upper(gender) = ?")
        params.append(gender.strip().upper())
    if quota:
        q = quota.strip().upper().replace("STATE LEVEL", "STATE")
        if q == "STATE_LEVEL":
            q = "STATE"
        clauses.append("upper(quota) = ?")
        params.append(q)
    if course_code:
        clauses.append("cast(course_code AS VARCHAR) = ?")
        params.append(str(course_code).strip())
    if course_name_contains and course_name_contains.strip():
        clauses.append("lower(coalesce(course_name,'')) LIKE ?")
        params.append(f"%{course_name_contains.strip().lower()}%")
    if institute_code is not None:
        clauses.append("institute_code = ?")
        params.append(int(institute_code))
    if year is not None:
        clauses.append("year = ?")
        params.append(int(year))
    if min_seats is not None:
        clauses.append("seats IS NOT NULL AND seats >= ?")
        params.append(int(min_seats))

    cols = (
        "year, institute_code, institute_name, choice_code, course_code, course_name, "
        "quota, category_code, gender, seats, si"
    )
    sql = f"SELECT {cols} FROM {TABLE_SEATS}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY seats DESC NULLS LAST, institute_code LIMIT ?"
    params.append(int(limit))

    con = _connect(path)
    try:
        df = con.execute(sql, params).df()
    finally:
        con.close()
    return _result(_json_rows(df, limit=limit))


def query_seats_with_cutoffs(
    *,
    category_code: str,
    year: int | None = None,
    branch_name: str | None = None,
    min_seats: int | None = 1,
    limit: int = DEFAULT_LIMIT,
    db_path: Any = None,
) -> dict[str, Any]:
    """
    Find seat-matrix cells for a CAP category (e.g. LNTD) and attach matching cutoff rows.
    """
    seats_payload = query_seats(
        category_code=category_code,
        year=year,
        course_name_contains=branch_name,
        min_seats=min_seats,
        limit=limit,
        db_path=db_path,
    )
    seat_rows = seats_payload.get("rows") or []
    if not seat_rows:
        return _result([], note=seats_payload.get("note") or "No seat rows for that category.")

    path = db_path or duckdb_path()
    if not table_exists(TABLE_CUTOFFS, db_path=path):
        return {
            **seats_payload,
            "note": "Seats found but cutoffs table missing.",
            "joined": [],
        }

    cat = category_code.strip().upper()
    joined: list[dict[str, Any]] = []
    con = _connect(path)
    try:
        for s in seat_rows:
            inst = int(s["institute_code"])
            course = str(s.get("course_code") or "")
            syear = s.get("year")
            # Cutoffs whose branch encodes this course under the institute
            sql = (
                f"SELECT year, round, institute_code, institute_name, branch_code, branch_name, "
                f"category_code, closing_percentile, closing_merit "
                f"FROM {TABLE_CUTOFFS} "
                f"WHERE institute_code = ? AND upper(category_code) = ?"
            )
            params: list[Any] = [inst, cat]
            if year is not None:
                sql += " AND year = ?"
                params.append(int(year))
            elif syear is not None:
                sql += " AND year = ?"
                params.append(int(syear))
            if branch_name:
                sql += " AND lower(branch_name) LIKE ?"
                params.append(f"%{branch_name.lower()}%")
            sql += " ORDER BY year DESC, round DESC LIMIT 20"
            cdf = con.execute(sql, params).df()
            if not cdf.empty and course:
                # Prefer rows whose derived course_code matches
                keep_idx = []
                for i, row in cdf.iterrows():
                    derived = course_code_from_cutoff_branch(
                        str(row["branch_code"]), int(row["institute_code"])
                    )
                    if derived == course:
                        keep_idx.append(i)
                if keep_idx:
                    cdf = cdf.loc[keep_idx]
            cut_rows = _json_rows(cdf, limit=5)
            joined.append(
                {
                    "seat": s,
                    "cutoffs": cut_rows,
                    "cutoff_count": len(cut_rows),
                }
            )
            if len(joined) >= limit:
                break
    finally:
        con.close()

    return {
        "row_count": len(joined),
        "rows": joined,
        "disclaimer": DISCLAIMER,
        "category_code": cat,
    }


def search_rulebook(
    query: str,
    *,
    max_excerpts: int = 5,
) -> dict[str, Any]:
    """Keyword search over indexed CET Information Brochure pages."""
    from cap_predictor.data.rulebook_store import load_rulebook_chunks
    from cap_predictor.models import StudentProfile

    q = (query or "").strip()
    if not q:
        return _result([], note="Empty rulebook query.")

    # Reuse excerpt scorer via a synthetic profile when query looks like a category
    upper = q.upper()
    profile = StudentProfile(
        percentile=50.0,
        category=upper if re.search(r"[A-Z]{3,}", upper) else "GOPENS",
        gender="M",
        home_university="ALL",
        seat_type="ALL",
        target_branch="ALL",
    )
    try:
        from cap_predictor.data.rulebook_store import retrieve_rulebook_excerpts

        excerpts = retrieve_rulebook_excerpts(profile, max_excerpts=max_excerpts)
    except Exception:  # noqa: BLE001
        excerpts = []

    # Also score raw chunks against free-text query
    chunks = load_rulebook_chunks(ensure_parsed=False)
    terms = [t for t in re.split(r"\W+", q) if len(t) >= 3]
    scored: list[tuple[float, dict]] = []
    for ch in chunks:
        text = ch.get("text") or ""
        low = text.lower()
        score = sum(1.0 for t in terms if t.lower() in low)
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: (-x[0], x[1].get("page", 0)))
    for score, ch in scored[:max_excerpts]:
        text = re.sub(r"\s+", " ", ch["text"]).strip()
        if len(text) > 700:
            text = text[:699].rsplit(" ", 1)[0] + "…"
        excerpts.append(
            {
                "page": ch.get("page"),
                "source_pdf": ch.get("source_pdf"),
                "score": round(score, 2),
                "text": text,
            }
        )

    # Dedupe by page
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for ex in excerpts:
        page = int(ex.get("page") or 0)
        if page in seen:
            continue
        seen.add(page)
        unique.append(ex)
        if len(unique) >= max_excerpts:
            break

    return {
        "row_count": len(unique),
        "rows": unique,
        "disclaimer": DISCLAIMER,
        "query": q,
    }


# Tool registry for MCP / Streamlit
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "query_cutoffs",
        "description": (
            "Search CAP cutoff rows. Use for percentile/merit filters "
            "(e.g. closing_percentile <= 71, merit 40000-60000), category codes (LNTD), branch, year, round."
        ),
        "parameters": [
            "max_closing_percentile",
            "min_closing_percentile",
            "min_closing_merit",
            "max_closing_merit",
            "category_code",
            "branch_name",
            "year",
            "round_num",
            "institute_code",
            "university_contains",
            "limit",
        ],
    },
    {
        "name": "query_seats",
        "description": (
            "Search seat-matrix inventory (seats available by quota/category/gender). "
            "category_code may be a CAP code like LNTD or matrix OPEN/SC/…"
        ),
        "parameters": [
            "category_code",
            "gender",
            "quota",
            "course_code",
            "course_name_contains",
            "institute_code",
            "year",
            "min_seats",
            "limit",
        ],
    },
    {
        "name": "query_seats_with_cutoffs",
        "description": (
            "Find institutes with seats for a CAP category (e.g. LNTD) and attach matching cutoff rows."
        ),
        "parameters": ["category_code", "year", "branch_name", "min_seats", "limit"],
    },
    {
        "name": "search_rulebook",
        "description": "Search the CET Information Brochure for official rule text (HU/OHU/TFWS/CAP process).",
        "parameters": ["query", "max_excerpts"],
    },
]


def run_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch a KB tool by name."""
    import inspect

    args = dict(arguments or {})
    dispatch = {
        "query_cutoffs": query_cutoffs,
        "query_seats": query_seats,
        "query_seats_with_cutoffs": query_seats_with_cutoffs,
        "search_rulebook": search_rulebook,
    }
    fn = dispatch.get(name)
    if fn is None:
        return _result([], note=f"Unknown tool: {name}")
    if name == "search_rulebook":
        return search_rulebook(
            str(args.get("query") or ""),
            max_excerpts=int(args.get("max_excerpts") or 5),
        )
    if name == "query_seats_with_cutoffs" and "category_code" not in args:
        return _result([], note="category_code is required")
    sig = inspect.signature(fn)
    allowed = {k: v for k, v in args.items() if k in sig.parameters}
    return fn(**allowed)
