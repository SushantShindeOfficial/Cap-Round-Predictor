"""Load structured CAP rules and retrieve brochure excerpts for counselling."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from cap_predictor.models import StudentProfile
from cap_predictor.parser.rulebook_parser import chunks_cache_path, parse_and_cache_rulebook
from cap_predictor.utils.paths import data_masters_dir, default_rulebook_pdf_paths

logger = logging.getLogger(__name__)


def cap_rules_path() -> Path:
    return data_masters_dir() / "cap_rules_2026.json"


@lru_cache(maxsize=1)
def load_cap_rules() -> dict[str, Any]:
    path = cap_rules_path()
    if not path.exists():
        logger.warning("Missing %s — rulebook structured notes unavailable", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load cap rules: %s", exc)
        return {}


def clear_rules_cache() -> None:
    load_cap_rules.cache_clear()


def seat_type_definition(seat_type: str, rules: dict[str, Any] | None = None) -> str | None:
    rules = rules if rules is not None else load_cap_rules()
    defs = rules.get("definitions") or {}
    st = seat_type.strip().upper()
    if st in {"ALL", "(ALL)", "*", "ANY"}:
        return (
            "Seat Type All: do not prefer a single HU/OHU/STATE section; "
            "definitions — "
            + "; ".join(f"{k}: {v}" for k, v in defs.items())
        )
    return defs.get(st)


def jurisdiction_for_university(university: str, rules: dict[str, Any] | None = None) -> list[str]:
    rules = rules if rules is not None else load_cap_rules()
    needle = university.strip().lower()
    if not needle or needle in {"all", "(all)"}:
        return []
    for row in rules.get("university_jurisdiction") or []:
        name = str(row.get("university") or "")
        if needle in name.lower() or name.lower() in needle:
            return list(row.get("districts") or [])
        # token overlap
        if "mumbai" in needle and "mumbai university" in name.lower():
            return list(row.get("districts") or [])
        if "pune" in needle and "pune university" in name.lower():
            return list(row.get("districts") or [])
    return []


def _category_keys(category: str) -> list[str]:
    raw = re.sub(r"\s+", "", category.strip().upper())
    keys: list[str] = []
    for key in ("TFWS", "EWS", "PWD", "ORPHAN", "DEF", "MI"):
        if key in raw or raw.startswith(key):
            keys.append(key)
    if raw.startswith("DEF") or "DEF" in raw[:6]:
        if "DEF" not in keys:
            keys.append("DEF")
    return keys


def structured_eligibility_notes(profile: StudentProfile) -> list[str]:
    """Brochure-backed soft eligibility notes (never hard-block)."""
    rules = load_cap_rules()
    if not rules:
        return []
    notes: list[str] = []
    ay = rules.get("academic_year") or "2026-27"
    src = "CET Information Brochure " + str(ay)
    elig = rules.get("eligibility") or {}

    for key in _category_keys(profile.category):
        block = elig.get(key) or {}
        summary = block.get("summary")
        if summary:
            notes.append(f"Eligibility ({key}, {src}): {summary}")

    st_def = seat_type_definition(profile.seat_type, rules)
    if st_def and profile.seat_type.strip().upper() not in {"ALL"}:
        notes.append(f"Seat type ({profile.seat_type}, {src}): {st_def}")

    districts = jurisdiction_for_university(profile.home_university, rules)
    if districts:
        notes.append(
            f"Home University jurisdiction ({profile.home_university}, {src}): "
            f"districts — {', '.join(districts)}."
        )

    for line in rules.get("cap_process_summary") or []:
        if line:
            notes.append(f"CAP process ({src}): {line}")
            break  # one process note by default; more via excerpts

    disclaimer = rules.get("disclaimer")
    if disclaimer and notes:
        notes.append(disclaimer)
    return notes


def load_rulebook_chunks(*, ensure_parsed: bool = True) -> list[dict]:
    path = chunks_cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if ensure_parsed and default_rulebook_pdf_paths():
        return parse_and_cache_rulebook(force=False)
    return []


def _query_terms(profile: StudentProfile) -> list[str]:
    terms: list[str] = []
    cat = profile.category.strip().upper()
    for key in _category_keys(cat):
        terms.append(key)
        block = (load_cap_rules().get("eligibility") or {}).get(key) or {}
        terms.extend(block.get("keywords") or [])
    st = profile.seat_type.strip().upper()
    if st in {"HU", "OHU", "STATE"}:
        terms.extend(
            {
                "HU": ["Home University", "HU"],
                "OHU": ["Outside Home University", "OHU", "Other than Home"],
                "STATE": ["State Level", "State level seats"],
            }[st]
        )
    if profile.home_university and profile.home_university.upper() not in {"ALL"}:
        terms.append(profile.home_university)
        terms.append("Jurisdiction")
        terms.append("Home University")
    terms.extend(["CAP Round", "allotment", "Inter-Se-Merit"])
    # unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def retrieve_rulebook_excerpts(
    profile: StudentProfile,
    *,
    max_excerpts: int = 4,
    max_chars: int = 700,
) -> list[dict[str, Any]]:
    """Keyword retrieve short brochure page excerpts for LLM grounding."""
    chunks = load_rulebook_chunks(ensure_parsed=True)
    if not chunks:
        return []
    terms = _query_terms(profile)
    if not terms:
        return []

    scored: list[tuple[float, dict]] = []
    for ch in chunks:
        text = ch.get("text") or ""
        low = text.lower()
        score = 0.0
        for t in terms:
            if t.lower() in low:
                score += 1.0 + (0.5 if len(t) > 8 else 0.0)
        if score <= 0:
            continue
        scored.append((score, ch))

    scored.sort(key=lambda x: (-x[0], x[1].get("page", 0)))
    excerpts: list[dict[str, Any]] = []
    for score, ch in scored[:max_excerpts]:
        text = re.sub(r"\s+", " ", ch["text"]).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1].rsplit(" ", 1)[0] + "…"
        excerpts.append(
            {
                "page": ch.get("page"),
                "source_pdf": ch.get("source_pdf"),
                "score": round(score, 2),
                "text": text,
            }
        )
    return excerpts
