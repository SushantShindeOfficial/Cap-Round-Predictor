"""Fuzzy match student Home University to institute university strings."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

# Common Maharashtra university aliases → canonical tokens for matching
UNIVERSITY_ALIASES: dict[str, str] = {
    "mu": "mumbai university",
    "university of mumbai": "mumbai university",
    "mumbai univ": "mumbai university",
    "uom": "mumbai university",
    "sppu": "savitribai phule pune university",
    "pune university": "savitribai phule pune university",
    "university of pune": "savitribai phule pune university",
    "unipune": "savitribai phule pune university",
    "batu": "dr babasaheb ambedkar technological university",
    "dbatu": "dr babasaheb ambedkar technological university",
    "sgbau": "sant gadge baba amravati university",
    "amravati university": "sant gadge baba amravati university",
    "rtmnu": "rashtrasant tukadoji maharaj nagpur university",
    "nagpur university": "rashtrasant tukadoji maharaj nagpur university",
    "srtmu": "swami ramanand teerth marathwada university",
    "nanded university": "swami ramanand teerth marathwada university",
    "su": "shivaji university",
    "shivaji univ": "shivaji university",
    "bautu": "dr babasaheb ambedkar technological university",
    "autonomous institute": "autonomous institute",
    "autonomous": "autonomous institute",
}

HU_MATCH_THRESHOLD = 75.0


def normalize_university(name: str | None) -> str:
    if not name:
        return ""
    text = name.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in UNIVERSITY_ALIASES:
        return UNIVERSITY_ALIASES[text]
    # Try alias without "university" suffix noise
    compact = text.replace(" university", "").strip()
    if compact in UNIVERSITY_ALIASES:
        return UNIVERSITY_ALIASES[compact]
    return text


def score_university(student_hu: str, institute_university: str | None) -> float:
    """Return 0–100 fuzzy score between student HU and institute university label."""
    a = normalize_university(student_hu)
    b = normalize_university(institute_university)
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    return float(fuzz.token_set_ratio(a, b))


def is_hu_affiliated(
    student_hu: str,
    institute_university: str | None,
    *,
    threshold: float = HU_MATCH_THRESHOLD,
) -> bool:
    return score_university(student_hu, institute_university) >= threshold
