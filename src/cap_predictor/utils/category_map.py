"""Maharashtra CAP category code vocabulary and student → code mapping."""

from __future__ import annotations

import re
from typing import Iterable

# Known reservation bodies (longest first for parsing / repair)
RESERVATION_BODIES: tuple[str, ...] = (
    "PWDOPEN",
    "PWDOBC",
    "PWDRSC",
    "PWDROBC",
    "PWDRST",
    "PWDRVJ",
    "PWDRNT1",
    "PWDRNT2",
    "PWDRNT3",
    "DEFOPEN",
    "DEFOBC",
    "DEFRSC",
    "DEFROBC",
    "DEFRST",
    "DEFRVJ",
    "DEFRNT1",
    "DEFRNT2",
    "DEFRNT3",
    "DEFRSEBC",
    "SEBC",
    "OPEN",
    "NT1",
    "NT2",
    "NT3",
    "OBC",
    "SC",
    "ST",
    "VJ",
    "EWS",
    "TFWS",
    "ORPHAN",
    "MI",
)

GENDER_PREFIXES = ("G", "L")
QUOTA_SUFFIXES = ("H", "O", "S", "AI")

SEAT_TYPE_SECTIONS: dict[str, str] = {
    "State Level": "STATE_LEVEL",
    "Home University Seats Allotted to Home University Candidates": "HU_TO_HU",
    "Home University Seats Allotted to Other Than Home University Candidates": "HU_TO_OHU",
    "Other Than Home University Seats Allotted to Other Than Home University Candidates": "OHU_TO_OHU",
    "Other Than Home University Seats Allotted to Home University Candidates": "OHU_TO_HU",
}

CATEGORY_TOKEN_RE = re.compile(
    r"^(?P<code>"
    r"(?:G|L)?"
    r"(?:PWDOPEN|PWDOBC|PWDRSC|PWDROBC|PWDRST|PWDRVJ|PWDRNT1|PWDRNT2|PWDRNT3|"
    r"DEFOPEN|DEFOBC|DEFRSC|DEFROBC|DEFRST|DEFRVJ|DEFRNT1|DEFRNT2|DEFRNT3|DEFRSEBC|"
    r"SEBC|OPEN|NT1|NT2|NT3|OBC|SC|ST|VJ|EWS|TFWS|ORPHAN|MI)"
    r"(?:H|O|S|AI)?"
    r")$"
)

# Student category aliases → reservation body
CATEGORY_ALIASES: dict[str, str] = {
    "OPEN": "OPEN",
    "GENERAL": "OPEN",
    "GEN": "OPEN",
    "SC": "SC",
    "ST": "ST",
    "VJ": "VJ",
    "DT": "VJ",
    "NT": "NT1",
    "NT-B": "NT1",
    "NT1": "NT1",
    "NT-C": "NT2",
    "NT2": "NT2",
    "NT-D": "NT3",
    "NT3": "NT3",
    "OBC": "OBC",
    "SEBC": "SEBC",
    "EWS": "EWS",
    "TFWS": "TFWS",
    "ORPHAN": "ORPHAN",
}

# Bodies that commonly appear as G/L + H/O/S in CAP PDFs
_GENDERED_BODIES: tuple[str, ...] = (
    "OPEN",
    "SC",
    "ST",
    "VJ",
    "NT1",
    "NT2",
    "NT3",
    "OBC",
    "SEBC",
)

# PWD / DEF style codes (usually State Level; include H/O/S for completeness)
_SPECIAL_BODIES: tuple[str, ...] = (
    "PWDOPEN",
    "PWDOBC",
    "PWDRSC",
    "PWDROBC",
    "DEFOPEN",
    "DEFOBC",
    "DEFROBC",
    "DEFRSC",
    "DEFRSEBC",
)


def build_cap_category_seats() -> list[str]:
    """Full CAP seat-code dropdown: G/L × body × H/O/S plus state-only specials."""
    seats: list[str] = []
    for body in _GENDERED_BODIES:
        for gender in ("G", "L"):
            for suffix in ("H", "O", "S"):
                seats.append(f"{gender}{body}{suffix}")
    for body in _SPECIAL_BODIES:
        for suffix in ("H", "O", "S"):
            seats.append(f"{body}{suffix}")
    for special in ("TFWS", "EWS", "ORPHAN"):
        seats.append(special)
        seats.append(f"{special}S")
    # Deduplicate preserve order
    seen: set[str] = set()
    out: list[str] = []
    for code in seats:
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


# Exact CAP seat-column codes shown in UI dropdowns (PDF header style)
CAP_CATEGORY_SEATS: list[str] = build_cap_category_seats()


def normalize_student_category(category: str) -> str:
    key = re.sub(r"\s+", "", category.strip().upper())
    key = key.replace("_", "").replace("/", "")
    return CATEGORY_ALIASES.get(key, key)


def is_full_cap_category_code(category: str) -> bool:
    """True when input is an exact PDF seat code (e.g. GOPENS), not OPEN/OBC."""
    key = re.sub(r"\s+", "", category.strip().upper())
    if key in CAP_CATEGORY_SEATS:
        return True
    # Any other valid CAP token that is not a short reservation alias
    if key in CATEGORY_ALIASES:
        return False
    return bool(CATEGORY_TOKEN_RE.match(key))


def parse_category_code(code: str) -> dict[str, str | None]:
    """Split a CAP category code into gender / reservation / quota parts."""
    raw = code.strip().upper()
    gender: str | None = None
    quota: str | None = None
    body = raw

    if raw.startswith(("G", "L")) and len(raw) > 1:
        # Prefer longest matching body after gender prefix
        for reservation in RESERVATION_BODIES:
            for suffix in (*QUOTA_SUFFIXES, ""):
                candidate = f"{raw[0]}{reservation}{suffix}"
                if candidate == raw:
                    return {
                        "gender_flag": raw[0],
                        "reservation": reservation,
                        "quota_suffix": suffix or None,
                        "category_code": raw,
                    }
        gender = raw[0]
        body = raw[1:]

    for reservation in RESERVATION_BODIES:
        for suffix in (*QUOTA_SUFFIXES, ""):
            candidate = f"{reservation}{suffix}"
            if candidate == body:
                return {
                    "gender_flag": gender,
                    "reservation": reservation,
                    "quota_suffix": suffix or None,
                    "category_code": raw,
                }

    # Fallback: strip trailing known suffix
    for suffix in QUOTA_SUFFIXES:
        if body.endswith(suffix) and len(body) > len(suffix):
            return {
                "gender_flag": gender,
                "reservation": body[: -len(suffix)],
                "quota_suffix": suffix,
                "category_code": raw,
            }

    return {
        "gender_flag": gender,
        "reservation": body,
        "quota_suffix": None,
        "category_code": raw,
    }


def is_plausible_category_token(token: str) -> bool:
    t = token.strip().upper()
    if not t or t in {"STAGE", "I", "II", "III"}:
        return False
    if CATEGORY_TOKEN_RE.match(t):
        return True
    # Partial tokens that appear when PDF wraps (e.g. PWDROBC, S)
    if t in QUOTA_SUFFIXES:
        return True
    return any(t.startswith(r) or r.startswith(t) for r in RESERVATION_BODIES)


def repair_category_headers(tokens: Iterable[str]) -> list[str]:
    """Join split category tokens such as PWDROBC + S → PWDROBCS."""
    parts = [t.strip().upper() for t in tokens if t and t.strip()]
    repaired: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        if (
            i + 1 < len(parts)
            and parts[i + 1] in QUOTA_SUFFIXES
            and not cur.endswith(tuple(QUOTA_SUFFIXES))
            and is_plausible_category_token(cur + parts[i + 1])
        ):
            merged = cur + parts[i + 1]
            if CATEGORY_TOKEN_RE.match(merged) or parse_category_code(merged)["reservation"]:
                repaired.append(merged)
                i += 2
                continue
        repaired.append(cur)
        i += 1
    return [t for t in repaired if CATEGORY_TOKEN_RE.match(t) or t in {"EWS", "TFWS", "ORPHAN"}]


# Special / certificate seats — excluded from Category=(All) unless chosen exactly
SPECIAL_SEAT_MARKERS: tuple[str, ...] = (
    "TFWS",
    "EWS",
    "ORPHAN",
    "MI",
    "PWD",
    "DEF",
)


def is_special_cap_code(code: str) -> bool:
    """True for TFWS / EWS / ORPHAN / MI / PWD* / DEF* seat codes."""
    raw = re.sub(r"\s+", "", code.strip().upper())
    if not raw:
        return False
    if raw in {"TFWS", "EWS", "ORPHAN", "MI"} or raw.startswith(("TFWS", "EWS", "ORPHAN")):
        return True
    if "PWD" in raw or raw.startswith("DEF") or "DEF" in raw:
        return True
    return False


def is_open_cap_code(code: str) -> bool:
    """True for OPEN seat columns: GOPENH/GOPENO/GOPENS/LOPEN* (not OBC/SC/…)."""
    raw = re.sub(r"\s+", "", code.strip().upper())
    return bool(re.fullmatch(r"[GL]?OPEN[HOSA]?", raw)) or raw in {
        "GOPENH",
        "GOPENO",
        "GOPENS",
        "LOPENH",
        "LOPENO",
        "LOPENS",
        "GOPEN",
        "LOPEN",
        "OPEN",
    }


def general_codes_for_gender(
    available_codes: list[str],
    gender: str,
    *,
    seat_type: str = "ALL",
) -> list[str]:
    """
    When Category is All: use OPEN seats only for the selected gender (+ seat type).

    This keeps the flow aligned with a typical student's percentile comparison:
    do not mix SC/ST/OBC/NT/SEBC cutoffs into an Open counselling run.
    Pick those exact CAP codes explicitly when the student belongs to that category.
    """
    g = gender.strip().upper()
    all_gender = g in {"ALL", "(ALL)", "*", "ANY"}
    female = g in {"F", "FEMALE", "L", "LADIES"}
    st = seat_type.strip().upper().replace(" ", "_")

    allowed_suffixes: set[str] | None
    if st in {"HU", "HOME", "HOME_UNIVERSITY", "H"}:
        allowed_suffixes = {"H", "S"}
    elif st in {"OHU", "OTHER", "OTHER_THAN_HOME", "O"}:
        allowed_suffixes = {"O", "S"}
    elif st in {"STATE", "STATE_LEVEL", "S"}:
        allowed_suffixes = {"S"}
    else:
        allowed_suffixes = None  # H/O/S all

    out: list[str] = []
    for code in available_codes:
        raw = str(code).strip().upper()
        if not raw or is_special_cap_code(raw):
            continue
        if not is_open_cap_code(raw):
            continue
        # Gender: male -> GOPEN*; female -> LOPEN* and GOPEN*; ALL -> both
        if raw.startswith("L") and not (female or all_gender):
            continue
        if allowed_suffixes is not None and raw[-1] in {"H", "O", "S"}:
            if raw[-1] not in allowed_suffixes:
                continue
        out.append(raw)

    seen: set[str] = set()
    deduped: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped
def applicable_category_codes(
    *,
    category: str,
    gender: str,
    seat_type: str,
) -> list[str]:
    """
    Build likely PDF category codes for a student profile.

    If `category` is already a full CAP seat code (e.g. GOPENS), return it
    as an exact match list. Otherwise expand short aliases (OPEN → GOPENH/…).

    category ALL returns [] — caller must use general_codes_for_gender on data codes.
    """
    raw = re.sub(r"\s+", "", category.strip().upper())
    if raw in {"ALL", "(ALL)", "*", "ANY"}:
        return []  # caller: general_codes_for_gender
    if is_full_cap_category_code(raw):
        return [raw]

    reservation = normalize_student_category(category)
    g = gender.strip().upper()
    female = g in {"F", "FEMALE", "L", "LADIES"}
    all_gender = g in {"ALL", "(ALL)", "*", "ANY"}
    st = seat_type.strip().upper().replace(" ", "_")

    suffixes: list[str]
    if st in {"HU", "HOME", "HOME_UNIVERSITY", "H"}:
        suffixes = ["H", "S"]
    elif st in {"OHU", "OTHER", "OTHER_THAN_HOME", "O"}:
        suffixes = ["O", "S"]
    elif st in {"STATE", "STATE_LEVEL", "S"}:
        suffixes = ["S"]
    else:
        suffixes = ["H", "O", "S", ""]

    codes: list[str] = []

    # Special state-level-only style codes
    if reservation in {"EWS", "TFWS", "ORPHAN"}:
        codes.append(reservation)
        for sfx in ("S", ""):
            codes.append(f"{reservation}{sfx}")
    else:
        if all_gender:
            gender_prefixes = ["G", "L"]
        elif female:
            gender_prefixes = ["L", "G"]
        else:
            gender_prefixes = ["G"]
        for gp in gender_prefixes:
            for sfx in suffixes:
                codes.append(f"{gp}{reservation}{sfx}")

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def match_section_header(line: str) -> str | None:
    text = " ".join(line.split())
    for header, key in SEAT_TYPE_SECTIONS.items():
        if text == header or text.startswith(header):
            return key
    return None


def category_suffix_conflicts_seat_type(category: str, seat_type: str) -> str | None:
    """Return a warning if exact CAP code suffix conflicts with Seat Type."""
    raw = re.sub(r"\s+", "", category.strip().upper())
    if not is_full_cap_category_code(raw) or raw in {"ALL", "(ALL)"}:
        return None
    st = seat_type.strip().upper()
    if st in {"ALL", "ANY", ""}:
        return None
    suffix = raw[-1] if raw[-1] in {"H", "O", "S"} else None
    if suffix == "H" and st not in {"HU", "HOME", "HOME_UNIVERSITY", "H"}:
        return f"Category {raw} is an HU (H) seat code but Seat Type is {seat_type}."
    if suffix == "O" and st not in {"OHU", "OTHER", "OTHER_THAN_HOME", "O"}:
        return f"Category {raw} is an OHU (O) seat code but Seat Type is {seat_type}."
    if suffix == "S" and st not in {"STATE", "STATE_LEVEL", "S", "ALL"}:
        return (
            f"Category {raw} is a State-Level (S) seat code but Seat Type is {seat_type}. "
            "Results will still match that exact code where present."
        )
    return None
