"""Map CAP cutoff category codes to seat-matrix cells."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cap_predictor.utils.category_map import parse_category_code

# Cutoff reservation body → seat-matrix category column
RESERVATION_TO_MATRIX: dict[str, str] = {
    "OPEN": "OPEN",
    "SC": "SC",
    "ST": "ST",
    "VJ": "VJDT",
    "NT1": "NTB",
    "NT2": "NTC",
    "NT3": "NTD",
    "OBC": "OBC",
    "SEBC": "SEBC",
    "EWS": "EWS",
    "TFWS": "TFWS",
    "ORPHAN": "ORPHAN",
}


@dataclass(frozen=True)
class SeatCellKey:
    quota: str  # STATE | HU | OHU | EWS | TFWS | PWD | DEF | ORPHAN
    category_code: str
    gender: str  # G | L | ALL


def map_cutoff_category_to_seat_cell(category_code: str) -> SeatCellKey | None:
    """
    Map e.g. GOPENH → HU / OPEN / G, GOPENS → STATE / OPEN / G.
    """
    raw = re.sub(r"\s+", "", category_code.strip().upper())
    if not raw:
        return None

    if raw in {"EWS", "EWSS"} or raw.startswith("EWS"):
        return SeatCellKey("EWS", "EWS", "ALL")
    if raw in {"TFWS", "TFWSS"} or raw.startswith("TFWS"):
        return SeatCellKey("TFWS", "TFWS", "ALL")
    if raw.startswith("ORPHAN"):
        return SeatCellKey("ORPHAN", "ORPHAN", "ALL")
    if "PWD" in raw:
        return SeatCellKey("PWD", "OPEN" if "OPEN" in raw else _pwd_body(raw), "ALL")
    if raw.startswith("DEF") or "DEF" in raw[:6]:
        return SeatCellKey("DEF", "OPEN" if "OPEN" in raw else _def_body(raw), "ALL")

    parts = parse_category_code(raw)
    reservation = str(parts.get("reservation") or "")
    gender_flag = parts.get("gender_flag")
    suffix = parts.get("quota_suffix")

    matrix_cat = RESERVATION_TO_MATRIX.get(reservation)
    if not matrix_cat:
        # Try without trailing junk
        for key, val in RESERVATION_TO_MATRIX.items():
            if reservation.startswith(key):
                matrix_cat = val
                break
    if not matrix_cat:
        return None

    if suffix == "H":
        quota = "HU"
    elif suffix == "O":
        quota = "OHU"
    elif suffix == "S" or suffix is None:
        quota = "STATE"
    else:
        quota = "STATE"

    gender = "L" if gender_flag == "L" else "G" if gender_flag == "G" else "ALL"
    return SeatCellKey(quota=quota, category_code=matrix_cat, gender=gender)


def _pwd_body(raw: str) -> str:
    for key, val in RESERVATION_TO_MATRIX.items():
        if key in raw and key != "OPEN":
            return val
    return "OPEN"


def _def_body(raw: str) -> str:
    return _pwd_body(raw)


def course_code_from_cutoff_branch(branch_code: str, institute_code: int) -> str:
    """
    Derive 4-digit CET course code from cutoff branch_code.

    Examples:
    - 0100224210 / 100224210 → 2421
    - 301224510 → 2451
    """
    digits = re.sub(r"\D", "", str(branch_code))
    inst = f"{int(institute_code):05d}"
    if digits.startswith(inst) and len(digits) >= 9:
        return digits[5:9]
    # 4-digit institute prefix common in cutoffs
    inst4 = f"{int(institute_code):04d}"
    if digits.startswith(inst4) and len(digits) >= 8:
        return digits[4:8]
    if len(digits) >= 9:
        return digits[-5:-1]
    if len(digits) >= 4:
        return digits[:4]
    return digits
