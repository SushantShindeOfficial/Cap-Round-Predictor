"""Basic eligibility notes for special CAP seat types."""

from __future__ import annotations

from cap_predictor.models import StudentProfile


def eligibility_notes(profile: StudentProfile) -> list[str]:
    """
    Soft counselling warnings for TFWS / PWD / DEF / ORPHAN seat codes.

    These do not hard-block predictions; they remind users to verify certificates.
    """
    code = profile.category.strip().upper()
    notes: list[str] = []

    if code.startswith("TFWS") or code == "TFWS":
        notes.append(
            "Eligibility: TFWS seats usually require family income below the CET TFWS limit "
            "(commonly Rs 8 lakh). Confirm current-year CET rules and documents."
        )
    if "PWD" in code:
        notes.append(
            "Eligibility: PWD seats require a valid disability certificate as per CET CAP brochure. "
            "Verify percentage/type requirements before preferring these options."
        )
    if code.startswith("DEF") or "DEF" in code:
        notes.append(
            "Eligibility: Defence (DEF) seats require qualifying defence-category documents "
            "under CET CAP rules."
        )
    if code.startswith("ORPHAN") or code == "ORPHAN":
        notes.append(
            "Eligibility: ORPHAN seats require orphan status documentation as per CET CAP rules."
        )
    if code.startswith("EWS") or code == "EWS":
        notes.append(
            "Eligibility: EWS seats require a valid EWS certificate for the current year."
        )
    return notes
