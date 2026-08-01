"""Prompt templates for grounded report narration."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are a Maharashtra CAP counselling assistant.
You MUST use only the JSON facts provided by the user.
Refer to colleges ONLY by institute_code from allowed_institute_codes.
Do NOT invent institute codes, college names, branches, cutoffs, ranks, seat counts, or probabilities.
If available_seats / competition / confidence_pct appear in facts, you may cite them; otherwise do not invent seats.
Cutoff bands drive chance; seat matrix (when present) is inventory only — never invent seat numbers.
Do NOT claim guarantees of admission.
Explain Dream/Target/Safe bands using the provided band meanings.
Write clear counselling prose in English (short paragraphs + bullets).
When listing preferences, cite institute_code first (e.g. 3012), then the name only if it appears in facts."""


def compact_facts_for_llm(facts: dict[str, Any], *, max_colleges: int = 25) -> dict[str, Any]:
    """Reduce facts to code-centric payload for stronger grounding."""
    allowed: list[int] = []
    for key in ("preference_order", "safe", "target", "dream"):
        for row in facts.get(key, [])[:max_colleges]:
            code = row.get("institute_code")
            if isinstance(code, int) and code not in allowed:
                allowed.append(code)

    def slim(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in rows[:max_colleges]:
            out.append(
                {
                    "institute_code": r.get("institute_code"),
                    "branch_code": r.get("branch_code"),
                    "band": r.get("band"),
                    "chance_label": r.get("chance_label"),
                    "margin": r.get("margin"),
                    "closing_percentile": r.get("closing_percentile"),
                    "year": r.get("year"),
                    "round": r.get("round"),
                    "category_code": r.get("category_code"),
                    "evidence_confidence": r.get("evidence_confidence"),
                    "available_seats": r.get("available_seats"),
                    "competition": r.get("competition"),
                    "confidence_pct": r.get("confidence_pct"),
                    "final_score": r.get("final_score"),
                }
            )
        return out

    return {
        "profile": facts.get("profile", {}),
        "allowed_institute_codes": allowed,
        "counts": facts.get("counts", {}),
        "data_years": facts.get("data_years", []),
        "preference_order": slim(facts.get("preference_order", [])),
        "safe": slim(facts.get("safe", [])),
        "target": slim(facts.get("target", [])),
        "dream": slim(facts.get("dream", [])),
        "notes": facts.get("notes", []),
        "band_meanings": facts.get("band_meanings"),
    }


def build_user_prompt(facts: dict[str, Any]) -> str:
    payload = json.dumps(facts, indent=2, ensure_ascii=False)
    codes = facts.get("allowed_institute_codes") or []
    return (
        "Using ONLY the following prediction facts JSON, write counselling recommendations "
        "and a short rationale for the suggested preference order.\n"
        f"You may mention ONLY these institute_code values: {codes}\n\n"
        f"FACTS_JSON:\n{payload}\n"
    )
