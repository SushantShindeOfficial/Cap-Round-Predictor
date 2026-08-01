"""Ground LLM narrative strictly on PredictionResult facts."""

from __future__ import annotations

import logging
import re

from cap_predictor.llm.ollama_client import OllamaClient, OllamaError
from cap_predictor.models import PredictionResult
from cap_predictor.prompts.report_prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    compact_facts_for_llm,
)
from cap_predictor.utils.band_legend import BAND_MEANINGS, BAND_MEANINGS_TEXT

logger = logging.getLogger(__name__)


def allowed_institute_codes(result: PredictionResult) -> set[str]:
    codes: set[str] = set()
    for bucket in (
        result.eligible,
        result.safe,
        result.target,
        result.dream,
        result.preference_order,
    ):
        for c in bucket:
            codes.add(str(c.institute_code))
    return codes


def invented_institute_codes(text: str, result: PredictionResult) -> list[str]:
    """Flag 4-digit codes in narrative that are not in prediction facts."""
    allowed = allowed_institute_codes(result)
    found = set(re.findall(r"\b(\d{4})\b", text))
    # Ignore years that look like 2024/2025
    found = {c for c in found if not c.startswith("20")}
    return sorted(found - allowed)


def strip_invented_sentences(text: str, invented: list[str]) -> str:
    """Drop sentences that mention invented institute codes."""
    if not invented:
        return text
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if any(re.search(rf"\b{re.escape(code)}\b", sentence) for code in invented):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def template_narrative(result: PredictionResult) -> str:
    p = result.profile
    lines = [
        f"Based solely on parsed CAP cutoff data for years {result.data_years}, "
        f"a student with percentile {p.percentile}, category {p.category}, "
        f"gender {p.gender}, seat type {p.seat_type}, targeting '{p.target_branch}' "
        f"(year={p.year or 'All'}, round={p.round or 'All'}) "
        f"has {len(result.safe)} safe, {len(result.target)} target, and "
        f"{len(result.dream)} dream options among matched institute-branch pairs.",
        "",
        "Band meanings: " + BAND_MEANINGS_TEXT,
        "",
        "Counselling guidance:",
        "- Prefer listing Safe colleges first, then Target, then a few Dream choices.",
        "- Cutoffs change by round and year; treat margins as historical evidence, not guarantees.",
        "- Verify home-university and category seat allotment rules before locking preferences.",
    ]
    if result.notes:
        elig = [n for n in result.notes if n.lower().startswith("eligibility")]
        if elig:
            lines.append("")
            lines.append("Eligibility notes:")
            lines.extend(f"- {n}" for n in elig)
    if result.preference_order:
        lines.append("")
        lines.append("Suggested preference highlights (deterministic order):")
        for i, c in enumerate(result.preference_order[:10], start=1):
            lines.append(
                f"{i}. {c.institute_code} - {c.branch_name} "
                f"[{c.band.value}, margin {c.margin:+.2f}, {c.primary_year} R{c.primary_round} "
                f"{c.primary_category_code} close {c.primary_closing_percentile:.4f}, "
                f"confidence={c.evidence_confidence}]"
            )
    if result.notes:
        other = [n for n in result.notes if not n.lower().startswith("eligibility")]
        if other:
            lines.append("")
            lines.append("Data notes: " + " ".join(other))
    return "\n".join(lines)


def generate_grounded_narrative(
    result: PredictionResult,
    *,
    client: OllamaClient | None = None,
    use_llm: bool = True,
) -> str:
    if not use_llm:
        return template_narrative(result)

    client = client or OllamaClient()
    if not client.available():
        logger.warning("Ollama unavailable; using template narrative.")
        return template_narrative(result)

    facts = compact_facts_for_llm(result.facts_dict())
    facts["band_meanings"] = BAND_MEANINGS
    prompt = build_user_prompt(facts)
    try:
        text = client.generate(prompt, system=SYSTEM_PROMPT)
    except OllamaError as exc:
        logger.warning("%s — falling back to template.", exc)
        return template_narrative(result)

    invented = invented_institute_codes(text, result)
    if invented:
        logger.warning("LLM invented institute codes %s; regenerating once.", invented)
        stricter = (
            prompt
            + "\n\nSTRICT RETRY: You invented institute codes "
            + ", ".join(invented)
            + ". Rewrite using ONLY allowed_institute_codes from the JSON. "
            "Cite colleges by institute_code only."
        )
        try:
            text = client.generate(stricter, system=SYSTEM_PROMPT)
        except OllamaError:
            return template_narrative(result)
        invented = invented_institute_codes(text, result)
        if invented:
            logger.warning("LLM still invented codes %s; stripping then fallback check.", invented)
            text = strip_invented_sentences(text, invented)
            if invented_institute_codes(text, result) or len(text) < 40:
                return template_narrative(result)

    if len(text) < 40:
        return template_narrative(result)
    return text
