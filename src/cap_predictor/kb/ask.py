"""Streamlit / chat helper: map NL questions to KB tools via Ollama JSON."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from cap_predictor.kb.queries import TOOL_SPECS, run_tool
from cap_predictor.llm.ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)

TOOL_PICK_SYSTEM = """You are a router for Maharashtra CAP knowledge tools.
Given the user question, reply with ONLY a JSON object (no markdown):
{"tool":"<name>","arguments":{...}}
Allowed tools and argument keys:
""" + "\n".join(
    f"- {t['name']}: {t['description']} Args: {', '.join(t['parameters'])}" for t in TOOL_SPECS
) + """

Rules:
- Prefer query_cutoffs for percentile/merit/cutoff questions.
- Prefer query_seats or query_seats_with_cutoffs for seat inventory + LNTD/category seats.
- Prefer search_rulebook for definitions/rules from the brochure.
- Never invent institutes or numbers; only choose tools.
- Use null for unused optional args; omit unknown keys.
"""

ANSWER_SYSTEM = """You answer Maharashtra CAP questions using ONLY the TOOL_RESULT JSON.
Do not invent institutes, cutoffs, seats, or ranks.
If row_count is 0, say no matching rows were found in the parsed dataset.
Be concise; list institutes with codes when present.
Mention that data is historical CAP evidence, not a guarantee.
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def answer_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Route question → tool → optional LLM prose.

    Returns {answer, tool, arguments, result}.
    """
    q = (question or "").strip()
    if not q:
        return {
            "answer": "Please enter a question.",
            "tool": None,
            "arguments": {},
            "result": None,
        }

    client = client or OllamaClient()
    tool_name = "query_cutoffs"
    arguments: dict[str, Any] = {}

    if use_llm and client.available():
        try:
            raw = client.generate(
                f"User question:\n{q}\n\nJSON:",
                system=TOOL_PICK_SYSTEM,
            )
            picked = _extract_json(raw) or {}
            tool_name = str(picked.get("tool") or tool_name)
            arguments = dict(picked.get("arguments") or {})
        except OllamaError as exc:
            logger.warning("Tool pick failed: %s", exc)
            use_llm = False

    if not use_llm or not arguments:
        # Heuristic fallback without LLM
        tool_name, arguments = heuristic_route(q)

    result = run_tool(tool_name, arguments)
    answer = format_result_table(result, tool=tool_name)

    if use_llm and client.available():
        try:
            prose = client.generate(
                f"QUESTION:\n{q}\n\nTOOL:{tool_name}\nTOOL_RESULT:\n"
                f"{json.dumps(result, ensure_ascii=False, default=str)[:12000]}\n",
                system=ANSWER_SYSTEM,
            )
            if prose and len(prose) > 40:
                answer = prose
        except OllamaError as exc:
            logger.warning("Answer format failed: %s", exc)

    return {
        "answer": answer,
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
    }


def heuristic_route(question: str) -> tuple[str, dict[str, Any]]:
    """Simple keyword router when Ollama is unavailable."""
    q = question.lower()
    cat_m = re.search(r"\b(l[a-z]*ntd|gopens?|gopenh|gopeno|tfws|ews|lntd|lnto)\b", question, re.I)
    category = cat_m.group(1).upper() if cat_m else None
    if category == "LNTD" or "lntd" in q:
        category = "LNTD"

    if any(w in q for w in ("seat", "seats", "inventory", "lntd")) and (
        "cutoff" in q or "closing" in q or "what was" in q or "their cutoff" in q
    ):
        return "query_seats_with_cutoffs", {
            "category_code": category or "LNTD",
            "min_seats": 1,
            "limit": 50,
        }
    if any(w in q for w in ("seat", "seats", "inventory")) or (
        category and category.startswith("L") and "nt" in category.lower()
    ):
        args: dict[str, Any] = {"limit": 100, "min_seats": 1}
        if category:
            args["category_code"] = category
        return "query_seats", args

    if any(w in q for w in ("brochure", "rule", "eligibility", "what is hu", "ohu", "tfws mean")):
        return "search_rulebook", {"query": question, "max_excerpts": 5}

    args = {"limit": 100}
    if category:
        args["category_code"] = category
    pct = re.search(r"(?:below|under|less than|<=?)\s*(\d+(?:\.\d+)?)\s*%?", q)
    if pct:
        args["max_closing_percentile"] = float(pct.group(1))
    merit = re.search(r"(?:rank|merit).*?(\d{4,6})\s*(?:to|-|–)\s*(\d{4,6})", q)
    if merit:
        args["min_closing_merit"] = int(merit.group(1))
        args["max_closing_merit"] = int(merit.group(2))
    year_m = re.search(r"\b(202[4-9])\b", q)
    if year_m:
        args["year"] = int(year_m.group(1))
    return "query_cutoffs", args


def format_result_table(result: dict[str, Any] | None, *, tool: str) -> str:
    if not result:
        return "No result."
    rows = result.get("rows") or []
    note = result.get("note")
    lines = [
        f"Tool: {tool} — {result.get('row_count', len(rows))} row(s).",
        result.get("disclaimer") or "",
    ]
    if note:
        lines.append(f"Note: {note}")
    if not rows:
        lines.append("No matching rows in the parsed dataset.")
        return "\n".join(lines)

    # Flatten joined shape
    if tool == "query_seats_with_cutoffs":
        for i, item in enumerate(rows[:40], start=1):
            seat = item.get("seat") or {}
            cuts = item.get("cutoffs") or []
            lines.append(
                f"{i}. {seat.get('institute_code')} — {seat.get('institute_name')} "
                f"/ {seat.get('course_name')} seats={seat.get('seats')} "
                f"({seat.get('quota')}/{seat.get('category_code')}/{seat.get('gender')})"
            )
            for c in cuts[:2]:
                lines.append(
                    f"    cutoff {c.get('year')} R{c.get('round')} {c.get('category_code')}: "
                    f"%={c.get('closing_percentile')} merit={c.get('closing_merit')}"
                )
        return "\n".join(lines)

    if tool == "search_rulebook":
        for ex in rows[:5]:
            lines.append(f"— page {ex.get('page')}: {ex.get('text', '')[:400]}")
        return "\n".join(lines)

    for i, r in enumerate(rows[:50], start=1):
        if "closing_percentile" in r:
            lines.append(
                f"{i}. {r.get('institute_code')} — {r.get('institute_name')} / {r.get('branch_name')} "
                f"{r.get('year')} R{r.get('round')} {r.get('category_code')} "
                f"close%={r.get('closing_percentile')} merit={r.get('closing_merit')}"
            )
        else:
            lines.append(
                f"{i}. {r.get('institute_code')} — {r.get('institute_name')} / {r.get('course_name')} "
                f"{r.get('quota')}/{r.get('category_code')}/{r.get('gender')} seats={r.get('seats')}"
            )
    return "\n".join(lines)
