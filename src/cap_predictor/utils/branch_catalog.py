"""Branch name catalog and aliases for UI / predictor."""

from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz, process

# Sentinel used in profile / CLI / Streamlit for "no filter on this field"
ALL_SENTINEL = "ALL"
ALL_UI_LABEL = "(All)"

# Short aliases → search phrases used for fuzzy match against branch_name
BRANCH_ALIASES: dict[str, list[str]] = {
    "CSE": ["Computer Science", "Computer Engineering"],
    "CS": ["Computer Science", "Computer Engineering"],
    "COMP": ["Computer Engineering", "Computer Science"],
    "IT": ["Information Technology"],
    "ENTC": ["Electronics and Telecommunication", "Electronics & Telecommunication"],
    "EXTC": ["Electronics and Telecommunication"],
    "ECE": ["Electronics and Communication", "Electronics and Telecommunication"],
    "MECH": ["Mechanical Engineering"],
    "CIVIL": ["Civil Engineering"],
    "EE": ["Electrical Engineering"],
    "AIDS": ["Artificial Intelligence and Data Science", "AI and Data Science"],
    "AIML": ["Artificial Intelligence and Machine Learning", "AI and ML"],
}


def is_all_selection(value: str | None) -> bool:
    """True when the user chose All / (All) / * for a catalog field."""
    if value is None:
        return False
    text = value.strip().upper()
    return text in {"ALL", "(ALL)", "*", "ANY"}


BRANCH_MULTI_SEP = " || "


def encode_target_branches(selected: list[str]) -> str:
    """
    Encode UI multi-select into StudentProfile.target_branch.

    Empty or containing (All)/ALL → ALL. Otherwise join with ' || '.
    """
    cleaned = [s.strip() for s in selected if s and str(s).strip()]
    if not cleaned:
        return ALL_SENTINEL
    if any(is_all_selection(s) or s == ALL_UI_LABEL for s in cleaned):
        return ALL_SENTINEL
    # Dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for s in cleaned:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return BRANCH_MULTI_SEP.join(out) if len(out) > 1 else out[0]


def decode_target_branches(target_branch: str) -> list[str] | None:
    """
    Decode profile target_branch into include list.

    Returns None when ALL (no include filter).
    """
    if is_all_selection(target_branch):
        return None
    raw = target_branch.strip()
    if BRANCH_MULTI_SEP in raw:
        parts = [p.strip() for p in raw.split(BRANCH_MULTI_SEP) if p.strip()]
        return parts or None
    return [raw] if raw else None


def with_all_option(options: list[str], *, label: str = ALL_UI_LABEL) -> list[str]:
    """Prepend (All) for Streamlit selectboxes; de-dupe if already present."""
    cleaned = [o for o in options if not is_all_selection(o)]
    return [label] + cleaned


def ui_value_to_profile(value: str) -> str:
    """Map Streamlit (All) label to profile sentinel ALL."""
    if is_all_selection(value):
        return ALL_SENTINEL
    return str(value).strip()


def _distinct_sorted(cutoffs: pd.DataFrame, column: str) -> list[str]:
    if cutoffs is None or cutoffs.empty or column not in cutoffs.columns:
        return []
    names = (
        cutoffs[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(set(names), key=str.lower)


def load_branch_names(cutoffs: pd.DataFrame) -> list[str]:
    return _distinct_sorted(cutoffs, "branch_name")


def load_university_names(cutoffs: pd.DataFrame) -> list[str]:
    return _distinct_sorted(cutoffs, "university")


def load_category_codes(cutoffs: pd.DataFrame) -> list[str]:
    """Distinct category_code values as they appear in cutoff PDFs."""
    return _distinct_sorted(cutoffs, "category_code")


def merge_category_options(cutoffs: pd.DataFrame) -> list[str]:
    """
    PDF category codes first (as extracted), then any known CAP seat codes
    not yet seen — so the dropdown mirrors cutoffs without dropping vocabulary.
    """
    from cap_predictor.utils.category_map import CAP_CATEGORY_SEATS

    pdf_codes = load_category_codes(cutoffs)
    seen = {c.upper() for c in pdf_codes}
    extras = [c for c in CAP_CATEGORY_SEATS if c.upper() not in seen]
    return pdf_codes + extras


def resolve_branch_queries(target_branch: str) -> list[str]:
    """Return search phrases for a user branch input (including aliases)."""
    raw = target_branch.strip()
    if not raw:
        return []
    key = re_key(raw)
    queries = [raw]
    if key in BRANCH_ALIASES:
        queries.extend(BRANCH_ALIASES[key])
    # Also try uppercase alias lookup without punctuation
    upper = raw.upper().replace(".", "").replace(" ", "")
    if upper in BRANCH_ALIASES:
        queries.extend(BRANCH_ALIASES[upper])
    # Dedupe
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def re_key(text: str) -> str:
    return text.strip().upper().replace(".", "").replace(" ", "").replace("-", "")


def match_branch_names(
    target_branch: str,
    available_names: list[str],
    *,
    score_cutoff: int = 85,
    exact_catalog: bool = False,
) -> list[str]:
    """
    Return matched branch names from catalog for a user query.

    When `exact_catalog` is True (Streamlit selectbox value), only exact
    case-insensitive equality is used — prevents "Computer Engineering"
    matching "Civil Engineering" via shared token "Engineering".
    """
    if not available_names:
        return []
    if is_all_selection(target_branch):
        return list(available_names)

    raw = target_branch.strip()
    # Exact catalog pick (UI) or exact name typed
    exact = [n for n in available_names if n.lower() == raw.lower()]
    if exact or exact_catalog:
        return exact

    queries = resolve_branch_queries(raw)
    good: set[str] = set()
    for q in queries:
        q_l = q.lower()
        for name in available_names:
            n_l = name.lower()
            if q_l == n_l:
                good.add(name)
            elif q_l in n_l and len(q_l) >= 8:
                # Require distinctive query (avoid matching on bare "Engineering")
                good.add(name)
        matches = process.extract(q, available_names, scorer=fuzz.token_set_ratio, limit=20)
        for name, score, _ in matches:
            if score < score_cutoff:
                continue
            # Reject matches that only share generic tokens like Engineering
            if _shares_distinctive_token(q, name):
                good.add(name)
    return sorted(good, key=str.lower)


_GENERIC_BRANCH_TOKENS = {
    "engineering",
    "technology",
    "science",
    "and",
    "of",
    "the",
    "in",
}


def _shares_distinctive_token(query: str, name: str) -> bool:
    q_toks = {t for t in query.lower().replace("&", " ").replace("-", " ").split() if t}
    n_toks = {t for t in name.lower().replace("&", " ").replace("-", " ").split() if t}
    distinctive = (q_toks - _GENERIC_BRANCH_TOKENS) & (n_toks - _GENERIC_BRANCH_TOKENS)
    return bool(distinctive)
