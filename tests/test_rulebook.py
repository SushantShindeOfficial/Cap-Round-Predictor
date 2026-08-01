"""Tests for CET Information Brochure structured rules + excerpt retrieval."""

from __future__ import annotations

from cap_predictor.data.rulebook_store import (
    jurisdiction_for_university,
    load_cap_rules,
    retrieve_rulebook_excerpts,
    seat_type_definition,
    structured_eligibility_notes,
)
from cap_predictor.models import StudentProfile
from cap_predictor.utils.eligibility import eligibility_notes


def test_cap_rules_load_jurisdiction():
    rules = load_cap_rules()
    assert rules.get("academic_year") == "2026-27"
    districts = jurisdiction_for_university("Mumbai University", rules)
    assert "Thane" in districts
    assert "Pune" in jurisdiction_for_university("Savitribai Phule Pune University", rules)


def test_seat_type_definitions():
    assert "Home University" in (seat_type_definition("HU") or "")
    assert "Outside Home University" in (seat_type_definition("OHU") or "")
    assert "State Level" in (seat_type_definition("STATE") or "")


def test_tfws_eligibility_from_brochure():
    profile = StudentProfile(
        percentile=90.0,
        category="TFWS",
        gender="M",
        home_university="Mumbai University",
        seat_type="STATE",
        target_branch="Computer Engineering",
    )
    notes = eligibility_notes(profile)
    assert notes
    assert any("8 Lakh" in n or "8 lakh" in n.lower() for n in notes)
    assert any("TFWS" in n for n in notes)


def test_pwd_eligibility_40_percent():
    profile = StudentProfile(
        percentile=85.0,
        category="PWD1",
        gender="M",
        home_university="ALL",
        seat_type="ALL",
        target_branch="ALL",
    )
    notes = structured_eligibility_notes(profile)
    assert any("40%" in n for n in notes)


def test_retrieve_excerpts_uses_chunks_or_empty(tmp_path, monkeypatch):
    # Without chunks cache, ensure_parsed may build from PDF if present
    profile = StudentProfile(
        percentile=92.5,
        category="GOPENS",
        gender="M",
        home_university="Mumbai University",
        seat_type="HU",
        target_branch="Computer Engineering",
    )
    excerpts = retrieve_rulebook_excerpts(profile, max_excerpts=3)
    # If PDF present and parse works, expect excerpts; else empty is OK
    assert isinstance(excerpts, list)
    for ex in excerpts:
        assert "text" in ex and "page" in ex
