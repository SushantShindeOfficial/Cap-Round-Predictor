"""Pydantic models for CAP prediction pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator


class ChanceLabel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    VERY_LOW = "Very Low"


class Band(str, Enum):
    SAFE = "safe"
    TARGET = "target"
    DREAM = "dream"
    UNLIKELY = "unlikely"


class CutoffRecord(BaseModel):
    year: int
    round: int = Field(ge=1, le=4)
    institute_code: int
    institute_name: str
    university: str | None = None
    branch_code: str
    branch_name: str
    seat_type_section: str
    category_code: str
    gender_flag: str | None = None
    reservation: str | None = None
    quota_suffix: str | None = None
    stage: Literal["I", "II", "III"] = "I"
    closing_merit: int | None = None
    closing_percentile: float | None = None
    source_pdf: str
    page_number: int


class StudentProfile(BaseModel):
    percentile: float = Field(ge=0, le=100)
    category: str
    gender: str
    home_university: str
    seat_type: str = "HU"
    target_branch: str
    year: int | None = None
    round: int | None = Field(default=None, ge=1, le=4)

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, v: str) -> str:
        g = v.strip().upper()
        if g in {"ALL", "(ALL)", "*", "ANY"}:
            return "ALL"
        if g in {"M", "MALE"}:
            return "M"
        if g in {"F", "FEMALE", "L", "LADIES"}:
            return "F"
        return g

    @field_validator("category", "seat_type", "target_branch", "home_university")
    @classmethod
    def strip_text(cls, v: str) -> str:
        text = v.strip()
        upper = text.upper()
        if upper in {"ALL", "(ALL)", "*", "ANY"}:
            return "ALL"
        return text


class ClosingTrendPoint(BaseModel):
    year: int
    round: int
    closing_percentile: float
    category_code: str


class CutoffEvidence(BaseModel):
    year: int
    round: int
    category_code: str
    stage: str
    closing_merit: int | None
    closing_percentile: float | None
    seat_type_section: str
    page_number: int | None = None


class CollegePrediction(BaseModel):
    institute_code: int
    institute_name: str
    university: str | None = None
    branch_code: str
    branch_name: str
    band: Band
    chance_label: ChanceLabel
    margin: float
    primary_closing_percentile: float
    primary_year: int
    primary_round: int = 1
    primary_category_code: str
    rule_reason: str
    hu_match_score: float | None = None
    # low = single year+round point; medium = multi-round same year; high = multi-year
    evidence_confidence: Literal["low", "medium", "high"] = "low"
    closing_trend: list[ClosingTrendPoint] = Field(default_factory=list)
    evidence: list[CutoffEvidence] = Field(default_factory=list)
    # Seat matrix inventory overlay
    available_seats: int | None = None
    seat_matrix_year: int | None = None
    seat_quota: str | None = None
    open_seats: int | None = None
    reserved_seats: int | None = None
    competition: Literal["low", "medium", "high"] | None = None
    confidence_pct: float | None = None
    final_score: float | None = None
    college_tier: str | None = None


class SeatSummaryRow(BaseModel):
    seat_type_section: str
    category_code: str
    row_count: int


class PredictionResult(BaseModel):
    profile: StudentProfile
    eligible: list[CollegePrediction] = Field(default_factory=list)
    dream: list[CollegePrediction] = Field(default_factory=list)
    target: list[CollegePrediction] = Field(default_factory=list)
    safe: list[CollegePrediction] = Field(default_factory=list)
    unlikely: list[CollegePrediction] = Field(default_factory=list)
    preference_order: list[CollegePrediction] = Field(default_factory=list)
    seat_summary: list[SeatSummaryRow] = Field(default_factory=list)
    narrative: str | None = None
    data_years: list[int] = Field(default_factory=list)
    matched_cutoff_rows: int = 0
    notes: list[str] = Field(default_factory=list)

    def facts_dict(self, max_colleges: int = 40) -> dict[str, Any]:
        """Serialize grounded facts for the LLM (capped)."""

        def pack(items: list[CollegePrediction]) -> list[dict[str, Any]]:
            out = []
            for c in items[:max_colleges]:
                out.append(
                    {
                        "institute_code": c.institute_code,
                        "institute_name": c.institute_name,
                        "branch_code": c.branch_code,
                        "branch_name": c.branch_name,
                        "band": c.band.value,
                        "chance_label": c.chance_label.value,
                        "margin": round(c.margin, 4),
                        "closing_percentile": c.primary_closing_percentile,
                        "year": c.primary_year,
                        "round": c.primary_round,
                        "category_code": c.primary_category_code,
                        "hu_match_score": c.hu_match_score,
                        "evidence_confidence": c.evidence_confidence,
                        "rule_reason": c.rule_reason,
                        "available_seats": c.available_seats,
                        "seat_quota": c.seat_quota,
                        "competition": c.competition,
                        "confidence_pct": c.confidence_pct,
                        "final_score": c.final_score,
                        "college_tier": c.college_tier,
                    }
                )
            return out

        return {
            "profile": self.profile.model_dump(),
            "data_years": self.data_years,
            "matched_cutoff_rows": self.matched_cutoff_rows,
            "counts": {
                "eligible": len(self.eligible),
                "safe": len(self.safe),
                "target": len(self.target),
                "dream": len(self.dream),
                "unlikely": len(self.unlikely),
            },
            "safe": pack(self.safe),
            "target": pack(self.target),
            "dream": pack(self.dream),
            "preference_order": pack(self.preference_order),
            "seat_summary": [s.model_dump() for s in self.seat_summary[:50]],
            "notes": self.notes,
        }


class BasePredictor(Protocol):
    """Protocol so ML models can replace the rule engine later."""

    def predict(self, profile: StudentProfile, cutoffs: Any) -> PredictionResult: ...
