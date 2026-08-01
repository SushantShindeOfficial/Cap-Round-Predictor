"""Build flat ML-ready feature tables from prediction results / cutoffs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cap_predictor.models import PredictionResult


FEATURE_COLUMNS = [
    "institute_code",
    "institute_name",
    "university",
    "branch_code",
    "branch_name",
    "band",
    "chance_label",
    "student_percentile",
    "closing_percentile",
    "margin",
    "yoy_delta",
    "primary_year",
    "primary_round",
    "category_code",
    "seat_type_section",
    "hu_match_score",
    "evidence_confidence",
    "trend_points",
    "student_category",
    "student_seat_type",
    "student_gender",
    "home_university",
    "target_branch",
]


def _yoy_delta(result_college) -> float | None:
    trend = result_college.closing_trend or []
    by_year: dict[int, float] = {}
    for t in trend:
        # keep latest round per year
        prev = by_year.get(t.year)
        if prev is None:
            by_year[t.year] = t.closing_percentile
    years = sorted(by_year)
    if len(years) < 2:
        return None
    return by_year[years[-1]] - by_year[years[-2]]


def build_feature_frame(result: PredictionResult) -> pd.DataFrame:
    """One row per predicted institute-branch for ML / analytics."""
    rows: list[dict] = []
    profile = result.profile
    for c in result.preference_order + result.unlikely:
        seat_section = ""
        if c.evidence:
            seat_section = c.evidence[0].seat_type_section
        rows.append(
            {
                "institute_code": c.institute_code,
                "institute_name": c.institute_name,
                "university": c.university,
                "branch_code": c.branch_code,
                "branch_name": c.branch_name,
                "band": c.band.value,
                "chance_label": c.chance_label.value,
                "student_percentile": profile.percentile,
                "closing_percentile": c.primary_closing_percentile,
                "margin": c.margin,
                "yoy_delta": _yoy_delta(c),
                "primary_year": c.primary_year,
                "primary_round": c.primary_round,
                "category_code": c.primary_category_code,
                "seat_type_section": seat_section,
                "hu_match_score": c.hu_match_score,
                "evidence_confidence": c.evidence_confidence,
                "trend_points": len(c.closing_trend),
                "student_category": profile.category,
                "student_seat_type": profile.seat_type,
                "student_gender": profile.gender,
                "home_university": profile.home_university,
                "target_branch": profile.target_branch,
            }
        )
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    df = pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS)
    # Deduplicate preference+unlikely overlap
    return df.drop_duplicates(subset=["institute_code", "branch_code"], keep="first")


def write_feature_export(result: PredictionResult, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_feature_frame(result)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)
    return out_path
