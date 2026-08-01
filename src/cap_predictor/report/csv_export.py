"""Export prediction lists to CSV."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from cap_predictor.models import CollegePrediction, PredictionResult


def predictions_to_rows(
    items: list[CollegePrediction],
    *,
    student_percentile: float | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for c in items:
        rows.append(
            {
                "institute_code": c.institute_code,
                "institute_name": c.institute_name,
                "university": c.university or "",
                "branch_code": c.branch_code,
                "branch_name": c.branch_name,
                "band": c.band.value,
                "chance_label": c.chance_label.value,
                "student_percentile": student_percentile,
                "margin": round(c.margin, 4),
                "closing_percentile": c.primary_closing_percentile,
                "year": c.primary_year,
                "round": c.primary_round,
                "category_code": c.primary_category_code,
                "hu_match_score": c.hu_match_score,
                "evidence_confidence": c.evidence_confidence,
                "available_seats": c.available_seats,
                "seat_quota": c.seat_quota or "",
                "competition": c.competition or "",
                "confidence_pct": c.confidence_pct,
                "final_score": c.final_score,
                "college_tier": c.college_tier or "",
                "rule_reason": c.rule_reason,
            }
        )
    return rows


def result_to_csv_string(result: PredictionResult, *, include_unlikely: bool = False) -> str:
    pct = result.profile.percentile
    rows: list[dict] = []
    for bucket in (result.safe, result.target, result.dream):
        rows.extend(predictions_to_rows(bucket, student_percentile=pct))
    if include_unlikely:
        rows.extend(predictions_to_rows(result.unlikely, student_percentile=pct))
    if not rows:
        return "institute_code,institute_name,band\n"

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_result_csv(result: PredictionResult, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result_to_csv_string(result), encoding="utf-8")
    return out_path


def filter_predictions(
    items: list[CollegePrediction],
    *,
    university_contains: str | None = None,
    min_hu_score: float | None = None,
    bands: set[str] | None = None,
) -> list[CollegePrediction]:
    out: list[CollegePrediction] = []
    needle = (university_contains or "").strip().lower()
    for c in items:
        if bands and c.band.value not in bands:
            continue
        if needle and needle not in (c.university or "").lower() and needle not in c.institute_name.lower():
            continue
        if min_hu_score is not None and (c.hu_match_score or 0) < min_hu_score:
            continue
        out.append(c)
    return out
