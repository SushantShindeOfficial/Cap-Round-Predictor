"""Build labeled training features from cutoff table using rule bands as labels."""

from __future__ import annotations

import pandas as pd

from cap_predictor.predictor.engine import BandThresholds, classify_band


def synthesize_training_features(
    cutoffs: pd.DataFrame,
    *,
    percentiles: list[float] | None = None,
    sample_rows: int = 4000,
    thresholds: BandThresholds | None = None,
) -> pd.DataFrame:
    """
    Create supervised rows: features from cutoff cells × student percentiles,
    labels from the deterministic rule bands (bootstrap labels for ML).
    """
    thr = thresholds or BandThresholds()
    percentiles = percentiles or [70.0, 80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 99.0]
    if cutoffs.empty:
        return pd.DataFrame()

    work = cutoffs.dropna(subset=["closing_percentile"]).copy()
    if len(work) > sample_rows:
        work = work.sample(sample_rows, random_state=42)

    rows: list[dict] = []
    for _, r in work.iterrows():
        closing = float(r["closing_percentile"])
        for pct in percentiles:
            margin = pct - closing
            band = classify_band(pct, closing, thr)
            rows.append(
                {
                    "institute_code": int(r["institute_code"]),
                    "institute_name": r.get("institute_name"),
                    "university": r.get("university"),
                    "branch_code": r.get("branch_code"),
                    "branch_name": r.get("branch_name"),
                    "band": band.value,
                    "chance_label": None,
                    "student_percentile": pct,
                    "closing_percentile": closing,
                    "margin": margin,
                    "yoy_delta": 0.0,
                    "primary_year": int(r["year"]),
                    "primary_round": int(r["round"]),
                    "category_code": r.get("category_code"),
                    "seat_type_section": r.get("seat_type_section"),
                    "hu_match_score": 0.0,
                    "evidence_confidence": "low",
                    "trend_points": 1,
                    "student_category": r.get("category_code"),
                    "student_seat_type": "ALL",
                    "student_gender": "M",
                    "home_university": "",
                    "target_branch": r.get("branch_name"),
                }
            )
    return pd.DataFrame(rows)
