"""Attach seat-matrix inventory and compute competition / confidence / final_score."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from cap_predictor.data.college_master import college_tier_score, load_college_master
from cap_predictor.models import Band, CollegePrediction
from cap_predictor.utils.seat_matrix_map import (
    course_code_from_cutoff_branch,
    map_cutoff_category_to_seat_cell,
)

_CONF = {"high": 1.0, "medium": 0.6, "low": 0.3}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _margin_norm(margin: float) -> float:
    # map [-8, +15] → 0..1
    return _clamp01((margin + 8.0) / 23.0)


def _seats_norm(seats: int | None, max_seats: int) -> float:
    if seats is None or seats < 0:
        return 0.0
    denom = math.log1p(max(max_seats, 1))
    if denom <= 0:
        return 0.0
    return _clamp01(math.log1p(seats) / denom)


def competition_label(seats: int | None, si: int | None = None) -> str | None:
    if seats is None:
        return None
    if si and si > 0:
        ratio = seats / si
        if ratio >= 0.25:
            return "low"
        if ratio >= 0.1:
            return "medium"
        return "high"
    if seats >= 10:
        return "low"
    if seats >= 4:
        return "medium"
    return "high"


def confidence_pct(
    *,
    margin: float,
    seats: int | None,
    evidence_confidence: str,
) -> float:
    base = 55.0 + 25.0 * _margin_norm(margin)
    if seats is not None:
        base += min(15.0, seats * 0.8)
    base += 10.0 * _CONF.get(evidence_confidence, 0.3)
    return round(_clamp01(base / 100.0) * 100.0, 1)


def lookup_seats(
    seat_df: pd.DataFrame,
    *,
    institute_code: int,
    branch_code: str,
    category_code: str,
    year: int | None,
) -> tuple[int | None, int | None, str | None, int | None]:
    """
    Returns available_seats, matrix_year, quota, si for the mapped cell.
    """
    if seat_df is None or seat_df.empty:
        return None, None, None, None
    cell = map_cutoff_category_to_seat_cell(category_code)
    if cell is None:
        return None, None, None, None
    course = course_code_from_cutoff_branch(branch_code, institute_code)
    work = seat_df.copy()
    work["institute_code"] = work["institute_code"].astype(int)
    work["course_code"] = work["course_code"].astype(str)
    matched = work[
        (work["institute_code"] == int(institute_code))
        & (work["course_code"] == str(course))
        & (work["quota"] == cell.quota)
        & (work["category_code"] == cell.category_code)
        & (work["gender"] == cell.gender)
    ]
    if matched.empty:
        # fallback: same institute+course+category any gender for ALL specials
        matched = work[
            (work["institute_code"] == int(institute_code))
            & (work["course_code"] == str(course))
            & (work["quota"] == cell.quota)
            & (work["category_code"] == cell.category_code)
        ]
    if matched.empty:
        return None, None, cell.quota, None
    if year is not None:
        year_hit = matched[matched["year"] == int(year)]
        if not year_hit.empty:
            matched = year_hit
        else:
            matched = matched[matched["year"] == matched["year"].max()]
    else:
        matched = matched[matched["year"] == matched["year"].max()]
    top = matched.iloc[0]
    return (
        int(top["seats"]),
        int(top["year"]),
        cell.quota,
        int(top["si"]) if pd.notna(top.get("si")) else None,
    )


def open_reserved_for_program(
    seat_df: pd.DataFrame,
    *,
    institute_code: int,
    branch_code: str,
    year: int | None,
    quota: str | None,
) -> tuple[int | None, int | None]:
    if seat_df is None or seat_df.empty or not quota:
        return None, None
    course = course_code_from_cutoff_branch(branch_code, institute_code)
    work = seat_df[
        (seat_df["institute_code"].astype(int) == int(institute_code))
        & (seat_df["course_code"].astype(str) == str(course))
        & (seat_df["quota"] == quota)
    ]
    if year is not None:
        y = work[work["year"] == int(year)]
        if not y.empty:
            work = y
    if work.empty:
        return None, None
    open_seats = int(
        work.loc[work["category_code"] == "OPEN", "seats"].sum()
    ) if (work["category_code"] == "OPEN").any() else None
    reserved = int(
        work.loc[
            ~work["category_code"].isin({"OPEN", "TOTAL", "EWS", "TFWS", "ORPHAN"}),
            "seats",
        ].sum()
    )
    return open_seats, reserved


def enrich_predictions_with_seats(
    predictions: Iterable[CollegePrediction],
    seat_df: pd.DataFrame | None,
    *,
    master: pd.DataFrame | None = None,
) -> tuple[list[CollegePrediction], dict]:
    """
    Attach seat inventory fields and scores. Returns enriched list + stats.
    """
    preds = list(predictions)
    stats = {"matched": 0, "unmatched": 0, "zero_seats": 0}
    if not preds:
        return preds, stats

    master = master if master is not None else load_college_master()
    master_map: dict[int, dict] = {}
    if master is not None and not master.empty:
        for _, r in master.iterrows():
            master_map[int(r["institute_code"])] = r.to_dict()

    # First pass: lookup seats
    max_seats = 1
    for p in preds:
        seats, myear, quota, si = lookup_seats(
            seat_df if seat_df is not None else pd.DataFrame(),
            institute_code=p.institute_code,
            branch_code=p.branch_code,
            category_code=p.primary_category_code,
            year=p.primary_year,
        )
        p.available_seats = seats
        p.seat_matrix_year = myear
        p.seat_quota = quota
        if seats is None:
            stats["unmatched"] += 1
        else:
            stats["matched"] += 1
            max_seats = max(max_seats, seats)
            if seats == 0:
                stats["zero_seats"] += 1
        open_s, reserved_s = open_reserved_for_program(
            seat_df if seat_df is not None else pd.DataFrame(),
            institute_code=p.institute_code,
            branch_code=p.branch_code,
            year=myear or p.primary_year,
            quota=quota,
        )
        p.open_seats = open_s
        p.reserved_seats = reserved_s
        p.competition = competition_label(seats, si)  # type: ignore[assignment]
        p.confidence_pct = confidence_pct(
            margin=p.margin,
            seats=seats,
            evidence_confidence=p.evidence_confidence,
        )
        mrow = master_map.get(int(p.institute_code))
        if mrow:
            if mrow.get("government"):
                p.college_tier = "Government"
            elif mrow.get("autonomous"):
                p.college_tier = "Autonomous"
            elif mrow.get("private"):
                p.college_tier = "Private"
            else:
                p.college_tier = "Other"
        else:
            p.college_tier = None

    # Second pass: final_score with list-relative seats_norm
    for p in preds:
        tier = college_tier_score(master_map.get(int(p.institute_code)))
        stability = _CONF.get(p.evidence_confidence, 0.3)
        p.final_score = round(
            0.40 * _margin_norm(p.margin)
            + 0.30 * _seats_norm(p.available_seats, max_seats)
            + 0.20 * tier
            + 0.10 * stability,
            4,
        )
    return preds, stats


def filter_zero_seat_predictions(
    predictions: list[CollegePrediction],
) -> tuple[list[CollegePrediction], list[CollegePrediction]]:
    """Split into keep (seats null or >0) vs zero-seat excluded."""
    keep: list[CollegePrediction] = []
    excluded: list[CollegePrediction] = []
    for p in predictions:
        if p.available_seats == 0:
            excluded.append(p)
        else:
            keep.append(p)
    return keep, excluded
