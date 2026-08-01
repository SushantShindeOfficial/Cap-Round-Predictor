"""Preference-list ordering for counselling."""

from __future__ import annotations

from cap_predictor.models import Band, CollegePrediction

BAND_PRIORITY = {
    Band.SAFE: 0,
    Band.TARGET: 1,
    Band.DREAM: 2,
    Band.UNLIKELY: 3,
}


def build_preference_order(
    safe: list[CollegePrediction],
    target: list[CollegePrediction],
    dream: list[CollegePrediction],
    *,
    max_items: int = 50,
    prefer_hu: bool = False,
) -> list[CollegePrediction]:
    """Safe → Target → Dream; within band by final_score (or HU/margin fallback)."""
    combined = list(safe) + list(target) + list(dream)

    def sort_key(p: CollegePrediction) -> tuple:
        hu = p.hu_match_score if (prefer_hu and p.hu_match_score is not None) else 0.0
        score = p.final_score if p.final_score is not None else None
        if score is not None:
            return (
                BAND_PRIORITY[p.band],
                -score,
                -p.margin,
                p.institute_code,
                p.branch_code,
            )
        return (
            BAND_PRIORITY[p.band],
            -hu,
            -p.margin,
            p.institute_code,
            p.branch_code,
        )

    combined.sort(key=sort_key)
    return combined[:max_items]
