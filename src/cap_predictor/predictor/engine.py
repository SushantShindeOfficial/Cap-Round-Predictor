"""Deterministic CAP admission band prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from cap_predictor.models import (
    Band,
    ChanceLabel,
    ClosingTrendPoint,
    CollegePrediction,
    CutoffEvidence,
    PredictionResult,
    SeatSummaryRow,
    StudentProfile,
)
from cap_predictor.predictor.ranking import build_preference_order
from cap_predictor.predictor.seat_overlay import (
    enrich_predictions_with_seats,
    filter_zero_seat_predictions,
)
from cap_predictor.utils.branch_catalog import (
    is_all_selection,
    load_branch_names,
    match_branch_names,
)
from cap_predictor.utils.category_map import (
    applicable_category_codes,
    category_suffix_conflicts_seat_type,
    general_codes_for_gender,
    is_full_cap_category_code,
    normalize_student_category,
)
from cap_predictor.utils.eligibility import eligibility_notes
from cap_predictor.utils.university_match import HU_MATCH_THRESHOLD, score_university

logger = logging.getLogger(__name__)

DEFAULT_PREFERENCE_LIMIT = 30


@dataclass
class BandThresholds:
    safe_margin: float = 3.0
    target_below: float = 2.0
    dream_below: float = 8.0


@dataclass
class PredictorOptions:
    thresholds: BandThresholds | None = None
    max_preferences: int = DEFAULT_PREFERENCE_LIMIT


def classify_band(student_percentile: float, closing: float, thr: BandThresholds) -> Band:
    margin = student_percentile - closing
    if margin >= thr.safe_margin:
        return Band.SAFE
    if margin >= -thr.target_below:
        return Band.TARGET
    if margin >= -thr.dream_below:
        return Band.DREAM
    return Band.UNLIKELY


def chance_from_margin(margin: float) -> ChanceLabel:
    if margin >= 5:
        return ChanceLabel.HIGH
    if margin >= 0:
        return ChanceLabel.MEDIUM
    if margin >= -5:
        return ChanceLabel.LOW
    return ChanceLabel.VERY_LOW


def _filter_year_round(df: pd.DataFrame, profile: StudentProfile) -> pd.DataFrame:
    out = df
    if profile.year is not None:
        out = out[out["year"] == profile.year]
    if profile.round is not None:
        out = out[out["round"] == profile.round]
    return out


def _filter_branch(
    df: pd.DataFrame,
    target_branch: str,
    score_cutoff: int = 85,
) -> pd.DataFrame:
    if df.empty:
        return df
    if is_all_selection(target_branch):
        return df
    target = target_branch.strip()
    if target.isdigit() or (target.isalnum() and len(target) >= 6 and any(ch.isdigit() for ch in target)):
        by_code = df[df["branch_code"].astype(str).str.contains(target, case=False, na=False)]
        if not by_code.empty:
            return by_code

    names = load_branch_names(df)
    # Streamlit / exact catalog names: equality only
    exact_catalog = any(n.lower() == target.lower() for n in names)
    matched = match_branch_names(
        target,
        names,
        score_cutoff=score_cutoff,
        exact_catalog=exact_catalog,
    )
    if not matched:
        return df.iloc[0:0]
    return df[df["branch_name"].isin(matched)]


def _seat_section_filter(df: pd.DataFrame, seat_type: str) -> pd.DataFrame:
    st = seat_type.strip().upper().replace(" ", "_")
    if st in {"ALL", "ANY", ""}:
        return df
    if st in {"HU", "HOME", "HOME_UNIVERSITY", "H"}:
        preferred = ["HU_TO_HU", "STATE_LEVEL", "OHU_TO_HU"]
    elif st in {"OHU", "OTHER", "OTHER_THAN_HOME", "O"}:
        preferred = ["OHU_TO_OHU", "STATE_LEVEL", "HU_TO_OHU"]
    elif st in {"STATE", "STATE_LEVEL", "S"}:
        preferred = ["STATE_LEVEL"]
    else:
        return df
    subset = df[df["seat_type_section"].isin(preferred)]
    return subset if not subset.empty else df


def _prefer_hu_rows(
    grp: pd.DataFrame,
    hu_score: float,
    seat_type: str,
    *,
    home_university: str = "",
) -> pd.DataFrame:
    """When HU-affiliated and seat_type HU/ALL, prefer HU_TO_HU rows if present."""
    if is_all_selection(home_university):
        return grp
    st = seat_type.strip().upper().replace(" ", "_")
    prefer = st in {"HU", "HOME", "HOME_UNIVERSITY", "H", "ALL", "ANY", ""}
    if not prefer or hu_score < HU_MATCH_THRESHOLD:
        return grp
    hu_rows = grp[grp["seat_type_section"] == "HU_TO_HU"]
    return hu_rows if not hu_rows.empty else grp


def evidence_confidence_from_trend(trend: list[ClosingTrendPoint]) -> str:
    if not trend:
        return "low"
    years = {t.year for t in trend}
    rounds = {(t.year, t.round) for t in trend}
    if len(years) >= 2:
        return "high"
    if len(rounds) >= 2:
        return "medium"
    return "low"


def _build_closing_trend(grp: pd.DataFrame, category_codes: list[str]) -> list[ClosingTrendPoint]:
    points: list[ClosingTrendPoint] = []
    stage_i = grp[grp["stage"] == "I"]
    work = stage_i if not stage_i.empty else grp
    if category_codes:
        preferred = work[work["category_code"].isin(category_codes)]
        if not preferred.empty:
            work = preferred
    # One point per year/round (best matching category)
    for (year, rnd), sub in work.groupby(["year", "round"], sort=True):
        sub = sub.copy()
        sub["_rank"] = sub["category_code"].apply(
            lambda c: category_codes.index(c) if c in category_codes else 999
        )
        top = sub.sort_values(["_rank", "closing_percentile"], ascending=[True, False]).iloc[0]
        if pd.isna(top.get("closing_percentile")):
            continue
        points.append(
            ClosingTrendPoint(
                year=int(year),
                round=int(rnd),
                closing_percentile=float(top["closing_percentile"]),
                category_code=str(top["category_code"]),
            )
        )
    return points


class RuleBasedPredictor:
    """Rule-based predictor; swap via BasePredictor for ML later."""

    def __init__(
        self,
        thresholds: BandThresholds | None = None,
        *,
        max_preferences: int = DEFAULT_PREFERENCE_LIMIT,
        seat_matrix: pd.DataFrame | None = None,
    ) -> None:
        self.thresholds = thresholds or BandThresholds()
        self.max_preferences = max_preferences
        self.seat_matrix = seat_matrix

    def predict(self, profile: StudentProfile, cutoffs: pd.DataFrame) -> PredictionResult:
        notes: list[str] = []
        notes.extend(eligibility_notes(profile))
        if cutoffs is None or cutoffs.empty:
            return PredictionResult(
                profile=profile,
                notes=["No cutoff data available. Parse CAP PDFs first."],
            )

        filtered = _filter_year_round(cutoffs, profile)
        if filtered.empty:
            notes.append(
                f"No cutoff rows for year={profile.year} round={profile.round}. "
                "Try All years/rounds."
            )
            return PredictionResult(
                profile=profile,
                data_years=sorted(cutoffs["year"].dropna().unique().tolist()),
                notes=notes,
            )
        if profile.year is not None or profile.round is not None:
            notes.append(
                f"Applied filters: year={profile.year or 'All'}, round={profile.round or 'All'}."
            )

        branch_df = _filter_branch(filtered, profile.target_branch)
        if branch_df.empty:
            notes.append(
                f"No branches matched '{profile.target_branch}'. "
                "Pick a branch from the catalog or try CSE / Computer Engineering."
            )
            return PredictionResult(
                profile=profile,
                data_years=sorted(filtered["year"].dropna().unique().tolist()),
                notes=notes,
            )
        if is_all_selection(profile.target_branch):
            notes.append("Branch filter: All (no branch restriction).")
        if is_all_selection(profile.home_university):
            notes.append("Home University: All (no HU affiliation preference).")
        if is_all_selection(profile.gender) and not is_all_selection(profile.category):
            notes.append("Gender: All (G and L seat codes considered for aliases).")

        codes = applicable_category_codes(
            category=profile.category,
            gender=profile.gender,
            seat_type=profile.seat_type,
        )
        if is_all_selection(profile.category):
            available = sorted(
                branch_df["category_code"].dropna().astype(str).unique().tolist()
            )
            codes = general_codes_for_gender(
                available, profile.gender, seat_type=profile.seat_type
            )
            cat_df = branch_df[branch_df["category_code"].isin(codes)]
            notes.append(
                f"Percentile {profile.percentile} drives Safe/Target/Dream "
                f"(margin = percentile - closing). "
                "Category All = OPEN seats only for selected gender/seat type "
                "(not SC/ST/OBC/NT/MI/TFWS/EWS/DEF - pick those codes if eligible)."
            )
        elif not codes:
            cat_df = branch_df
            codes = sorted(branch_df["category_code"].dropna().astype(str).unique().tolist())
            notes.append("Category filter: All (all CAP seat codes in scope).")
        else:
            cat_df = branch_df[branch_df["category_code"].isin(codes)]
            if cat_df.empty:
                reservation = normalize_student_category(profile.category)
                cat_df = branch_df[branch_df["reservation"] == reservation]
                notes.append(
                    f"Exact category codes {codes} not found; "
                    f"fell back to reservation={reservation}."
                )
            if is_full_cap_category_code(profile.category):
                notes.append(
                    f"Exact CAP code {profile.category.upper()}: Gender/Seat Type do not "
                    "expand to other codes."
                )
            conflict = category_suffix_conflicts_seat_type(
                profile.category, profile.seat_type
            )
            if conflict:
                notes.append(conflict)

        cat_df = _seat_section_filter(cat_df, profile.seat_type)
        if cat_df.empty:
            notes.append("No cutoff rows after seat-type / category filters.")
            return PredictionResult(
                profile=profile,
                data_years=sorted(filtered["year"].dropna().unique().tolist()),
                matched_cutoff_rows=0,
                notes=notes,
            )

        stage_i = cat_df[cat_df["stage"] == "I"]
        work = stage_i if not stage_i.empty else cat_df

        years = sorted(work["year"].dropna().unique().tolist())
        primary_year = max(years) if years else None
        prefer_hu = profile.seat_type.strip().upper() in {
            "HU",
            "HOME",
            "HOME_UNIVERSITY",
            "H",
            "ALL",
            "",
        }

        predictions: list[CollegePrediction] = []
        grouped = work.groupby(["institute_code", "branch_code"], sort=False)

        for (inst_code, br_code), grp in grouped:
            inst_univ = None
            if "university" in grp.columns and pd.notna(grp.iloc[0].get("university")):
                inst_univ = str(grp.iloc[0]["university"])
            if is_all_selection(profile.home_university):
                hu_score = 0.0
            else:
                hu_score = score_university(profile.home_university, inst_univ)

            scoped = _prefer_hu_rows(
                grp,
                hu_score,
                profile.seat_type,
                home_university=profile.home_university,
            )
            primary = scoped
            if primary_year is not None:
                py = scoped[scoped["year"] == primary_year]
                if not py.empty:
                    primary = py
                    latest_round = int(primary["round"].max())
                    pr = primary[primary["round"] == latest_round]
                    if not pr.empty:
                        primary = pr

            primary = primary.copy()
            primary["_code_rank"] = primary["category_code"].apply(
                lambda c: codes.index(c) if c in codes else 999
            )
            primary = primary.sort_values(
                ["_code_rank", "closing_percentile"], ascending=[True, False]
            )
            top = primary.iloc[0]
            closing = top.get("closing_percentile")
            if pd.isna(closing):
                continue
            closing_f = float(closing)
            margin = float(profile.percentile) - closing_f
            band = classify_band(float(profile.percentile), closing_f, self.thresholds)
            chance = chance_from_margin(margin)

            evidence: list[CutoffEvidence] = []
            for _, row in (
                grp.sort_values(["year", "round", "stage"], ascending=[False, False, True])
                .head(8)
                .iterrows()
            ):
                evidence.append(
                    CutoffEvidence(
                        year=int(row["year"]),
                        round=int(row["round"]),
                        category_code=str(row["category_code"]),
                        stage=str(row["stage"]),
                        closing_merit=int(row["closing_merit"])
                        if pd.notna(row.get("closing_merit"))
                        else None,
                        closing_percentile=float(row["closing_percentile"])
                        if pd.notna(row.get("closing_percentile"))
                        else None,
                        seat_type_section=str(row["seat_type_section"]),
                        page_number=int(row["page_number"])
                        if pd.notna(row.get("page_number"))
                        else None,
                    )
                )

            closing_trend = _build_closing_trend(grp, codes)
            confidence = evidence_confidence_from_trend(closing_trend)

            trend_note = ""
            if primary_year is not None and len(years) > 1:
                older = min(years)
                older_rows = grp[(grp["year"] == older) & (grp["stage"] == "I")]
                if older_rows.empty:
                    older_rows = grp[grp["year"] == older]
                if not older_rows.empty and pd.notna(older_rows.iloc[0].get("closing_percentile")):
                    old_pct = float(older_rows.iloc[0]["closing_percentile"])
                    delta = closing_f - old_pct
                    trend_note = f" YoY closing change vs {older}: {delta:+.2f} pts."

            hu_note = ""
            if hu_score >= HU_MATCH_THRESHOLD and prefer_hu:
                hu_note = f" HU match score {hu_score:.0f} with '{inst_univ}'."

            reason = (
                f"Student {profile.percentile:.4f} vs {int(top['year'])} R{int(top['round'])} "
                f"{top['category_code']} Stage {top['stage']} closing "
                f"{closing_f:.4f} (margin {margin:+.4f}). "
                f"Evidence confidence: {confidence}.{trend_note}{hu_note}"
            )

            predictions.append(
                CollegePrediction(
                    institute_code=int(inst_code),
                    institute_name=str(top["institute_name"]),
                    university=inst_univ,
                    branch_code=str(br_code),
                    branch_name=str(top["branch_name"]),
                    band=band,
                    chance_label=chance,
                    margin=margin,
                    primary_closing_percentile=closing_f,
                    primary_year=int(top["year"]),
                    primary_round=int(top["round"]),
                    primary_category_code=str(top["category_code"]),
                    rule_reason=reason.strip(),
                    hu_match_score=hu_score,
                    evidence_confidence=confidence,  # type: ignore[arg-type]
                    closing_trend=closing_trend,
                    evidence=evidence,
                )
            )

        safe = [p for p in predictions if p.band == Band.SAFE]
        target = [p for p in predictions if p.band == Band.TARGET]
        dream = [p for p in predictions if p.band == Band.DREAM]
        unlikely = [p for p in predictions if p.band == Band.UNLIKELY]

        # Seat matrix inventory overlay (optional)
        if self.seat_matrix is not None and not self.seat_matrix.empty:
            predictions, sm_stats = enrich_predictions_with_seats(
                predictions, self.seat_matrix
            )
            keep, zeroed = filter_zero_seat_predictions(predictions)
            if zeroed:
                notes.append(
                    f"Excluded {len(zeroed)} options with 0 seats in seat matrix."
                )
            predictions = keep
            notes.append(
                f"Seat matrix: matched={sm_stats['matched']}, "
                f"unmatched={sm_stats['unmatched']}, zero_seats={sm_stats['zero_seats']}."
            )
            safe = [p for p in predictions if p.band == Band.SAFE]
            target = [p for p in predictions if p.band == Band.TARGET]
            dream = [p for p in predictions if p.band == Band.DREAM]
            unlikely = [p for p in predictions if p.band == Band.UNLIKELY]

        preference = build_preference_order(
            safe,
            target,
            dream,
            prefer_hu=prefer_hu,
            max_items=self.max_preferences,
        )
        if len(safe) + len(target) + len(dream) > self.max_preferences:
            notes.append(
                f"Preference list capped at top {self.max_preferences} "
                f"(Safe then Target then Dream fillers)."
            )

        seat_summary = (
            work.groupby(["seat_type_section", "category_code"], dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values("row_count", ascending=False)
        )
        seat_rows = [
            SeatSummaryRow(
                seat_type_section=str(r.seat_type_section),
                category_code=str(r.category_code),
                row_count=int(r.row_count),
            )
            for r in seat_summary.itertuples(index=False)
        ]

        notes.append(
            f"Matched {len(work)} cutoff rows across {len(predictions)} institute-branch pairs "
            f"for branch filter '{profile.target_branch}'."
        )

        def _sort_band(items: list[CollegePrediction]) -> list[CollegePrediction]:
            return sorted(
                items,
                key=lambda p: (
                    -(p.hu_match_score or 0) if prefer_hu else 0,
                    -p.margin,
                ),
            )

        return PredictionResult(
            profile=profile,
            eligible=[p for p in predictions if p.band != Band.UNLIKELY],
            dream=_sort_band(dream),
            target=_sort_band(target),
            safe=_sort_band(safe),
            unlikely=_sort_band(unlikely),
            preference_order=preference,
            seat_summary=seat_rows,
            data_years=years,
            matched_cutoff_rows=int(len(work)),
            notes=notes,
        )
