"""Streamlit UI: collect student inputs and generate CAP prediction report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cap_predictor.models import StudentProfile
from cap_predictor.parser.normalizer import load_all_cutoffs
from cap_predictor.pipeline import run_prediction_report
from cap_predictor.predictor.engine import BandThresholds
from cap_predictor.report.csv_export import filter_predictions, result_to_csv_string
from cap_predictor.utils.band_legend import BAND_MEANINGS
from cap_predictor.utils.branch_catalog import (
    ALL_UI_LABEL,
    load_branch_names,
    load_university_names,
    merge_category_options,
    ui_value_to_profile,
    with_all_option,
)

st.set_page_config(page_title="CAP Predictor", layout="wide")
st.title("Maharashtra CAP Prediction Report")
st.caption(
    "Enter student details below. Predictions are historical closing-percentile "
    "comparisons only — not admission guarantees."
)


@st.cache_data(show_spinner="Loading cutoff dataset...")
def _cached_cutoffs(force_reparse: bool = False) -> pd.DataFrame:
    return load_all_cutoffs(force_reparse=force_reparse)


with st.expander("What Safe / Target / Dream mean", expanded=True):
    st.markdown(
        "Margin = **student percentile - closing percentile**. "
        "These bands are historical comparisons, not admission guarantees."
    )
    st.dataframe(BAND_MEANINGS, use_container_width=True, hide_index=True)
    st.markdown(
        "**Evidence confidence:** `high` = multi-year, `medium` = multi-round same year, "
        "`low` = single year+round point."
    )

force_reparse_sidebar = st.sidebar.checkbox("Force re-parse PDFs", value=False)
use_ml = st.sidebar.checkbox("Use trained ML band model", value=False)
try:
    cutoffs_df = _cached_cutoffs(force_reparse_sidebar)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load cutoffs: {exc}")
    st.stop()

branch_options = load_branch_names(cutoffs_df)
university_options = load_university_names(cutoffs_df)
category_options = merge_category_options(cutoffs_df)
try:
    from cap_predictor.data.duckdb_store import duckdb_available, duckdb_path, list_distinct

    if duckdb_available() and duckdb_path().exists():
        db_branches = list_distinct("branch_name")
        db_univs = list_distinct("university")
        db_cats = list_distinct("category_code")
        if db_branches:
            branch_options = db_branches
        if db_univs:
            university_options = db_univs
        if db_cats:
            # Keep PDF codes first, then any CAP vocabulary extras
            from cap_predictor.utils.category_map import CAP_CATEGORY_SEATS

            seen = {c.upper() for c in db_cats}
            extras = [c for c in CAP_CATEGORY_SEATS if c.upper() not in seen]
            category_options = list(db_cats) + extras
        st.sidebar.caption("Catalog source: DuckDB")
except Exception:
    pass

# UI lists with (All) — underlying profile uses ALL
branch_ui = with_all_option(branch_options)
university_ui = with_all_option(university_options)
category_ui = with_all_option(category_options)
gender_ui = [ALL_UI_LABEL, "M", "F"]
seat_type_ui = ["ALL", "HU", "OHU", "STATE"]

years = sorted(int(y) for y in cutoffs_df["year"].dropna().unique().tolist()) if not cutoffs_df.empty else []
rounds = sorted(int(r) for r in cutoffs_df["round"].dropna().unique().tolist()) if not cutoffs_df.empty else []

default_branch = "Computer Engineering"
if default_branch not in branch_options and branch_options:
    computerish = [b for b in branch_options if "computer" in b.lower()]
    default_branch = computerish[0] if computerish else branch_options[0]
# index in UI list (after (All) at 0)
branch_index = (
    branch_ui.index(default_branch) if default_branch in branch_ui else (1 if len(branch_ui) > 1 else 0)
)
default_category = "GOPENS"
category_index = (
    category_ui.index(default_category)
    if default_category in category_ui
    else (1 if len(category_ui) > 1 else 0)
)
default_hu = next((u for u in university_options if "mumbai" in u.lower()), None)
hu_index = university_ui.index(default_hu) if default_hu and default_hu in university_ui else 0

st.sidebar.caption(
    f"Catalog: {len(university_options)} universities, "
    f"{len(branch_options)} branches, {len(category_options)} categories"
)

tab_predict, tab_whatif = st.tabs(["Predict", "What-if"])

with tab_predict:
    st.subheader("Student inputs")
    with st.form("student_input_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            percentile = st.number_input(
                "Percentile",
                min_value=0.0,
                max_value=100.0,
                value=92.5,
                step=0.01,
                help="MHT-CET / JEE percentile",
            )
            category = st.selectbox(
                "Category (CAP seat code)",
                category_ui,
                index=category_index,
                help=(
                    "Your CAP seat code. Default GOPENS for Open. "
                    "(All) = OPEN seats only (GOPENH/O/S) for your Gender — "
                    "bands still use your Percentile vs those closings."
                ),
            )
            gender = st.selectbox(
                "Gender",
                gender_ui,
                index=1 if "M" in gender_ui else 0,
                help="M/F, or (All) to include both G* and L* seats when expanding aliases.",
            )
            year_label = st.selectbox(
                "Year",
                ["All"] + [str(y) for y in years],
                index=0,
            )
        with c2:
            home_university = st.selectbox(
                "Home University",
                university_ui if university_ui else [ALL_UI_LABEL, "Mumbai University"],
                index=hu_index if university_ui else 0,
                help="All universities found in cutoff PDFs. (All) skips HU preference.",
            )
            seat_type = st.selectbox(
                "Seat Type",
                seat_type_ui,
                index=0,
            )
            branch_filter = st.text_input("Filter branches", value="")
            filtered_branches = branch_options
            if branch_filter.strip():
                q = branch_filter.strip().lower()
                filtered_branches = [b for b in branch_options if q in b.lower()] or branch_options
            branch_choices = with_all_option(filtered_branches)
            # Keep default branch if still in filtered list
            bi = (
                branch_choices.index(default_branch)
                if default_branch in branch_choices
                else (1 if len(branch_choices) > 1 else 0)
            )
            branch = st.selectbox(
                "Target Branch",
                branch_choices if branch_choices else [ALL_UI_LABEL, "Computer Engineering"],
                index=bi,
                help="All branch names from cutoff PDFs. (All) includes every branch.",
            )
            round_label = st.selectbox(
                "CAP Round",
                ["All"] + [str(r) for r in rounds],
                index=0,
            )

        with st.expander("Band thresholds (counsellor tuning)"):
            safe_margin = st.number_input("Safe margin (>=)", value=3.0, step=0.5)
            target_below = st.number_input("Target below cutoff (to -X)", value=2.0, step=0.5)
            dream_below = st.number_input("Dream below cutoff (to -X)", value=8.0, step=0.5)
            max_preferences = st.number_input(
                "Max preference list length",
                min_value=5,
                max_value=100,
                value=30,
                step=5,
            )

        with st.expander("Advanced options"):
            use_llm = st.checkbox("Use Ollama (qwen3:14b) for counselling text", value=False)
            export_excel = st.checkbox("Export Excel report", value=True)
            export_features = st.checkbox("Export ML feature parquet", value=True)
            # sidebar use_ml already controls ML; keep form value aligned
            _ = use_ml

        submitted = st.form_submit_button("Generate report", type="primary")

    if submitted:
        if not str(branch).strip():
            st.error("Please select Target Branch.")
            st.stop()
        if not str(home_university).strip():
            st.error("Please enter Home University.")
            st.stop()

        year_val = None if year_label == "All" else int(year_label)
        round_val = None if round_label == "All" else int(round_label)
        thresholds = BandThresholds(
            safe_margin=float(safe_margin),
            target_below=float(target_below),
            dream_below=float(dream_below),
        )

        profile = StudentProfile(
            percentile=float(percentile),
            category=ui_value_to_profile(str(category)),
            gender=ui_value_to_profile(str(gender)),
            home_university=ui_value_to_profile(str(home_university)),
            seat_type=seat_type,
            target_branch=ui_value_to_profile(str(branch)),
            year=year_val,
            round=round_val,
        )

        st.warning(
            "Disclaimer: results are historical cutoff comparisons only and are not "
            "admission guarantees. Always verify with official CET CAP allotment rules."
        )

        with st.spinner("Generating report..."):
            df = (
                _cached_cutoffs(force_reparse_sidebar)
                if not force_reparse_sidebar
                else load_all_cutoffs(force_reparse=True)
            )
            result, html_path, pdf_path, csv_path, excel_path, features_path = run_prediction_report(
                profile,
                cutoffs=df,
                use_llm=use_llm,
                export_pdf_file=True,
                export_csv_file=True,
                export_excel_file=export_excel,
                export_features_file=export_features,
                thresholds=thresholds,
                max_preferences=int(max_preferences),
                use_ml=use_ml,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Safe", len(result.safe))
        c2.metric("Target", len(result.target))
        c3.metric("Dream", len(result.dream))
        c4.metric("Matched rows", result.matched_cutoff_rows)

        with st.expander("Applied filters (from your inputs)", expanded=True):
            st.markdown(
                f"- **Percentile (drives all bands):** {profile.percentile}  \n"
                f"  Margin = **{profile.percentile} − closing percentile** for each college  \n"
                f"- **Category:** {profile.category}  \n"
                f"- **Gender:** {profile.gender}  \n"
                f"- **Home University:** {profile.home_university}  \n"
                f"- **Seat Type:** {profile.seat_type}  \n"
                f"- **Branch:** {profile.target_branch}  \n"
                f"- **Year / Round:** {profile.year or 'All'} / {profile.round or 'All'}"
            )
            if result.notes:
                st.markdown("**Engine notes:**")
                for n in result.notes:
                    st.markdown(f"- {n}")
            prefs_branches = sorted({c.branch_name for c in result.preference_order})
            if prefs_branches:
                st.caption("Branches in preference list: " + ", ".join(prefs_branches[:12]))

        st.subheader("Filter preference list")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            univ_filter = st.text_input("University / institute contains", value="")
        with fc2:
            min_hu = st.slider("Min HU match score", 0, 100, 0)
        with fc3:
            band_filter = st.multiselect(
                "Bands",
                ["safe", "target", "dream"],
                default=["safe", "target", "dream"],
            )

        filtered = filter_predictions(
            result.preference_order,
            university_contains=univ_filter or None,
            min_hu_score=float(min_hu) if min_hu > 0 else None,
            bands=set(band_filter) if band_filter else None,
        )

        st.subheader("Counselling narrative")
        st.write(result.narrative)

        st.subheader(f"Preference order ({len(filtered)} after filters)")
        prefs = [
            {
                "rank": i,
                "institute": f"{c.institute_code} - {c.institute_name}",
                "university": c.university or "",
                "branch": c.branch_name,
                "band": c.band.value,
                "confidence": c.evidence_confidence,
                "chance": c.chance_label.value,
                "margin": round(c.margin, 3),
                "closing": c.primary_closing_percentile,
                "seats": c.available_seats,
                "competition": c.competition or "",
                "conf_pct": c.confidence_pct,
                "score": c.final_score,
                "tier": c.college_tier or "",
                "year": c.primary_year,
                "round": c.primary_round,
                "hu_score": round(c.hu_match_score or 0, 1),
                "category": c.primary_category_code,
            }
            for i, c in enumerate(filtered[:40], start=1)
        ]
        st.dataframe(prefs, use_container_width=True)

        st.subheader("Closing trend (top 5 filtered)")
        for c in filtered[:5]:
            if not c.closing_trend:
                continue
            st.markdown(
                f"**{c.institute_code} — {c.institute_name}** / {c.branch_name} "
                f"(confidence: {c.evidence_confidence})"
            )
            trend_df = pd.DataFrame(
                [
                    {
                        "label": f"{t.year}-R{t.round}",
                        "closing_percentile": t.closing_percentile,
                    }
                    for t in c.closing_trend
                ]
            ).set_index("label")
            st.line_chart(trend_df)

        with st.expander("Why this college? (evidence)"):
            for c in filtered[:15]:
                st.markdown(
                    f"**{c.institute_code} — {c.institute_name}** "
                    f"({c.band.value}, confidence={c.evidence_confidence})  \n"
                    f"{c.rule_reason}"
                )

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button(
                "Download HTML",
                data=Path(html_path).read_bytes(),
                file_name=html_path.name,
                mime="text/html",
            )
        with d2:
            st.download_button(
                "Download CSV",
                data=result_to_csv_string(result),
                file_name=(csv_path.name if csv_path else "cap_report.csv"),
                mime="text/csv",
            )
        with d3:
            if excel_path and Path(excel_path).exists():
                st.download_button(
                    "Download Excel",
                    data=Path(excel_path).read_bytes(),
                    file_name=Path(excel_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        with d4:
            if pdf_path and Path(pdf_path).exists():
                st.download_button(
                    "Download PDF",
                    data=Path(pdf_path).read_bytes(),
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                )
        saved = [str(html_path)]
        if csv_path:
            saved.append(str(csv_path))
        if excel_path:
            saved.append(str(excel_path))
        if features_path:
            saved.append(str(features_path))
        st.success("Saved: " + " | ".join(saved))
    else:
        st.info(
            "Fill student inputs (including Year / Round filters and Target Branch), "
            "then click **Generate report**."
        )

with tab_whatif:
    st.subheader("What-if comparison")
    st.caption(
        "Compare two student profiles side-by-side (same branch/filters, different percentile "
        "or category). No report files are written — counts and top preferences only."
    )
    with st.form("whatif_form", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**Profile A**")
            pct_a = st.number_input("Percentile A", 0.0, 100.0, 90.0, 0.01, key="wa_pct")
            cat_a = st.selectbox(
                "Category A",
                category_ui,
                index=category_index,
                key="wa_cat",
            )
            gender_a = st.selectbox("Gender A", gender_ui, index=1, key="wa_gender")
        with right:
            st.markdown("**Profile B**")
            pct_b = st.number_input("Percentile B", 0.0, 100.0, 95.0, 0.01, key="wb_pct")
            cat_b = st.selectbox(
                "Category B",
                category_ui,
                index=category_index,
                key="wb_cat",
            )
            gender_b = st.selectbox("Gender B", gender_ui, index=1, key="wb_gender")

        shared1, shared2 = st.columns(2)
        with shared1:
            hu_shared = st.selectbox(
                "Home University (both)",
                university_ui if university_ui else [ALL_UI_LABEL, "Mumbai University"],
                index=hu_index if university_ui else 0,
                key="w_hu",
            )
            seat_shared = st.selectbox("Seat Type (both)", seat_type_ui, key="w_seat")
            year_w = st.selectbox(
                "Year (both)",
                ["All"] + [str(y) for y in years],
                index=0,
                key="w_year",
            )
        with shared2:
            branch_w = st.selectbox(
                "Target Branch (both)",
                branch_ui if branch_ui else [ALL_UI_LABEL, "Computer Engineering"],
                index=branch_index if branch_ui else 0,
                key="w_branch",
            )
            round_w = st.selectbox(
                "CAP Round (both)",
                ["All"] + [str(r) for r in rounds],
                index=0,
                key="w_round",
            )
            max_pref_w = st.number_input("Max preferences", 5, 50, 15, 5, key="w_max")

        compare = st.form_submit_button("Compare profiles", type="primary")

    if compare:
        year_val = None if year_w == "All" else int(year_w)
        round_val = None if round_w == "All" else int(round_w)
        df = _cached_cutoffs(force_reparse_sidebar)
        results = []
        for label, pct, cat, gen in [
            ("A", pct_a, cat_a, gender_a),
            ("B", pct_b, cat_b, gender_b),
        ]:
            profile = StudentProfile(
                percentile=float(pct),
                category=ui_value_to_profile(str(cat)),
                gender=ui_value_to_profile(str(gen)),
                home_university=ui_value_to_profile(str(hu_shared)),
                seat_type=seat_shared,
                target_branch=ui_value_to_profile(str(branch_w)),
                year=year_val,
                round=round_val,
            )
            result, *_ = run_prediction_report(
                profile,
                cutoffs=df,
                use_llm=False,
                export_pdf_file=False,
                export_csv_file=False,
                export_excel_file=False,
                export_features_file=False,
                max_preferences=int(max_pref_w),
                use_ml=use_ml,
            )
            results.append((label, result))

        col_a, col_b = st.columns(2)
        for col, (label, result) in zip([col_a, col_b], results):
            with col:
                st.markdown(f"### Profile {label}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Safe", len(result.safe))
                m2.metric("Target", len(result.target))
                m3.metric("Dream", len(result.dream))
                rows = [
                    {
                        "rank": i,
                        "institute": f"{c.institute_code} - {c.institute_name}",
                        "band": c.band.value,
                        "margin": round(c.margin, 3),
                        "closing": c.primary_closing_percentile,
                    }
                    for i, c in enumerate(result.preference_order[: int(max_pref_w)], start=1)
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Set Profile A and B, then click **Compare profiles**.")
