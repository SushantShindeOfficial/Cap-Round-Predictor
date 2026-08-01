"""Prediction engine unit tests."""

import pandas as pd

from cap_predictor.models import Band, ChanceLabel, CollegePrediction, StudentProfile
from cap_predictor.predictor.engine import BandThresholds, RuleBasedPredictor, classify_band
from cap_predictor.predictor.ranking import build_preference_order


def test_classify_band_thresholds():
    thr = BandThresholds()
    assert classify_band(95, 90, thr) == Band.SAFE
    assert classify_band(91, 90, thr) == Band.TARGET
    assert classify_band(85, 90, thr) == Band.DREAM
    assert classify_band(80, 90, thr) == Band.UNLIKELY


def _sample_df() -> pd.DataFrame:
    rows = [
        {
            "year": 2025,
            "round": 1,
            "institute_code": 3012,
            "institute_name": "VJTI Mumbai",
            "university": "Mumbai University",
            "branch_code": "301224510",
            "branch_name": "Computer Engineering",
            "seat_type_section": "HU_TO_HU",
            "category_code": "GOPENH",
            "gender_flag": "G",
            "reservation": "OPEN",
            "quota_suffix": "H",
            "stage": "I",
            "closing_merit": 500,
            "closing_percentile": 99.0,
            "source_pdf": "t.pdf",
            "page_number": 1,
        },
        {
            "year": 2025,
            "round": 4,
            "institute_code": 3012,
            "institute_name": "VJTI Mumbai",
            "university": "Mumbai University",
            "branch_code": "301224510",
            "branch_name": "Computer Engineering",
            "seat_type_section": "HU_TO_HU",
            "category_code": "GOPENH",
            "gender_flag": "G",
            "reservation": "OPEN",
            "quota_suffix": "H",
            "stage": "I",
            "closing_merit": 600,
            "closing_percentile": 98.5,
            "source_pdf": "t.pdf",
            "page_number": 1,
        },
        {
            "year": 2025,
            "round": 1,
            "institute_code": 1002,
            "institute_name": "GCOE Amravati",
            "university": "Autonomous Institute",
            "branch_code": "100224210",
            "branch_name": "Computer Science and Engineering",
            "seat_type_section": "STATE_LEVEL",
            "category_code": "GOPENS",
            "gender_flag": "G",
            "reservation": "OPEN",
            "quota_suffix": "S",
            "stage": "I",
            "closing_merit": 8000,
            "closing_percentile": 90.0,
            "source_pdf": "t.pdf",
            "page_number": 2,
        },
        {
            "year": 2025,
            "round": 1,
            "institute_code": 2008,
            "institute_name": "Lower Tier College",
            "university": "SGBAU",
            "branch_code": "200824210",
            "branch_name": "Computer Science and Engineering",
            "seat_type_section": "STATE_LEVEL",
            "category_code": "GOPENS",
            "gender_flag": "G",
            "reservation": "OPEN",
            "quota_suffix": "S",
            "stage": "I",
            "closing_merit": 40000,
            "closing_percentile": 85.0,
            "source_pdf": "t.pdf",
            "page_number": 3,
        },
        {
            "year": 2024,
            "round": 1,
            "institute_code": 2008,
            "institute_name": "Lower Tier College",
            "university": "SGBAU",
            "branch_code": "200824210",
            "branch_name": "Computer Science and Engineering",
            "seat_type_section": "STATE_LEVEL",
            "category_code": "GOPENS",
            "gender_flag": "G",
            "reservation": "OPEN",
            "quota_suffix": "S",
            "stage": "I",
            "closing_merit": 41000,
            "closing_percentile": 84.0,
            "source_pdf": "t.pdf",
            "page_number": 4,
        },
    ]
    return pd.DataFrame(rows)


def test_predictor_bands_and_preference():
    profile = StudentProfile(
        percentile=92.0,
        category="OPEN",
        gender="M",
        home_university="Mumbai University",
        seat_type="ALL",
        target_branch="Computer",
    )
    result = RuleBasedPredictor().predict(profile, _sample_df())
    assert result.matched_cutoff_rows >= 2
    assert 2008 in {p.institute_code for p in result.safe}
    assert result.preference_order
    assert result.preference_order[0].band == Band.SAFE


def test_year_filter():
    profile = StudentProfile(
        percentile=92.0,
        category="GOPENS",
        gender="M",
        home_university="Mumbai University",
        seat_type="ALL",
        target_branch="Computer Science and Engineering",
        year=2024,
    )
    result = RuleBasedPredictor().predict(profile, _sample_df())
    assert result.data_years == [2024]
    assert all(p.primary_year == 2024 for p in result.preference_order)


def test_hu_match_score_populated():
    profile = StudentProfile(
        percentile=99.2,
        category="GOPENH",
        gender="M",
        home_university="Mumbai University",
        seat_type="HU",
        target_branch="Computer Engineering",
    )
    result = RuleBasedPredictor().predict(profile, _sample_df())
    vjti = next(p for p in result.preference_order if p.institute_code == 3012)
    assert vjti.hu_match_score is not None and vjti.hu_match_score >= 75
    assert vjti.closing_trend
    assert any(t.round == 4 for t in vjti.closing_trend)


def test_preference_order_safe_first():
    def mk(code: int, band: Band, margin: float, hu: float = 0) -> CollegePrediction:
        return CollegePrediction(
            institute_code=code,
            institute_name=f"C{code}",
            branch_code=str(code),
            branch_name="CSE",
            band=band,
            chance_label=ChanceLabel.MEDIUM,
            margin=margin,
            primary_closing_percentile=90,
            primary_year=2025,
            primary_round=1,
            primary_category_code="GOPENS",
            rule_reason="t",
            hu_match_score=hu,
        )

    order = build_preference_order(
        [mk(1, Band.SAFE, 1.0, hu=50)],
        [mk(2, Band.TARGET, 5.0, hu=90)],
        [mk(3, Band.DREAM, 10.0, hu=100)],
        prefer_hu=True,
    )
    assert [c.institute_code for c in order] == [1, 2, 3]


def test_preference_list_cap():
    profile = StudentProfile(
        percentile=99.9,
        category="GOPENS",
        gender="M",
        home_university="Mumbai University",
        seat_type="ALL",
        target_branch="Computer",
    )
    # Duplicate many safe-ish rows by expanding sample with lower cutoffs
    df = _sample_df()
    extra = []
    for i in range(40):
        row = df.iloc[-1].to_dict()
        row["institute_code"] = 5000 + i
        row["institute_name"] = f"College {i}"
        row["branch_code"] = f"5000{i:04d}210"
        row["closing_percentile"] = 50.0
        extra.append(row)
    import pandas as pd

    big = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    result = RuleBasedPredictor(max_preferences=10).predict(profile, big)
    assert len(result.preference_order) <= 10
    assert any("capped" in n.lower() for n in result.notes)


def test_branch_all_includes_multiple_branches():
    profile = StudentProfile(
        percentile=92.0,
        category="GOPENS",
        gender="M",
        home_university="ALL",
        seat_type="ALL",
        target_branch="ALL",
    )
    result = RuleBasedPredictor().predict(profile, _sample_df())
    assert result.matched_cutoff_rows >= 1
    assert any("Branch filter: All" in n for n in result.notes)


def test_category_all_excludes_special_seats():
    df = _sample_df()
    # Inject a low MI closing that would otherwise dominate Safe
    extra = df.iloc[0].to_dict()
    extra["institute_code"] = 9999
    extra["institute_name"] = "Minority Fake College"
    extra["branch_code"] = "999924510"
    extra["branch_name"] = "Computer Engineering"
    extra["category_code"] = "MI"
    extra["reservation"] = "MI"
    extra["closing_percentile"] = 1.0
    import pandas as pd

    big = pd.concat([df, pd.DataFrame([extra])], ignore_index=True)
    profile = StudentProfile(
        percentile=92.5,
        category="ALL",
        gender="M",
        home_university="Mumbai University",
        seat_type="ALL",
        target_branch="Computer Engineering",
    )
    result = RuleBasedPredictor().predict(profile, big)
    assert 9999 not in {p.institute_code for p in result.preference_order}
    assert any("OPEN seats only" in n for n in result.notes)


def test_exact_computer_engineering_excludes_civil():
    df = _sample_df()
    # Add civil with low closing under GOPENS
    row = df.iloc[-1].to_dict()
    row["institute_code"] = 8888
    row["institute_name"] = "Civil Only College"
    row["branch_code"] = "888824210"
    row["branch_name"] = "Civil Engineering"
    row["category_code"] = "GOPENS"
    row["closing_percentile"] = 10.0
    import pandas as pd

    big = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    profile = StudentProfile(
        percentile=92.5,
        category="GOPENS",
        gender="M",
        home_university="Mumbai University",
        seat_type="ALL",
        target_branch="Computer Engineering",
    )
    result = RuleBasedPredictor().predict(profile, big)
    assert all(p.branch_name != "Civil Engineering" for p in result.preference_order)
    assert 8888 not in {p.institute_code for p in result.preference_order}


def test_category_all_skips_code_filter():
    profile = StudentProfile(
        percentile=92.0,
        category="ALL",
        gender="ALL",
        home_university="Mumbai University",
        seat_type="ALL",
        target_branch="Computer",
    )
    result = RuleBasedPredictor().predict(profile, _sample_df())
    assert result.matched_cutoff_rows >= 1
    assert any("OPEN seats only" in n or "Category filter: All general" in n or "Percentile" in n for n in result.notes)


def test_profile_normalizes_all_ui_label():
    profile = StudentProfile(
        percentile=90.0,
        category="(All)",
        gender="(All)",
        home_university="(All)",
        seat_type="ALL",
        target_branch="(All)",
    )
    assert profile.category == "ALL"
    assert profile.gender == "ALL"
    assert profile.home_university == "ALL"
    assert profile.target_branch == "ALL"


def test_multi_branch_include_and_exclude():
    df = _sample_df()
    civil = df.iloc[0].copy()
    civil["institute_code"] = 7777
    civil["institute_name"] = "Civil College"
    civil["branch_code"] = "777719110"
    civil["branch_name"] = "Civil Engineering"
    civil["category_code"] = "GOPENS"
    civil["closing_percentile"] = 80.0
    df = pd.concat([df, pd.DataFrame([civil])], ignore_index=True)

    profile = StudentProfile(
        percentile=92.0,
        category="GOPENS",
        gender="M",
        home_university="ALL",
        seat_type="ALL",
        target_branch="Computer Engineering || Computer Science and Engineering || Civil Engineering",
        exclude_branches=["Civil Engineering"],
    )
    result = RuleBasedPredictor(max_preferences=100).predict(profile, df)
    names = {p.branch_name for p in result.preference_order + result.safe + result.target + result.dream}
    assert "Civil Engineering" not in names
    assert names  # some computer-ish options remain


def test_max_preferences_honored():
    df = _sample_df()
    profile = StudentProfile(
        percentile=92.0,
        category="GOPENS",
        gender="M",
        home_university="ALL",
        seat_type="ALL",
        target_branch="ALL",
    )
    result = RuleBasedPredictor(max_preferences=2).predict(profile, df)
    assert len(result.preference_order) <= 2
