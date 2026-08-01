"""Branch catalog tests."""

import pandas as pd

from cap_predictor.utils.branch_catalog import (
    is_all_selection,
    load_category_codes,
    load_university_names,
    match_branch_names,
    merge_category_options,
    resolve_branch_queries,
    ui_value_to_profile,
    with_all_option,
)
from cap_predictor.utils.category_map import CAP_CATEGORY_SEATS


def test_cse_alias_expands():
    queries = resolve_branch_queries("CSE")
    assert any("Computer" in q for q in queries)


def test_match_computer_engineering():
    names = [
        "Civil Engineering",
        "Computer Engineering",
        "Computer Science and Engineering",
        "Mechanical Engineering",
        "Automobile Engineering",
    ]
    matched = match_branch_names("CSE", names)
    assert "Computer Engineering" in matched or "Computer Science and Engineering" in matched
    assert "Civil Engineering" not in matched
    assert "Automobile Engineering" not in matched
    assert "Mechanical Engineering" not in matched


def test_exact_catalog_computer_engineering_no_civil():
    names = [
        "Civil Engineering",
        "Computer Engineering",
        "Automobile Engineering",
        "Mechanical Engineering",
    ]
    matched = match_branch_names(
        "Computer Engineering", names, exact_catalog=True
    )
    assert matched == ["Computer Engineering"]


def test_is_all_selection():
    assert is_all_selection("(All)")
    assert is_all_selection("ALL")
    assert is_all_selection("*")
    assert not is_all_selection("Mumbai University")
    assert ui_value_to_profile("(All)") == "ALL"


def test_with_all_option_prepends():
    opts = with_all_option(["A", "B"])
    assert opts[0] == "(All)"
    assert opts[1:] == ["A", "B"]


def test_merge_category_options_pdf_first():
    cutoffs = pd.DataFrame(
        {
            "category_code": ["GOPENS", "ZZZPDFONLY", "GOPENS"],
            "university": ["Mumbai University", "Pune University", "Mumbai University"],
            "branch_name": [
                "Computer Engineering",
                "Civil Engineering",
                "Computer Engineering",
            ],
        }
    )
    cats = merge_category_options(cutoffs)
    assert "GOPENS" in cats
    assert "ZZZPDFONLY" in cats
    assert any(c in CAP_CATEGORY_SEATS for c in cats)
    assert "Mumbai University" in load_university_names(cutoffs)
    assert "ZZZPDFONLY" in load_category_codes(cutoffs)
