"""Tests for category mapping helpers."""

from cap_predictor.utils.category_map import (
    CAP_CATEGORY_SEATS,
    applicable_category_codes,
    general_codes_for_gender,
    is_full_cap_category_code,
    is_special_cap_code,
    parse_category_code,
    repair_category_headers,
)


def test_repair_split_pwdrobcs():
    assert repair_category_headers(["PWDROBC", "S", "EWS"]) == ["PWDROBCS", "EWS"]


def test_parse_gopenh():
    parts = parse_category_code("GOPENH")
    assert parts["gender_flag"] == "G"
    assert parts["reservation"] == "OPEN"
    assert parts["quota_suffix"] == "H"


def test_applicable_codes_male_open_hu():
    codes = applicable_category_codes(category="OPEN", gender="M", seat_type="HU")
    assert "GOPENH" in codes
    assert "GOPENS" in codes
    assert "LOPENH" not in codes


def test_applicable_codes_female_includes_l():
    codes = applicable_category_codes(category="OBC", gender="F", seat_type="HU")
    assert "LOBCH" in codes
    assert "GOBCH" in codes


def test_applicable_codes_all_gender_includes_g_and_l():
    codes = applicable_category_codes(category="OPEN", gender="ALL", seat_type="HU")
    assert "GOPENH" in codes
    assert "LOPENH" in codes


def test_applicable_category_all_returns_empty():
    assert applicable_category_codes(category="ALL", gender="M", seat_type="ALL") == []


def test_general_codes_for_gender_excludes_specials():
    available = [
        "GOPENS",
        "GOPENH",
        "LOPENS",
        "MI",
        "TFWS",
        "EWS",
        "DEFOPENS",
        "PWDOPENS",
        "GOBCS",
        "GSCH",
        "GNT1H",
    ]
    male = general_codes_for_gender(available, "M")
    assert "GOPENS" in male and "GOPENH" in male
    assert "LOPENS" not in male
    assert "GOBCS" not in male and "GSCH" not in male and "GNT1H" not in male
    assert "MI" not in male and "TFWS" not in male and "EWS" not in male
    female = general_codes_for_gender(available, "F")
    assert "LOPENS" in female and "GOPENS" in female
    assert "GSCH" not in female


def test_general_codes_respects_seat_type_hu():
    available = ["GOPENH", "GOPENO", "GOPENS", "GSCH"]
    hu = general_codes_for_gender(available, "M", seat_type="HU")
    assert "GOPENH" in hu and "GOPENS" in hu
    assert "GOPENO" not in hu
    assert "GSCH" not in hu


def test_is_special_cap_code():
    assert is_special_cap_code("MI")
    assert is_special_cap_code("TFWS")
    assert is_special_cap_code("DEFOPENS")
    assert not is_special_cap_code("GOPENS")


def test_full_dropdown_includes_hos_variants():
    for code in ["GOPENH", "GOPENO", "GOPENS", "LOPENH", "LOPENO", "LOPENS", "GOBCH", "EWS"]:
        assert code in CAP_CATEGORY_SEATS


def test_exact_cap_code_not_expanded():
    assert is_full_cap_category_code("GOPENS")
    assert is_full_cap_category_code("GOPENH")
    assert not is_full_cap_category_code("OPEN")
    assert applicable_category_codes(category="GOPENS", gender="M", seat_type="ALL") == [
        "GOPENS"
    ]
    assert applicable_category_codes(category="GOPENH", gender="M", seat_type="HU") == [
        "GOPENH"
    ]
    assert applicable_category_codes(category="LOPENS", gender="M", seat_type="HU") == [
        "LOPENS"
    ]
    assert applicable_category_codes(category="EWS", gender="F", seat_type="STATE") == ["EWS"]
