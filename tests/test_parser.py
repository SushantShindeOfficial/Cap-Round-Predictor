"""Parser fixture tests."""

from pathlib import Path

from cap_predictor.parser.pdf_parser import parse_pdf_text_fixture


FIXTURE = Path(__file__).parent / "fixtures" / "sample_cap_text.txt"


def test_parse_text_fixture_rows():
    text = FIXTURE.read_text(encoding="utf-8")
    records = parse_pdf_text_fixture(text, year=2024, round_num=1)
    assert len(records) >= 8
    cse = [r for r in records if r.branch_code == "100224210"]
    assert cse
    gopens = next(r for r in cse if r.category_code == "GOPENS")
    assert gopens.closing_merit == 7872
    assert abs(gopens.closing_percentile - 97.3911937) < 1e-6
    assert gopens.seat_type_section == "STATE_LEVEL"


def test_parse_hu_section():
    text = FIXTURE.read_text(encoding="utf-8")
    records = parse_pdf_text_fixture(text)
    food = [r for r in records if "Food" in r.branch_name]
    assert food
    assert food[0].seat_type_section == "HU_TO_HU"
    assert any(r.category_code == "GOPENH" for r in food)
