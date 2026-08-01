"""Parser validation helper tests."""

from cap_predictor.parser.pdf_parser import ParseStats
from cap_predictor.parser.validation import write_validation_report


def test_parse_stats_to_dict():
    stats = ParseStats(pages=10, rows=100, pages_with_rows=8, pages_without_rows=2)
    stats.institutes.add(1002)
    stats.unknown_tokens["FOO"] = 3
    d = stats.to_dict()
    assert d["empty_page_pct"] == 20.0
    assert d["unknown_token_count"] == 3


def test_write_validation_report(tmp_path):
    payload = {"generated_at": "t", "totals": {"pdfs": 0}, "pdfs": []}
    out = write_validation_report(payload, tmp_path / "v.json")
    assert out.exists()
    assert "totals" in out.read_text(encoding="utf-8")
