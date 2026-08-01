"""Slow smoke test against real CAP PDF (first pages)."""

from pathlib import Path

import pytest

from cap_predictor.parser.pdf_parser import parse_pdf
from cap_predictor.utils.paths import knowledge_base_dir


@pytest.mark.slow
def test_parse_first_pages_known_gopens():
    pdf = knowledge_base_dir() / "2024ENGG_CAP1_CutOff.pdf"
    if not pdf.exists():
        pytest.skip("CAP PDF not present in Knowledge Base")

    records, stats = parse_pdf(pdf, max_pages=20)
    assert stats.rows > 0
    civil = [
        r
        for r in records
        if r.institute_code == 1002
        and "Civil" in r.branch_name
        and r.category_code == "GOPENS"
        and r.stage == "I"
    ]
    assert civil, "Expected GOPENS Civil Engineering row for institute 1002"
    assert civil[0].closing_percentile is not None
    assert abs(civil[0].closing_percentile - 88.5013511) < 0.05
