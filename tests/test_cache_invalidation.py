"""Parse cache invalidation tests."""

import json
from pathlib import Path

from cap_predictor.parser.normalizer import _cache_is_fresh, _pdf_fingerprint


def test_cache_freshness_detects_mtime_change(tmp_path: Path):
    pdf = tmp_path / "2025ENGG_CAP1_CutOff.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    parquet = tmp_path / "2025ENGG_CAP1_CutOff.parquet"
    parquet.write_bytes(b"parquet-bytes")
    meta = tmp_path / "2025ENGG_CAP1_CutOff.meta.json"
    fp = _pdf_fingerprint(pdf)
    meta.write_text(json.dumps({**fp, "rows": 1}), encoding="utf-8")
    assert _cache_is_fresh(pdf, parquet) is True

    # Change PDF size/mtime
    pdf.write_bytes(b"%PDF-1.4 fake larger content")
    assert _cache_is_fresh(pdf, parquet) is False
