"""Optional HTML → PDF export via WeasyPrint."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_pdf(html_path: Path | str, pdf_path: Path | str | None = None) -> Path | None:
    html_path = Path(html_path)
    pdf_path = Path(pdf_path) if pdf_path else html_path.with_suffix(".pdf")
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "WeasyPrint unavailable (%s). HTML report kept at %s",
            exc,
            html_path,
        )
        return None

    try:
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        logger.info("Wrote PDF %s", pdf_path)
        return pdf_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF export failed (%s). HTML remains at %s", exc, html_path)
        return None
