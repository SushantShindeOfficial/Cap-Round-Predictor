"""Render prediction reports to HTML."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from cap_predictor.models import PredictionResult
from cap_predictor.utils.band_legend import BAND_MEANINGS


def _template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def _evidence_rows(result: PredictionResult, limit: int = 60) -> list[dict]:
    rows: list[dict] = []
    for c in result.preference_order + result.safe + result.target + result.dream:
        for e in c.evidence:
            rows.append(
                {
                    "institute": f"{c.institute_code} — {c.institute_name}",
                    "year": e.year,
                    "round": e.round,
                    "category_code": e.category_code,
                    "stage": e.stage,
                    "closing_merit": e.closing_merit,
                    "closing_percentile": e.closing_percentile,
                    "seat_type_section": e.seat_type_section,
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def render_html(result: PredictionResult) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_template_dir())),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.globals["evidence_rows"] = _evidence_rows
    template = env.get_template("report.html")
    return template.render(
        result=result,
        profile=result.profile,
        band_meanings=BAND_MEANINGS,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def write_html_report(result: PredictionResult, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(result)
    out_path.write_text(html, encoding="utf-8")
    return out_path
