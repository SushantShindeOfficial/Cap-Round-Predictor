"""Shared prediction → report pipeline used by CLI and Streamlit."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from cap_predictor.llm.grounded import generate_grounded_narrative
from cap_predictor.ml.features import write_feature_export
from cap_predictor.models import PredictionResult, StudentProfile
from cap_predictor.parser.normalizer import load_all_cutoffs
from cap_predictor.predictor.engine import BandThresholds, RuleBasedPredictor
from cap_predictor.report.csv_export import write_result_csv
from cap_predictor.report.excel_export import write_excel_report
from cap_predictor.report.html_renderer import write_html_report
from cap_predictor.report.pdf_export import export_pdf
from cap_predictor.utils.paths import reports_dir


def run_prediction_report(
    profile: StudentProfile,
    *,
    cutoffs: pd.DataFrame | None = None,
    seat_matrix: pd.DataFrame | None = None,
    force_reparse: bool = False,
    use_llm: bool = True,
    out_dir: Path | str | None = None,
    export_pdf_file: bool = True,
    export_csv_file: bool = True,
    export_excel_file: bool = True,
    export_features_file: bool = True,
    thresholds: BandThresholds | None = None,
    max_preferences: int = 30,
    use_ml: bool = False,
) -> tuple[PredictionResult, Path, Path | None, Path | None, Path | None, Path | None]:
    """
    Run predictor and write HTML (+ optional PDF/CSV/Excel/features).

    Returns
    -------
    result, html_path, pdf_path, csv_path, excel_path, features_path
    """
    df = cutoffs if cutoffs is not None else load_all_cutoffs(force_reparse=force_reparse)
    sm = seat_matrix
    if sm is None:
        try:
            from cap_predictor.parser.seat_matrix_store import load_all_seat_matrix

            sm = load_all_seat_matrix(force_reparse=False)
        except Exception:  # noqa: BLE001
            sm = None

    if use_ml:
        try:
            from cap_predictor.ml.band_model import MLBandPredictor

            result = MLBandPredictor().predict(profile, df)
        except Exception as exc:  # noqa: BLE001
            result = RuleBasedPredictor(
                thresholds=thresholds,
                max_preferences=max_preferences,
                seat_matrix=sm,
            ).predict(profile, df)
            result.notes.append(f"ML model unavailable ({exc}); used rule engine.")
    else:
        result = RuleBasedPredictor(
            thresholds=thresholds,
            max_preferences=max_preferences,
            seat_matrix=sm,
        ).predict(profile, df)
    result.narrative = generate_grounded_narrative(result, use_llm=use_llm)

    destination = Path(out_dir) if out_dir else reports_dir()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = destination / f"cap_report_{stamp}"

    html_path = write_html_report(result, Path(str(stem) + ".html"))
    pdf_path = export_pdf(html_path) if export_pdf_file else None
    csv_path = write_result_csv(result, Path(str(stem) + ".csv")) if export_csv_file else None
    excel_path = write_excel_report(result, Path(str(stem) + ".xlsx")) if export_excel_file else None
    features_path = (
        write_feature_export(result, Path(str(stem) + "_features.parquet"))
        if export_features_file
        else None
    )
    return result, html_path, pdf_path, csv_path, excel_path, features_path
