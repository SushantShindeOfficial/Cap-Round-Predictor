"""Excel counselling report export."""

from __future__ import annotations

import logging
from pathlib import Path

from cap_predictor.models import PredictionResult
from cap_predictor.report.csv_export import predictions_to_rows
from cap_predictor.utils.band_legend import BAND_MEANINGS

logger = logging.getLogger(__name__)


def write_excel_report(result: PredictionResult, out_path: Path | str) -> Path | None:
    """Write multi-sheet .xlsx report. Returns path or None if openpyxl unavailable."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        logger.warning("pandas unavailable for Excel export: %s", exc)
        return None

    try:
        profile_df = pd.DataFrame([result.profile.model_dump()])
        legend_df = pd.DataFrame(BAND_MEANINGS)
        pct = result.profile.percentile
        pref_df = pd.DataFrame(
            predictions_to_rows(result.preference_order, student_percentile=pct)
        )
        safe_df = pd.DataFrame(predictions_to_rows(result.safe, student_percentile=pct))
        target_df = pd.DataFrame(predictions_to_rows(result.target, student_percentile=pct))
        dream_df = pd.DataFrame(predictions_to_rows(result.dream, student_percentile=pct))
        notes_df = pd.DataFrame({"notes": result.notes or ["(none)"]})

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            profile_df.to_excel(writer, sheet_name="Profile", index=False)
            legend_df.to_excel(writer, sheet_name="Band_Meanings", index=False)
            pref_df.to_excel(writer, sheet_name="Preference_Order", index=False)
            safe_df.to_excel(writer, sheet_name="Safe", index=False)
            target_df.to_excel(writer, sheet_name="Target", index=False)
            dream_df.to_excel(writer, sheet_name="Dream", index=False)
            notes_df.to_excel(writer, sheet_name="Notes", index=False)
        logger.info("Wrote Excel report %s", out_path)
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Excel export failed (%s). Install openpyxl if missing.", exc)
        return None
