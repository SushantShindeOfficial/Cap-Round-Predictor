"""FastAPI service for CAP prediction reports."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cap_predictor.models import StudentProfile
from cap_predictor.parser.normalizer import load_all_cutoffs
from cap_predictor.pipeline import run_prediction_report
from cap_predictor.predictor.engine import BandThresholds

app = FastAPI(
    title="Maharashtra CAP Predictor API",
    version="0.5.0",
    description="Offline CAP counselling prediction API (rule-based + optional ML).",
)


class PredictRequest(BaseModel):
    percentile: float = Field(ge=0, le=100)
    category: str
    gender: str
    home_university: str
    seat_type: str = "ALL"
    target_branch: str
    year: int | None = None
    round: int | None = Field(default=None, ge=1, le=4)
    use_llm: bool = False
    use_ml: bool = False
    max_preferences: int = 30
    safe_margin: float = 3.0
    target_below: float = 2.0
    dream_below: float = 8.0


class PredictResponse(BaseModel):
    counts: dict[str, int]
    preference_order: list[dict[str, Any]]
    notes: list[str]
    narrative: str | None
    html_path: str
    csv_path: str | None = None
    excel_path: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        profile = StudentProfile(
            percentile=req.percentile,
            category=req.category,
            gender=req.gender,
            home_university=req.home_university,
            seat_type=req.seat_type,
            target_branch=req.target_branch,
            year=req.year,
            round=req.round,
        )
        cutoffs = load_all_cutoffs(force_reparse=False)
        thresholds = BandThresholds(
            safe_margin=req.safe_margin,
            target_below=req.target_below,
            dream_below=req.dream_below,
        )
        result, html_path, _pdf, csv_path, excel_path, _feat = run_prediction_report(
            profile,
            cutoffs=cutoffs,
            use_llm=req.use_llm,
            export_pdf_file=False,
            export_csv_file=True,
            export_excel_file=True,
            export_features_file=False,
            thresholds=thresholds,
            max_preferences=req.max_preferences,
            use_ml=req.use_ml,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prefs = [
        {
            "institute_code": c.institute_code,
            "institute_name": c.institute_name,
            "branch_name": c.branch_name,
            "band": c.band.value,
            "margin": c.margin,
            "closing_percentile": c.primary_closing_percentile,
            "year": c.primary_year,
            "round": c.primary_round,
            "category_code": c.primary_category_code,
            "evidence_confidence": c.evidence_confidence,
        }
        for c in result.preference_order
    ]
    return PredictResponse(
        counts={
            "safe": len(result.safe),
            "target": len(result.target),
            "dream": len(result.dream),
            "unlikely": len(result.unlikely),
            "preferences": len(result.preference_order),
        },
        preference_order=prefs,
        notes=result.notes,
        narrative=result.narrative,
        html_path=str(html_path),
        csv_path=str(csv_path) if csv_path else None,
        excel_path=str(excel_path) if excel_path else None,
    )


def main() -> None:
    import uvicorn

    uvicorn.run("cap_predictor.api:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
