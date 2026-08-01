"""Load real allotment outcome labels for supervised band training."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cap_predictor.utils.paths import project_root

logger = logging.getLogger(__name__)

# Expected columns in data/labels/allotments.csv
REQUIRED_COLUMNS = {
    "student_percentile",
    "institute_code",
    "branch_code",
    "category_code",
    "year",
    "round",
    "allotted",  # 1/0 or true/false
}

OPTIONAL_COLUMNS = {
    "closing_percentile",
    "home_university",
    "gender",
    "seat_type",
    "band",  # optional explicit band override
}


def labels_dir() -> Path:
    path = project_root() / "data" / "labels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_labels_path() -> Path:
    return labels_dir() / "allotments.csv"


def example_labels_path() -> Path:
    return labels_dir() / "allotments.example.csv"


def write_example_labels(path: Path | None = None) -> Path:
    out = path or example_labels_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame(
        [
            {
                "student_percentile": 96.5,
                "institute_code": 3012,
                "branch_code": "301224510",
                "category_code": "GOPENH",
                "year": 2024,
                "round": 1,
                "allotted": 1,
                "closing_percentile": 95.2,
                "home_university": "Mumbai University",
                "gender": "M",
                "seat_type": "HU",
                "band": "safe",
            },
            {
                "student_percentile": 88.0,
                "institute_code": 3012,
                "branch_code": "301224510",
                "category_code": "GOPENH",
                "year": 2024,
                "round": 1,
                "allotted": 0,
                "closing_percentile": 95.2,
                "home_university": "Mumbai University",
                "gender": "M",
                "seat_type": "HU",
                "band": "dream",
            },
        ]
    )
    sample.to_csv(out, index=False)
    return out


def load_outcome_labels(path: Path | None = None) -> pd.DataFrame:
    path = path or default_labels_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Outcome labels not found at {path}. "
            f"Copy {example_labels_path().name} to allotments.csv and fill real outcomes."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Labels file missing columns: {sorted(missing)}")
    df["allotted"] = df["allotted"].astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )
    return df


def labels_to_training_features(
    labels: pd.DataFrame,
    cutoffs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join allotment outcomes to cutoff closings and derive band labels.

    Band mapping when `band` column absent:
    - allotted & margin >= 0 -> safe/target by margin thresholds
    - allotted & margin < 0 -> target/dream
    - not allotted -> dream/unlikely by margin
    """
    from cap_predictor.predictor.engine import BandThresholds, classify_band

    thr = BandThresholds()
    work = labels.copy()
    cut = cutoffs.copy()
    cut["institute_code"] = cut["institute_code"].astype(int)
    cut["branch_code"] = cut["branch_code"].astype(str)
    cut["category_code"] = cut["category_code"].astype(str)
    work["institute_code"] = work["institute_code"].astype(int)
    work["branch_code"] = work["branch_code"].astype(str)
    work["category_code"] = work["category_code"].astype(str)

    # Prefer Stage I closings
    stage_i = cut[cut["stage"] == "I"] if "stage" in cut.columns else cut
    join_cols = ["year", "round", "institute_code", "branch_code", "category_code"]
    merged = work.merge(
        stage_i[join_cols + ["closing_percentile", "institute_name", "university",
                             "branch_name", "seat_type_section"]],
        on=join_cols,
        how="left",
        suffixes=("", "_cut"),
    )
    if "closing_percentile_cut" in merged.columns:
        merged["closing_percentile"] = merged["closing_percentile"].fillna(
            merged["closing_percentile_cut"]
        )

    rows: list[dict] = []
    for _, r in merged.iterrows():
        if pd.isna(r.get("closing_percentile")):
            continue
        pct = float(r["student_percentile"])
        closing = float(r["closing_percentile"])
        margin = pct - closing
        if pd.notna(r.get("band")) and str(r["band"]).strip():
            band = str(r["band"]).strip().lower()
        else:
            band = classify_band(pct, closing, thr).value
            # nudge using allotted outcome
            if bool(r["allotted"]) and band == "unlikely":
                band = "dream"
            if (not bool(r["allotted"])) and band == "safe":
                band = "target"
        rows.append(
            {
                "institute_code": int(r["institute_code"]),
                "institute_name": r.get("institute_name"),
                "university": r.get("university"),
                "branch_code": r.get("branch_code"),
                "branch_name": r.get("branch_name"),
                "band": band,
                "chance_label": None,
                "student_percentile": pct,
                "closing_percentile": closing,
                "margin": margin,
                "yoy_delta": 0.0,
                "primary_year": int(r["year"]),
                "primary_round": int(r["round"]),
                "category_code": r.get("category_code"),
                "seat_type_section": r.get("seat_type_section"),
                "hu_match_score": 0.0,
                "evidence_confidence": "high",
                "trend_points": 1,
                "student_category": r.get("category_code"),
                "student_seat_type": r.get("seat_type") or "ALL",
                "student_gender": r.get("gender") or "M",
                "home_university": r.get("home_university") or "",
                "target_branch": r.get("branch_name"),
                "allotted": bool(r["allotted"]),
            }
        )
    logger.info("Built %s training rows from outcome labels", len(rows))
    return pd.DataFrame(rows)
