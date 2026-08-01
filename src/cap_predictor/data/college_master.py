"""College master metadata loader (seeded from seat-matrix / cutoff headers)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cap_predictor.utils.paths import data_masters_dir

logger = logging.getLogger(__name__)

MASTER_COLUMNS = [
    "institute_code",
    "college_name",
    "district",
    "autonomous",
    "government",
    "private",
    "status_raw",
    "naac",
    "nba",
    "fees",
    "website",
    "lat",
    "lon",
]


def college_master_path() -> Path:
    return data_masters_dir() / "college_master.csv"


def _flags_from_status(status: str) -> tuple[bool, bool, bool]:
    s = (status or "").lower()
    government = "government" in s and "un-aided" not in s
    autonomous = "autonomous" in s
    private = "un-aided" in s or "private" in s
    if not government and not private:
        private = True
    return autonomous, government, private


def seed_college_master_from_seat_matrix(seat_df: pd.DataFrame) -> Path:
    """Build/update college_master.csv from distinct institutes in seat matrix."""
    path = college_master_path()
    if seat_df is None or seat_df.empty:
        if not path.exists():
            pd.DataFrame(columns=MASTER_COLUMNS).to_csv(path, index=False)
        return path

    work = seat_df.copy()
    work["institute_code"] = work["institute_code"].astype(int)
    # Prefer row with non-empty status
    work["_status_len"] = work["status"].fillna("").astype(str).str.len()
    work = work.sort_values(["institute_code", "_status_len"], ascending=[True, False])
    uniq = work.drop_duplicates("institute_code")

    rows = []
    for _, r in uniq.iterrows():
        status = str(r.get("status") or "")
        auto, gov, priv = _flags_from_status(status)
        rows.append(
            {
                "institute_code": int(r["institute_code"]),
                "college_name": r.get("institute_name") or "",
                "district": "",
                "autonomous": auto,
                "government": gov,
                "private": priv,
                "status_raw": status,
                "naac": "",
                "nba": "",
                "fees": "",
                "website": "",
                "lat": "",
                "lon": "",
            }
        )
    new_df = pd.DataFrame(rows)

    if path.exists():
        old = pd.read_csv(path)
        # Preserve enrichment columns when institute already present
        if not old.empty and "institute_code" in old.columns:
            old["institute_code"] = old["institute_code"].astype(int)
            merged = new_df.merge(
                old,
                on="institute_code",
                how="left",
                suffixes=("", "_old"),
            )
            for col in ("district", "naac", "nba", "fees", "website", "lat", "lon", "college_name"):
                old_col = f"{col}_old"
                if old_col in merged.columns:
                    merged[col] = merged[col].where(
                        merged[col].astype(str).str.len() > 0, merged[old_col]
                    )
            keep = [c for c in MASTER_COLUMNS if c in merged.columns]
            new_df = merged[keep]

    new_df = new_df.reindex(columns=MASTER_COLUMNS)
    new_df.to_csv(path, index=False)
    logger.info("Wrote college master %s (%s institutes)", path, len(new_df))
    return path


def load_college_master() -> pd.DataFrame:
    path = college_master_path()
    if not path.exists():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    df = pd.read_csv(path)
    if "institute_code" in df.columns:
        df["institute_code"] = df["institute_code"].astype(int)
    return df


def college_tier_score(master_row: dict | None) -> float:
    """0..1 tier for ranking: Government/Autonomous preferred."""
    if not master_row:
        return 0.5
    score = 0.4
    if master_row.get("government"):
        score += 0.35
    if master_row.get("autonomous"):
        score += 0.2
    if master_row.get("private") and not master_row.get("government"):
        score -= 0.05
    return max(0.0, min(1.0, score))
