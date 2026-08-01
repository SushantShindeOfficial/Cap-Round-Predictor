"""Optional ML band predictor (XGBoost preferred, sklearn fallback)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cap_predictor.ml.features import FEATURE_COLUMNS, build_feature_frame
from cap_predictor.models import Band, PredictionResult, StudentProfile
from cap_predictor.utils.paths import data_processed_dir

logger = logging.getLogger(__name__)

BAND_TO_INT = {
    Band.SAFE.value: 0,
    Band.TARGET.value: 1,
    Band.DREAM.value: 2,
    Band.UNLIKELY.value: 3,
}
INT_TO_BAND = {v: k for k, v in BAND_TO_INT.items()}

FEATURE_NUM = [
    "student_percentile",
    "closing_percentile",
    "margin",
    "yoy_delta",
    "primary_year",
    "primary_round",
    "hu_match_score",
    "trend_points",
]


def models_dir() -> Path:
    path = data_processed_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work["yoy_delta"] = work["yoy_delta"].fillna(0.0)
    work["hu_match_score"] = work["hu_match_score"].fillna(0.0)
    work["trend_points"] = work["trend_points"].fillna(0)
    x = work[FEATURE_NUM].astype(float)
    y = work["band"].map(BAND_TO_INT)
    mask = y.notna()
    return x.loc[mask], y.loc[mask].astype(int)


def _make_model():
    try:
        from xgboost import XGBClassifier  # type: ignore

        logger.info("Using XGBClassifier")
        return XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            verbosity=0,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

        logger.info("XGBoost unavailable; using sklearn GradientBoostingClassifier")
        return GradientBoostingClassifier(random_state=42)


def train_band_model_from_features(
    features_df: pd.DataFrame,
    *,
    model_path: Path | None = None,
) -> dict:
    if features_df.empty or "band" not in features_df.columns:
        raise ValueError("features_df must include band labels")
    x, y = _prepare_xy(features_df)
    if len(x) < 20:
        raise ValueError(f"Need at least 20 labeled rows to train; got {len(x)}")

    model = _make_model()
    model.fit(x, y)

    out = model_path or (models_dir() / "band_model.json")
    # Persist via joblib if available, else sklearn-compatible pickle path noted in meta
    try:
        import joblib  # type: ignore

        bin_path = out.with_suffix(".joblib")
        joblib.dump(model, bin_path)
        artifact = str(bin_path)
    except Exception:
        import pickle

        bin_path = out.with_suffix(".pkl")
        bin_path.write_bytes(pickle.dumps(model))
        artifact = str(bin_path)

    # simple train accuracy
    pred = model.predict(x)
    acc = float((pred == y.to_numpy()).mean())
    meta = {
        "artifact": artifact,
        "backend": type(model).__name__,
        "features": FEATURE_NUM,
        "rows": int(len(x)),
        "train_accuracy": round(acc, 4),
        "band_mapping": BAND_TO_INT,
    }
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Trained band model acc=%.3f -> %s", acc, artifact)
    return meta


def load_band_model(meta_path: Path | None = None):
    meta_file = meta_path or (models_dir() / "band_model.json")
    if not meta_file.exists():
        raise FileNotFoundError(f"No trained model meta at {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    artifact = Path(meta["artifact"])
    try:
        import joblib  # type: ignore

        model = joblib.load(artifact)
    except Exception:
        import pickle

        model = pickle.loads(artifact.read_bytes())
    return model, meta


class MLBandPredictor:
    """
    Wraps a trained classifier to re-label RuleBasedPredictor outputs.

    Does not invent colleges; only re-scores bands for existing predictions.
    """

    def __init__(self, meta_path: Path | None = None) -> None:
        self.model, self.meta = load_band_model(meta_path)

    def predict(self, profile: StudentProfile, cutoffs) -> PredictionResult:
        from cap_predictor.predictor.engine import RuleBasedPredictor

        base = RuleBasedPredictor().predict(profile, cutoffs)
        feats = build_feature_frame(base)
        if feats.empty:
            base.notes.append("ML model loaded but no feature rows to re-score.")
            return base

        x = feats.copy()
        x["yoy_delta"] = x["yoy_delta"].fillna(0.0)
        x["hu_match_score"] = x["hu_match_score"].fillna(0.0)
        x["trend_points"] = x["trend_points"].fillna(0)
        proba = None
        try:
            preds = self.model.predict(x[FEATURE_NUM].astype(float))
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(x[FEATURE_NUM].astype(float))
        except Exception as exc:  # noqa: BLE001
            base.notes.append(f"ML re-score failed ({exc}); using rule bands.")
            return base

        # Map predictions back onto college objects
        key_to_band = {
            (int(r.institute_code), str(r.branch_code)): INT_TO_BAND.get(int(p), r.band.value)
            for r, p in zip(feats.itertuples(index=False), preds)
        }

        def relabel(items):
            for c in items:
                new_band = key_to_band.get((c.institute_code, c.branch_code))
                if new_band:
                    c.band = Band(new_band)
                    c.rule_reason = (
                        c.rule_reason + f" ML band override -> {new_band}."
                    ).strip()

        for bucket in (base.safe, base.target, base.dream, base.unlikely, base.preference_order, base.eligible):
            relabel(bucket)

        # Rebuild buckets from preference+unlikely universe
        all_cols = {
            (c.institute_code, c.branch_code): c
            for c in (base.preference_order + base.unlikely + base.eligible)
        }
        values = list(all_cols.values())
        base.safe = [c for c in values if c.band == Band.SAFE]
        base.target = [c for c in values if c.band == Band.TARGET]
        base.dream = [c for c in values if c.band == Band.DREAM]
        base.unlikely = [c for c in values if c.band == Band.UNLIKELY]
        from cap_predictor.predictor.ranking import build_preference_order

        base.preference_order = build_preference_order(base.safe, base.target, base.dream)
        base.eligible = [c for c in values if c.band != Band.UNLIKELY]
        base.notes.append(
            f"ML band model applied ({self.meta.get('backend')}, "
            f"train_acc={self.meta.get('train_accuracy')})."
        )
        if proba is not None:
            base.notes.append("ML probabilities computed for feature rows.")
        return base
