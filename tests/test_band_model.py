"""ML band model training tests (sklearn fallback)."""

import pandas as pd

from cap_predictor.ml.band_model import train_band_model_from_features
from cap_predictor.ml.training_data import synthesize_training_features
from cap_predictor.predictor.engine import classify_band, BandThresholds


def test_synthesize_and_train(tmp_path):
    cutoffs = pd.DataFrame(
        [
            {
                "year": 2025,
                "round": 1,
                "institute_code": 1000 + i,
                "institute_name": f"C{i}",
                "university": "MU",
                "branch_code": f"{1000+i}24510",
                "branch_name": "Computer Engineering",
                "seat_type_section": "STATE_LEVEL",
                "category_code": "GOPENS",
                "gender_flag": "G",
                "reservation": "OPEN",
                "quota_suffix": "S",
                "stage": "I",
                "closing_merit": i,
                "closing_percentile": 70 + (i % 25),
                "source_pdf": "t.pdf",
                "page_number": 1,
            }
            for i in range(50)
        ]
    )
    feats = synthesize_training_features(cutoffs, sample_rows=50)
    assert len(feats) >= 20
    # labels should match rule engine
    thr = BandThresholds()
    sample = feats.iloc[0]
    assert sample["band"] == classify_band(
        float(sample["student_percentile"]),
        float(sample["closing_percentile"]),
        thr,
    ).value

    meta = train_band_model_from_features(feats, model_path=tmp_path / "band_model.json")
    assert meta["rows"] >= 20
    assert meta["train_accuracy"] >= 0.9
