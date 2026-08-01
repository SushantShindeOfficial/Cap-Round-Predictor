"""Band legend copy shared by Streamlit, CLI, and HTML reports."""

from __future__ import annotations

BAND_MEANINGS: list[dict[str, str]] = [
    {
        "band": "Safe",
        "meaning": "You're clearly above last year's/round's cutoff",
        "rule": "margin >= +3",
    },
    {
        "band": "Target",
        "meaning": "You're around the cutoff — realistic shot",
        "rule": "margin between -2 and +3",
    },
    {
        "band": "Dream",
        "meaning": "You're below cutoff, but not far — stretch option",
        "rule": "margin between -8 and -2",
    },
    {
        "band": "Unlikely",
        "meaning": "Too far below",
        "rule": "margin < -8",
    },
]

BAND_MEANINGS_TEXT = (
    "Safe: You're clearly above last year's/round's cutoff (margin >= +3). "
    "Target: You're around the cutoff - realistic shot (margin between -2 and +3). "
    "Dream: You're below cutoff, but not far - stretch option (margin between -8 and -2). "
    "Unlikely: Too far below (margin < -8). "
    "Margin = student percentile - closing percentile. Historical comparison only."
)
