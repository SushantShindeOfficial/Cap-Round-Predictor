# Knowledge Base (local data)

Place CET Cell CAP cutoff PDFs, seat-matrix PDFs, and the Information Brochure here. Large PDFs are **not** committed to GitHub (see `.gitignore`).

## Expected layout

```
Knowledge Base/
  2024ENGG_CAP1_CutOff.pdf
  ...
  Seat Matrix/
    2024SeatMatrix.pdf
    2025SeatMatrix.pdf
    seat-matrix-2026.pdf
  rulebook/
    Information_Brochure_UG_&_PG_2026-27.pdf
```

Then parse:

```powershell
$env:PYTHONPATH = "src"
python -m cap_predictor.cli parse
python -m cap_predictor.cli parse-seat-matrix
python -m cap_predictor.cli parse-rulebook
```
