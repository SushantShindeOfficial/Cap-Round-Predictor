# Knowledge Base (local data)

Place CET Cell CAP cutoff and seat-matrix PDFs here. They are **not** committed to GitHub (see `.gitignore`).

## Expected layout

```
Knowledge Base/
  2024ENGG_CAP1_CutOff.pdf
  2024ENGG_CAP2_CutOff.pdf
  ...
  2025ENGG_CAP4_CutOff.pdf
  Seat Matrix/
    2024SeatMatrix.pdf
    2025SeatMatrix.pdf
```

Then parse:

```powershell
$env:PYTHONPATH = "src"
python -m cap_predictor.cli parse
python -m cap_predictor.cli parse-seat-matrix
```
