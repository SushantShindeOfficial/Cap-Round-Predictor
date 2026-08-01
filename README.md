# Maharashtra CAP Round Prediction Report Generator

Offline Python MVP that parses CET CAP cutoff PDFs, runs a deterministic rule-based predictor, optionally narrates with local Ollama (`qwen3:14b`), and writes an HTML/PDF counselling report.

## Project root

`E:\Sushant\LLMCHATBOT_v1_F\Normal Cap Predictor`

## Data

Cutoff PDFs live in `Knowledge Base/` (auto-discovered). Currently includes:

- 2024: CAP1, CAP2, CAP3
- 2025: CAP1, CAP2, CAP3, CAP4

The parser is round-aware (year/round inferred from filenames). Prediction prefers the **latest year**, then the **latest round**, for primary chance bands; older rounds/years remain as evidence.

## Setup

One-click (recommended on Windows):

```powershell
cd "E:\Sushant\LLMCHATBOT_v1_F\Normal Cap Predictor"
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Manual:

```powershell
cd "E:\Sushant\LLMCHATBOT_v1_F\Normal Cap Predictor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Parse PDFs

```powershell
$env:PYTHONPATH = "src"
python -m cap_predictor.cli parse --force-reparse
```

Cached parquet files land in `data/processed/`.

### Seat matrix (inventory overlay)

Place CAP Round I seat-matrix PDFs under `Knowledge Base/Seat Matrix/` (e.g. `2024SeatMatrix.pdf`, `2025SeatMatrix.pdf`). Parse once:

```powershell
python -m cap_predictor.cli parse-seat-matrix
# smoke: python -m cap_predictor.cli parse-seat-matrix --max-pages 30
# or with cutoffs: python -m cap_predictor.cli parse --with-seat-matrix
```

This writes `data/processed/seat_matrix_*.parquet` (SHA256-validated cache), `seat_matrix_all.parquet`, and seeds `data/masters/college_master.csv`.

**Cutoffs drive Dream/Target/Safe bands.** Seat matrix attaches `available_seats`, competition, confidence %, and a weighted `final_score` for preference ranking. Options with **0 seats** for the mapped cell are excluded from the preference list. The LLM may cite seat fields from facts only — it must not invent seat counts.
## Predict from user inputs

The report is driven by these student fields:

| Field | Example |
|-------|---------|
| Percentile | 92.5 |
| Category (CAP seat code) | GOPENS / GOPENH / GOPENO |
| Gender | M |
| Home University | Mumbai University |
| Seat Type | ALL |
| Target Branch | Computer Engineering (from catalog) |
| Year | All or 2025 |
| Round | All or 1–4 |

### Band meanings (Safe / Target / Dream)

Margin = student percentile − closing percentile (historical comparison only):

| Band | Meaning | Rule |
|------|---------|------|
| Safe | Clearly above last year's/round's cutoff | margin ≥ +3 |
| Target | Around the cutoff — realistic shot | margin between −2 and +3 |
| Dream | Below cutoff, but not far — stretch option | margin between −8 and −2 |
| Unlikely | Too far below | margin < −8 |

### Option A — Streamlit form (recommended)

```powershell
$env:PYTHONPATH = "src"
streamlit run src/cap_predictor/app_streamlit.py
```

Enter student fields (including Year/Round and CAP seat code), tune band thresholds if needed, then click **Generate report**. Download HTML/CSV/PDF from the page (also saved under `outputs/reports/`).

Dropdowns list **every university, branch, and category code** found in the cutoff PDFs (plus known CAP seat codes). Choose **(All)** on Home University, Gender, Branch, or Category to skip that filter.

**Category `(All)`** means general G/L seats for the selected Gender only — it does **not** include MI / TFWS / EWS / DEF / PWD / ORPHAN (pick those codes explicitly). Target Branch from the dropdown is an **exact** match (e.g. Computer Engineering will not pull Civil Engineering).

Evidence confidence: **high** = multi-year, **medium** = multi-round same year, **low** = single point. Cache auto-refreshes when a Knowledge Base PDF changes size/mtime.

Exports per run: HTML, CSV, Excel (`.xlsx`), and ML feature parquet (`*_features.parquet`). Preference list defaults to top **30** (Safe → Target → Dream).

### Phase 4 ops / ML

```powershell
# Build DuckDB store (also auto-synced on parse)
python -m cap_predictor.cli build-db

# Parser quality report (empty-page %, rows/page, unknown tokens)
python -m cap_predictor.cli validate --max-pages 30

# Train band model (XGBoost if installed, else sklearn)
python -m cap_predictor.cli train-model

# Predict with ML band re-score
python -m cap_predictor.cli predict --use-ml --percentile 92.5 --category GOPENS --gender M `
  --home-university "Mumbai University" --seat-type ALL --branch "Computer Engineering" --no-llm
```

### Phase 5 — API, outcome labels, What-if

**Streamlit launchers**

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
# What-if tab: compare two profiles side-by-side (counts + top preferences)
```

**Outcome-labeled training**

```powershell
# Example schema written on train (or copy manually):
# data/labels/allotments.example.csv  ->  data/labels/allotments.csv

python -m cap_predictor.cli train-model
# Uses allotments.csv when present; else synthetic rule labels.
# Force synthetic: python -m cap_predictor.cli train-model --synthetic
```

**FastAPI**

```powershell
powershell -ExecutionPolicy Bypass -File .\serve-api.ps1
# or:
python -m cap_predictor.cli serve --host 127.0.0.1 --port 8080
# Health: GET  http://127.0.0.1:8080/health
# Predict: POST http://127.0.0.1:8080/predict  (OpenAPI docs at /docs)
```

CI runs unit tests (excluding slow PDF smokes) on push/PR via `.github/workflows/ci.yml`.

### Option B — Interactive CLI (prompts for missing fields)

```powershell
python -m cap_predictor.cli predict --interactive --no-llm
```

### Option C — CLI with all fields passed

```powershell
python -m cap_predictor.cli predict `
  --percentile 92.5 `
  --category GOPENS `
  --gender M `
  --home-university "Mumbai University" `
  --seat-type ALL `
  --branch "Computer Engineering" `
  --year 2025 `
  --round 4 `
  --no-llm
```

List branches from cache: `python -m cap_predictor.cli list-branches`

Omit `--no-llm` when Ollama is running with `qwen3:14b`:

```powershell
ollama pull qwen3:14b
ollama serve
```

Reports are written to `outputs/reports/`.
## Tests

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

## Notes

- Admission chances are **historical closing-percentile comparisons**, not guarantees.
- LLM output is constrained to colleges/cutoffs present in the parsed dataset.
- PDF export needs WeasyPrint system deps on Windows; HTML is always produced.
