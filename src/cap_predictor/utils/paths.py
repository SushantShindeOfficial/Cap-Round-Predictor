"""Resolve project root and standard directories."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return Normal Cap Predictor root (contains Knowledge Base/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Knowledge Base").is_dir():
            return parent
    # Fallback: src/cap_predictor/utils -> parents[3]
    return here.parents[3]


def knowledge_base_dir() -> Path:
    return project_root() / "Knowledge Base"


def data_processed_dir() -> Path:
    path = project_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_masters_dir() -> Path:
    path = project_root() / "data" / "masters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir() -> Path:
    path = project_root() / "outputs" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def seat_matrix_dir() -> Path:
    return knowledge_base_dir() / "Seat Matrix"


def rulebook_dir() -> Path:
    return knowledge_base_dir() / "rulebook"


def default_pdf_paths() -> list[Path]:
    kb = knowledge_base_dir()
    paths = {
        *kb.glob("*CAP*_CutOff.pdf"),
        *kb.glob("*CAP*_Cutoff.pdf"),
        *kb.glob("*CutOff*.pdf"),
    }
    return sorted(p for p in paths if p.is_file())


def default_seat_matrix_pdf_paths() -> list[Path]:
    sm = seat_matrix_dir()
    if not sm.is_dir():
        return []
    paths = {
        *sm.glob("*SeatMatrix*.pdf"),
        *sm.glob("*Seat*Matrix*.pdf"),
        *sm.glob("*seat*matrix*.pdf"),
        *sm.glob("seat-matrix-*.pdf"),
    }
    return sorted(p for p in paths if p.is_file())


def default_rulebook_pdf_paths() -> list[Path]:
    rb = rulebook_dir()
    if not rb.is_dir():
        return []
    paths = {
        *rb.glob("*Information*Brochure*.pdf"),
        *rb.glob("*Brochure*.pdf"),
        *rb.glob("*.pdf"),
    }
    return sorted(p for p in paths if p.is_file())
