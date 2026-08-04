"""CLI entrypoint for CAP prediction reports."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from cap_predictor.models import StudentProfile
from cap_predictor.parser.normalizer import load_all_cutoffs
from cap_predictor.pipeline import run_prediction_report
from cap_predictor.predictor.engine import BandThresholds
from cap_predictor.utils.band_legend import BAND_MEANINGS_TEXT
from cap_predictor.utils.branch_catalog import load_branch_names
from cap_predictor.utils.category_map import CAP_CATEGORY_SEATS
from cap_predictor.utils.logging_config import setup_logging
from cap_predictor.utils.paths import default_pdf_paths

logger = logging.getLogger(__name__)


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{label}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        print(f"  Please enter {label}.")


def _parse_optional_int(raw: str) -> int | None:
    text = raw.strip().lower()
    if not text or text in {"all", "none", "*"}:
        return None
    return int(text)


def collect_student_profile_interactive(
    *,
    percentile: float | None = None,
    category: str | None = None,
    gender: str | None = None,
    home_university: str | None = None,
    seat_type: str | None = None,
    branch: str | None = None,
    year: int | None = None,
    round_num: int | None = None,
    year_provided: bool = False,
    round_provided: bool = False,
    cutoffs: pd.DataFrame | None = None,
) -> StudentProfile:
    """Ask the user for any missing student fields, then build StudentProfile."""
    from cap_predictor.utils.branch_catalog import (
        load_branch_names,
        load_university_names,
        merge_category_options,
    )

    print("\nEnter student details for the CAP prediction report\n")
    print("Band meanings:", BAND_MEANINGS_TEXT, "\n")
    print("Tip: type ALL for Home University / Gender / Branch / Category to skip that filter.\n")

    df = cutoffs if cutoffs is not None else pd.DataFrame()
    univs = load_university_names(df) if not df.empty else []
    branches = load_branch_names(df) if not df.empty else []
    cats = merge_category_options(df) if not df.empty else list(CAP_CATEGORY_SEATS)

    if percentile is None:
        percentile = float(_prompt("Percentile", "92.5"))
    if not category:
        print(f"CAP seat codes ({len(cats)} from PDFs + vocabulary), e.g. GOPENH/GOPENO/GOPENS/ALL:")
        print("  " + ", ".join(cats[:24]) + (", ..." if len(cats) > 24 else ""))
        category = _prompt("Category (CAP seat code or ALL)", "GOPENS").upper()
    if not gender:
        gender = _prompt("Gender (M/F/ALL)", "M")
    if not home_university:
        if univs:
            print(f"Home universities from cutoffs ({len(univs)}):")
            print("  " + ", ".join(univs[:20]) + (", ..." if len(univs) > 20 else ""))
        home_university = _prompt("Home University (or ALL)", "Mumbai University")
    if not seat_type:
        seat_type = _prompt("Seat Type (HU/OHU/STATE/ALL)", "ALL")
    if not branch:
        if branches:
            print(f"Branches from cutoffs ({len(branches)}):")
            print("  " + ", ".join(branches[:20]) + (", ..." if len(branches) > 20 else ""))
        branch = _prompt("Target Branch (or ALL)", "Computer Engineering")
    if not year_provided:
        year = _parse_optional_int(_prompt("Year (All or e.g. 2025)", "All"))
    if not round_provided:
        round_num = _parse_optional_int(_prompt("CAP Round (All or 1-4)", "All"))

    return StudentProfile(
        percentile=float(percentile),
        category=category,
        gender=gender,
        home_university=home_university,
        seat_type=seat_type,
        target_branch=branch,
        year=year,
        round=round_num,
    )


def cmd_parse(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.pdf] if args.pdf else default_pdf_paths()
    df = load_all_cutoffs(paths, force_reparse=args.force_reparse, max_pages=args.max_pages)
    print(f"Parsed/loaded {len(df)} cutoff rows from {len(paths)} PDF(s).")
    if not df.empty:
        print(df.groupby(["year", "round"]).size().to_string())
    if getattr(args, "with_seat_matrix", False):
        # Cutoff --pdf paths must not be reused for seat-matrix PDFs
        sm_args = argparse.Namespace(
            pdf=None,
            force_reparse=bool(args.force_reparse),
            max_pages=getattr(args, "max_pages", None),
            log_level=getattr(args, "log_level", "INFO"),
        )
        return cmd_parse_seat_matrix(sm_args)
    return 0


def cmd_parse_seat_matrix(args: argparse.Namespace) -> int:
    from cap_predictor.data.college_master import seed_college_master_from_seat_matrix
    from cap_predictor.parser.seat_matrix_store import load_all_seat_matrix
    from cap_predictor.utils.paths import default_seat_matrix_pdf_paths

    paths = [Path(p) for p in args.pdf] if getattr(args, "pdf", None) else default_seat_matrix_pdf_paths()
    if not paths:
        print("No seat matrix PDFs found under Knowledge Base/Seat Matrix/")
        return 1
    df = load_all_seat_matrix(
        paths,
        force_reparse=bool(getattr(args, "force_reparse", False)),
        max_pages=getattr(args, "max_pages", None),
    )
    print(f"Parsed/loaded {len(df)} seat-matrix rows from {len(paths)} PDF(s).")
    if not df.empty and "year" in df.columns:
        print(df.groupby("year").size().to_string())
    master_path = seed_college_master_from_seat_matrix(df)
    print(f"College master: {master_path}")
    return 0


def cmd_parse_rulebook(args: argparse.Namespace) -> int:
    from cap_predictor.parser.rulebook_parser import parse_and_cache_rulebook
    from cap_predictor.utils.paths import default_rulebook_pdf_paths

    paths = [Path(p) for p in args.pdf] if getattr(args, "pdf", None) else default_rulebook_pdf_paths()
    if not paths:
        print("No rulebook PDFs found under Knowledge Base/rulebook/")
        return 1
    chunks = parse_and_cache_rulebook(
        paths,
        force=bool(getattr(args, "force_reparse", False)),
        max_pages=getattr(args, "max_pages", None),
    )
    print(f"Indexed {len(chunks)} rulebook page chunks from {len(paths)} PDF(s).")
    for p in paths:
        print(f"  {p.name}")
    return 0


def cmd_build_db(args: argparse.Namespace) -> int:
    from cap_predictor.data.duckdb_store import duckdb_path, sync_knowledge_base_to_duckdb

    path = sync_knowledge_base_to_duckdb(force_reparse_cutoffs=args.force_reparse)
    print(f"DuckDB ready: {path}")
    print(f"Default path: {duckdb_path()}")
    print("Tables: cutoffs + seat_matrix (from seat_matrix_all.parquet when present)")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from cap_predictor.parser.validation import validate_pdfs, write_validation_report

    paths = [Path(p) for p in args.pdf] if args.pdf else None
    payload = validate_pdfs(paths, max_pages=args.max_pages)
    out = write_validation_report(payload, args.out)
    totals = payload["totals"]
    print(
        f"Validation: pdfs={totals['pdfs']} pages={totals['pages']} rows={totals['rows']} "
        f"empty_page_pct={totals['empty_page_pct']} rows_per_page={totals['rows_per_page']}"
    )
    for item in payload["pdfs"]:
        print(
            f"  {item['source_pdf']}: rows={item['rows']} empty%={item['empty_page_pct']} "
            f"rpp={item['rows_per_page']} flag={item['quality_flag']}"
        )
    print(f"Report: {out}")
    return 0


def cmd_train_model(args: argparse.Namespace) -> int:
    from cap_predictor.ml.band_model import train_band_model_from_features
    from cap_predictor.ml.outcome_labels import (
        default_labels_path,
        labels_to_training_features,
        load_outcome_labels,
        write_example_labels,
    )
    from cap_predictor.ml.training_data import synthesize_training_features

    write_example_labels()  # ensure example exists for users
    df = load_all_cutoffs(force_reparse=False)
    source = "synthetic-rules"
    labels_path = Path(args.labels) if args.labels else default_labels_path()
    if labels_path.exists() and not args.synthetic:
        labels = load_outcome_labels(labels_path)
        features = labels_to_training_features(labels, df)
        source = f"outcome-labels:{labels_path.name}"
        if len(features) < 20:
            print(
                f"Only {len(features)} labeled rows from outcomes; "
                "augmenting with synthetic rule features."
            )
            synth = synthesize_training_features(df, sample_rows=args.sample_rows)
            features = pd.concat([features, synth], ignore_index=True)
            source += "+synthetic"
    else:
        features = synthesize_training_features(df, sample_rows=args.sample_rows)
        if not labels_path.exists():
            print(
                f"No {labels_path.name} found — trained on synthetic rule labels. "
                f"See data/labels/allotments.example.csv"
            )

    meta = train_band_model_from_features(features)
    print(
        f"Trained {meta['backend']} on {meta['rows']} rows "
        f"(source={source}); train_accuracy={meta['train_accuracy']}"
    )
    print(f"Artifact: {meta['artifact']}")
    return 0


def cmd_list_branches(args: argparse.Namespace) -> int:
    names: list[str] = []
    try:
        from cap_predictor.data.duckdb_store import duckdb_available, duckdb_path, list_distinct

        if duckdb_available() and duckdb_path().exists():
            names = list_distinct("branch_name")
    except Exception:  # noqa: BLE001
        names = []
    if not names:
        df = load_all_cutoffs(force_reparse=False)
        names = load_branch_names(df)
    for name in names:
        print(name)
    print(f"\n{len(names)} branches", file=sys.stderr)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.pdf] if args.pdf else None
    df = load_all_cutoffs(
        paths, force_reparse=args.force_reparse, max_pages=args.max_pages
    )

    missing = any(
        v is None
        for v in (
            args.percentile,
            args.category,
            args.gender,
            args.home_university,
            args.branch,
        )
    )
    if args.interactive or missing:
        profile = collect_student_profile_interactive(
            percentile=args.percentile,
            category=args.category,
            gender=args.gender,
            home_university=args.home_university,
            seat_type=args.seat_type,
            branch=args.branch,
            year=args.year,
            round_num=args.round,
            year_provided=args.year is not None,
            round_provided=args.round is not None,
            cutoffs=df,
        )
    else:
        profile = StudentProfile(
            percentile=args.percentile,
            category=args.category,
            gender=args.gender,
            home_university=args.home_university,
            seat_type=args.seat_type or "ALL",
            target_branch=args.branch,
            year=args.year,
            round=args.round,
        )

    print(
        "\nGenerating report for:\n"
        f"  Percentile       : {profile.percentile}\n"
        f"  Category         : {profile.category}\n"
        f"  Gender           : {profile.gender}\n"
        f"  Home University  : {profile.home_university}\n"
        f"  Seat Type        : {profile.seat_type}\n"
        f"  Target Branch    : {profile.target_branch}\n"
        f"  Year             : {profile.year or 'All'}\n"
        f"  Round            : {profile.round or 'All'}\n"
    )
    print("Band meanings:", BAND_MEANINGS_TEXT)

    thresholds = BandThresholds(
        safe_margin=args.safe_margin,
        target_below=args.target_below,
        dream_below=args.dream_below,
    )
    result, html_path, pdf_path, csv_path, excel_path, features_path = run_prediction_report(
        profile,
        cutoffs=df,
        use_llm=not args.no_llm,
        out_dir=args.out,
        export_pdf_file=not args.no_pdf,
        export_csv_file=not args.no_csv,
        export_excel_file=not args.no_excel,
        export_features_file=not args.no_features,
        thresholds=thresholds,
        max_preferences=args.max_preferences,
        use_ml=args.use_ml,
    )

    print(f"HTML report: {html_path}")
    if pdf_path:
        print(f"PDF report:  {pdf_path}")
    if csv_path:
        print(f"CSV report:  {csv_path}")
    if excel_path:
        print(f"Excel report: {excel_path}")
    if features_path:
        print(f"ML features:  {features_path}")
    print(
        f"Bands - safe={len(result.safe)} target={len(result.target)} "
        f"dream={len(result.dream)} unlikely={len(result.unlikely)} "
        f"preferences={len(result.preference_order)}"
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Starting API on http://{args.host}:{args.port}  (docs: /docs)")
    uvicorn.run(
        "cap_predictor.api:app",
        host=args.host,
        port=int(args.port),
        reload=False,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cap-predictor",
        description="Maharashtra CAP Round Prediction Report Generator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Parse CAP cutoff PDFs into parquet cache")
    p_parse.add_argument("--log-level", default="INFO")
    p_parse.add_argument("--pdf", nargs="*", help="PDF paths (default: Knowledge Base/)")
    p_parse.add_argument("--force-reparse", action="store_true")
    p_parse.add_argument("--max-pages", type=int, default=None)
    p_parse.add_argument(
        "--with-seat-matrix",
        action="store_true",
        help="Also parse Seat Matrix PDFs and seed college_master.csv",
    )
    p_parse.set_defaults(func=cmd_parse)

    p_sm = sub.add_parser(
        "parse-seat-matrix",
        help="Parse CAP Round I seat matrix PDFs into parquet + seed college master",
    )
    p_sm.add_argument("--log-level", default="INFO")
    p_sm.add_argument(
        "--pdf",
        nargs="*",
        help="Seat matrix PDF paths (default: Knowledge Base/Seat Matrix/)",
    )
    p_sm.add_argument("--force-reparse", action="store_true")
    p_sm.add_argument("--max-pages", type=int, default=None, help="Limit pages per PDF (smoke)")
    p_sm.set_defaults(func=cmd_parse_seat_matrix)

    p_rb = sub.add_parser(
        "parse-rulebook",
        help="Index CET Information Brochure into searchable chunks for counselling",
    )
    p_rb.add_argument("--log-level", default="INFO")
    p_rb.add_argument(
        "--pdf",
        nargs="*",
        help="Rulebook PDF paths (default: Knowledge Base/rulebook/)",
    )
    p_rb.add_argument("--force-reparse", action="store_true")
    p_rb.add_argument("--max-pages", type=int, default=None, help="Limit pages (smoke)")
    p_rb.set_defaults(func=cmd_parse_rulebook)

    p_branches = sub.add_parser("list-branches", help="List distinct branch names from cache")
    p_branches.add_argument("--log-level", default="INFO")
    p_branches.set_defaults(func=cmd_list_branches)

    p_db = sub.add_parser("build-db", help="Build/refresh DuckDB cutoff store from parquet cache")
    p_db.add_argument("--log-level", default="INFO")
    p_db.add_argument("--force-reparse", action="store_true")
    p_db.set_defaults(func=cmd_build_db)

    p_val = sub.add_parser("validate", help="Parser quality report for CAP PDFs")
    p_val.add_argument("--log-level", default="INFO")
    p_val.add_argument("--pdf", nargs="*", help="PDF paths (default: Knowledge Base/)")
    p_val.add_argument("--max-pages", type=int, default=None, help="Limit pages per PDF (smoke)")
    p_val.add_argument("--out", default=None, help="Validation JSON output path")
    p_val.set_defaults(func=cmd_validate)

    p_train = sub.add_parser(
        "train-model",
        help="Train ML band model from outcome labels (preferred) or synthetic rules",
    )
    p_train.add_argument("--log-level", default="INFO")
    p_train.add_argument("--sample-rows", type=int, default=4000)
    p_train.add_argument(
        "--labels",
        default=None,
        help="Path to allotments.csv (default data/labels/allotments.csv)",
    )
    p_train.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic rule labels even if allotments.csv exists",
    )
    p_train.set_defaults(func=cmd_train_model)

    p_pred = sub.add_parser(
        "predict",
        help="Generate prediction report from student inputs "
        "(prompt interactively if fields are omitted)",
    )
    p_pred.add_argument("--log-level", default="INFO")
    p_pred.add_argument("--interactive", "-i", action="store_true", help="Prompt for all student fields")
    p_pred.add_argument("--percentile", type=float, default=None)
    p_pred.add_argument("--category", default=None, help="CAP seat code or ALL")
    p_pred.add_argument("--gender", default=None, help="M | F | ALL")
    p_pred.add_argument("--home-university", default=None, help="University name or ALL")
    p_pred.add_argument("--seat-type", default=None, help="HU | OHU | STATE | ALL")
    p_pred.add_argument("--branch", default=None, help="Target branch name/code or ALL")
    p_pred.add_argument("--year", type=int, default=None, help="Filter year (omit for All)")
    p_pred.add_argument("--round", type=int, default=None, dest="round", help="Filter CAP round 1-4")
    p_pred.add_argument("--pdf", nargs="*", help="PDF paths (default: Knowledge Base/)")
    p_pred.add_argument("--force-reparse", action="store_true")
    p_pred.add_argument("--max-pages", type=int, default=None)
    p_pred.add_argument("--out", default=None, help="Output directory for reports")
    p_pred.add_argument("--no-llm", action="store_true", help="Skip Ollama; use template text")
    p_pred.add_argument("--no-pdf", action="store_true", help="Skip PDF export")
    p_pred.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    p_pred.add_argument("--no-excel", action="store_true", help="Skip Excel export")
    p_pred.add_argument("--no-features", action="store_true", help="Skip ML feature parquet export")
    p_pred.add_argument("--max-preferences", type=int, default=30, help="Cap preference list length")
    p_pred.add_argument("--safe-margin", type=float, default=3.0, help="Safe band: margin >= this")
    p_pred.add_argument("--target-below", type=float, default=2.0, help="Target band: down to -this")
    p_pred.add_argument("--dream-below", type=float, default=8.0, help="Dream band: down to -this")
    p_pred.add_argument("--use-ml", action="store_true", help="Re-score bands with trained ML model")
    p_pred.set_defaults(func=cmd_predict)

    p_serve = sub.add_parser("serve", help="Start FastAPI prediction API (uvicorn)")
    p_serve.add_argument("--log-level", default="INFO")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Command failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
