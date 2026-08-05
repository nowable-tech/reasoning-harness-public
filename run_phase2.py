#!/usr/bin/env python3
"""
Phase 2 entry point — anchored rubric edition.

Patches src.judge._RUBRIC_TEMPLATE with the anchored rubric from
src/judge_rubric.py before any judge call is made, then runs the
standard Phase 2 flow from run.py.

Usage:
  python3 run_phase2.py --validate-judges [--source-run-id RUN_ID]
  python3 run_phase2.py --judge            [--source-run-id RUN_ID]

After --validate-judges this script also:
  • Writes a human-readable HTML review (results/phase2/<run_id>.html)
  • Prints per-judge per-dimension score variance
  • STOPS — does not proceed to --judge without explicit invocation
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Ensure project root is in sys.path.
# pathlib.resolve() calls os.path.realpath() which follows symlinks and can fail
# with EPERM on macOS Documents folders. os.path.abspath() is safe: Python 3.11+
# sets __file__ to an absolute path, so abspath() is a no-op (no getcwd call).
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
del _os

# ── Step 1: patch the rubric BEFORE importing run.py ────────────────────────
# run.py does `from src.judge import build_rubric_prompt`.
# build_rubric_prompt looks up _RUBRIC_TEMPLATE in src.judge's module namespace
# at CALL time, so patching before the first call is sufficient.
import src.judge as _judge_module
from src.judge_rubric import ANCHORED_RUBRIC_TEMPLATE
_judge_module._RUBRIC_TEMPLATE = ANCHORED_RUBRIC_TEMPLATE

# ── Step 2: import run — it will use the patched judge module ────────────────
import run as _run

# ── Step 3: HTML + variance helpers ─────────────────────────────────────────
from src.report import generate_validation_html, print_variance_table
from src.storage import PHASE2_DIR, RESULTS_DIR


def _latest_validate_jsonl() -> pathlib.Path | None:
    files = sorted(PHASE2_DIR.glob("*_validate.jsonl"))
    return files[-1] if files else None


def _latest_full_jsonl() -> pathlib.Path | None:
    full_dir = RESULTS_DIR / "full"
    files = sorted(full_dir.glob("*_full.jsonl"))
    return files[-1] if files else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 legibility scoring with anchored rubric"
    )
    parser.add_argument(
        "--validate-judges", action="store_true",
        help="Score Gemma traces only; generate HTML review; stop for confirmation"
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Full Phase 2 scoring across all raw-trace models"
    )
    parser.add_argument(
        "--source-run-id",
        help="Which Phase 1 run to read traces from (default: most recent full run)"
    )
    args = parser.parse_args()

    if args.validate_judges:
        # Run the standard validation gate
        _run.run_validate_judges(source_run_id=args.source_run_id)

        # Locate the JSONL just written
        validate_jsonl = _latest_validate_jsonl()
        if validate_jsonl is None:
            print("ERROR: could not find validation JSONL in results/phase2/", file=sys.stderr)
            sys.exit(1)

        # Phase 1 JSONL for trace excerpts in the HTML
        phase1_jsonl = _latest_full_jsonl()

        # Generate HTML
        html_path = generate_validation_html(validate_jsonl, phase1_jsonl=phase1_jsonl)
        print(f"\n  HTML review written → {html_path}")

        # Print variance table
        print_variance_table(validate_jsonl)

        print()
        print("  ─────────────────────────────────────────────────────────────")
        print("  STOP — review the HTML and variance above before proceeding.")
        print("  If the anchored rubric looks correct, run:")
        print()
        print("    python3 run_phase2.py --judge")
        print("  ─────────────────────────────────────────────────────────────")

    elif args.judge:
        _run.run_judge(source_run_id=args.source_run_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
