#!/usr/bin/env python3
"""
Post-hoc analysis over full-run JSONL records.

Usage:
    python3 analyse.py                          # uses most-recent results/full/*.jsonl
    python3 analyse.py --run-id 20260625T181036_full

Outputs two re-cut aggregates — no new model calls:
  A) Language matrix (raw models only as findings; claude/gpt explicitly excluded)
  B) Absolute reasoning-token volume by reasoning_load
     (raw cluster × facit prompts only, so output length doesn't dilute difficulty)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

RESULTS_DIR = Path(__file__).parent / "results" / "full"

# Prompts where carries_correctness=True and output is short.
# Using these isolates the difficulty signal from reasoning_load without it
# being washed out by the long, open-ended output of P1/P2/P9/P10.
FACIT_PROMPTS: set[str] = {"P3", "P4", "P5", "P6", "P7", "P8"}

# Display order within the raw-cluster findings
RAW_FINDING_MODELS: list[str] = ["deepseek_v4", "glm_5_2", "kimi_k2_7", "gemma_4"]
RAW_CLUSTER: set[str] = {"deepseek_v4", "glm_5_2", "kimi_k2_7"}   # comparable reas_share

# Models explicitly excluded from language findings
NON_FINDING_MODELS: dict[str, str] = {
    "claude_sonnet_4_6": "summarized — summary language, not raw reasoning trace",
    "gpt_5_5":           "count_only — no trace text exposed",
}

# da_framed_neutral prompts contain code/math; the segment classifier strips those
# segments before language detection, so results are lower-confidence there.
DA_FRAMED_NEUTRAL_PROMPTS: set[str] = {"P5", "P6", "P7", "P8"}

LOAD_ORDER: list[str] = ["low", "medium", "high", "very_high"]

ABBREV: dict[str, str] = {
    "deepseek_v4":      "deepseek",
    "glm_5_2":          "glm",
    "kimi_k2_7":        "kimi",
    "gemma_4":          "gemma",
    "claude_sonnet_4_6":"claude",
    "gpt_5_5":          "gpt",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _lang(row: dict) -> str:
    """Return the primary_trace_language label for a row, with confidence suffix."""
    lm = row.get("language_metric", {})
    lang = lm.get("primary_trace_language")
    conf = lm.get("switch_count_confidence", "")
    if lang is None:
        return "—" if conf == "no_trace" else "?"
    return lang


def print_aggregate_a(rows: list[dict], all_pids: list[str]) -> None:
    """
    Language matrix: raw models as findings, claude/gpt explicitly as non-findings.
    """
    by_key: dict[tuple[str, str], dict] = {
        (r["prompt_id"], r["model_key"]): r for r in rows
    }

    # --- Per-prompt metadata ---
    probe_of: dict[str, str] = {r["prompt_id"]: r["language_probe"] for r in rows}

    abbrevs = [ABBREV.get(k, k[:9]) for k in RAW_FINDING_MODELS]
    col_w = 9
    hdr = "  ".join(f"{a:<{col_w}}" for a in abbrevs)

    W = 94
    print(f"\n{'═'*W}")
    print(f"  RECUT AGGREGATE A — Trace language by prompt")
    print(f"{'═'*W}")

    # --- Findings block ---
    print(f"\n  RAW-TRACE FINDINGS  (deepseek / glm / kimi / gemma)")
    print(f"\n  {'Prompt':<8}  {'Probe':<22}  {hdr}")
    print(f"  {'─'*80}")

    for pid in all_pids:
        probe = probe_of.get(pid, "?")
        is_neutral = pid in DA_FRAMED_NEUTRAL_PROMPTS
        probe_label = (probe + "†") if is_neutral else probe
        cells = []
        for model in RAW_FINDING_MODELS:
            r = by_key.get((pid, model))
            cells.append(f"{_lang(r) if r else '—':<{col_w}}")
        print(f"  {pid:<8}  {probe_label:<22}  {'  '.join(cells)}")

    print(f"  {'─'*80}")
    print()
    print(
        "  † da_framed_neutral (P5–P8): prompts contain code/math. The segment classifier"
    )
    print(
        "    strips these before language detection — results are lower-confidence for"
    )
    print(
        "    these prompts. Switch counts and primary-language labels may undercount."
    )

    # --- Non-findings block ---
    print(f"\n  NON-FINDINGS — excluded from language conclusions")
    print(f"\n  {'Model':<22}  {'Regime':<14}  Reason")
    print(f"  {'─'*78}")
    for model, reason in NON_FINDING_MODELS.items():
        regime = next(
            (r["regime"] for r in rows if r["model_key"] == model), "?"
        )
        print(f"  {model:<22}  {regime:<14}  {reason}")
    print(f"  {'─'*78}")

    # --- Interpretation ---
    print()
    print(f"  {'─'*W}")
    print(f"  INTERPRETATION")
    print(f"  {'─'*W}")
    print()
    print(
        "  Baseline (expected): Models reason in their strongest-represented language."
        "\n  For Chinese lab models (deepseek, glm, kimi) that means English or Chinese;"
        "\n  for Gemma (Google) it means English. This baseline holds throughout the run."
    )
    print()
    print(
        "  Bounded leakage: Simple, conversational Danish prompts (P1/P2 — da_simple)"
        "\n  pulled Danish into the raw trace on all three Chinese models. Harder and"
        "\n  more structured prompts — legal, math, logic, code (P3–P8) — and open"
        "\n  long-form prompts (P9/P10) stayed on the English/Chinese substrate."
        "\n  The leakage is small and bounded: it appears on easy linguistic prompts"
        "\n  only, and disappears when reasoning load or structural demand increases."
    )
    print()
    print(
        "  Implication: Because the reasoning phase runs in English/Chinese, it carries"
        "\n  little Danish token tax. The value of an extended Danish tokenizer is"
        "\n  concentrated on input and output — not on the thinking phase."
    )
    print(f"\n{'═'*W}")


def print_aggregate_b(rows: list[dict], all_pids: list[str]) -> None:
    """
    Absolute reasoning-token volume by reasoning_load.
    Raw cluster only; facit prompts only (P3–P8) so short output keeps the
    difficulty signal clean.
    """
    facit_raw = [
        r for r in rows
        if r["model_key"] in RAW_CLUSTER
        and r["prompt_id"] in FACIT_PROMPTS
    ]

    by_pid_model: dict[tuple[str, str], dict] = {
        (r["prompt_id"], r["model_key"]): r for r in facit_raw
    }

    probe_of: dict[str, str] = {r["prompt_id"]: r["language_probe"] for r in rows}
    load_of:  dict[str, str] = {r["prompt_id"]: r["reasoning_load"]  for r in rows}

    facit_pids = sorted(
        FACIT_PROMPTS,
        key=lambda p: (LOAD_ORDER.index(load_of.get(p, "low")), int(p[1:]))
    )

    raw_abbrevs = [ABBREV.get(k, k) for k in sorted(RAW_CLUSTER)]  # alphabetical
    raw_models_sorted = sorted(RAW_CLUSTER)

    col_w = 7
    hdr = "  ".join(f"{a:>{col_w}}" for a in raw_abbrevs)

    W = 80
    print(f"\n{'═'*W}")
    print(f"  RECUT AGGREGATE B — Reasoning token VOLUME by difficulty")
    print(f"  Raw cluster: deepseek / glm / kimi")
    print(f"  Facit prompts only (P3–P8, carries_correctness=True)")
    print(f"  Short output keeps difficulty signal from being diluted by answer length.")
    print(f"{'═'*W}")

    # Per-prompt breakdown
    print(f"\n  {'Prompt':<7}  {'Load':<10}  {'Probe':<20}  {hdr}  {'Avg':>{col_w}}")
    print(f"  {'─'*72}")

    load_buckets: dict[str, list[int]] = {k: [] for k in LOAD_ORDER}

    for pid in facit_pids:
        load = load_of.get(pid, "?")
        probe = probe_of.get(pid, "?")
        vals: list[int] = []
        cells: list[str] = []
        for model in raw_models_sorted:
            r = by_pid_model.get((pid, model))
            if r:
                tok = r["tokens"]["reasoning"]
                vals.append(tok)
                cells.append(f"{tok:>{col_w}}")
                load_buckets.setdefault(load, []).append(tok)
            else:
                cells.append(f"{'—':>{col_w}}")
        avg_str = f"{round(mean(vals)):>{col_w}}" if vals else f"{'—':>{col_w}}"
        print(f"  {pid:<7}  {load:<10}  {probe:<20}  {'  '.join(cells)}  {avg_str}")

    print(f"  {'─'*72}")

    # By-load summary
    print(f"\n  By reasoning_load  (avg / min / max across facit prompts × 3 models):")
    print(f"\n  {'Load':<12}  {'n':>4}  {'Avg tokens':>11}  {'Min':>7}  {'Max':>7}")
    print(f"  {'─'*50}")
    for load in LOAD_ORDER:
        toks = load_buckets.get(load, [])
        if not toks:
            continue
        print(
            f"  {load:<12}  {len(toks):>4}  {round(mean(toks)):>11,}"
            f"  {min(toks):>7,}  {max(toks):>7,}"
        )
    print(f"  {'─'*50}")

    # Highlight variance within 'high'
    high_toks = load_buckets.get("high", [])
    if high_toks:
        print()
        print(
            f"  Note: 'high' spans {min(high_toks):,}–{max(high_toks):,} tokens"
            f" ({max(high_toks)//max(min(high_toks),1)}× range)."
        )
        print(
            "  P3 (legal deadline, da_forcing) drives deepseek to heavy reasoning;"
        )
        print(
            "  P5 (arithmetic, da_framed_neutral) elicits minimal reasoning despite"
        )
        print(
            "  the same 'high' label — difficulty type matters, not just load level."
        )

    print(f"\n{'═'*W}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc analysis of full-run JSONL")
    parser.add_argument(
        "--run-id",
        metavar="RUN_ID",
        default=None,
        help="Full run ID (e.g. 20260625T181036_full); defaults to most-recent",
    )
    args = parser.parse_args()

    if args.run_id:
        jsonl_path = RESULTS_DIR / f"{args.run_id}.jsonl"
    else:
        candidates = sorted(RESULTS_DIR.glob("*_full.jsonl"))
        if not candidates:
            print(f"No full-run JSONL found in {RESULTS_DIR}")
            return 1
        jsonl_path = candidates[-1]

    if not jsonl_path.exists():
        print(f"JSONL not found: {jsonl_path}")
        return 1

    rows = load_jsonl(jsonl_path)
    print(f"\nSource: {jsonl_path.name}  ({len(rows)} records)")

    all_pids = sorted({r["prompt_id"] for r in rows}, key=lambda p: int(p[1:]))

    print_aggregate_a(rows, all_pids)
    print_aggregate_b(rows, all_pids)

    return 0


if __name__ == "__main__":
    sys.exit(main())
