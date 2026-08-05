#!/usr/bin/env python3
"""
Phase 3 — Correctness scoring.

Reads Phase 1 answer_text from results/full/*.jsonl. NO new panel model calls.
Grades 6 facit prompts (P3–P8) across all 6 models using two paths:
  Programmatic (exact): P5 (math), P6 (logic), P7 (JSON)
  LLM grader (blind):   P3 (legal), P4 (legal), P8 (code bug)

Stops after generating the HTML review for P3/P4 — requires Lars's confirmation
on the legal verdicts before results are considered final.

Usage:
  python3 run_phase3.py [--source-run-id RUN_ID]
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import pathlib
import sys
from datetime import datetime, timezone

import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
del _os

import yaml
from dotenv import load_dotenv
load_dotenv()

from src.grader import (
    CORRECTNESS_PROMPTS,
    LLM_GRADER_PROMPTS,
    PROGRAMMATIC_PROMPTS,
    GradeResult,
    grade_llm,
    grade_programmatic,
)
from src.config_loader import load_panel
from src.storage import RESULTS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PHASE3_DIR = RESULTS_DIR / "phase3"

# Grader model — reuses the Phase 2 judge infrastructure.
# gemini_3_1_pro: handles Danish, reliable JSON output, already in panel.yaml.
GRADER_PANEL_KEY = "gemini_3_1_pro"
GRADER_PRICING_KEY = "gemini_3_1_pro"

# Display order — all 8 models; Claude, GPT, Opus re-enter here (answer is public)
MODEL_ORDER: list[str] = [
    "deepseek_v4", "glm_5_2", "kimi_k2_7", "gemma_4",
    "claude_sonnet_4_6", "gpt_5_5", "opus_4_8", "mistral_medium_3_5",
]

VERDICT_WEIGHT: dict[str, float] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_facit_prompts() -> dict[str, dict]:
    """Load ALL prompt fields including facit — only valid on the grading path."""
    data_dir = pathlib.Path(__file__).parent / "data"
    with open(data_dir / "prompts.yaml") as f:
        raw = yaml.safe_load(f)
    return {entry["id"]: entry for entry in raw["prompts"]}


def _load_phase1_jsonl(source_run_id: str | None = None) -> tuple[list[dict], str]:
    full_dir = RESULTS_DIR / "full"
    if source_run_id:
        path = full_dir / f"{source_run_id}.jsonl"
    else:
        candidates = sorted(full_dir.glob("*_full.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No full-run JSONL in {full_dir}")
        path = candidates[-1]
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows, path.stem


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

def _save_grade(
    run_id: str,
    grade: GradeResult,
    prompt_text: str,
    reasoning_tokens: int,
) -> None:
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "phase": 3,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_id": grade.prompt_id,
        "model_key": grade.model_key,
        "verdict": grade.verdict,
        "extracted_or_justification": grade.extracted,
        "grading_method": grade.grading_method,
        "grader_model": grade.grader_model,
        "cost_usd": grade.cost_usd,
        "parse_ok": grade.parse_ok,
        "parse_error": grade.parse_error,
        "reasoning_tokens": reasoning_tokens,
    }
    out_path = PHASE3_DIR / f"{run_id}_correctness.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# HTML report (P3 and P4 — legal prompts requiring human confirmation)
# ─────────────────────────────────────────────────────────────────────────────

_HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 24px; background: #f8f9fa; color: #212529; }
h1 { border-bottom: 2px solid #dee2e6; padding-bottom: 8px; }
h2 { color: #343a40; font-size: 1.05rem; margin: 36px 0 6px; }
.prompt-text { font-size: .85rem; background: #fff; border-left: 4px solid #adb5bd;
               padding: 10px 14px; white-space: pre-wrap; margin: 6px 0 12px;
               border-radius: 0 4px 4px 0; max-height: 180px; overflow-y: auto; }
.facit-block { font-size: .85rem; background: #e8f4f8; border-left: 4px solid #0d6efd;
               padding: 10px 14px; white-space: pre-wrap; margin: 6px 0 20px;
               border-radius: 0 4px 4px 0; }
.model-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 36px; }
.model-card { background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 14px; }
.model-card h4 { margin: 0 0 6px; font-size: .9rem; color: #343a40; }
.verdict { display: inline-block; border-radius: 4px; padding: 2px 10px;
           font-size: .8rem; font-weight: bold; margin: 2px 0 8px; }
.v-correct   { background: #d1e7dd; color: #0f5132; }
.v-partial   { background: #fff3cd; color: #664d03; }
.v-incorrect { background: #f8d7da; color: #842029; }
.just { font-size: .82rem; color: #495057; margin-bottom: 8px; font-style: italic; }
details.ans-wrap { margin-top: 8px; }
details.ans-wrap summary { cursor: pointer; color: #6c757d; font-size: .8rem; user-select: none; }
details.ans-wrap summary:hover { color: #0d6efd; }
.ans-text { background: #f1f3f5; font-size: .8rem; font-family: monospace;
            white-space: pre-wrap; padding: 8px 10px; border-radius: 4px;
            max-height: 320px; overflow-y: auto; margin-top: 4px; }
"""


def _generate_html(
    run_id: str,
    p1_run_id: str,
    results: dict[tuple[str, str], GradeResult],
    facit_prompts: dict[str, dict],
    phase1_index: dict[tuple[str, str], dict],
    model_order: list[str] | None = None,
) -> pathlib.Path:
    """Generate HTML review for P3 and P4 (legal prompts) for human confirmation."""
    display_order = model_order if model_order is not None else MODEL_ORDER
    blocks: list[str] = []
    for pid in ["P3", "P4"]:
        fp = facit_prompts[pid]
        prompt_escaped = html_lib.escape(fp.get("prompt", ""))
        facit_escaped = html_lib.escape(fp.get("facit", ""))
        p_type = html_lib.escape(fp.get("type", ""))
        p_load = html_lib.escape(fp.get("reasoning_load", ""))

        cards: list[str] = []
        for mk in display_order:
            grade = results.get((pid, mk))
            p1_row = phase1_index.get((mk, pid))
            if grade is None or p1_row is None:
                cards.append(
                    f'<div class="model-card"><h4>{html_lib.escape(mk)}</h4>'
                    f'<em style="color:#6c757d">no data</em></div>'
                )
                continue
            v_cls = f"v-{grade.verdict}"
            just_e = html_lib.escape(grade.extracted)
            ans_e = html_lib.escape(p1_row.get("answer_text", ""))
            ans_len = len(p1_row.get("answer_text", ""))
            cards.append(
                f'<div class="model-card">'
                f'<h4>{html_lib.escape(mk)}</h4>'
                f'<span class="verdict {v_cls}">{grade.verdict.upper()}</span>'
                f'<div class="just">{just_e}</div>'
                f'<details class="ans-wrap">'
                f'<summary>Full answer ({ans_len:,} chars) — click to expand</summary>'
                f'<div class="ans-text">{ans_e}</div>'
                f'</details>'
                f'</div>'
            )

        blocks.append(
            f'<h2>{html_lib.escape(pid)} — {p_type}  '
            f'<span style="color:#6c757d;font-weight:normal">load: {p_load}</span></h2>'
            f'<p style="font-size:.85rem;color:#495057">Prompt:</p>'
            f'<div class="prompt-text">{prompt_escaped}</div>'
            f'<p style="font-size:.85rem;color:#495057"><strong>Grading key (facit):</strong></p>'
            f'<div class="facit-block">{facit_escaped}</div>'
            f'<div class="model-grid">{"".join(cards)}</div>'
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_out = "\n".join([
        "<!DOCTYPE html><html lang='da'><head>",
        "<meta charset='utf-8'>",
        f"<title>Phase 3 Correctness — {run_id}</title>",
        f"<style>{_HTML_CSS}</style>",
        "</head><body>",
        "<h1>Phase 3 — Correctness Review (P3 &amp; P4)</h1>",
        f"<p><b>Run:</b> {html_lib.escape(run_id)} &nbsp; "
        f"<b>Source:</b> {html_lib.escape(p1_run_id)} &nbsp; "
        f"<b>Generated:</b> {generated_at}</p>",
        "<p><em><strong>Action required:</strong> Review each model's answer against "
        "the facit and confirm the LLM verdicts are correct before finalizing. "
        "P5/P6/P7 are programmatic and need no confirmation.</em></p>",
        "<hr>",
        *blocks,
        "</body></html>",
    ])

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PHASE3_DIR / f"{run_id}_correctness.html"
    out_path.write_text(html_out, encoding="utf-8")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — Correctness scoring")
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="Phase 1 full run ID (default: most recent results/full/*.jsonl)",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        phase1_rows, p1_run_id = _load_phase1_jsonl(args.source_run_id)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)

    facit_prompts = _load_facit_prompts()
    panel = load_panel()

    grader_cfg = panel.get(GRADER_PANEL_KEY, {})
    grader_or_id = grader_cfg.get("openrouter_model_id")
    if not grader_or_id:
        print(f"\n  ERROR: no openrouter_model_id for grader '{GRADER_PANEL_KEY}' in panel.yaml\n")
        sys.exit(1)

    # Build (model_key, prompt_id) → Phase 1 row index
    phase1_index: dict[tuple[str, str], dict] = {
        (r["model_key"], r["prompt_id"]): r for r in phase1_rows
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_phase3"

    print(f"\n{'═'*100}")
    print(f"  PHASE 3 — Correctness Scoring   run_id={run_id}")
    print(f"  Source: {p1_run_id}")
    print(f"  Grader: {GRADER_PANEL_KEY} ({grader_or_id})")
    print(f"  Prompts: {', '.join(CORRECTNESS_PROMPTS)}  ({len(CORRECTNESS_PROMPTS)} facit prompts)")
    models_in_source = {r["model_key"] for r in phase1_rows}
    effective_order = [m for m in MODEL_ORDER if m in models_in_source]
    print(f"  Models: {', '.join(effective_order)}  (correctness scores answers, not traces)")
    print(f"{'═'*100}")
    print(
        f"\n  NOTE: Correctness judges the ANSWER, which every model exposes."
        f"\n  Closed/summarized traces are no barrier — all models in the source JSONL are scored.\n"
    )

    # ── Grade ─────────────────────────────────────────────────────────────────
    # results[(prompt_id, model_key)] = GradeResult
    results: dict[tuple[str, str], GradeResult] = {}
    total_cost = 0.0

    for pid in CORRECTNESS_PROMPTS:
        fp = facit_prompts.get(pid, {})
        facit = str(fp.get("facit", ""))
        prompt_text = fp.get("prompt", "")
        p_type = fp.get("type", "?")
        method = "programmatic" if pid in PROGRAMMATIC_PROMPTS else f"llm_grader ({GRADER_PANEL_KEY})"

        print(f"\n{'─'*100}")
        print(f"  [{pid}]  {p_type}  —  {method}")
        print(f"  {'Model':<22}  {'Verdict':<12}  {'Cost':>9}  Extracted / Justification")
        print(f"  {'·'*90}")

        for mk in effective_order:
            p1_row = phase1_index.get((mk, pid))
            if p1_row is None:
                print(f"  {mk:<22}  {'MISSING':<12}  {'—':>9}  no Phase 1 record")
                continue

            answer = p1_row.get("answer_text") or ""
            reasoning_tokens: int = p1_row.get("tokens", {}).get("reasoning", 0)

            if pid in PROGRAMMATIC_PROMPTS:
                grade = grade_programmatic(pid, answer)
            else:
                grade = grade_llm(pid, answer, facit, grader_or_id, GRADER_PRICING_KEY)

            grade.model_key = mk
            results[(pid, mk)] = grade
            total_cost += grade.cost_usd
            _save_grade(run_id, grade, prompt_text, reasoning_tokens)

            cost_str = f"${grade.cost_usd:.5f}" if grade.cost_usd else "—"
            detail = grade.extracted[:80]
            warn = "  ← PARSE ERR" if not grade.parse_ok else ""
            print(f"  {mk:<22}  {grade.verdict:<12}  {cost_str:>9}  {detail}{warn}")

    print(f"\n{'─'*100}")
    print(f"  Total grader cost: ${total_cost:.5f}")

    # ── Aggregate 1: correctness rate per model ───────────────────────────────
    W = 100
    print(f"\n{'═'*W}")
    print(f"  AGGREGATE 1 — Correctness rate per model  (across {len(CORRECTNESS_PROMPTS)} facit prompts)")
    print(f"  correct=1.0 pt  partial=0.5 pt  incorrect=0 pt")
    print(f"{'═'*W}")
    print(f"  {'Model':<22}  {'Correct':>8}  {'Partial':>8}  {'Incorrect':>10}  "
          f"{'Score/6':>8}  {'Score%':>8}")
    print(f"  {'─'*75}")

    for mk in effective_order:
        counts = {"correct": 0, "partial": 0, "incorrect": 0}
        score = 0.0
        n = 0
        for pid in CORRECTNESS_PROMPTS:
            g = results.get((pid, mk))
            if g:
                counts[g.verdict] += 1
                score += VERDICT_WEIGHT[g.verdict]
                n += 1
        pct = (score / len(CORRECTNESS_PROMPTS)) * 100 if n > 0 else 0
        print(
            f"  {mk:<22}  {counts['correct']:>8}  {counts['partial']:>8}  "
            f"{counts['incorrect']:>10}  {score:>7.1f}/{len(CORRECTNESS_PROMPTS)}"
            f"  {pct:>7.0f}%"
        )
    print(f"  {'─'*75}")

    # ── Aggregate 2: correctness × economy cross-axis ─────────────────────────
    print(f"\n{'═'*W}")
    print(f"  AGGREGATE 2 — Correctness × Economy  (the primary cross-axis finding)")
    print(f"  Reasoning tokens = avg across ALL 10 Phase 1 prompts (from {p1_run_id})")
    print(f"  FIREWALL: do not merge these columns with legibility (Phase 2) scores.")
    print(f"{'═'*W}")
    print(
        f"  {'Model':<22}  {'Avg Reas Tok':>12}  {'Regime':<13}  "
        f"{'Score%':>8}  {'Verdict':<9}  Interpretation"
    )
    print(f"  {'─'*88}")

    REGIME: dict[str, str] = {
        "deepseek_v4": "raw",
        "glm_5_2": "raw",
        "kimi_k2_7": "raw",
        "gemma_4": "raw_anchor",
        "claude_sonnet_4_6": "summarized",
        "gpt_5_5": "count_only",
        "opus_4_8": "summarized",
        "mistral_medium_3_5": "raw",
    }

    for mk in effective_order:
        # Average reasoning tokens across all 10 prompts
        tok_rows = [r for r in phase1_rows if r["model_key"] == mk]
        avg_tok = (
            sum(r["tokens"]["reasoning"] for r in tok_rows) / len(tok_rows)
            if tok_rows else 0
        )
        # Correctness score
        score = sum(
            VERDICT_WEIGHT.get(results[(pid, mk)].verdict, 0)
            for pid in CORRECTNESS_PROMPTS
            if (pid, mk) in results
        )
        n_graded = sum(1 for pid in CORRECTNESS_PROMPTS if (pid, mk) in results)
        pct = (score / len(CORRECTNESS_PROMPTS)) * 100 if n_graded else 0

        # Simple "efficient and right" / "expensive and right" / etc. label
        high_tok = avg_tok > 2000
        high_score = pct >= 75
        if high_score and not high_tok:
            interp = "efficient AND right"
        elif high_score and high_tok:
            interp = "expensive AND right"
        elif not high_score and not high_tok:
            interp = "efficient BUT wrong/partial"
        else:
            interp = "expensive AND wrong/partial"

        regime = REGIME.get(mk, "?")
        print(
            f"  {mk:<22}  {avg_tok:>12,.0f}  {regime:<13}  "
            f"{pct:>7.0f}%  {score:>5.1f}/{len(CORRECTNESS_PROMPTS):<3}  {interp}"
        )
    print(f"  {'─'*88}")
    print(
        f"\n  † Claude: reasoning tokens = thinking-budget allocation (summarized, not raw CoT)"
        f"\n  ‡ GPT-5.5: reasoning tokens from API usage field (trace text hidden)\n"
    )

    # ── HTML for P3/P4 + STOP ─────────────────────────────────────────────────
    html_path = _generate_html(run_id, p1_run_id, results, facit_prompts, phase1_index,
                                model_order=effective_order)
    print(f"\n{'═'*W}")
    print(f"  HTML review written → {html_path}")
    print()
    print(f"  STOP — review P3 and P4 verdicts in the HTML before finalizing.")
    print(f"  These are the two Danish legal prompts graded by LLM. Confirm that")
    print(f"  the verdicts match your reading of the answers against the facit.")
    print()
    print(f"  Objective prompts P5/P6/P7 are programmatic and need no confirmation.")
    print(f"  P8 (code bug) is LLM-graded — also visible in the JSONL if you want to check.")
    print()
    print(f"  Records saved → results/phase3/{run_id}_correctness.jsonl")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
