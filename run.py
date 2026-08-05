#!/usr/bin/env python3
"""
Reasoning Benchmark — orchestration entry point.

Usage:
    python3 run.py --smoke                    # run smoke test across all available models
    python3 run.py --smoke --model MODEL_KEY  # smoke test one model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.accounting import build_account
from src.adapters import PROVIDER_MAP, CredentialMissingError
from src.adapters.base import AdapterError, ModelResponse
from src.config_loader import load_experiment, load_multilang_prompts, load_panel, load_prompts
from src.cost import compute_cost
from src.judge import (
    DIMENSIONS,
    JudgeResponse,
    build_rubric_prompt,
    call_judge_openrouter,
    compute_agreement,
)
from src.heavy_grader import grade as grade_heavy
from src.heavy_tasks import TASK_KEYS as HEAVY_TASK_KEYS
from src.heavy_tasks import load_heavy_tasks
from src.language_metric import measure_trace_language
from src.metrics import correct_per_dollar
from src.model_resolver import assert_no_silent_direct_route, print_resolution_table, resolve_models
from src.storage import (
    HEAVY_DIR,
    PHASE2_DIR,
    RESULTS_DIR,
    TOOLS_DIR,
    TOOLS3_DIR,
    VARIANCE_DIR,
    save_heavy_result,
    save_heavy_trace,
    save_langcost_result,
    save_langcost_trace,
    save_phase2_result,
    save_result,
    save_tools_result,
    save_tools_trace,
    save_trace,
    save_variance_result,
)
from src.tool_loop import ToolsNotSupportedError
from src.tools import available_tool_defs, search_available

# ---------------------------------------------------------------------------
# Smoke test constants
# ---------------------------------------------------------------------------

SMOKE_PROMPT = (
    "What is 15 + 27? Work through this step by step, then give your final answer."
)

SMOKE_ROLES = {"scored", "anchor"}

# Models where reasoning_tokens MUST be > 0 — zero is a regression.
# gpt_5_5: count_only (reasoning count reported, no text)
# deepseek_v4 / glm_5_2 / kimi_k2_7 / mistral_medium_3_5: raw (text present)
# claude_sonnet_4_6 / opus_4_8: summarized (thinking block present)
# gemma_4: raw (tokens estimated from text)
MUST_HAVE_REASONING_TOKENS: set[str] = {
    "deepseek_v4",
    "glm_5_2",
    "kimi_k2_7",
    "gpt_5_5",
    "claude_sonnet_4_6",
    "gemma_4",
    # opus_4_8 excluded: when routed via OpenRouter (no ANTHROPIC_API_KEY),
    # the thinking block is not consistently forwarded. trace_status=absent on
    # low-complexity prompts is accurate data, not a configuration error.
    "mistral_medium_3_5",
}

# ---------------------------------------------------------------------------
# Trace exposure verification
# ---------------------------------------------------------------------------

def _verify_trace(
    response: ModelResponse, expected_exposure: Optional[str]
) -> tuple[bool, str]:
    if expected_exposure is None:
        return True, "n/a (judge stub)"

    status = response.trace_status
    has_trace = response.raw_reasoning_trace is not None
    rtoks = response.reasoning_tokens

    if expected_exposure == "raw":
        ok = has_trace and status == "raw"
        detail = f"trace={'present' if has_trace else 'ABSENT'}, status={status!r}"
    elif expected_exposure == "summarized":
        ok = has_trace and status == "summarized"
        detail = f"trace={'present' if has_trace else 'ABSENT'}, status={status!r}"
    elif expected_exposure == "count_only":
        ok = not has_trace and status == "count_only" and rtoks > 0
        detail = (
            f"trace={'PRESENT' if has_trace else 'absent'}, "
            f"status={status!r}, reasoning_tokens={rtoks}"
        )
    elif expected_exposure == "absent":
        ok = status == "absent"
        detail = f"status={status!r}"
    else:
        ok = False
        detail = f"unknown expected_exposure={expected_exposure!r}"

    return ok, detail


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _fmt_row(
    key: str,
    version: str,
    inp: int,
    reas: int,
    out: int,
    status: str,
    cost: float,
    latency: float,
    verify: str,
    reas_assert: str,
) -> str:
    return (
        f"{key:<22} {version:<28} {inp:>7} {reas:>9} {out:>7}  "
        f"{status:<12} ${cost:>8.5f}  {latency:>7.2f}s  {verify:<20}  {reas_assert}"
    )


def run_smoke(model_filter: Optional[str] = None) -> int:
    """
    Run the smoke test. Returns exit code (0 = all checks passed, 1 = failures).
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_smoke"

    target_keys = [
        k
        for k, cfg in panel.items()
        if cfg.get("role") in SMOKE_ROLES
        and (model_filter is None or k == model_filter)
    ]

    # ------------------------------------------------------------------
    # Step 1: Print model resolution table + fail loudly on bad IDs
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print(f"  Reasoning Benchmark — Smoke Test   run_id={run_id}")
    print(f"  Prompt: {SMOKE_PROMPT!r}")
    print(f"  reasoning_effort={reasoning_effort!r}  (experiment condition, locked across all models)")
    print(f"{'='*110}")

    resolved = resolve_models(panel, target_keys)
    hard_errors = print_resolution_table(resolved)
    if hard_errors > 0:
        print(
            f"\n  !! {hard_errors} model(s) could not be resolved to the intended ID. "
            "Fix panel.yaml before running.\n"
        )
        return 1

    # ------------------------------------------------------------------
    # Step 2: Run each model and collect results
    # ------------------------------------------------------------------
    print(
        f"\n{'Model':<22} {'Version':<28} {'Input':>7} {'Reasoning':>9} {'Output':>7}  "
        f"{'TraceStatus':<12} {'Cost(USD)':>10}  {'Latency':>8}  {'Exposure':20}  {'TokenAssert (src)'}"
    )
    print("-" * 140)

    has_failure = False

    for key in target_keys:
        cfg = panel[key]
        provider = cfg["provider"]
        expected_exposure: Optional[str] = cfg.get("trace_exposure")
        thinking_budget: int = cfg.get("thinking_budget", 4096)

        cls = PROVIDER_MAP.get(provider)
        if cls is None:
            print(f"{key:<22} SKIPPED (unknown provider: {provider})")
            continue

        try:
            adapter = cls(key, cfg)
        except CredentialMissingError as e:
            print(f"{key:<22} SKIPPED — {e}")
            continue

        try:
            response = adapter.call(
                SMOKE_PROMPT,
                thinking_budget=thinking_budget,
                reasoning_effort=reasoning_effort,
            )
        except AdapterError as e:
            print(f"{key:<22} ERROR    — {e}")
            has_failure = True
            continue
        except Exception as e:
            print(f"{key:<22} ERROR    — unexpected: {e}")
            has_failure = True
            continue

        account = build_account(response)
        cost_usd, snapshot_date = compute_cost(key, account)

        save_result(
            run_id=run_id,
            model_key=key,
            prompt=SMOKE_PROMPT,
            response=response,
            account=account,
            cost_usd=cost_usd,
            pricing_snapshot_date=snapshot_date,
            thinking_budget=thinking_budget,
            reasoning_effort=reasoning_effort,
        )

        # Exposure check
        exp_ok, exp_detail = _verify_trace(response, expected_exposure)
        exposure_label = "PASS" if exp_ok else f"MISMATCH ({exp_detail})"
        if not exp_ok:
            has_failure = True

        # Reasoning-token assertion — hard failure if zero where we expect non-zero.
        # Include reasoning_source so "measured" and "estimated" values are never
        # silently mixed in the same comparison column.
        reas_label = ""
        if key in MUST_HAVE_REASONING_TOKENS:
            if account.reasoning_tokens == 0:
                reas_label = "FAIL: reasoning_tokens=0"
                has_failure = True
            else:
                src_tag = "api" if response.reasoning_source == "api" else "est"
                reas_label = f"OK ({account.reasoning_tokens}, {src_tag})"

        print(
            _fmt_row(
                key,
                response.model_version[:27],
                account.input_tokens,
                account.reasoning_tokens,
                account.output_tokens,
                response.trace_status,
                cost_usd,
                response.latency_s,
                exposure_label,
                reas_label,
            )
        )

    print("-" * 140)
    status_line = "ALL CHECKS PASSED" if not has_failure else "FAILURES DETECTED — see FAIL/MISMATCH rows"
    print(f"\n  {status_line}")
    print(f"  Results written to results/{run_id}.jsonl\n")
    return 1 if has_failure else 0


# ---------------------------------------------------------------------------
# Pilot (Phase 1)
# ---------------------------------------------------------------------------

# Models that must have reasoning_tokens > 0 for the pilot to be valid.
# gpt_5_5 and claude_sonnet_4_6 are excluded: their reasoning is always on
# (gpt: count-only; claude: budget-forced). Only the raw-trace models need
# this assertion to catch a dropped include_reasoning flag.
PILOT_MUST_REASON: set[str] = {"deepseek_v4", "glm_5_2", "kimi_k2_7", "gemma_4"}

# Pilot prompt IDs (default); override with --prompts P3,P5
PILOT_DEFAULT_PROMPTS = ["P3", "P5"]


# ---------------------------------------------------------------------------
# Full run constants
# ---------------------------------------------------------------------------

# Trace-exposure regime for each model.
# reasoning_share is comparable ONLY within regime="raw".
# "raw_anchor" (gemma) is raw text but separate from the frontier raw cluster.
# "summarized" and "count_only" are NOT comparable to raw — annotate accordingly.
REGIME_MAP: dict[str, str] = {
    "deepseek_v4": "raw",
    "glm_5_2": "raw",
    "kimi_k2_7": "raw",
    "gemma_4": "raw_anchor",
    "claude_sonnet_4_6": "summarized",
    "gpt_5_5": "count_only",
    "opus_4_8": "summarized",
    "mistral_medium_3_5": "raw",
}

# Canonical display order — groups regimes together
FULL_MODEL_ORDER: list[str] = [
    "deepseek_v4",
    "glm_5_2",
    "kimi_k2_7",
    "gemma_4",
    "claude_sonnet_4_6",
    "gpt_5_5",
    "opus_4_8",
    "mistral_medium_3_5",
]

# Models that expose trace text — language metric is applicable to these
# opus_4_8: summarized (measured on the summary, noted in output)
# mistral_medium_3_5: raw
TRACE_TEXT_MODELS: set[str] = {
    "deepseek_v4",
    "glm_5_2",
    "kimi_k2_7",
    "gemma_4",
    "claude_sonnet_4_6",
    "opus_4_8",
    "mistral_medium_3_5",
}

# ---------------------------------------------------------------------------
# Phase 2 constants
# ---------------------------------------------------------------------------

# Models scored for legibility (raw trace exists)
LEGIBILITY_SCORED: list[str] = [
    "deepseek_v4", "glm_5_2", "kimi_k2_7", "gemma_4", "mistral_medium_3_5"
]

# Models explicitly excluded from Phase 2, with the reason
LEGIBILITY_EXCLUDED: dict[str, str] = {
    "claude_sonnet_4_6": (
        "summarized — scoring would measure Anthropic's summarizer, "
        "not the model's raw reasoning process"
    ),
    "opus_4_8": (
        "summarized — same constraint as claude_sonnet_4_6; "
        "legibility scores would reflect Anthropic's summarizer, not Opus's raw CoT"
    ),
    "gpt_5_5": (
        "count_only — no trace text exists to score"
    ),
}

# Judge models (from panel.yaml)
JUDGE_KEYS: list[str] = ["minimax", "gemini_3_1_pro"]

# Maps judge panel key → pricing.yaml key (for cost calculation)
JUDGE_PRICING_KEY: dict[str, str] = {
    "minimax": "minimax",
    "gemini_3_1_pro": "gemini_3_1_pro",
}

# High-disagreement threshold (per-dimension absolute diff)
HIGH_DISAGREEMENT_THRESHOLD: int = 2


# ---------------------------------------------------------------------------
# Language-cost experiment constants
# ---------------------------------------------------------------------------

# Only the five open models with exposed raw trace text.
# Opus, Sonnet, GPT are excluded — hidden/summarized trace cannot be measured
# for thinking language.
LANGCOST_MODELS: list[str] = [
    "deepseek_v4",
    "glm_5_2",
    "kimi_k2_7",
    "gemma_4",
    "mistral_medium_3_5",
]

LANGCOST_LANGS: list[str] = ["da", "en", "zh"]

# Generous and equal thinking budget for all models.
# Prevents a budget-ceiling artefact where Danish (more tokens) is cut off
# while English is not — which would produce a spurious "da harder" result.
LANGCOST_THINKING_BUDGET: int = 16384

# Pilot runs M1 only; full run covers all six tasks.
LANGCOST_PILOT_TASK: str = "M1"

# Steering prefixes for --steer mode (Step 4 scaffold — do not run together
# with the unsteered baseline; keep run_ids separate).
STEER_PREFIX: dict[str, str] = {
    "da": "Tænk og ræsonnér på dansk.\n\n",
    "en": "Think and reason in English.\n\n",
    "zh": "请用中文思考和推理。\n\n",
}


def run_pilot(prompt_ids: list[str]) -> int:
    """
    Phase 1 pilot: run selected prompts across all scored+anchor models.
    Economy axis only — no judges, no correctness scoring.
    Returns exit code (0 = pass, 1 = failures).
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")

    # load_prompts() strips facit — it must never appear on the request path.
    prompts = load_prompts()

    # Validate requested prompt IDs.
    missing = [pid for pid in prompt_ids if pid not in prompts]
    if missing:
        print(f"\n  ERROR: unknown prompt ID(s): {', '.join(missing)}")
        print(f"  Available: {', '.join(sorted(prompts))}\n")
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_pilot"
    pilot_dir = RESULTS_DIR / "pilot"
    traces_dir = pilot_dir / f"{run_id}_traces"

    model_keys = [
        k for k, cfg in panel.items() if cfg.get("role") in SMOKE_ROLES
    ]

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    print(f"\n{'='*110}")
    print(f"  Reasoning Benchmark — Phase 1 Pilot   run_id={run_id}")
    prompt_labels = ", ".join(
        f"{pid} ({prompts[pid].get('type','?')}/{prompts[pid].get('language_probe','?')})"
        for pid in prompt_ids
    )
    print(f"  Prompts: {prompt_labels}")
    print(f"  reasoning_effort={reasoning_effort!r}  |  economy axis only — no judges, no scoring")
    print(f"{'='*110}")

    # Model resolution
    resolved = resolve_models(panel, model_keys)
    hard_errors = print_resolution_table(resolved)
    if hard_errors > 0:
        print(f"\n  !! {hard_errors} model(s) not resolved. Fix panel.yaml.\n")
        return 1

    # ----------------------------------------------------------------
    # Per-prompt runs
    # ----------------------------------------------------------------
    has_failure = False
    # costs[model_key] = list of per-call cost floats
    costs_by_model: dict[str, list[float]] = {k: [] for k in model_keys}

    col_hdr = (
        f"\n{'Model':<22} {'Input':>7} {'Reasoning':>9} {'Output':>7}  "
        f"{'Reas%':>6}  {'Src':<4}  {'Cost(USD)':>10}  {'Latency':>8}  "
        f"{'TraceStatus':<13} {'TokenAssert'}"
    )
    col_sep = "-" * 120

    for pid in prompt_ids:
        p = prompts[pid]

        # Runtime guard: facit must not be present on the request path.
        assert "facit" not in p, (
            f"CRITICAL SECURITY VIOLATION: facit present in request-path object for {pid}"
        )
        prompt_text: str = p["prompt"]
        p_type = p.get("type", "")
        p_probe = p.get("language_probe", "")

        print(f"\n{'─'*110}")
        print(f"  [{pid}]  {p_type} / {p_probe}")
        print(f"  Prompt (first 120 chars): {prompt_text[:120].strip()!r}")
        print(f"{'─'*110}")
        print(col_hdr)
        print(col_sep)

        for key in model_keys:
            cfg = panel[key]
            provider = cfg["provider"]
            thinking_budget: int = cfg.get("thinking_budget", 4096)

            cls = PROVIDER_MAP.get(provider)
            if cls is None:
                print(f"{key:<22} SKIPPED (unknown provider: {provider})")
                continue

            try:
                adapter = cls(key, cfg)
            except CredentialMissingError as e:
                print(f"{key:<22} SKIPPED — {e}")
                continue

            try:
                response = adapter.call(
                    prompt_text,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                )
            except AdapterError as e:
                print(f"{key:<22} ERROR — {e}")
                has_failure = True
                continue
            except Exception as e:
                print(f"{key:<22} ERROR — unexpected: {e}")
                has_failure = True
                continue

            account = build_account(response)
            cost_usd, snapshot_date = compute_cost(key, account)
            costs_by_model[key].append(cost_usd)

            # Save structured record.
            save_result(
                run_id=run_id,
                model_key=key,
                prompt=prompt_text,
                response=response,
                account=account,
                cost_usd=cost_usd,
                pricing_snapshot_date=snapshot_date,
                thinking_budget=thinking_budget,
                reasoning_effort=reasoning_effort,
                results_dir=pilot_dir,
                extra={"prompt_id": pid, "prompt_type": p_type, "language_probe": p_probe},
            )

            # Save human-readable trace.
            save_trace(
                traces_dir=traces_dir,
                run_id=run_id,
                model_key=key,
                prompt_id=pid,
                prompt_meta=p,
                prompt_text=prompt_text,
                answer_text=response.answer_text,
                reasoning_trace=response.raw_reasoning_trace,
                trace_status=response.trace_status,
                reasoning_tokens=account.reasoning_tokens,
                reasoning_source=response.reasoning_source,
            )

            # Token assertion for raw-trace models.
            if key in PILOT_MUST_REASON:
                if account.reasoning_tokens == 0:
                    tok_assert = "FAIL: reasoning_tokens=0"
                    has_failure = True
                else:
                    src_tag = "api" if response.reasoning_source == "api" else "est"
                    tok_assert = f"OK ({account.reasoning_tokens}, {src_tag})"
            else:
                src_tag = "api" if response.reasoning_source == "api" else "est"
                tok_assert = f"{account.reasoning_tokens} ({src_tag})"

            reas_share_pct = account.reasoning_share * 100
            src_tag = "api" if response.reasoning_source == "api" else "est"

            print(
                f"{key:<22} {account.input_tokens:>7} {account.reasoning_tokens:>9} "
                f"{account.output_tokens:>7}  {reas_share_pct:>5.1f}%  {src_tag:<4}  "
                f"${cost_usd:>8.5f}  {response.latency_s:>7.2f}s  "
                f"{response.trace_status:<13} {tok_assert}"
            )

        print(col_sep)

    # ----------------------------------------------------------------
    # Per-model summary
    # ----------------------------------------------------------------
    n_prompts = len(prompt_ids)
    print(f"\n{'─'*110}")
    print(f"  Per-model summary  ({n_prompts} prompt(s) × {len(model_keys)} models)")
    print(f"{'─'*110}")
    print(f"  {'Model':<22}  {'Calls':>5}  {'Total $':>10}  {'Avg $/prompt':>14}")
    print(f"  {'-'*60}")
    total_pilot_cost = 0.0
    avg_by_model: dict[str, float] = {}
    for key in model_keys:
        call_costs = costs_by_model[key]
        if not call_costs:
            print(f"  {key:<22}  {'—':>5}  {'—':>10}  {'—':>14}")
            avg_by_model[key] = 0.0
            continue
        total = sum(call_costs)
        avg = total / len(call_costs)
        avg_by_model[key] = avg
        total_pilot_cost += total
        print(f"  {key:<22}  {len(call_costs):>5}  ${total:>9.5f}  ${avg:>13.5f}")
    print(f"  {'-'*60}")
    print(f"  {'PILOT TOTAL':<22}  {'':>5}  ${total_pilot_cost:>9.5f}")

    # ----------------------------------------------------------------
    # Cost projection — full run (10 × 6)
    # ----------------------------------------------------------------
    full_prompts = 10
    full_models = len([k for k in model_keys if avg_by_model.get(k, 0) > 0])
    total_projected = sum(avg_by_model[k] * full_prompts for k in model_keys)

    print(f"\n{'═'*70}")
    print(f"  COST PROJECTION — Full run ({full_prompts} prompts × {full_models} models)")
    print(f"  Based on pilot averages ({n_prompts} prompt(s) per model)")
    print(f"{'═'*70}")
    print(f"  {'Model':<22}  {'Avg $/prompt':>14}  {f'{full_prompts}×6 projected':>16}")
    print(f"  {'-'*58}")
    for key in model_keys:
        avg = avg_by_model.get(key, 0.0)
        proj = avg * full_prompts
        if avg > 0:
            print(f"  {key:<22}  ${avg:>13.5f}  ${proj:>15.4f}")
        else:
            print(f"  {key:<22}  {'—':>14}  {'—':>16}")
    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<22}  {'':>14}  ${total_projected:>15.4f}")
    print(f"{'═'*70}")

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    status_line = "ALL ASSERTIONS PASSED" if not has_failure else "FAILURES — see FAIL rows above"
    print(f"\n  {status_line}")
    print(f"  Structured records  → results/pilot/{run_id}.jsonl")
    print(f"  Raw traces          → results/pilot/{run_id}_traces/<model>_<prompt>.txt\n")

    return 1 if has_failure else 0


# ---------------------------------------------------------------------------
# Full run (Phase 1 — all 10 prompts × all scored+anchor models)
# ---------------------------------------------------------------------------

_FULL_COL_HDR = (
    f"  {'Model':<22} {'Inp':>6} {'Reas':>7} {'Out':>6}  {'Reas%':>7}  "
    f"{'Regime':<12} {'Src':<4}  {'Cost($)':>10}  {'Lat':>7}  "
    f"{'TrStatus':<12} {'Lang':<7} {'SW':>3}  Conf"
)
_FULL_COL_SEP = "  " + "─" * 132


def run_full(model_filter: Optional[list[str]] = None, allow_direct: bool = False) -> int:
    """
    Phase 1 full economy run: all 10 prompts × all scored+anchor models.
    Adds segment-aware language metrics to every exposed trace.
    Regime-separates reasoning_share: raw cluster (deepseek/glm/kimi) only.

    model_filter: if given, restrict to these model keys (e.g. ["opus_4_8", "mistral_medium_3_5"]).
    Returns exit code (0 = all assertions passed, 1 = any failure/error).
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")
    prompts = load_prompts()

    all_prompt_ids = sorted(prompts.keys(), key=lambda p: int(p[1:]))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_full"
    full_dir = RESULTS_DIR / "full"
    traces_dir = full_dir / f"{run_id}_traces"

    model_keys_ordered = [
        k for k in FULL_MODEL_ORDER
        if k in panel and panel[k].get("role") in SMOKE_ROLES
        and (model_filter is None or k in model_filter)
    ]

    n_calls = len(all_prompt_ids) * len(model_keys_ordered)
    print(f"\n{'═'*120}")
    print(f"  Reasoning Benchmark — Phase 1 Full Economy Run   run_id={run_id}")
    print(
        f"  Prompts: {', '.join(all_prompt_ids)}"
        f"  ({len(all_prompt_ids)} × {len(model_keys_ordered)} models = {n_calls} calls)"
    )
    print(f"  reasoning_effort={reasoning_effort!r}  |  economy axis only — no judges, no scoring")
    print(f"{'═'*120}")

    resolved = resolve_models(panel, model_keys_ordered)
    hard_errors = print_resolution_table(resolved)
    assert_no_silent_direct_route(panel, model_keys_ordered, allow_direct)
    if hard_errors > 0:
        print(f"\n  !! {hard_errors} model(s) not resolved. Fix panel.yaml.\n")
        return 1

    has_failure = False
    # Collect per-call data for aggregate tables at the end
    agg: list[dict] = []

    # --- Per-prompt loop ---
    for pid in all_prompt_ids:
        p = prompts[pid]
        assert "facit" not in p, (
            f"CRITICAL SECURITY VIOLATION: facit in request-path object for {pid}"
        )
        prompt_text: str = p["prompt"]
        p_type = p.get("type", "?")
        p_probe = p.get("language_probe", "?")
        p_load = p.get("reasoning_load", "?")
        p_correct = p.get("carries_correctness", False)

        print(f"\n{'═'*120}")
        print(
            f"  [{pid}]  {p_type} / {p_probe}"
            f"  (load: {p_load}, carries_correctness: {p_correct})"
        )
        print(f"  {prompt_text[:110].strip()!r}")
        print(f"{'═'*120}")
        print(_FULL_COL_HDR)
        print(_FULL_COL_SEP)

        for key in model_keys_ordered:
            cfg = panel[key]
            provider = cfg["provider"]
            thinking_budget: int = cfg.get("thinking_budget", 4096)
            regime = REGIME_MAP.get(key, "unknown")

            cls = PROVIDER_MAP.get(provider)
            if cls is None:
                print(f"  {key:<22} SKIPPED (unknown provider: {provider})")
                continue

            try:
                adapter = cls(key, cfg)
            except CredentialMissingError as e:
                print(f"  {key:<22} SKIPPED — {e}")
                continue

            try:
                response = adapter.call(
                    prompt_text,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                )
            except AdapterError as e:
                print(f"  {key:<22} ERROR — {e}")
                has_failure = True
                continue
            except Exception as e:
                print(f"  {key:<22} ERROR — unexpected: {e}")
                has_failure = True
                continue

            account = build_account(response)
            cost_usd, snapshot_date = compute_cost(key, account)

            # Language metric — only for models that expose trace text
            if key in TRACE_TEXT_MODELS and response.raw_reasoning_trace:
                lm = measure_trace_language(response.raw_reasoning_trace)
            else:
                lm = measure_trace_language(None)

            # Persist JSONL record
            save_result(
                run_id=run_id,
                model_key=key,
                prompt=prompt_text,
                response=response,
                account=account,
                cost_usd=cost_usd,
                pricing_snapshot_date=snapshot_date,
                thinking_budget=thinking_budget,
                reasoning_effort=reasoning_effort,
                results_dir=full_dir,
                extra={
                    "prompt_id": pid,
                    "prompt_type": p_type,
                    "language_probe": p_probe,
                    "reasoning_load": p_load,
                    "regime": regime,
                    "language_metric": lm,
                },
            )

            # Persist human-readable trace
            save_trace(
                traces_dir=traces_dir,
                run_id=run_id,
                model_key=key,
                prompt_id=pid,
                prompt_meta=p,
                prompt_text=prompt_text,
                answer_text=response.answer_text,
                reasoning_trace=response.raw_reasoning_trace,
                trace_status=response.trace_status,
                reasoning_tokens=account.reasoning_tokens,
                reasoning_source=response.reasoning_source,
            )

            # Collect for aggregates
            agg.append({
                "pid": pid,
                "reasoning_load": p_load,
                "language_probe": p_probe,
                "model_key": key,
                "regime": regime,
                "reasoning_share": account.reasoning_share,
                "cost_usd": cost_usd,
                "primary_lang": lm.get("primary_trace_language"),
                "sw_count": lm.get("language_switch_count", 0),
                "sw_conf": lm.get("switch_count_confidence", "?"),
            })

            # --- Format output row ---
            reas_share_pct = account.reasoning_share * 100
            share_str = f"{reas_share_pct:.1f}%"
            if regime == "summarized":
                reas_pct_str = share_str + "†"   # not comparable to raw
            elif regime == "count_only":
                reas_pct_str = share_str + "‡"   # text hidden
            else:
                reas_pct_str = share_str

            src_tag = "api" if response.reasoning_source == "api" else "est"

            if key in TRACE_TEXT_MODELS:
                lang_str = lm.get("primary_trace_language") or "?"
                sw_str = str(lm.get("language_switch_count", 0))
                conf_str = (lm.get("switch_count_confidence") or "?")
            else:
                lang_str = sw_str = conf_str = "—"

            # Reasoning-token assertion: warn if zero where non-zero is expected
            tok_warn = ""
            if key in MUST_HAVE_REASONING_TOKENS and account.reasoning_tokens == 0:
                tok_warn = " ← WARN: reasoning_tokens=0"
                has_failure = True

            print(
                f"  {key:<22} {account.input_tokens:>6} {account.reasoning_tokens:>7}"
                f" {account.output_tokens:>6}  {reas_pct_str:>7}  "
                f"{regime:<12} {src_tag:<4}  ${cost_usd:>9.5f}  {response.latency_s:>6.2f}s  "
                f"{response.trace_status:<12} {lang_str:<7} {sw_str:>3}  {conf_str}"
                f"{tok_warn}"
            )

        print(_FULL_COL_SEP)
        print(
            "  † Claude: reas% reflects thinking-budget allocation, not raw CoT fraction  "
            "‡ GPT-5.5: reasoning text hidden; token count from API usage field"
        )

    # ─────────────────────────────────────────────────────────────
    # AGGREGATE 1 — Reasoning share by reasoning_load (raw cluster)
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  AGGREGATE 1 — Reasoning share by reasoning_load")
    print(f"  Raw cluster only: deepseek_v4, glm_5_2, kimi_k2_7")
    print(f"{'═'*80}")
    print(f"  {'Load':<14} {'n':>5}  {'Avg Reas%':>10}  {'Min%':>8}  {'Max%':>8}")
    print(f"  {'─'*52}")
    raw_rows = [r for r in agg if r["regime"] == "raw"]
    for load in ("low", "medium", "high", "very_high"):
        rows = [r for r in raw_rows if r["reasoning_load"] == load]
        if not rows:
            continue
        shares = [r["reasoning_share"] * 100 for r in rows]
        avg = sum(shares) / len(shares)
        print(
            f"  {load:<14} {len(shares):>5}  {avg:>9.1f}%"
            f"  {min(shares):>7.1f}%  {max(shares):>7.1f}%"
        )
    print(f"  {'─'*52}")

    # ─────────────────────────────────────────────────────────────
    # AGGREGATE 2 — Cost per prompt (all models)
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*120}")
    print(f"  AGGREGATE 2 — Cost per prompt (USD)")
    print(f"{'═'*120}")
    # Abbreviate long model keys for header
    abbrevs = {
        "deepseek_v4": "deepseek",
        "glm_5_2": "glm",
        "kimi_k2_7": "kimi",
        "gemma_4": "gemma",
        "claude_sonnet_4_6": "claude",
        "gpt_5_5": "gpt",
        "opus_4_8": "opus",
        "mistral_medium_3_5": "mistral",
    }
    hdr_cells = "  ".join(f"{abbrevs.get(k, k[:8]):<9}" for k in model_keys_ordered)
    print(f"  {'Prompt':<7}  {'Load':<10} {hdr_cells}  {'Total':>10}")
    print(f"  {'─'*100}")

    model_totals: dict[str, float] = {k: 0.0 for k in model_keys_ordered}
    grand_total = 0.0

    for pid in all_prompt_ids:
        pid_by_model = {r["model_key"]: r for r in agg if r["pid"] == pid}
        load = prompts[pid].get("reasoning_load", "?")
        row_total = 0.0
        cells = []
        for k in model_keys_ordered:
            r = pid_by_model.get(k)
            if r:
                c = r["cost_usd"]
                row_total += c
                model_totals[k] += c
                cells.append(f"${c:.5f}")
            else:
                cells.append(f"{'—':<9}")
        grand_total += row_total
        print(f"  {pid:<7}  {load:<10} {'  '.join(cells)}  ${row_total:.5f}")

    print(f"  {'─'*100}")
    total_cells = "  ".join(f"${model_totals[k]:.5f}" for k in model_keys_ordered)
    print(f"  {'TOTAL':<7}  {'':10} {total_cells}  ${grand_total:.5f}")
    print(f"{'═'*120}")

    # ─────────────────────────────────────────────────────────────
    # AGGREGATE 3 — Primary trace language by prompt
    # ─────────────────────────────────────────────────────────────
    trace_models = [k for k in model_keys_ordered if k in TRACE_TEXT_MODELS]
    abbrev_hdrs = "  ".join(f"{abbrevs.get(k, k[:8]):<7}" for k in trace_models)

    print(f"\n{'═'*100}")
    print(f"  AGGREGATE 3 — Primary trace language by prompt")
    print(f"  (gpt_5_5 excluded — trace text not exposed)")
    print(f"{'═'*100}")
    print(f"  {'Prompt':<7}  {'Probe':<18}  {abbrev_hdrs}")
    print(f"  {'─'*85}")

    da_prompts: set[str] = set()
    for pid in all_prompt_ids:
        pid_by_model = {r["model_key"]: r for r in agg if r["pid"] == pid}
        probe = prompts[pid].get("language_probe", "?")
        cells = []
        for k in trace_models:
            r = pid_by_model.get(k)
            if r:
                lang = r["primary_lang"] or "?"
                cells.append(f"{lang:<7}")
                if lang == "da":
                    da_prompts.add(pid)
            else:
                cells.append(f"{'—':<7}")
        print(f"  {pid:<7}  {probe:<18}  {'  '.join(cells)}")

    print(f"  {'─'*85}")
    if da_prompts:
        print(
            f"\n  Danish (da) detected as primary trace language in:"
            f" {', '.join(sorted(da_prompts, key=lambda p: int(p[1:])))}"
        )
    else:
        print(f"\n  No prompt pulled Danish as the primary trace language in any model.")
    print(f"{'═'*100}")

    # ─────────────────────────────────────────────────────────────
    # Footer
    # ─────────────────────────────────────────────────────────────
    status_line = (
        "ALL ASSERTIONS PASSED"
        if not has_failure
        else "WARNINGS/ERRORS detected — review rows above"
    )
    print(f"\n  {status_line}")
    print(f"  Structured records  → results/full/{run_id}.jsonl")
    print(f"  Raw traces          → results/full/{run_id}_traces/<model>_<prompt>.txt")
    print(f"  Calls completed: {len(agg)} / {n_calls}")
    print()
    return 1 if has_failure else 0


# ---------------------------------------------------------------------------
# --juni-recap-mistral — cap-artifact check for Mistral's June Phase 1 numbers.
# June (20260626T130846_full) hit the completion cap (thinking_budget 4096 +
# 512 = 4608) on 3/10 rows (P1, P9, P10). P1 is reasoning_load="low" — one of
# only two "light task" data points — so Mistral's light-task reasoning median
# is an undercount. Re-runs all 10 prompts, 1 pass each (as in June), at
# thinking_budget=16384, to de-censor it. Everything else (pinned dated model
# string, reasoning_effort, prompts) is identical to June.
# ---------------------------------------------------------------------------

JUNI_RECAP_MISTRAL_THINKING_BUDGET = 16384
JUNI_RECAP_MISTRAL_BASELINE_RUN_ID = "20260626T130846_full"


def run_juni_recap_mistral(allow_direct: bool = False) -> int:
    """
    Re-run mistral_medium_3_5 on P1-P10 (1 pass each) at thinking_budget=16384.
    10 calls. Reuses save_result()'s "extra" dict to log finish_reason,
    native_finish_reason, and a computed truncated flag without changing
    save_result's schema for every other caller (--smoke/--pilot/--full) —
    those phases can run non-Mistral models via the direct Anthropic SDK
    path, whose max_tokens offset (thinking_budget+4096, adaptive) differs
    from the OpenRouter convention (thinking_budget+512) this run relies on.
    Mistral has no direct-provider path — it is always OpenRouter, always
    +512 — so that offset is exact here, not assumed.
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")
    prompts = load_prompts()
    all_prompt_ids = sorted(prompts.keys(), key=lambda p: int(p[1:]))

    model_key = "mistral_medium_3_5"
    cfg = panel[model_key]
    provider = cfg["provider"]
    cls = PROVIDER_MAP.get(provider)
    thinking_budget = JUNI_RECAP_MISTRAL_THINKING_BUDGET
    request_max_tokens = thinking_budget + 512

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_juni_recap_mistral"
    full_dir = RESULTS_DIR / "full"
    traces_dir = full_dir / f"{run_id}_traces"

    print(f"\n{'═'*110}")
    print(f"  Juni Recap — Mistral (--juni-recap-mistral)   run_id={run_id}")
    print(f"  Cap-artifact check: thinking_budget={thinking_budget} (panel.yaml's 4096 untouched)")
    print(f"  Baseline: results/full/{JUNI_RECAP_MISTRAL_BASELINE_RUN_ID}.jsonl")
    print(f"  {len(all_prompt_ids)} prompts × 1 pass = {len(all_prompt_ids)} calls, mistral_medium_3_5 only")
    print(f"  reasoning_effort={reasoning_effort!r}")
    print(f"{'═'*110}")

    try:
        old_rows, _ = _load_phase1_jsonl(JUNI_RECAP_MISTRAL_BASELINE_RUN_ID)
    except FileNotFoundError as e:
        print(f"  !! Cannot load baseline for comparison: {e}")
        return 1
    old_by_pid = {r["prompt_id"]: r for r in old_rows if r.get("model_key") == model_key}

    noise_floor = _relative_noise_floor(_load_pooled_baseline()).get(model_key)

    resolved = resolve_models(panel, [model_key])
    print_resolution_table(resolved)
    assert_no_silent_direct_route(panel, [model_key], allow_direct)

    has_failure = False
    drift_failed = False
    agg: list[dict] = []

    col_hdr = (
        f"  {'Prompt':<7} {'Load':<10} {'Inp':>6} {'OldReas':>8} {'NewReas':>8} {'Out':>6}  "
        f"{'Cost($)':>10}  {'Trunc':<6} {'FinishReason':<14}"
    )
    print(col_hdr)
    print("  " + "─" * 100)

    for pid in all_prompt_ids:
        p = prompts[pid]
        assert "facit" not in p, (
            f"CRITICAL SECURITY VIOLATION: facit in request-path object for {pid}"
        )
        prompt_text: str = p["prompt"]
        p_load = p.get("reasoning_load", "?")

        try:
            adapter = cls(model_key, cfg)
        except CredentialMissingError as e:
            print(f"  {pid:<7} SKIPPED — {e}")
            has_failure = True
            continue

        try:
            response = adapter.call(
                prompt_text, thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
            )
        except AdapterError as e:
            print(f"  {pid:<7} ERROR — {e}")
            has_failure = True
            continue

        account = build_account(response)
        cost_usd, snapshot_date = compute_cost(model_key, account)
        completion_tokens = account.reasoning_tokens + account.output_tokens
        truncated = completion_tokens >= request_max_tokens

        lm = (
            measure_trace_language(response.raw_reasoning_trace)
            if model_key in TRACE_TEXT_MODELS and response.raw_reasoning_trace
            else measure_trace_language(None)
        )

        save_result(
            run_id=run_id, model_key=model_key, prompt=prompt_text, response=response,
            account=account, cost_usd=cost_usd, pricing_snapshot_date=snapshot_date,
            thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
            results_dir=full_dir,
            extra={
                "prompt_id": pid,
                "prompt_type": p.get("type", "?"),
                "language_probe": p.get("language_probe", "?"),
                "reasoning_load": p_load,
                "regime": REGIME_MAP.get(model_key, "unknown"),
                "language_metric": lm,
                "finish_reason": response.finish_reason,
                "native_finish_reason": response.native_finish_reason,
                "request_max_tokens": request_max_tokens,
                "truncated": truncated,
            },
        )
        save_trace(
            traces_dir=traces_dir, run_id=run_id, model_key=model_key, prompt_id=pid,
            prompt_meta=p, prompt_text=prompt_text, answer_text=response.answer_text,
            reasoning_trace=response.raw_reasoning_trace, trace_status=response.trace_status,
            reasoning_tokens=account.reasoning_tokens, reasoning_source=response.reasoning_source,
        )

        old_row = old_by_pid.get(pid)
        old_reas = (old_row.get("tokens") or {}).get("reasoning") if old_row else None
        old_total = (
            (old_row.get("tokens") or {}).get("reasoning", 0) + (old_row.get("tokens") or {}).get("output", 0)
            if old_row else None
        )
        old_truncated = old_total is not None and old_total >= 4608  # June's cap: 4096+512

        agg.append({
            "pid": pid, "reasoning_load": p_load, "reasoning": account.reasoning_tokens,
            "output": account.output_tokens, "cost_usd": cost_usd, "truncated": truncated,
            "old_reasoning": old_reas, "old_truncated": old_truncated,
            "finish_reason": response.finish_reason,
        })

        print(
            f"  {pid:<7} {p_load:<10} {account.input_tokens:>6} {(old_reas if old_reas is not None else '—'):>8}"
            f" {account.reasoning_tokens:>8} {account.output_tokens:>6}  ${cost_usd:>9.5f}  "
            f"{'YES' if truncated else 'no':<6} {response.finish_reason!r:<14}"
        )

    # ═════════════════════════════════════════════════════════
    # REPORT
    # ═════════════════════════════════════════════════════════
    import statistics as _stats

    print(f"\n{'═'*100}")
    print(f"  1. TRUNCATED — computed (completion_tokens >= {request_max_tokens}), not inferred")
    print(f"{'═'*100}")
    truncated_rows = [r for r in agg if r["truncated"]]
    if truncated_rows:
        for r in truncated_rows:
            print(f"    !! {r['pid']}  reas+out={r['reasoning']+r['output']}  finish_reason={r['finish_reason']!r}")
    else:
        print(f"  None — no row hit {request_max_tokens} at thinking_budget={thinking_budget}.")

    print(f"\n{'═'*100}")
    print(f"  2. REASONING TOKENS — June (thinking_budget=4096) vs recap (thinking_budget=16384)")
    print(f"{'═'*100}")
    print(f"  {'Prompt':<7} {'Load':<10} {'OldReas':>8} {'OldTrunc':>9}  {'NewReas':>8} {'NewTrunc':>9}  {'Δ':>7}")
    for r in agg:
        old_r = r["old_reasoning"]
        delta = (r["reasoning"] - old_r) if old_r is not None else None
        print(
            f"  {r['pid']:<7} {r['reasoning_load']:<10} "
            f"{(old_r if old_r is not None else '—'):>8} {('YES' if r['old_truncated'] else 'no'):>9}  "
            f"{r['reasoning']:>8} {('YES' if r['truncated'] else 'no'):>9}  "
            f"{(f'{delta:+d}' if delta is not None else '—'):>7}"
        )

    print(f"\n{'═'*100}")
    print(f"  3. LIGHT-TASK REASONING MEDIAN (reasoning_load == 'low': P1, P2)")
    print(f"{'═'*100}")
    light_new = [r["reasoning"] for r in agg if r["reasoning_load"] == "low"]
    light_old = [r["old_reasoning"] for r in agg if r["reasoning_load"] == "low" and r["old_reasoning"] is not None]
    if light_new:
        print(f"  Old median: {_stats.median(light_old):.0f}  (n={len(light_old)})")
        print(f"  New median: {_stats.median(light_new):.0f}  (n={len(light_new)})")
    else:
        print("  No reasoning_load=='low' rows found.")

    print(f"\n{'═'*100}")
    print(f"  4. DRIFT CONTROL — the 7 rows NOT capped in June must reproduce within variance")
    print(f"  Tolerance: mistral_medium_3_5's own relative noise floor from --variance/--tools3 pooled data"
          f" ({noise_floor:.0%})" if noise_floor else "  !! No noise floor available for mistral_medium_3_5 — cannot judge drift quantitatively.")
    print(f"{'═'*100}")
    for r in agg:
        if r["old_truncated"] or r["old_reasoning"] is None:
            continue  # only the 7 rows June did NOT cap
        old_r = r["old_reasoning"]
        rel_delta = abs(r["reasoning"] - old_r) / old_r if old_r > 0 else None
        if noise_floor is not None and rel_delta is not None:
            verdict = "OK" if rel_delta <= noise_floor else "DRIFT"
            if verdict == "DRIFT":
                drift_failed = True
            print(f"  {r['pid']:<7}  old={old_r}  new={r['reasoning']}  Δrel={rel_delta:.0%}  [{verdict}]")
        else:
            print(f"  {r['pid']:<7}  old={old_r}  new={r['reasoning']}  (no verdict — missing noise floor)")
    if drift_failed:
        print(f"\n  !!!! DRIFT DETECTED on a row June never capped — stop, do not interpret the light-task median above.")
    else:
        print(f"\n  No drift — uncapped June rows reproduced within Mistral's own noise floor.")

    total_cost = sum(r["cost_usd"] for r in agg)
    print(f"\n{'═'*100}")
    print(f"  Total cost: ${total_cost:.5f}  ({len(agg)} calls)")
    status_line = "ALL CALLS COMPLETED" if not has_failure else "ERRORS DETECTED — see rows above"
    print(f"  {status_line}")
    print(f"  Structured records → results/full/{run_id}.jsonl")
    print(f"  Raw traces         → results/full/{run_id}_traces/<model>_<prompt>.txt")
    print()
    return 1 if (has_failure or drift_failed) else 0


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def _load_phase1_jsonl(run_id: Optional[str] = None) -> tuple[list[dict], str]:
    """
    Load the Phase 1 full-run JSONL. Returns (rows, run_id).
    Uses most-recent results/full/*.jsonl when run_id is None.
    """
    full_dir = RESULTS_DIR / "full"
    if run_id:
        path = full_dir / f"{run_id}.jsonl"
    else:
        candidates = sorted(full_dir.glob("*_full.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No full-run JSONL found in {full_dir}")
        path = candidates[-1]
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 JSONL not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(__import__("json").loads(line))
    return rows, path.stem  # stem = run_id without .jsonl


def _judge_one(
    judge_key: str,
    judge_cfg: dict,
    prompt_text: str,
    trace_text: str,
) -> JudgeResponse:
    """Call one judge on one (prompt, trace) pair."""
    rubric = build_rubric_prompt(prompt_text, trace_text)
    model_id = judge_cfg["openrouter_model_id"]
    pricing_key = JUDGE_PRICING_KEY[judge_key]
    return call_judge_openrouter(model_id, rubric, pricing_key)


def _print_judge_row(
    judge_key: str,
    jr: JudgeResponse,
) -> None:
    """Print a compact two-line judge result block."""
    if jr.parse_ok:
        r = jr.scores.get("redundancy", "?")
        c = jr.scores.get("coherence", "?")
        r_just = jr.justifications.get("redundancy", "")
        c_just = jr.justifications.get("coherence", "")
        print(f"  {judge_key:<18}  Redund={r}  Coher={c}"
              f"  ({jr.latency_s:.1f}s  ${jr.cost_usd:.5f})")
        print(f"    redundancy:  {r_just[:90]}")
        print(f"    coherence:   {c_just[:90]}")
    else:
        print(f"  {judge_key:<18}  PARSE ERROR — {jr.parse_error[:80]}")


# ---------------------------------------------------------------------------
# Phase 2 — validate-judges gate (Gemma only)
# ---------------------------------------------------------------------------

def run_validate_judges(source_run_id: Optional[str] = None) -> int:
    """
    Gate step: run both judges on Gemma's traces only (English/Danish — readable
    by Lars). Print a side-by-side of trace excerpt + judge scores + justifications
    so the human can verify judge quality before trusting them on Chinese traces.

    STOPS HERE. Does not proceed to Chinese traces. Run --judge after confirmation.
    """
    panel = load_panel()

    try:
        phase1_rows, p1_run_id = _load_phase1_jsonl(source_run_id)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}\n")
        return 1

    gemma_rows = [r for r in phase1_rows if r["model_key"] == "gemma_4"]
    gemma_rows.sort(key=lambda r: int(r["prompt_id"][1:]))

    if not gemma_rows:
        print("\n  ERROR: No gemma_4 records in Phase 1 JSONL.\n")
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_validate"

    print(f"\n{'═'*100}")
    print(f"  PHASE 2 — JUDGE VALIDATION GATE")
    print(f"  Source: {p1_run_id}  |  Scoring: gemma_4 only ({len(gemma_rows)} traces)")
    print(f"  Both judges read Gemma traces (English) — Lars can verify scores before trusting on Chinese.")
    print(f"  LEGIBILITY ONLY: correctness and faithfulness are out of scope.")
    print(f"{'═'*100}")

    has_error = False
    validation_records: list[dict] = []

    for row in gemma_rows:
        pid = row["prompt_id"]
        p_type = row.get("prompt_type", "?")
        p_probe = row.get("language_probe", "?")
        p_load = row.get("reasoning_load", "?")
        prompt_text = row["prompt"]
        trace_text = row.get("raw_reasoning_trace") or ""

        print(f"\n{'─'*100}")
        print(f"  [{pid}]  {p_type} / {p_probe}  (load: {p_load})")
        print(f"  Prompt: {prompt_text[:90].strip()!r}")

        if not trace_text.strip():
            print(f"  SKIPPED — no trace text (trace_status={row.get('trace_status')})")
            continue

        trace_excerpt = trace_text[:500].replace("\n", " ↵ ")
        print(f"\n  Trace excerpt (first 500 chars):")
        print(f"  {trace_excerpt!r}")
        print()

        judge_responses: dict[str, JudgeResponse] = {}
        for judge_key in JUDGE_KEYS:
            judge_cfg = panel.get(judge_key, {})
            if not judge_cfg.get("openrouter_model_id"):
                print(f"  {judge_key}: SKIPPED — no openrouter_model_id in panel.yaml")
                continue
            try:
                jr = _judge_one(judge_key, judge_cfg, prompt_text, trace_text)
            except Exception as e:
                print(f"  {judge_key}: ERROR — {e}")
                has_error = True
                continue
            judge_responses[judge_key] = jr
            _print_judge_row(judge_key, jr)

        # Agreement between the two judges on this trace — only when both parsed OK
        agr_val: Optional[dict] = None
        if len(judge_responses) == 2:
            jr0 = judge_responses[JUDGE_KEYS[0]]
            jr1 = judge_responses[JUDGE_KEYS[1]]
            if jr0.parse_ok and jr1.parse_ok:
                agr_val = compute_agreement(jr0.scores, jr1.scores)
                flag = "  ← HIGH DISAGREEMENT" if agr_val["high_disagreement"] else ""
                print(
                    f"\n  Agreement:  "
                    + "  ".join(
                        f"{dim}=Δ{agr_val['dim_diffs'].get(dim,'?')}"
                        for dim in DIMENSIONS
                    )
                    + f"  mean_diff={agr_val['mean_diff']}{flag}"
                )
            else:
                failed = [k for k in JUDGE_KEYS if not judge_responses[k].parse_ok]
                print(f"\n  Agreement:  n/a ({', '.join(failed)} parse failed)")

        # Save validation records
        for judge_key, jr in judge_responses.items():
            save_phase2_result(
                run_id=run_id,
                source_run_id=p1_run_id,
                model_key="gemma_4",
                prompt_id=pid,
                prompt_type=p_type,
                language_probe=p_probe,
                reasoning_load=p_load,
                judge_key=judge_key,
                judge_response=jr,
                agreement=agr_val,
                phase1_language_metric=row.get("language_metric"),
                trace_status=row.get("trace_status", "raw"),
                results_dir=PHASE2_DIR,
            )
            validation_records.append({"pid": pid, "judge": judge_key, "jr": jr})

    print(f"\n{'═'*100}")
    print(f"  GATE REACHED")
    print(f"  {len(gemma_rows)} Gemma traces scored by both judges.")
    print(f"  Validation records saved → results/phase2/{run_id}.jsonl")
    print()
    print(f"  Review the trace excerpts and judge scores above.")
    print(f"  If the scores match your reading of the traces, proceed with:")
    print()
    print(f"    python3 run.py --judge")
    print()
    print(f"  This will score all four raw-trace models (deepseek/glm/kimi + gemma).")
    print(f"  Claude and GPT are excluded — no scorable trace text exists for them.")
    print(f"{'═'*100}\n")

    return 1 if has_error else 0


# ---------------------------------------------------------------------------
# Phase 2 — full legibility scoring
# ---------------------------------------------------------------------------

def run_judge(source_run_id: Optional[str] = None) -> int:
    """
    Phase 2 full legibility scoring.
    Both judges score all 4 raw-trace models (deepseek/glm/kimi + gemma).
    Claude and GPT are explicitly excluded — reasons stated in output.
    LEGIBILITY ONLY — no correctness, no faithfulness, no economy mixing.
    """
    import statistics

    panel = load_panel()

    try:
        phase1_rows, p1_run_id = _load_phase1_jsonl(source_run_id)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}\n")
        return 1

    # Compute which scored models actually appear in this source JSONL.
    # This allows running --judge on a partial JSONL (e.g. two new models only)
    # without noisy "MISSING" rows for the models that weren't in that run.
    models_in_jsonl: set[str] = {r["model_key"] for r in phase1_rows}
    effective_scored: list[str] = [k for k in LEGIBILITY_SCORED if k in models_in_jsonl]

    # Build index: (model_key, prompt_id) → row
    p1_index: dict[tuple[str, str], dict] = {
        (r["model_key"], r["prompt_id"]): r
        for r in phase1_rows
        if r["model_key"] in effective_scored
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_phase2"
    all_pids = sorted(
        {r["prompt_id"] for r in phase1_rows},
        key=lambda p: int(p[1:])
    )

    n_total = len(effective_scored) * len(all_pids) * len(JUDGE_KEYS)

    print(f"\n{'═'*110}")
    print(f"  PHASE 2 — Legibility Scoring   run_id={run_id}")
    print(f"  Source: {p1_run_id}")
    print(f"  Models: {', '.join(effective_scored)}")
    print(f"  Judges: {', '.join(JUDGE_KEYS)}")
    print(f"  Calls: {len(effective_scored)} models × {len(all_pids)} prompts × {len(JUDGE_KEYS)} judges = {n_total}")
    print(f"  LEGIBILITY ONLY — correctness, faithfulness, and economy are out of scope.")
    print(f"{'═'*110}")

    # Explicit exclusion note
    print(f"\n  EXCLUDED FROM LEGIBILITY SCORING (transparency finding):")
    for mk, reason in LEGIBILITY_EXCLUDED.items():
        print(f"    {mk:<24}  {reason}")
    print(
        f"\n  This is a structural finding: the qualitative legibility axis cannot run"
        f"\n  on closed or summarized traces. Only raw-trace models can be assessed.\n"
    )

    has_error = False
    # scored_data[(model_key, pid, judge_key)] = JudgeResponse
    scored_data: dict[tuple[str, str, str], JudgeResponse] = {}
    # agg list for aggregate tables
    agg_records: list[dict] = []

    SEP = "─" * 110

    for model_key in effective_scored:
        print(f"\n{'═'*110}")
        print(f"  Model: {model_key}")
        print(f"{'═'*110}")
        print(f"\n  {'Prompt':<8}  {'Load':<10}  {'Judge':<18}  {'Redund':>7}  {'Coher':>7}  {'Δ':>4}  {'Cost':>10}  Justifications")
        print(f"  {SEP}")

        for pid in all_pids:
            p1_row = p1_index.get((model_key, pid))
            if p1_row is None:
                print(f"  {pid:<8}  MISSING in Phase 1 JSONL — skipping")
                continue

            trace_text = p1_row.get("raw_reasoning_trace") or ""
            prompt_text = p1_row["prompt"]
            p_type = p1_row.get("prompt_type", "?")
            p_probe = p1_row.get("language_probe", "?")
            p_load = p1_row.get("reasoning_load", "?")

            if not trace_text.strip():
                print(f"  {pid:<8}  {p_load:<10}  SKIPPED — no trace text")
                continue

            prompt_responses: dict[str, JudgeResponse] = {}

            for judge_key in JUDGE_KEYS:
                judge_cfg = panel.get(judge_key, {})
                if not judge_cfg.get("openrouter_model_id"):
                    print(f"  {pid:<8}  {p_load:<10}  {judge_key:<18}  SKIPPED — no openrouter_model_id")
                    continue

                try:
                    jr = _judge_one(judge_key, judge_cfg, prompt_text, trace_text)
                except Exception as e:
                    print(f"  {pid:<8}  {p_load:<10}  {judge_key:<18}  ERROR — {e}")
                    has_error = True
                    continue

                scored_data[(model_key, pid, judge_key)] = jr
                prompt_responses[judge_key] = jr

            # Compute agreement only when both judges responded AND both parsed OK
            agr: Optional[dict] = None
            if len(prompt_responses) == 2:
                jr0 = prompt_responses[JUDGE_KEYS[0]]
                jr1 = prompt_responses[JUDGE_KEYS[1]]
                if jr0.parse_ok and jr1.parse_ok:
                    agr = compute_agreement(jr0.scores, jr1.scores)

            # Print row(s) + save
            for i, judge_key in enumerate(JUDGE_KEYS):
                jr = prompt_responses.get(judge_key)
                if jr is None:
                    continue

                # Save
                save_phase2_result(
                    run_id=run_id,
                    source_run_id=p1_run_id,
                    model_key=model_key,
                    prompt_id=pid,
                    prompt_type=p_type,
                    language_probe=p_probe,
                    reasoning_load=p_load,
                    judge_key=judge_key,
                    judge_response=jr,
                    agreement=agr if i == 0 else None,
                    phase1_language_metric=p1_row.get("language_metric"),
                    trace_status=p1_row.get("trace_status", "raw"),
                    results_dir=PHASE2_DIR,
                )

                # Print
                if jr.parse_ok:
                    r = jr.scores.get("redundancy", "?")
                    c = jr.scores.get("coherence", "?")
                    delta_str = ""
                    if agr and i == 1:
                        d = agr["max_diff"]
                        flag = "!" if agr["high_disagreement"] else " "
                        delta_str = f"{d}{flag}"
                    r_just = jr.justifications.get("redundancy", "")[:40]
                    c_just = jr.justifications.get("coherence", "")[:40]
                    print(
                        f"  {pid:<8}  {p_load:<10}  {judge_key:<18}  {r:>7}  {c:>7}  "
                        f"{delta_str:>4}  ${jr.cost_usd:>8.5f}  "
                        f"R:{r_just!r} / C:{c_just!r}"
                    )
                else:
                    print(
                        f"  {pid:<8}  {p_load:<10}  {judge_key:<18}  "
                        f"PARSE ERROR — {jr.parse_error[:60]}"
                    )
                    has_error = True

                # Collect for aggregates
                if jr.parse_ok:
                    agg_records.append({
                        "model_key": model_key,
                        "prompt_id": pid,
                        "prompt_type": p_type,
                        "reasoning_load": p_load,
                        "judge_key": judge_key,
                        "redundancy": jr.scores.get("redundancy", 0),
                        "coherence": jr.scores.get("coherence", 0),
                        "cost_usd": jr.cost_usd,
                        "high_disagreement": agr["high_disagreement"] if agr else False,
                        "max_diff": agr["max_diff"] if agr else 0,
                    })

        print(f"  {SEP}")

    # ─────────────────────────────────────────────────────────────────────────
    # AGGREGATE TABLES
    # ─────────────────────────────────────────────────────────────────────────

    def avg(vals: list) -> str:
        return f"{statistics.mean(vals):.2f}" if vals else "—"

    # --- Table 1: Average legibility scores by model and judge ---
    print(f"\n{'═'*90}")
    print(f"  AGGREGATE 1 — Average legibility by model  (1–5 scale; higher coher = better, lower redund = better)")
    print(f"{'═'*90}")
    print(f"  {'Model':<22}  {'Judge':<18}  {'Avg Redund':>10}  {'Avg Coher':>10}  {'Calls':>6}")
    print(f"  {'─'*70}")
    for model_key in effective_scored:
        for judge_key in JUDGE_KEYS:
            recs = [r for r in agg_records if r["model_key"] == model_key and r["judge_key"] == judge_key]
            r_vals = [r["redundancy"] for r in recs]
            c_vals = [r["coherence"] for r in recs]
            print(
                f"  {model_key:<22}  {judge_key:<18}  {avg(r_vals):>10}  "
                f"{avg(c_vals):>10}  {len(recs):>6}"
            )
        print(f"  {'·'*70}")
    print(f"  {'─'*70}")

    # --- Table 2: Average legibility by prompt type and judge ---
    print(f"\n{'═'*90}")
    print(f"  AGGREGATE 2 — Average legibility by prompt type (raw-trace models pooled)")
    print(f"{'═'*90}")
    print(f"  {'Type':<22}  {'Load':<12}  {'Judge':<18}  {'Avg Redund':>10}  {'Avg Coher':>10}  {'n':>4}")
    print(f"  {'─'*78}")
    prompt_types: list[str] = []
    seen_types: set[str] = set()
    for r in agg_records:
        key = (r["prompt_type"], r["reasoning_load"])
        if key not in seen_types:
            seen_types.add(key)
            prompt_types.append(key)
    prompt_types.sort(key=lambda x: x[1])  # sort by load
    for (p_type, p_load) in prompt_types:
        for judge_key in JUDGE_KEYS:
            recs = [
                r for r in agg_records
                if r["prompt_type"] == p_type and r["judge_key"] == judge_key
            ]
            r_vals = [r["redundancy"] for r in recs]
            c_vals = [r["coherence"] for r in recs]
            print(
                f"  {p_type:<22}  {p_load:<12}  {judge_key:<18}  {avg(r_vals):>10}  "
                f"{avg(c_vals):>10}  {len(recs):>4}"
            )
    print(f"  {'─'*78}")

    # --- Table 3: Inter-judge agreement by model ---
    print(f"\n{'═'*80}")
    print(f"  AGGREGATE 3 — Inter-judge agreement by model")
    print(f"  (Δ = |minimax_score − gemini_score|; high disagreement = max Δ ≥ {HIGH_DISAGREEMENT_THRESHOLD})")
    print(f"{'═'*80}")
    print(f"  {'Model':<22}  {'Avg ΔRedund':>12}  {'Avg ΔCoher':>11}  {'Max Δ':>7}  {'Flagged':>8}")
    print(f"  {'─'*65}")
    flagged_traces: list[dict] = []
    for model_key in effective_scored:
        mm_recs = {r["prompt_id"]: r for r in agg_records if r["model_key"] == model_key and r["judge_key"] == JUDGE_KEYS[0]}
        gm_recs = {r["prompt_id"]: r for r in agg_records if r["model_key"] == model_key and r["judge_key"] == JUDGE_KEYS[1]}
        shared_pids = set(mm_recs) & set(gm_recs)

        r_diffs, c_diffs, max_diffs = [], [], []
        for pid in shared_pids:
            rd = abs(mm_recs[pid]["redundancy"] - gm_recs[pid]["redundancy"])
            cd = abs(mm_recs[pid]["coherence"] - gm_recs[pid]["coherence"])
            r_diffs.append(rd)
            c_diffs.append(cd)
            max_d = max(rd, cd)
            max_diffs.append(max_d)
            if max_d >= HIGH_DISAGREEMENT_THRESHOLD:
                flagged_traces.append({
                    "model": model_key, "pid": pid,
                    "redund_diff": rd, "coher_diff": cd,
                    "minimax_r": mm_recs[pid]["redundancy"],
                    "minimax_c": mm_recs[pid]["coherence"],
                    "gemini_r": gm_recs[pid]["redundancy"],
                    "gemini_c": gm_recs[pid]["coherence"],
                })

        n_flagged = sum(1 for d in max_diffs if d >= HIGH_DISAGREEMENT_THRESHOLD)
        print(
            f"  {model_key:<22}  {avg(r_diffs):>12}  {avg(c_diffs):>11}  "
            f"{max(max_diffs) if max_diffs else 0:>7}  {n_flagged:>8}"
        )
    print(f"  {'─'*65}")

    if flagged_traces:
        print(f"\n  High-disagreement traces (Δ ≥ {HIGH_DISAGREEMENT_THRESHOLD} — legibility signal least reliable here):")
        print(f"  {'Model':<22}  {'Pid':<6}  {'Dim':<10}  {'MiniMax':>8}  {'Gemini':>7}  {'Diff':>5}")
        print(f"  {'·'*62}")
        for t in flagged_traces:
            if t["redund_diff"] >= HIGH_DISAGREEMENT_THRESHOLD:
                print(f"  {t['model']:<22}  {t['pid']:<6}  {'redundancy':<10}  {t['minimax_r']:>8}  {t['gemini_r']:>7}  {t['redund_diff']:>5}")
            if t["coher_diff"] >= HIGH_DISAGREEMENT_THRESHOLD:
                print(f"  {t['model']:<22}  {t['pid']:<6}  {'coherence':<10}  {t['minimax_c']:>8}  {t['gemini_c']:>7}  {t['coher_diff']:>5}")

    # --- Cost summary ---
    total_cost = sum(r["cost_usd"] for r in agg_records)
    print(f"\n  Phase 2 total judge cost: ${total_cost:.5f}")

    # --- Scope firewall reminder ---
    print(f"\n{'═'*90}")
    print(f"  SCOPE FIREWALL")
    print(f"  These scores measure LEGIBILITY only: is the trace readable and monitorable?")
    print(f"  They do NOT measure:")
    print(f"    - Correctness  (whether the answer is right — Phase 3)")
    print(f"    - Faithfulness (whether the trace drove the output — not measured here)")
    print(f"  Do not mix legibility scores into economy or correctness tables.")
    print(f"{'═'*90}")

    # --- Footer ---
    status = "ALL CALLS COMPLETED" if not has_error else "ERRORS DETECTED — see rows above"
    print(f"\n  {status}")
    print(f"  Records → results/phase2/{run_id}.jsonl")
    print(f"  Calls completed: {len(agg_records)} judge scores recorded")
    print()

    return 1 if has_error else 0


# ---------------------------------------------------------------------------
# Language-cost run (Step 2)
# ---------------------------------------------------------------------------

def run_langcost(full: bool = False, steer: bool = False, allow_direct: bool = False) -> int:
    """
    Language-cost experiment: same content in da/en/zh × 5 open models.

    Pilot (default): M1 × 3 langs × 5 models = 15 calls, then cost projection
    for the full 90-call grid and stop.  Lars confirms, then runs --langcost-full.

    Full (--langcost-full): M1–M6 × 3 langs × 5 models = 90 calls.

    Steer (--steer, scaffold — do not mix with unsteered baseline):
    prepends a short language instruction before each prompt, e.g.
    "Tænk og ræsonnér på dansk." See STEER_PREFIX dict above.
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")
    prompts = load_multilang_prompts()

    all_task_ids = sorted(prompts.keys(), key=lambda t: int(t[1:]))
    task_ids = all_task_ids if full else [LANGCOST_PILOT_TASK]

    steer_suffix = "_steer" if steer else ""
    mode_suffix = ("_langcost_full" if full else "_langcost_pilot") + steer_suffix
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + mode_suffix
    mode_label = "Full" if full else "Pilot"

    langcost_dir = RESULTS_DIR / "langcost"
    traces_dir = langcost_dir / f"{run_id}_traces"

    n_calls = len(task_ids) * len(LANGCOST_LANGS) * len(LANGCOST_MODELS)
    n_full_grid = len(all_task_ids) * len(LANGCOST_LANGS) * len(LANGCOST_MODELS)

    print(f"\n{'═'*112}")
    print(f"  Language-Cost Experiment — {mode_label}   run_id={run_id}")
    if steer:
        print(f"  STEER MODE: language-instruction prefix prepended to each prompt")
    print(
        f"  Grid: {len(task_ids)} task(s) × {len(LANGCOST_LANGS)} lang(s) × "
        f"{len(LANGCOST_MODELS)} models = {n_calls} calls"
    )
    print(
        f"  thinking_budget={LANGCOST_THINKING_BUDGET}  "
        f"(generous+equal; prevents budget-ceiling artefact for longer-token languages)"
    )
    print(f"  reasoning_effort={reasoning_effort!r}  (locked — experiment condition)")
    print(f"  Models: {', '.join(LANGCOST_MODELS)}")
    print(f"{'═'*112}")

    resolved = resolve_models(panel, LANGCOST_MODELS)
    hard_errors = print_resolution_table(resolved)
    assert_no_silent_direct_route(panel, LANGCOST_MODELS, allow_direct)
    if hard_errors > 0:
        print(f"\n  !! {hard_errors} model(s) not resolved. Fix panel.yaml.\n")
        return 1

    has_failure = False
    agg: list[dict] = []

    col_hdr = (
        f"  {'Model':<22} {'Inp':>6} {'Reas':>8} {'Out':>6}  {'Reas%':>6}  "
        f"{'ReasChars':>10}  {'Cost($)':>10}  {'Lat':>7}  {'TraceLang':<10} {'Status'}"
    )
    col_sep = "  " + "─" * 114

    for task_id in task_ids:
        p = prompts[task_id]
        assert "facit" not in p, (
            f"CRITICAL SECURITY VIOLATION: facit in request-path object for {task_id}"
        )
        p_type = p.get("type", "?")
        p_load = p.get("reasoning_load", "?")

        for lang in LANGCOST_LANGS:
            prompt_text: str = p["variants"][lang]
            if steer:
                effective_prompt = STEER_PREFIX[lang] + prompt_text
            else:
                effective_prompt = prompt_text

            print(f"\n{'═'*112}")
            print(
                f"  [{task_id}]  lang={lang}  type={p_type}  load={p_load}"
                + ("  [STEERED]" if steer else "")
            )
            print(f"  Prompt (first 100 chars): {prompt_text[:100].strip()!r}")
            print(f"{'═'*112}")
            print(col_hdr)
            print(col_sep)

            for key in LANGCOST_MODELS:
                cfg = panel.get(key)
                if not cfg:
                    print(f"  {key:<22} SKIPPED — not in panel.yaml")
                    continue
                provider = cfg["provider"]
                regime = REGIME_MAP.get(key, "raw")

                cls = PROVIDER_MAP.get(provider)
                if cls is None:
                    print(f"  {key:<22} SKIPPED (unknown provider: {provider})")
                    continue

                try:
                    adapter = cls(key, cfg)
                except CredentialMissingError as e:
                    print(f"  {key:<22} SKIPPED — {e}")
                    continue

                try:
                    response = adapter.call(
                        effective_prompt,
                        thinking_budget=LANGCOST_THINKING_BUDGET,
                        reasoning_effort=reasoning_effort,
                    )
                except AdapterError as e:
                    print(f"  {key:<22} ERROR — {e}")
                    has_failure = True
                    continue
                except Exception as e:
                    print(f"  {key:<22} ERROR — unexpected: {e}")
                    has_failure = True
                    continue

                account = build_account(response)
                cost_usd, snapshot_date = compute_cost(key, account)

                reasoning_chars = len(response.raw_reasoning_trace or "")
                output_chars = len(response.answer_text or "")
                lm = measure_trace_language(response.raw_reasoning_trace)

                save_langcost_result(
                    run_id=run_id,
                    model_key=key,
                    task_id=task_id,
                    prompt_lang=lang,
                    prompt_text=effective_prompt,
                    response=response,
                    account=account,
                    cost_usd=cost_usd,
                    pricing_snapshot_date=snapshot_date,
                    thinking_budget=LANGCOST_THINKING_BUDGET,
                    reasoning_effort=reasoning_effort,
                    reasoning_chars=reasoning_chars,
                    output_chars=output_chars,
                    regime=regime,
                    language_metric=lm,
                    results_dir=langcost_dir,
                )

                save_langcost_trace(
                    traces_dir=traces_dir,
                    model_key=key,
                    task_id=task_id,
                    prompt_lang=lang,
                    prompt_text=effective_prompt,
                    answer_text=response.answer_text,
                    reasoning_trace=response.raw_reasoning_trace,
                    trace_status=response.trace_status,
                    reasoning_tokens=account.reasoning_tokens,
                    reasoning_source=response.reasoning_source,
                )

                agg.append({
                    "task_id": task_id,
                    "lang": lang,
                    "model_key": key,
                    "reasoning_tokens": account.reasoning_tokens,
                    "reasoning_chars": reasoning_chars,
                    "cost_usd": cost_usd,
                    "primary_trace_lang": lm.get("primary_trace_language"),
                })

                reas_share_pct = account.reasoning_share * 100
                lang_str = lm.get("primary_trace_language") or "?"

                print(
                    f"  {key:<22} {account.input_tokens:>6} {account.reasoning_tokens:>8}"
                    f" {account.output_tokens:>6}  {reas_share_pct:>5.1f}%  "
                    f"{reasoning_chars:>10}  ${cost_usd:>9.5f}  {response.latency_s:>6.2f}s  "
                    f"{lang_str:<10} {response.trace_status}"
                )

            print(col_sep)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary / projection
    # ─────────────────────────────────────────────────────────────────────────

    def _avg(seq: list) -> Optional[float]:
        return sum(seq) / len(seq) if seq else None

    if not full:
        # Pilot: cost per (model, lang) + projection to full grid
        print(f"\n{'═'*90}")
        print(f"  PILOT SUMMARY — M1 cost per model × language")
        print(f"{'═'*90}")
        print(f"  {'Model':<22}  {'da':>11}  {'en':>11}  {'zh':>11}  {'Total':>11}")
        print(f"  {'─'*66}")
        model_pilot_totals: dict[str, float] = {}
        for key in LANGCOST_MODELS:
            costs = {
                lang: sum(r["cost_usd"] for r in agg if r["model_key"] == key and r["lang"] == lang)
                for lang in LANGCOST_LANGS
            }
            total = sum(costs.values())
            model_pilot_totals[key] = total
            print(
                f"  {key:<22}  "
                + "  ".join(f"${costs[l]:>9.5f}" for l in LANGCOST_LANGS)
                + f"  ${total:>9.5f}"
            )
        print(f"  {'─'*66}")
        pilot_grand = sum(model_pilot_totals.values())
        print(f"  {'PILOT TOTAL':<22}  {'':>11}  {'':>11}  {'':>11}  ${pilot_grand:>9.5f}")

        proj_label = f"{len(all_task_ids)} tasks"
        print(f"\n{'═'*90}")
        print(
            f"  COST PROJECTION — Full grid "
            f"({len(all_task_ids)} tasks × {len(LANGCOST_LANGS)} langs × "
            f"{len(LANGCOST_MODELS)} models = {n_full_grid} calls)"
        )
        print(f"  Based on M1 pilot averages × {len(all_task_ids)} tasks")
        print(f"{'═'*90}")
        print(f"  {'Model':<22}  {'Pilot (1 task)':>16}  {'Projected ({})'.format(proj_label):>22}")
        print(f"  {'─'*64}")
        total_projected = 0.0
        for key in LANGCOST_MODELS:
            pilot_cost = model_pilot_totals.get(key, 0.0)
            proj = pilot_cost * len(all_task_ids)
            total_projected += proj
            marker = "—" if pilot_cost == 0 else f"${proj:>20.4f}"
            print(f"  {key:<22}  ${pilot_cost:>15.5f}  {marker}")
        print(f"  {'─'*64}")
        print(f"  {'TOTAL':<22}  {'':>16}  ${total_projected:>21.4f}")
        print(f"{'═'*90}")

        print(f"\n  Pilot complete. Review costs above.")
        print(f"  To run the full 90-call grid:")
        steer_flag = " --steer" if steer else ""
        print(f"    python3 run.py --langcost-full{steer_flag}")
        print(f"  Records: results/langcost/{run_id}.jsonl")
        print(f"  Traces:  results/langcost/{run_id}_traces/")
        print()

    else:
        # Full run: aggregate tables
        print(f"\n{'═'*112}")
        print(f"  AGGREGATE 1 — Reasoning tokens per model × language (avg over {len(all_task_ids)} tasks)")
        print(f"{'═'*112}")
        print(f"  {'Model':<22}  {'da avg':>10}  {'en avg':>10}  {'zh avg':>10}  {'da/en':>7}  {'zh/en':>7}")
        print(f"  {'─'*72}")

        for key in LANGCOST_MODELS:
            tok = {
                lang: _avg([r["reasoning_tokens"] for r in agg if r["model_key"] == key and r["lang"] == lang])
                for lang in LANGCOST_LANGS
            }
            da_en = f"{tok['da']/tok['en']:.2f}" if tok["da"] and tok["en"] else "—"
            zh_en = f"{tok['zh']/tok['en']:.2f}" if tok["zh"] and tok["en"] else "—"
            print(
                f"  {key:<22}  "
                + "  ".join(f"{tok[l]:>9.0f}" if tok[l] is not None else f"{'—':>10}" for l in LANGCOST_LANGS)
                + f"  {da_en:>7}  {zh_en:>7}"
            )
        print(f"  {'─'*72}")

        print(f"\n{'═'*112}")
        print(f"  AGGREGATE 2 — Reasoning chars per model × language (avg over {len(all_task_ids)} tasks)")
        print(f"{'═'*112}")
        print(f"  {'Model':<22}  {'da avg':>10}  {'en avg':>10}  {'zh avg':>10}  {'da/en':>7}  {'zh/en':>7}")
        print(f"  {'─'*72}")

        for key in LANGCOST_MODELS:
            chrs = {
                lang: _avg([r["reasoning_chars"] for r in agg if r["model_key"] == key and r["lang"] == lang])
                for lang in LANGCOST_LANGS
            }
            da_en = f"{chrs['da']/chrs['en']:.2f}" if chrs["da"] and chrs["en"] else "—"
            zh_en = f"{chrs['zh']/chrs['en']:.2f}" if chrs["zh"] and chrs["en"] else "—"
            print(
                f"  {key:<22}  "
                + "  ".join(f"{chrs[l]:>9.0f}" if chrs[l] is not None else f"{'—':>10}" for l in LANGCOST_LANGS)
                + f"  {da_en:>7}  {zh_en:>7}"
            )
        print(f"  {'─'*72}")

        # Crosstable: prompt_lang × primary_trace_language
        from collections import Counter
        print(f"\n{'═'*80}")
        print(f"  AGGREGATE 3 — Krydstabel: prompt_lang × primary_trace_language")
        print(f"  (pooled over alle modeller og alle {len(all_task_ids)} opgaver)")
        print(f"{'═'*80}")
        for pl in LANGCOST_LANGS:
            trace_langs = [r["primary_trace_lang"] or "?" for r in agg if r["lang"] == pl]
            counts: Counter = Counter(trace_langs)
            total = len(trace_langs)
            print(f"  prompt_lang={pl}  (n={total}):")
            for tl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                pct = cnt / total * 100 if total else 0
                print(f"    trace_lang={tl:<8}  {cnt:>4}/{total}  ({pct:.0f}%)")
        print(f"{'═'*80}")

        print(f"\n  Full run complete.")
        print(f"  Records: results/langcost/{run_id}.jsonl")
        print(f"  Traces:  results/langcost/{run_id}_traces/")
        print(f"  Generate analysis report:")
        print(f"    python3 run.py --langcost-report --source-run-id {run_id}")
        print()

    return 1 if has_failure else 0


# ---------------------------------------------------------------------------
# Language-cost report (Step 3)
# ---------------------------------------------------------------------------

def run_langcost_report(source_run_id: Optional[str] = None) -> int:
    """
    Read a completed langcost full-run JSONL and write results/syntese/sprogets_pris_data.md.

    Computes:
      1. Sprog-tax i tokens (avg + median per model × lang, avg-based ratios)
      2. Sprog-tax i tegn  (avg + median per model × lang, avg-based ratios)
      3. Outlier-robusthed (two runs that skew the averages, with corrected ratios)
      4. Dekomponering: tegn per token — encoding tax vs genuine extra thinking
      5. Krydstabel per model + pooled summary
      6. Pris per sprog per model
    """
    import json
    import statistics
    from collections import Counter, defaultdict

    langcost_dir = RESULTS_DIR / "langcost"
    syntese_dir = RESULTS_DIR / "syntese"

    if source_run_id:
        path = langcost_dir / f"{source_run_id}.jsonl"
    else:
        candidates = sorted(langcost_dir.glob("*_langcost_full*.jsonl"))
        if not candidates:
            print(f"\n  ERROR: No *_langcost_full*.jsonl found in {langcost_dir}")
            print(f"  Run the full grid first: python3 run.py --langcost-full")
            return 1
        path = candidates[-1]
        source_run_id = path.stem

    if not path.exists():
        print(f"\n  ERROR: File not found: {path}")
        return 1

    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print(f"\n  ERROR: No records in {path}")
        return 1

    print(f"\n  Loaded {len(rows)} records from {path.name}")

    by_model_lang: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_model_lang[(r["model_key"], r["prompt_lang"])].append(r)

    def safe_avg(vals: list) -> Optional[float]:
        return statistics.mean(vals) if vals else None

    def safe_med(vals: list) -> Optional[float]:
        return statistics.median(vals) if vals else None

    def fmt_ratio(a: Optional[float], b: Optional[float]) -> str:
        if a is not None and b is not None and b > 0:
            return f"{a / b:.2f}"
        return "—"

    def fmt_num(v: Optional[float]) -> str:
        return f"{v:.0f}" if v is not None else "—"

    # ─── Build markdown ───
    L: list[str] = []

    L.append("# Sprogets pris — data-note")
    L.append("")
    L.append(
        "> **Forbehold:** Resultaterne er betinget af oversættelsernes troskab, "
        "særligt de kinesiske (zh) varianter. De kinesiske prompts er maskinoversat "
        "med engelsk som pivot-sprog og er ikke verificeret af en kyndig taler. "
        "Alle konklusioner fra zh-sessioner er indikative, ikke endelige, "
        "indtil oversættelserne er efterprøvet. "
        "Derudover endte 14 af de 30 kinesiske kald med at tænke på engelsk, "
        "så zh-kolonnen blander ægte kinesisk tænkning med tilbagefald, "
        "og zh-tallene er kun indikative."
    )
    L.append("")
    L.append(f"**Kilde:** `{source_run_id}.jsonl` — {len(rows)} records")
    L.append(f"**Modeller:** {', '.join(LANGCOST_MODELS)}")
    L.append(f"**Sprog:** {', '.join(LANGCOST_LANGS)}")
    L.append(f"**Opgaver:** M1–M6 (6 kultur-neutrale opgaver)")
    L.append(f"**thinking_budget:** {LANGCOST_THINKING_BUDGET} (generøst og ens for alle modeller)")
    L.append("")

    # ─── 1. Sprog-tax i tokens ───
    L.append("## 1. Sprog-tax i tokens (reasoning_tokens)")
    L.append("")
    L.append(
        "Gennemsnit og median per model og sprog over de 6 opgaver. "
        "Medianen er robust mod enkeltstående outlier-kørsler. "
        "da/en og zh/en er gennemsnitsbaserede forhold: >1 = det sprog bruger flere tokens end engelsk."
    )
    L.append("")
    L.append("| Model | da avg | da med | en avg | en med | zh avg | zh med | da/en | zh/en |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in LANGCOST_MODELS:
        t_avg = {
            lang: safe_avg([r["tokens"]["reasoning"] for r in by_model_lang[(key, lang)]])
            for lang in LANGCOST_LANGS
        }
        t_med = {
            lang: safe_med([r["tokens"]["reasoning"] for r in by_model_lang[(key, lang)]])
            for lang in LANGCOST_LANGS
        }
        L.append(
            f"| {key} "
            + "".join(
                f"| {fmt_num(t_avg[l])} | {fmt_num(t_med[l])} "
                for l in LANGCOST_LANGS
            )
            + f"| {fmt_ratio(t_avg['da'], t_avg['en'])} | {fmt_ratio(t_avg['zh'], t_avg['en'])} |"
        )
    L.append("")

    # ─── 2. Sprog-tax i tegn ───
    L.append("## 2. Sprog-tax i tegn (reasoning_chars)")
    L.append("")
    L.append(
        "Gennemsnit og median for antal tegn i det rå trace-tekst. "
        "Tegn er et tokenizer-uafhængigt mål: en stigning her er ægte ekstra tekst, ikke blot kodning."
    )
    L.append("")
    L.append("| Model | da avg | da med | en avg | en med | zh avg | zh med | da/en | zh/en |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in LANGCOST_MODELS:
        c_avg = {
            lang: safe_avg([r["reasoning_chars"] for r in by_model_lang[(key, lang)]])
            for lang in LANGCOST_LANGS
        }
        c_med = {
            lang: safe_med([r["reasoning_chars"] for r in by_model_lang[(key, lang)]])
            for lang in LANGCOST_LANGS
        }
        L.append(
            f"| {key} "
            + "".join(
                f"| {fmt_num(c_avg[l])} | {fmt_num(c_med[l])} "
                for l in LANGCOST_LANGS
            )
            + f"| {fmt_ratio(c_avg['da'], c_avg['en'])} | {fmt_ratio(c_avg['zh'], c_avg['en'])} |"
        )
    L.append("")

    # ─── 3. Outlier-robusthed ───
    L.append("## 3. Outlier-robusthed")
    L.append("")
    L.append(
        "To enkeltkørsler er markant atypiske og forvrider gennemsnitstallene i tabel 1 og 2. "
        "Median-kolonnerne ovenfor er immune, men forholdene (da/en, zh/en) er gennemsnitsbaserede "
        "og påvirkes."
    )
    L.append("")

    # Compute the specific outlier-corrected ratios
    # Mistral en/da: drop M1 from BOTH sides
    mis_en_no_m1 = [
        r["tokens"]["reasoning"] for r in by_model_lang[("mistral_medium_3_5", "en")]
        if r["task_id"] != "M1"
    ]
    mis_da_no_m1 = [
        r["tokens"]["reasoning"] for r in by_model_lang[("mistral_medium_3_5", "da")]
        if r["task_id"] != "M1"
    ]
    mis_en_avg_all = safe_avg([r["tokens"]["reasoning"] for r in by_model_lang[("mistral_medium_3_5", "en")]])
    mis_da_avg_all = safe_avg([r["tokens"]["reasoning"] for r in by_model_lang[("mistral_medium_3_5", "da")]])
    mis_ratio_all = mis_en_avg_all / mis_da_avg_all if mis_en_avg_all and mis_da_avg_all else None
    mis_ratio_no_m1 = statistics.mean(mis_en_no_m1) / statistics.mean(mis_da_no_m1) if mis_en_no_m1 and mis_da_no_m1 else None

    # DeepSeek zh/en: drop M6 from BOTH sides
    ds_zh_no_m6 = [
        r["tokens"]["reasoning"] for r in by_model_lang[("deepseek_v4", "zh")]
        if r["task_id"] != "M6"
    ]
    ds_en_no_m6 = [
        r["tokens"]["reasoning"] for r in by_model_lang[("deepseek_v4", "en")]
        if r["task_id"] != "M6"
    ]
    ds_zh_avg_all = safe_avg([r["tokens"]["reasoning"] for r in by_model_lang[("deepseek_v4", "zh")]])
    ds_en_avg_all = safe_avg([r["tokens"]["reasoning"] for r in by_model_lang[("deepseek_v4", "en")]])
    ds_ratio_all = ds_zh_avg_all / ds_en_avg_all if ds_zh_avg_all and ds_en_avg_all else None
    ds_ratio_no_m6 = statistics.mean(ds_zh_no_m6) / statistics.mean(ds_en_no_m6) if ds_zh_no_m6 and ds_en_no_m6 else None

    L.append("**Outlier 1 — mistral_medium_3_5, M1, lang=en:**")
    L.append(
        f"13 606 reasoning-tokens og 48 814 tegn — ca. 10× mere end Mistrals "
        f"typiske engelske kørsel (median: 1 444 tokens). "
        f"Denne ene kørsel puster Mistrals en-gennemsnit op fra 2 160 til 4 068 tokens "
        f"og trækker en/da-forholdet fra "
        f"{'~' + f'{mis_ratio_no_m1:.1f}' if mis_ratio_no_m1 else '?'} til "
        f"{'~' + f'{mis_ratio_all:.1f}' if mis_ratio_all else '?'} (M1 fratrukket begge sider). "
        f"**Retningen holder dog på alle seks opgaver:** Mistral bruger konsekvent "
        f"flere tokens på engelsk end på dansk, uanset om M1 medregnes eller ej."
    )
    L.append("")
    L.append("**Outlier 2 — deepseek_v4, M6, lang=zh:**")
    L.append(
        f"7 137 reasoning-tokens og 9 357 tegn — "
        f"ca. 20× mere end DeepSeeks typiske kinesiske kørsel (median: 344 tokens). "
        f"Denne ene kørsel driver næsten hele zh/en-forholdet: "
        f"{'~' + f'{ds_ratio_all:.2f}' if ds_ratio_all else '?'} med M6, men "
        f"{'~' + f'{ds_ratio_no_m6:.2f}' if ds_ratio_no_m6 else '?'} uden "
        f"(M6 fratrukket begge sider). "
        f"DeepSeeks kinesiske zh/en-forhold på 4,56 bør ikke tolkes som et generelt mønster."
    )
    L.append("")

    # ─── 4. Dekomponering ───
    L.append("## 4. Dekomponering: tegn per reasoning-token")
    L.append("")
    L.append(
        "Hvis **tokens stiger** men **tegn ikke gør** (tegn/token-forholdet falder), "
        "er det en ren **kodningsskat** — det samme indhold kræver flere tokens at "
        "repræsentere på det pågældende sprog. "
        "Hvis **både tokens og tegn stiger** (forholdet er stabilt), er det "
        "**ægte ekstra tænkning** — modellen ræsonnerer faktisk mere."
    )
    L.append("")
    L.append("| Model | da tg/tok | en tg/tok | zh tg/tok | Konklusion (da vs en) |")
    L.append("|---|---:|---:|---:|---|")
    for key in LANGCOST_MODELS:
        tok = {
            lang: safe_avg([r["tokens"]["reasoning"] for r in by_model_lang[(key, lang)]])
            for lang in LANGCOST_LANGS
        }
        chrs = {
            lang: safe_avg([r["reasoning_chars"] for r in by_model_lang[(key, lang)]])
            for lang in LANGCOST_LANGS
        }
        cpt = {
            lang: (chrs[lang] / tok[lang]) if (tok[lang] and tok[lang] > 0) else None
            for lang in LANGCOST_LANGS
        }

        if tok["da"] and tok["en"] and chrs["da"] and chrs["en"]:
            tok_ratio = tok["da"] / tok["en"]
            chr_ratio = chrs["da"] / chrs["en"]
            if tok_ratio > 1.05 and chr_ratio < 1.05:
                conclusion = "Kodningsskat (tok↑, tegn≈)"
            elif tok_ratio > 1.05 and chr_ratio > 1.05:
                conclusion = "Ægte ekstra tænkning (tok↑ og tegn↑)"
            elif tok_ratio < 0.95:
                conclusion = "da billigere end en"
            else:
                conclusion = "Ingen klar forskel"
        else:
            conclusion = "Utilstrækkeligt data"

        L.append(
            f"| {key} "
            + "".join(f"| {cpt[l]:.1f} " if cpt[l] is not None else "| — " for l in LANGCOST_LANGS)
            + f"| {conclusion} |"
        )
    L.append("")
    L.append(
        "_Note: Kinesisk bruger typisk 1,5–3 tegn per token (effektiv tokenisering af "
        "unicode-tegn), mod 3–6 tegn per token for latin-baserede sprog. "
        "En lav zh tg/tok kan afspejle tokenizerens effektivitet, ikke kortere tænkning._"
    )
    L.append("")

    # ─── 5. Krydstabel per model + poolet total ───
    L.append("## 5. Krydstabel: prompt_lang × primary_trace_language")
    L.append("")
    L.append(
        "Substrat-kontrollen: får dansk prompt modellen til at tænke på dansk, "
        "eller falder den tilbage på engelsk? Her vises mønstret per model "
        "(n=6 per celle — ét kald per opgave), så modelspecifik adfærd er synlig."
    )
    L.append("")
    L.append("| Model | da→ (6 kald) | en→ (6 kald) | zh→ (6 kald) |")
    L.append("|---|---|---|---|")

    pooled: dict[str, Counter] = {lang: Counter() for lang in LANGCOST_LANGS}

    for key in LANGCOST_MODELS:
        cells = []
        for pl in LANGCOST_LANGS:
            recs = by_model_lang[(key, pl)]
            counts: Counter = Counter(
                (r.get("language_metric") or {}).get("primary_trace_language") or "?"
                for r in recs
            )
            pooled[pl].update(counts)
            parts = " ".join(
                f"{tl}:{c}" for tl, c in sorted(counts.items(), key=lambda x: -x[1])
            )
            cells.append(parts)
        L.append(f"| {key} | {cells[0]} | {cells[1]} | {cells[2]} |")

    L.append("")
    L.append("**Poolet total (alle modeller, alle 6 opgaver, n=30 per sprog):**")
    pool_parts = []
    for pl in LANGCOST_LANGS:
        total = sum(pooled[pl].values())
        top = sorted(pooled[pl].items(), key=lambda x: -x[1])
        pct_str = " / ".join(
            f"{tl}: {c/total*100:.0f}%"
            for tl, c in top
        )
        pool_parts.append(f"{pl}→ {pct_str}")
    L.append("  \n".join(pool_parts))
    L.append("")

    # ─── 6. Pris per sprog per model ───
    L.append("## 6. Pris per sprog per model (USD)")
    L.append("")
    L.append("Samlet pris over alle 6 opgaver, fordelt på sprog og model.")
    L.append("")
    L.append("| Model | da | en | zh | Total |")
    L.append("|---|---:|---:|---:|---:|")
    for key in LANGCOST_MODELS:
        costs = {
            lang: sum(r["cost_usd"] for r in by_model_lang[(key, lang)])
            for lang in LANGCOST_LANGS
        }
        total = sum(costs.values())
        L.append(
            f"| {key} "
            + "".join(f"| ${costs[l]:.5f} " for l in LANGCOST_LANGS)
            + f"| ${total:.5f} |"
        )
    lang_totals = {
        lang: sum(r["cost_usd"] for r in rows if r["prompt_lang"] == lang)
        for lang in LANGCOST_LANGS
    }
    grand_total = sum(lang_totals.values())
    L.append(
        "| **Total** "
        + "".join(f"| **${lang_totals[l]:.5f}** " for l in LANGCOST_LANGS)
        + f"| **${grand_total:.5f}** |"
    )
    L.append("")

    # ─── Footer ───
    L.append("---")
    L.append("")
    L.append(f"*Auto-genereret af `run.py --langcost-report`. Kilde: `{source_run_id}`.*")

    syntese_dir.mkdir(parents=True, exist_ok=True)
    report_path = syntese_dir / "sprogets_pris_data.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"  Report written → {report_path}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Tool-offload experiment (--tools)
# ---------------------------------------------------------------------------

# Same 8 models as the Phase 1 full run — provider tool-call support is
# discovered empirically at run time (ToolsNotSupportedError -> n/a row),
# not hand-curated here.
TOOLS_MODELS: list[str] = FULL_MODEL_ORDER

# P1, P2, P9, P10 are open/no-facit tasks — no tool call is expected. A tool
# call here is flagged as mis-routing, not suppressed.
TOOLS_CONTROL_GROUP: set[str] = {"P1", "P2", "P9", "P10"}

KNOWN_TOOL_NAMES: set[str] = {"python_exec", "web_search"}


def _tool_names_used(response) -> tuple[str, ...]:
    return tuple(sorted({tc["name"] for tc in response.tool_calls}))


def run_tools(allow_direct: bool = False) -> int:
    """
    Tool-offload experiment: two arms (baseline, tools) per (model, prompt),
    run fresh in the same invocation so there is no cross-run drift.

    Arm A (baseline): identical to the existing --full call() path.
    Arm B (tools): python_exec + web_search offered; harness executes every
    tool call itself; a tools-omitted continuation call forces a final answer.

    Closed-book rule is unchanged — only `prompt` is ever sent to the model.
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")
    prompts = load_prompts()

    all_prompt_ids = sorted(prompts.keys(), key=lambda p: int(p[1:]))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_tools"
    traces_dir = TOOLS_DIR / f"{run_id}_traces"

    model_keys = [
        k for k in TOOLS_MODELS
        if k in panel and panel[k].get("role") in SMOKE_ROLES
    ]

    tool_defs = available_tool_defs()
    tools_available_names = [t["name"] for t in tool_defs]

    n_calls = len(all_prompt_ids) * len(model_keys) * 2
    print(f"\n{'═'*120}")
    print(f"  Tool-Offload Experiment (--tools)   run_id={run_id}")
    print(
        f"  Prompts: {', '.join(all_prompt_ids)}  ({len(all_prompt_ids)} × {len(model_keys)} models × 2 arms"
        f" = {n_calls} rows)"
    )
    print(f"  Tools offered to Arm B: {', '.join(tools_available_names)}")
    if not search_available():
        print(f"  !! web_search PENDING — SEARCH_API_KEY not set. repl-offload measurement proceeds unaffected.")
    print(f"  reasoning_effort={reasoning_effort!r}  |  closed-book rule unchanged (facit never sent)")
    print(f"{'═'*120}")

    resolved = resolve_models(panel, model_keys)
    hard_errors = print_resolution_table(resolved)
    assert_no_silent_direct_route(panel, model_keys, allow_direct)
    if hard_errors > 0:
        print(f"\n  !! {hard_errors} model(s) not resolved. Fix panel.yaml.\n")
        return 1

    has_failure = False
    agg: list[dict] = []              # one entry per (model, pid, arm) that produced a response
    unexpected_serverside: list[dict] = []  # raw tool events whose name we never declared
    control_group_hits: list[dict] = []     # tool calls on no-facit control prompts

    col_hdr = (
        f"  {'Model':<22} {'Arm':<9} {'Inp':>6} {'Reas':>7} {'Out':>6}  "
        f"{'Cost($)':>10}  {'#API':>4}  {'Tools called':<28} {'Status'}"
    )
    col_sep = "  " + "─" * 116

    for pid in all_prompt_ids:
        p = prompts[pid]
        assert "facit" not in p, (
            f"CRITICAL SECURITY VIOLATION: facit in request-path object for {pid}"
        )
        prompt_text: str = p["prompt"]
        p_type = p.get("type", "?")
        p_load = p.get("reasoning_load", "?")
        is_control = pid in TOOLS_CONTROL_GROUP

        print(f"\n{'═'*120}")
        print(f"  [{pid}]  {p_type}  (load: {p_load})" + ("  [control: no tool call expected]" if is_control else ""))
        print(f"  {prompt_text[:110].strip()!r}")
        print(f"{'═'*120}")
        print(col_hdr)
        print(col_sep)

        for key in model_keys:
            cfg = panel[key]
            provider = cfg["provider"]
            thinking_budget: int = cfg.get("thinking_budget", 4096)

            cls = PROVIDER_MAP.get(provider)
            if cls is None:
                print(f"  {key:<22} SKIPPED (unknown provider: {provider})")
                continue

            try:
                adapter = cls(key, cfg)
            except CredentialMissingError as e:
                print(f"  {key:<22} SKIPPED — {e}")
                continue

            # ---------------- Arm A: baseline ----------------
            try:
                resp_a = adapter.call(
                    prompt_text,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                )
            except AdapterError as e:
                print(f"  {key:<22} {'baseline':<9} ERROR — {e}")
                has_failure = True
                continue
            except Exception as e:
                print(f"  {key:<22} {'baseline':<9} ERROR — unexpected: {e}")
                has_failure = True
                continue

            account_a = build_account(resp_a)
            cost_a, snapshot_date = compute_cost(key, account_a)

            save_tools_result(
                run_id=run_id,
                model_key=key,
                prompt_id=pid,
                arm="baseline",
                status="ok",
                response=resp_a,
                account=account_a,
                cost_usd=cost_a,
                pricing_snapshot_date=snapshot_date,
                thinking_budget=thinking_budget,
                reasoning_effort=reasoning_effort,
                tools_available=[],
                extra={"prompt_type": p_type, "reasoning_load": p_load, "is_control": is_control},
            )
            save_tools_trace(
                traces_dir=traces_dir, model_key=key, prompt_id=pid, arm="baseline",
                prompt_text=prompt_text, response=resp_a, status="ok",
            )
            agg.append({
                "pid": pid, "model_key": key, "arm": "baseline", "is_control": is_control,
                "input": account_a.input_tokens, "reasoning": account_a.reasoning_tokens,
                "output": account_a.output_tokens, "cost_usd": cost_a, "tools_used": (),
            })
            print(
                f"  {key:<22} {'baseline':<9} {account_a.input_tokens:>6} {account_a.reasoning_tokens:>7}"
                f" {account_a.output_tokens:>6}  ${cost_a:>9.5f}  {1:>4}  {'—':<28} ok"
            )

            # ---------------- Arm B: tools ----------------
            try:
                resp_b = adapter.call_with_tools(
                    prompt_text,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                )
            except ToolsNotSupportedError as e:
                save_tools_result(
                    run_id=run_id, model_key=key, prompt_id=pid, arm="tools",
                    status="n/a_no_tool_support", response=None, account=None, cost_usd=None,
                    pricing_snapshot_date=snapshot_date, thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort, tools_available=tools_available_names,
                    extra={"prompt_type": p_type, "reasoning_load": p_load, "is_control": is_control, "error": str(e)},
                )
                print(f"  {key:<22} {'tools':<9} {'n/a':>6} {'n/a':>7} {'n/a':>6}  {'—':>10}  {'—':>4}  {'—':<28} n/a (no tool support)")
                continue
            except NotImplementedError:
                save_tools_result(
                    run_id=run_id, model_key=key, prompt_id=pid, arm="tools",
                    status="n/a_no_tool_support", response=None, account=None, cost_usd=None,
                    pricing_snapshot_date=snapshot_date, thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort, tools_available=tools_available_names,
                    extra={"prompt_type": p_type, "reasoning_load": p_load, "is_control": is_control,
                           "error": "call_with_tools not implemented for this adapter"},
                )
                print(f"  {key:<22} {'tools':<9} SKIPPED — call_with_tools not implemented")
                continue
            except AdapterError as e:
                save_tools_result(
                    run_id=run_id, model_key=key, prompt_id=pid, arm="tools",
                    status="error", response=None, account=None, cost_usd=None,
                    pricing_snapshot_date=snapshot_date, thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort, tools_available=tools_available_names,
                    extra={"prompt_type": p_type, "reasoning_load": p_load, "is_control": is_control, "error": str(e)},
                )
                print(f"  {key:<22} {'tools':<9} ERROR — {e}")
                has_failure = True
                continue
            except Exception as e:
                print(f"  {key:<22} {'tools':<9} ERROR — unexpected: {e}")
                has_failure = True
                continue

            account_b = build_account(resp_b)
            cost_b, snapshot_date_b = compute_cost(key, account_b)

            save_tools_result(
                run_id=run_id,
                model_key=key,
                prompt_id=pid,
                arm="tools",
                status="ok",
                response=resp_b,
                account=account_b,
                cost_usd=cost_b,
                pricing_snapshot_date=snapshot_date_b,
                thinking_budget=thinking_budget,
                reasoning_effort=reasoning_effort,
                tools_available=tools_available_names,
                extra={"prompt_type": p_type, "reasoning_load": p_load, "is_control": is_control},
            )
            save_tools_trace(
                traces_dir=traces_dir, model_key=key, prompt_id=pid, arm="tools",
                prompt_text=prompt_text, response=resp_b, status="ok",
            )

            tools_used = _tool_names_used(resp_b)
            agg.append({
                "pid": pid, "model_key": key, "arm": "tools", "is_control": is_control,
                "input": account_b.input_tokens, "reasoning": account_b.reasoning_tokens,
                "output": account_b.output_tokens, "cost_usd": cost_b, "tools_used": tools_used,
            })

            if is_control and tools_used:
                control_group_hits.append({"pid": pid, "model_key": key, "tools_used": tools_used})

            for ev in resp_b.raw_tool_events:
                if ev.get("name") not in KNOWN_TOOL_NAMES:
                    unexpected_serverside.append({"pid": pid, "model_key": key, "event": ev})

            tools_str = ", ".join(tools_used) if tools_used else "—"
            print(
                f"  {key:<22} {'tools':<9} {account_b.input_tokens:>6} {account_b.reasoning_tokens:>7}"
                f" {account_b.output_tokens:>6}  ${cost_b:>9.5f}  {resp_b.n_api_calls:>4}  {tools_str:<28} ok"
            )

        print(col_sep)

    # ─────────────────────────────────────────────────────────────
    # AGGREGATE 1 — Which model called which tool, per prompt
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print(f"  AGGREGATE 1 — Tool usage matrix (Arm B), per prompt")
    print(f"  (— = answered directly, n/a = tool-calling not supported by this model/endpoint)")
    print(f"{'═'*100}")
    tools_rows = {(r["pid"], r["model_key"]): r for r in agg if r["arm"] == "tools"}
    abbrevs = {
        "deepseek_v4": "deepseek", "glm_5_2": "glm", "kimi_k2_7": "kimi", "gemma_4": "gemma",
        "claude_sonnet_4_6": "claude", "gpt_5_5": "gpt", "opus_4_8": "opus", "mistral_medium_3_5": "mistral",
    }
    hdr_cells = "  ".join(f"{abbrevs.get(k, k[:8]):<10}" for k in model_keys)
    print(f"  {'Prompt':<8} {hdr_cells}")
    print(f"  {'─'*(9 + 12*len(model_keys))}")
    for pid in all_prompt_ids:
        cells = []
        for k in model_keys:
            r = tools_rows.get((pid, k))
            if r is None:
                cells.append(f"{'n/a':<10}")
            elif r["tools_used"]:
                cells.append(f"{','.join(r['tools_used'])[:10]:<10}")
            else:
                cells.append(f"{'—':<10}")
        marker = "  ← control" if pid in TOOLS_CONTROL_GROUP else ""
        print(f"  {pid:<8} {'  '.join(cells)}{marker}")

    # ─────────────────────────────────────────────────────────────
    # AGGREGATE 2 — Arm A vs Arm B, per model (avg delta over shared prompts)
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*110}")
    print(f"  AGGREGATE 2 — Arm A (baseline) → Arm B (tools), per model")
    print(f"  Δ = tools − baseline, avg over prompts where BOTH arms produced a response (n/a excluded)")
    print(f"{'═'*110}")
    print(f"  {'Model':<22} {'n':>3}  {'ΔInput':>9}  {'ΔReasoning':>11}  {'ΔOutput':>9}  {'ΔCost($)':>10}")
    print(f"  {'─'*80}")
    for key in model_keys:
        base_by_pid = {r["pid"]: r for r in agg if r["arm"] == "baseline" and r["model_key"] == key}
        tool_by_pid = {r["pid"]: r for r in agg if r["arm"] == "tools" and r["model_key"] == key}
        shared = sorted(set(base_by_pid) & set(tool_by_pid), key=lambda p: int(p[1:]))
        if not shared:
            print(f"  {key:<22} {'—':>3}  {'—':>9}  {'—':>11}  {'—':>9}  {'—':>10}")
            continue
        d_in = [tool_by_pid[p]["input"] - base_by_pid[p]["input"] for p in shared]
        d_reas = [tool_by_pid[p]["reasoning"] - base_by_pid[p]["reasoning"] for p in shared]
        d_out = [tool_by_pid[p]["output"] - base_by_pid[p]["output"] for p in shared]
        d_cost = [tool_by_pid[p]["cost_usd"] - base_by_pid[p]["cost_usd"] for p in shared]
        n = len(shared)
        print(
            f"  {key:<22} {n:>3}  {sum(d_in)/n:>+9.1f}  {sum(d_reas)/n:>+11.1f}  "
            f"{sum(d_out)/n:>+9.1f}  {sum(d_cost)/n:>+10.5f}"
        )
    print(f"  {'─'*80}")

    # ─────────────────────────────────────────────────────────────
    # AGGREGATE 3 — Same delta, pooled and split by which tool was actually called
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*110}")
    print(f"  AGGREGATE 3 — Δ (tools − baseline), pooled across models, split by tool actually called")
    print(f"{'═'*110}")
    print(f"  {'Group':<26} {'n':>4}  {'ΔInput':>9}  {'ΔReasoning':>11}  {'ΔOutput':>9}  {'ΔCost($)':>10}")
    print(f"  {'─'*76}")

    def _group_label(tools_used: tuple) -> str:
        if not tools_used:
            return "no tool called"
        if set(tools_used) == {"python_exec"}:
            return "repl only"
        if set(tools_used) == {"web_search"}:
            return "search only"
        return "repl + search"

    pooled_pairs: dict[str, list[dict]] = {}
    for key in model_keys:
        base_by_pid = {r["pid"]: r for r in agg if r["arm"] == "baseline" and r["model_key"] == key}
        tool_by_pid = {r["pid"]: r for r in agg if r["arm"] == "tools" and r["model_key"] == key}
        for pid in set(base_by_pid) & set(tool_by_pid):
            group = _group_label(tool_by_pid[pid]["tools_used"])
            pooled_pairs.setdefault(group, []).append({
                "d_in": tool_by_pid[pid]["input"] - base_by_pid[pid]["input"],
                "d_reas": tool_by_pid[pid]["reasoning"] - base_by_pid[pid]["reasoning"],
                "d_out": tool_by_pid[pid]["output"] - base_by_pid[pid]["output"],
                "d_cost": tool_by_pid[pid]["cost_usd"] - base_by_pid[pid]["cost_usd"],
            })

    for group in ("repl only", "search only", "repl + search", "no tool called"):
        rows = pooled_pairs.get(group, [])
        if not rows:
            print(f"  {group:<26} {0:>4}  {'—':>9}  {'—':>11}  {'—':>9}  {'—':>10}")
            continue
        n = len(rows)
        print(
            f"  {group:<26} {n:>4}  "
            f"{sum(r['d_in'] for r in rows)/n:>+9.1f}  "
            f"{sum(r['d_reas'] for r in rows)/n:>+11.1f}  "
            f"{sum(r['d_out'] for r in rows)/n:>+9.1f}  "
            f"{sum(r['d_cost'] for r in rows)/n:>+10.5f}"
        )
    print(f"  {'─'*76}")

    # ─────────────────────────────────────────────────────────────
    # Control-group + unexpected server-side tool findings
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print(f"  CONTROL GROUP (P1/P2/P9/P10 — no tool call expected)")
    print(f"{'═'*100}")
    if control_group_hits:
        print(f"  MIS-ROUTING DETECTED — a model called a tool on an open/no-facit prompt:")
        for h in control_group_hits:
            print(f"    {h['model_key']:<22} {h['pid']:<6} tools_used={h['tools_used']}")
    else:
        print(f"  No tool calls on the control prompts — as expected.")

    print(f"\n{'═'*100}")
    print(f"  UNEXPECTED TOOL EVENTS (name not in {sorted(KNOWN_TOOL_NAMES)} — e.g. provider server-side tools)")
    print(f"{'═'*100}")
    if unexpected_serverside:
        for u in unexpected_serverside:
            print(f"    {u['model_key']:<22} {u['pid']:<6} {u['event']}")
    else:
        print(f"  None observed.")

    # ─────────────────────────────────────────────────────────────
    # Footer
    # ─────────────────────────────────────────────────────────────
    status_line = "ALL CALLS COMPLETED" if not has_failure else "ERRORS DETECTED — see rows above"
    print(f"\n{'═'*100}")
    print(f"  {status_line}")
    print(f"  Structured records  → results/tools/{run_id}.jsonl")
    print(f"  Raw traces          → results/tools/{run_id}_traces/<model>_<prompt>_<arm>.txt")
    print()
    return 1 if has_failure else 0


# ---------------------------------------------------------------------------
# Variance repro run (--variance)
# ---------------------------------------------------------------------------

# Same 8 models, pinned to fully-dated openrouter_model_id strings in panel.yaml
# (set 2026-07-08 to reproduce June exactly after the tools-run alias contamination).
VARIANCE_MODELS: list[str] = FULL_MODEL_ORDER

# Substrings that mark an API error as "this exact pinned string is dead", not a
# transient failure. Matched case-insensitively against the raised error text.
DEAD_PIN_MARKERS: tuple[str, ...] = (
    "not found",
    "not a valid model",
    "does not exist",
    "no endpoints found",
    "invalid model",
    "no allowed providers",
    "model_not_found",
)


def _classify_pin_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(marker in msg for marker in DEAD_PIN_MARKERS):
        return "dead_pin"
    return "error"


def run_variance(passes: int = 2, allow_direct: bool = False) -> int:
    """
    Pure variance repro: baseline only (no tools), all 8 models pinned to fully
    dated openrouter_model_id strings (identical to June), run PASSES times.

    Forces the OpenRouter route for Anthropic models by temporarily removing
    ANTHROPIC_API_KEY from the environment for the duration of this run — the
    same routing June used for Sonnet, and the routing the tools-run silently
    dropped (which is exactly the contamination this run exists to undo).

    No pre-flight catalog check gates this run — dated snapshot slugs may not
    appear in OpenRouter's general /models listing even when callable. Instead:
    a pinned string that errors with a "model not found"-shaped message is
    reported loudly as dead and skipped for the rest of the run; anything else
    is a transient error and does not disable the model.
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")
    prompts = load_prompts()

    all_prompt_ids = sorted(prompts.keys(), key=lambda p: int(p[1:]))

    model_keys = [
        k for k in VARIANCE_MODELS
        if k in panel and panel[k].get("role") in SMOKE_ROLES
    ]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_variance"

    n_calls = len(all_prompt_ids) * len(model_keys) * passes
    print(f"\n{'═'*120}")
    print(f"  Variance Repro Run (--variance)   run_id={run_id}")
    print(
        f"  Prompts: {len(all_prompt_ids)} × models: {len(model_keys)} × passes: {passes}"
        f" = {n_calls} calls  |  baseline only, closed-book, no tools"
    )
    print(f"  reasoning_effort={reasoning_effort!r}")
    print(f"  Pinned versions (panel.yaml openrouter_model_id):")
    for key in model_keys:
        print(f"    {key:<22} {panel[key].get('openrouter_model_id')}")
    print(f"{'═'*120}")

    # Force the OpenRouter route for Anthropic models (claude_sonnet_4_6, opus_4_8) —
    # same route June used. Direct ANTHROPIC_API_KEY would silently reintroduce the
    # undated-alias contamination this run exists to eliminate.
    saved_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    if saved_anthropic_key:
        print(f"  !! ANTHROPIC_API_KEY temporarily removed from env — forcing OpenRouter route for this run.")

    try:
        resolved = resolve_models(panel, model_keys)
        print_resolution_table(resolved)
        assert_no_silent_direct_route(panel, model_keys, allow_direct)
        print(
            f"\n  NOTE: catalog mismatches above are not a gate — dated snapshot slugs"
            f" can be absent from OpenRouter's /models listing while still callable."
            f" resp.model (model_version) is NOT proof either way: diagnostics on"
            f" 2026-07-14 showed OpenRouter echoes the same canonical label whether the"
            f" dated pin or the undated slug is sent. The actual gate is"
            f" assert_model_pin_honored() (adapters/base.py), which checks the request"
            f" variable itself (request_model_id) against panel.yaml, before the call."
        )

        has_failure = False
        dead_pins: dict[str, str] = {}   # model_key -> error text
        rows: list[dict] = []            # for the closing summary table

        col_hdr = (
            f"  {'Model':<22} {'Pass':>4} {'Inp':>6} {'Reas':>7} {'Out':>6}  "
            f"{'Cost($)':>10}  {'ServedBy':<14} {'ModelVersion':<42} Status"
        )
        col_sep = "  " + "─" * 118

        for pass_index in range(1, passes + 1):
            for pid in all_prompt_ids:
                p = prompts[pid]
                assert "facit" not in p, (
                    f"CRITICAL SECURITY VIOLATION: facit in request-path object for {pid}"
                )
                prompt_text: str = p["prompt"]

                print(f"\n{'═'*120}")
                print(f"  Pass {pass_index}/{passes}  [{pid}]  {prompt_text[:90].strip()!r}")
                print(f"{'═'*120}")
                print(col_hdr)
                print(col_sep)

                for key in model_keys:
                    if key in dead_pins:
                        print(f"  {key:<22} {pass_index:>4}  SKIPPED — dead pin ({dead_pins[key][:60]})")
                        continue

                    cfg = panel[key]
                    provider = cfg["provider"]
                    thinking_budget: int = cfg.get("thinking_budget", 4096)
                    pinned_id = cfg.get("openrouter_model_id", cfg.get("model_id"))

                    cls = PROVIDER_MAP.get(provider)
                    if cls is None:
                        print(f"  {key:<22} SKIPPED (unknown provider: {provider})")
                        continue

                    try:
                        adapter = cls(key, cfg)
                    except CredentialMissingError as e:
                        print(f"  {key:<22} SKIPPED — {e}")
                        continue

                    try:
                        response = adapter.call(
                            prompt_text,
                            thinking_budget=thinking_budget,
                            reasoning_effort=reasoning_effort,
                        )
                    except AdapterError as e:
                        classification = _classify_pin_error(e)
                        if classification == "dead_pin":
                            dead_pins[key] = str(e)
                            print(f"\n  !!!!!! DEAD PIN !!!!!!  {key} — {pinned_id!r} is not callable:")
                            print(f"  !!!!!!                  {e}")
                            print(f"  !!!!!!                  No fallback to alias. Skipping {key} for the rest of this run.\n")
                        else:
                            print(f"  {key:<22} {pass_index:>4}  ERROR — {e}")
                        save_variance_result(
                            run_id=run_id, model_key=key, prompt_id=pid, pass_index=pass_index,
                            pinned_model_id=pinned_id, status=classification, response=None, account=None,
                            cost_usd=None, pricing_snapshot_date=None, thinking_budget=thinking_budget,
                            reasoning_effort=reasoning_effort, extra={"error": str(e)},
                        )
                        has_failure = True
                        continue
                    except Exception as e:
                        print(f"  {key:<22} {pass_index:>4}  ERROR — unexpected: {e}")
                        save_variance_result(
                            run_id=run_id, model_key=key, prompt_id=pid, pass_index=pass_index,
                            pinned_model_id=pinned_id, status="error", response=None, account=None,
                            cost_usd=None, pricing_snapshot_date=None, thinking_budget=thinking_budget,
                            reasoning_effort=reasoning_effort, extra={"error": str(e)},
                        )
                        has_failure = True
                        continue

                    account = build_account(response)
                    cost_usd, snapshot_date = compute_cost(key, account)

                    version_flag = "" if response.model_version == pinned_id else "  ← DRIFT vs pin"
                    if version_flag:
                        has_failure = True

                    save_variance_result(
                        run_id=run_id, model_key=key, prompt_id=pid, pass_index=pass_index,
                        pinned_model_id=pinned_id, status="ok", response=response, account=account,
                        cost_usd=cost_usd, pricing_snapshot_date=snapshot_date,
                        thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
                    )
                    rows.append({
                        "model_key": key, "pass_index": pass_index, "pid": pid,
                        "reasoning": account.reasoning_tokens, "reasoning_source": response.reasoning_source,
                        "served_by": response.served_by, "model_version": response.model_version,
                        "cost_usd": cost_usd,
                    })

                    print(
                        f"  {key:<22} {pass_index:>4} {account.input_tokens:>6} {account.reasoning_tokens:>7}"
                        f" {account.output_tokens:>6}  ${cost_usd:>9.5f}  {str(response.served_by):<14} "
                        f"{response.model_version:<42} ok{version_flag}"
                    )

                print(col_sep)

        # ─────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────
        print(f"\n{'═'*100}")
        print(f"  SUMMARY")
        print(f"{'═'*100}")
        if dead_pins:
            print(f"  DEAD PINS (never resolved, no fallback used):")
            for k, err in dead_pins.items():
                print(f"    {k:<22} {panel[k].get('openrouter_model_id')}")
                print(f"      {err}")
        else:
            print(f"  All {len(model_keys)} pinned strings resolved and were callable.")

        served_by_seen: dict[str, set] = {}
        for r in rows:
            served_by_seen.setdefault(r["model_key"], set()).add(r["served_by"])
        print(f"\n  Underlying backend(s) reported by OpenRouter per model (fingerprint, not a version guarantee):")
        for key in model_keys:
            backends = served_by_seen.get(key)
            print(f"    {key:<22} {sorted(b for b in backends if b) if backends else '—'}")

        text_estimate_cells = [r for r in rows if r["reasoning_source"] != "api"]
        if text_estimate_cells:
            print(f"\n  {len(text_estimate_cells)} row(s) used reasoning_source=text_estimate (not comparable to API-reported counts):")
            for r in text_estimate_cells:
                print(f"    {r['model_key']:<22} pass={r['pass_index']} {r['pid']}")

        status_line = "ALL CALLS COMPLETED" if not has_failure else "ISSUES DETECTED — see DEAD PINS / DRIFT / ERROR rows above"
        print(f"\n  {status_line}")
        print(f"  Structured records → results/variance/{run_id}.jsonl")
        print()
        return 1 if has_failure else 0

    finally:
        if saved_anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_anthropic_key


# ---------------------------------------------------------------------------
# Invited/forced tool-offload re-run (--tools3)
# ---------------------------------------------------------------------------
#
# Replaces the first --tools measurement (20260708T112015): that run only
# observed SPONTANEOUS tool use (schema declared, tool_choice left on "auto",
# no textual invitation). Gemma and Mistral never called anything there, but
# both are documented to support function calling — silence is not evidence
# of incapacity when the setup itself may be weak. This run separates three
# questions: CAN the model tool-call at all (forced), WILL it when invited
# realistically (invited_auto), and is any of it just our own routing bug.

TOOLS3_TOOL_RELEVANT_PROMPTS: dict[str, str] = {
    # prompt_id -> which tool it plausibly calls for. Confirmed against
    # data/prompts.yaml types: P3/P5/P6/P7/P8 are computation/code/structure
    # (repl-shaped); P4 is da_legal/very_high citing specific §-rules a model
    # may want to verify (search-shaped) — also the prompt Kimi searched on
    # in the first --tools run.
    "P3": "repl", "P5": "repl", "P6": "repl", "P7": "repl", "P8": "repl",
    "P4": "search",
}
TOOLS3_CONTROL_GROUP: set[str] = {"P1", "P2", "P9", "P10"}

TOOLS3_INVITATION = (
    "\n\nDu har adgang til to værktøjer: python_exec (kør Python for eksakt "
    "beregning) og web_search (slå fakta op). Brug dem hvis de hjælper med at "
    "svare korrekt."
)

# Baseline is REUSED, not re-run — same pinned-version pooling rule as --variance:
# exact model_version match to panel.yaml's openrouter_model_id, reasoning_source=="api".
TOOLS3_BASELINE_SOURCES: tuple[tuple[Path, Optional[str]], ...] = (
    (RESULTS_DIR / "full" / "combined_8models_full.jsonl", None),
    (TOOLS_DIR / "20260708T112015_tools.jsonl", "baseline"),
    (VARIANCE_DIR / "20260708T143500_variance.jsonl", None),
)


def _load_pooled_baseline() -> dict[tuple[str, str], list[dict]]:
    """
    (model_key, prompt_id) -> list of {input, reasoning, output, cost_usd},
    pooled across June + the first --tools run's baseline arm + both
    --variance passes — the exact same multi-source pooling the --variance
    report used. This IS the baseline for --tools3; it is not re-run.
    """
    import json
    from collections import defaultdict

    panel = load_panel()
    pins = {k: panel[k].get("openrouter_model_id") for k in FULL_MODEL_ORDER if k in panel}
    pooled: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for path, arm_filter in TOOLS3_BASELINE_SOURCES:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                mk = r.get("model_key")
                if mk not in pins or r.get("model_version") != pins.get(mk):
                    continue
                if arm_filter is not None and r.get("arm") != arm_filter:
                    continue
                tok = r.get("tokens")
                if not tok or tok.get("reasoning_source") != "api":
                    continue
                pooled[(mk, r["prompt_id"])].append({
                    "input": tok["input"], "reasoning": tok["reasoning"],
                    "output": tok["output"], "cost_usd": r.get("cost_usd") or 0.0,
                })
    return pooled


def _relative_noise_floor(pooled: dict[tuple[str, str], list[dict]]) -> dict[str, float]:
    """Per-model median per-cell coefficient of variation ((max-min)/median) in
    reasoning tokens, from the pooled baseline — the threshold --tools3 tests against."""
    import statistics
    from collections import defaultdict

    by_model: dict[str, list[float]] = defaultdict(list)
    for (mk, _pid), rows in pooled.items():
        vals = [r["reasoning"] for r in rows]
        if len(vals) < 2:
            continue
        med = statistics.median(vals)
        if med > 0:
            by_model[mk].append((max(vals) - min(vals)) / med)
    return {mk: statistics.median(cvs) for mk, cvs in by_model.items() if cvs}


def _tools3_smoke_test(panel: dict, prompts: dict, reasoning_effort: str) -> bool:
    """
    Mandatory gate: does tool_choice="required" actually get enforced for
    Gemma and Mistral through our OpenRouter route? If a forced call comes
    back with zero tool_calls, that is our translation/routing failing, not
    the model declining — the caller must stop the whole run on failure.
    """
    print(f"\n{'═'*100}")
    print(f"  SMOKE TEST — tool_choice=required on Gemma + Mistral (gate before the full run)")
    print(f"{'═'*100}")
    ok = True
    for key in ("gemma_4", "mistral_medium_3_5"):
        cfg = panel[key]
        cls = PROVIDER_MAP[cfg["provider"]]
        thinking_budget = cfg.get("thinking_budget", 4096)
        prompt_text = prompts["P5"]["prompt"] + TOOLS3_INVITATION
        try:
            adapter = cls(key, cfg)
            resp = adapter.call_with_tools(
                prompt_text, thinking_budget=thinking_budget,
                reasoning_effort=reasoning_effort, tool_choice="required",
            )
        except Exception as e:
            print(f"  {key:<22} ERROR — {e}")
            ok = False
            continue
        called = bool(resp.tool_calls)
        status = "OK — required respected" if called else "FAIL — required NOT respected (adapter/routing defect)"
        print(f"  {key:<22} tool_calls={resp.tool_calls}  served_by={resp.served_by}  {status}")
        if not called:
            ok = False
    print(f"{'═'*100}")
    return ok


def run_tools3(repeats: int = 5, allow_direct: bool = False) -> int:
    """
    Three-condition tool-offload re-run on pinned dated versions:
      - baseline: REUSED from --variance + June + first-run (not re-run here).
      - invited_auto: prompt explicitly invites both tools, tool_choice="auto".
      - forced: tool_choice="required" on the 6 tool-relevant prompts only.

    Gate: a mandatory smoke test on Gemma/Mistral (forced) must pass before
    anything else runs — if required isn't enforced there, it's our adapter
    translation, not model capability, and the run stops.
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")
    prompts = load_prompts()

    model_keys = [
        k for k in FULL_MODEL_ORDER
        if k in panel and panel[k].get("role") in SMOKE_ROLES
    ]

    # Force OpenRouter route for Anthropic models — same mechanism as --variance.
    saved_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    if saved_anthropic_key:
        print(f"\n  !! ANTHROPIC_API_KEY temporarily removed from env — forcing OpenRouter route for this run.")

    try:
        if not _tools3_smoke_test(panel, prompts, reasoning_effort):
            print(
                "\n  STOP — tool_choice=required was not respected by Gemma and/or Mistral "
                "through the current route/adapter. This is being treated as OUR translation "
                "failing, not a model capability finding. Fix the adapter/schema before running "
                "the rest of --tools3.\n"
            )
            return 1

        pooled_baseline = _load_pooled_baseline()
        noise_floor = _relative_noise_floor(pooled_baseline)

        tool_defs = available_tool_defs()
        tools_available_names = [t["name"] for t in tool_defs]

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_tools3"
        traces_dir = TOOLS3_DIR / f"{run_id}_traces"

        # Cell list: (model, prompt, condition)
        cells: list[tuple[str, str, str]] = []
        for mk in model_keys:
            for pid in TOOLS3_TOOL_RELEVANT_PROMPTS:
                cells.append((mk, pid, "invited_auto"))
                cells.append((mk, pid, "forced"))
            for pid in TOOLS3_CONTROL_GROUP:
                cells.append((mk, pid, "invited_auto"))
        cells.sort(key=lambda c: (int(c[1][1:]), c[2], c[0]))

        n_calls = len(cells) * repeats
        print(f"\n{'═'*120}")
        print(f"  Invited/Forced Tool-Offload Re-run (--tools3)   run_id={run_id}")
        print(f"  Tool-relevant prompts: {TOOLS3_TOOL_RELEVANT_PROMPTS}")
        print(f"  Control prompts (invited_auto only): {sorted(TOOLS3_CONTROL_GROUP)}")
        print(f"  Cells: {len(cells)}  ×  repeats: {repeats} = {n_calls} rows  |  baseline REUSED, not re-run")
        print(f"  Tools offered: {', '.join(tools_available_names)}")
        if not search_available():
            print(f"  !! web_search PENDING — SEARCH_API_KEY not set.")
        print(f"  Noise floor (relative, from pooled baseline): {{{', '.join(f'{k}: {v:.0%}' for k, v in sorted(noise_floor.items(), key=lambda x: x[1]))}}}")
        print(f"  Versions: panel.yaml pinned dated strings (identical to --variance). No alias.")
        print(f"{'═'*120}")

        resolved = resolve_models(panel, model_keys)
        print_resolution_table(resolved)
        assert_no_silent_direct_route(panel, model_keys, allow_direct)
        print(
            f"\n  NOTE: catalog mismatches are informational only — the real protection is"
            f" assert_model_pin_honored() checking request_model_id (the request variable"
            f" itself, NOT resp.model — see adapters/base.py) before every call, hard-"
            f" stopping on mismatch. A retired-pin silent-reroute incident on 2026-07-14"
            f" is why this exists; resp.model was proven that day to be identical whether"
            f" the dated pin or the undated slug is sent, so it cannot serve as this check.\n"
        )

        has_failure = False
        agg: list[dict] = []
        required_violations: list[dict] = []

        col_hdr = (
            f"  {'Pass':>4} {'Inp':>6} {'Reas':>7} {'Out':>6}  {'Cost($)':>10}  "
            f"{'ServedBy':<14} {'Tools called':<20} Status"
        )

        for (model_key, pid, condition) in cells:
            cfg = panel[model_key]
            provider = cfg["provider"]
            thinking_budget: int = cfg.get("thinking_budget", 4096)
            p = prompts[pid]
            assert "facit" not in p, (
                f"CRITICAL SECURITY VIOLATION: facit in request-path object for {pid}"
            )
            prompt_text: str = p["prompt"] + TOOLS3_INVITATION
            tool_choice = "auto" if condition == "invited_auto" else "required"

            cls = PROVIDER_MAP.get(provider)
            if cls is None:
                print(f"\n  {model_key} / {pid} / {condition}  SKIPPED (unknown provider: {provider})")
                continue

            print(f"\n{'═'*120}")
            print(f"  {model_key} / {pid} / {condition}  (tool_choice={tool_choice!r})")
            print(f"  {prompt_text[:100].strip()!r}")
            print(f"{'═'*120}")
            print(col_hdr)
            print("  " + "─" * 100)

            for pass_index in range(1, repeats + 1):
                try:
                    adapter = cls(model_key, cfg)
                except CredentialMissingError as e:
                    print(f"  {pass_index:>4}  SKIPPED — {e}")
                    continue

                try:
                    resp = adapter.call_with_tools(
                        prompt_text, thinking_budget=thinking_budget,
                        reasoning_effort=reasoning_effort, tool_choice=tool_choice,
                    )
                except ToolsNotSupportedError as e:
                    save_tools_result(
                        run_id=run_id, model_key=model_key, prompt_id=pid, arm=condition,
                        status="n/a_no_tool_support", response=None, account=None, cost_usd=None,
                        pricing_snapshot_date=None, thinking_budget=thinking_budget,
                        reasoning_effort=reasoning_effort, tools_available=tools_available_names,
                        extra={"pass_index": pass_index, "tool_choice_sent": tool_choice, "error": str(e)},
                        results_dir=TOOLS3_DIR,
                    )
                    print(f"  {pass_index:>4}  n/a — no tool support")
                    continue
                except AdapterError as e:
                    save_tools_result(
                        run_id=run_id, model_key=model_key, prompt_id=pid, arm=condition,
                        status="error", response=None, account=None, cost_usd=None,
                        pricing_snapshot_date=None, thinking_budget=thinking_budget,
                        reasoning_effort=reasoning_effort, tools_available=tools_available_names,
                        extra={"pass_index": pass_index, "tool_choice_sent": tool_choice, "error": str(e)},
                        results_dir=TOOLS3_DIR,
                    )
                    print(f"  {pass_index:>4}  ERROR — {e}")
                    has_failure = True
                    continue
                except Exception as e:
                    print(f"  {pass_index:>4}  ERROR — unexpected: {e}")
                    has_failure = True
                    continue

                account = build_account(resp)
                cost_usd, snapshot_date = compute_cost(model_key, account)
                tools_used = _tool_names_used(resp)

                save_tools_result(
                    run_id=run_id, model_key=model_key, prompt_id=pid, arm=condition, status="ok",
                    response=resp, account=account, cost_usd=cost_usd, pricing_snapshot_date=snapshot_date,
                    thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
                    tools_available=tools_available_names,
                    extra={"pass_index": pass_index, "served_by": resp.served_by, "tool_choice_sent": tool_choice},
                    results_dir=TOOLS3_DIR,
                )
                save_tools_trace(
                    traces_dir=traces_dir, model_key=model_key, prompt_id=pid, arm=condition,
                    prompt_text=prompt_text, response=resp, status="ok", pass_index=pass_index,
                )

                if condition == "forced" and not tools_used:
                    required_violations.append({"model_key": model_key, "pid": pid, "pass": pass_index})
                    print(f"  {pass_index:>4}  !!!! REQUIRED NOT RESPECTED — no tool called despite tool_choice=required !!!!")

                agg.append({
                    "model_key": model_key, "pid": pid, "condition": condition, "pass": pass_index,
                    "input": account.input_tokens, "reasoning": account.reasoning_tokens,
                    "output": account.output_tokens, "cost_usd": cost_usd, "tools_used": tools_used,
                })
                tools_str = ", ".join(tools_used) if tools_used else "—"
                print(
                    f"  {pass_index:>4} {account.input_tokens:>6} {account.reasoning_tokens:>7}"
                    f" {account.output_tokens:>6}  ${cost_usd:>9.5f}  {str(resp.served_by):<14} {tools_str:<20} ok"
                )

        # ═════════════════════════════════════════════════════════
        # REPORT
        # ═════════════════════════════════════════════════════════
        import statistics as _stats

        def _median(vals):
            return _stats.median(vals) if vals else None

        def _rows(mk, pid, condition):
            return [r for r in agg if r["model_key"] == mk and r["pid"] == pid and r["condition"] == condition]

        # --- 1. CAN vs WILL ---
        print(f"\n{'═'*100}")
        print(f"  1. KAN vs VIL — call rate per model, tool-relevant prompts only")
        print(f"{'═'*100}")
        print(f"  {'Model':<22} {'invited_auto (VIL)':>20}  {'forced (KAN)':>14}")
        for mk in model_keys:
            inv_rows = [r for r in agg if r["model_key"] == mk and r["condition"] == "invited_auto" and r["pid"] in TOOLS3_TOOL_RELEVANT_PROMPTS]
            forced_rows = [r for r in agg if r["model_key"] == mk and r["condition"] == "forced"]
            inv_rate = f"{sum(1 for r in inv_rows if r['tools_used'])}/{len(inv_rows)}" if inv_rows else "—"
            forced_rate = f"{sum(1 for r in forced_rows if r['tools_used'])}/{len(forced_rows)}" if forced_rows else "—"
            flag = "  <<<< check" if mk in ("gemma_4", "mistral_medium_3_5") else ""
            print(f"  {mk:<22} {inv_rate:>20}  {forced_rate:>14}{flag}")

        # --- 2. Offload economy (forced vs reused baseline), per tool-relevant cell ---
        print(f"\n{'═'*130}")
        print(f"  2. OFFLOAD ECONOMY — forced vs reused baseline, per (model, prompt)")
        print(f"{'═'*130}")
        print(
            f"  {'Model':<20} {'Pid':<4} {'BaseReas':>9} {'ForcedReas':>10} {'ΔReas':>8} {'ΔReas%':>8}  "
            f"{'ΔInput':>8} {'ΔOutput':>8} {'ΔCost':>9}  {'Decision':<28}"
        )
        print("  " + "─" * 126)
        cell_deltas: list[dict] = []
        for mk in model_keys:
            for pid in TOOLS3_TOOL_RELEVANT_PROMPTS:
                base_rows = pooled_baseline.get((mk, pid), [])
                forced_rows = _rows(mk, pid, "forced")
                if not base_rows or not forced_rows:
                    continue
                base_reas = _median([r["reasoning"] for r in base_rows])
                forced_reas = _median([r["reasoning"] for r in forced_rows])
                d_reas = forced_reas - base_reas
                d_reas_pct = (d_reas / base_reas) if base_reas > 0 else None
                d_in = _median([r["input"] for r in forced_rows]) - _median([r["input"] for r in base_rows])
                d_out = _median([r["output"] for r in forced_rows]) - _median([r["output"] for r in base_rows])
                d_cost = _median([r["cost_usd"] for r in forced_rows]) - _median([r["cost_usd"] for r in base_rows])
                nf = noise_floor.get(mk)
                if nf is not None and d_reas_pct is not None:
                    decision = f"EXCEEDS ({abs(d_reas_pct):.0%} > {nf:.0%})" if abs(d_reas_pct) > nf else f"within noise (<= {nf:.0%})"
                else:
                    decision = "no noise floor data"
                cell_deltas.append({
                    "model_key": mk, "pid": pid, "kind": TOOLS3_TOOL_RELEVANT_PROMPTS[pid],
                    "d_reas": d_reas, "d_reas_pct": d_reas_pct, "d_in": d_in, "d_out": d_out, "d_cost": d_cost,
                    "exceeds_noise": nf is not None and d_reas_pct is not None and abs(d_reas_pct) > nf,
                })
                pct_str = f"{d_reas_pct:+.0%}" if d_reas_pct is not None else "—"
                print(
                    f"  {mk:<20} {pid:<4} {base_reas:>9.1f} {forced_reas:>10.1f} {d_reas:>+8.1f} {pct_str:>8}  "
                    f"{d_in:>+8.1f} {d_out:>+8.1f} {d_cost:>+9.5f}  {decision:<28}"
                )

        # --- 3. Invitation's own effect on the control group ---
        print(f"\n{'═'*100}")
        print(f"  3. INVITATION'S OWN EFFECT — invited_auto vs baseline on control cells where NO tool was called")
        print(f"{'═'*100}")
        print(f"  {'Model':<22} {'n cells':>8}  {'Median ΔReas':>13}")
        for mk in model_keys:
            deltas = []
            for pid in TOOLS3_CONTROL_GROUP:
                base_rows = pooled_baseline.get((mk, pid), [])
                inv_rows = [r for r in _rows(mk, pid, "invited_auto") if not r["tools_used"]]
                if base_rows and inv_rows:
                    deltas.append(_median([r["reasoning"] for r in inv_rows]) - _median([r["reasoning"] for r in base_rows]))
            if deltas:
                print(f"  {mk:<22} {len(deltas):>8}  {_median(deltas):>+13.1f}")
            else:
                print(f"  {mk:<22} {'0':>8}  {'—':>13}")

        # --- 4. Split repl vs search ---
        print(f"\n{'═'*100}")
        print(f"  4. SPLIT — repl-offload vs search-offload (forced condition, median ΔReas)")
        print(f"{'═'*100}")
        for kind in ("repl", "search"):
            rows = [c for c in cell_deltas if c["kind"] == kind]
            if not rows:
                print(f"  {kind:<10} n=0")
                continue
            n_exceed = sum(1 for r in rows if r["exceeds_noise"])
            print(f"  {kind:<10} n={len(rows):<3} median ΔReas={_median([r['d_reas'] for r in rows]):+.1f}  "
                  f"exceeding noise floor: {n_exceed}/{len(rows)}")

        # --- 5. Required violations ---
        print(f"\n{'═'*100}")
        print(f"  5. REQUIRED NOT RESPECTED (possible adapter/routing defect)")
        print(f"{'═'*100}")
        if required_violations:
            for v in required_violations:
                print(f"    {v['model_key']:<22} {v['pid']:<6} pass={v['pass']}")
        else:
            print(f"  None — every forced call produced >=1 tool call.")

        status_line = "ALL CALLS COMPLETED" if not has_failure else "ERRORS DETECTED — see rows above"
        print(f"\n{'═'*100}")
        print(f"  {status_line}")
        print(f"  Structured records → results/tools3/{run_id}.jsonl")
        print(f"  Raw traces         → results/tools3/{run_id}_traces/<model>_<prompt>_<condition>_pass<N>.txt")
        print()
        return 1 if has_failure else 0

    finally:
        if saved_anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_anthropic_key


# ---------------------------------------------------------------------------
# --heavy phase (heavy tasks: code + finance_calc + finance_interp)
# ---------------------------------------------------------------------------
#
# Three locked tasks from established, external datasets (HumanEval, FinQA —
# see src/heavy_tasks.py). Goal: HOW models solve heavy tasks — tokens, cost,
# tool behavior, reasoning — not primarily whether they can. Correctness is
# graded as quality control (src/heavy_grader.py), not the headline metric.
#
# Two conditions only — baseline (closed-book) and invited_auto (both tools
# offered, tool_choice=auto). No forced condition: --tools3 already showed
# forcing tool use kills Claude's reasoning trace and gets ignored by two
# other models, so it would not measure genuine tool-adoption behavior here.

HEAVY_MODELS: list[str] = FULL_MODEL_ORDER
HEAVY_CONDITIONS: tuple[str, ...] = ("baseline", "invited_auto")

# Reuse the --tools3 invitation verbatim — "fælles invitationslinje identisk
# for alle" was already satisfied by that constant; no reason to re-derive it.
HEAVY_INVITATION = TOOLS3_INVITATION


def run_heavy(repeats: int = 5, allow_direct: bool = False) -> int:
    """
    3 locked tasks (HumanEval/94, FinQA CDNS calc, FinQA AMAT interp) x
    2 conditions (baseline, invited_auto) x 8 scored+anchor models x
    `repeats` passes. Every row is graded for correctness as quality control.

    Forces the OpenRouter route for Anthropic models (pops ANTHROPIC_API_KEY
    for the run) — same reasoning as --variance/--tools3: panel.yaml's pinned
    dated openrouter_model_id must be used, never the undated direct alias.
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")

    tasks_safe = load_heavy_tasks(with_facit=False)
    tasks_facit = load_heavy_tasks(with_facit=True)

    model_keys = [
        k for k in HEAVY_MODELS
        if k in panel and panel[k].get("role") in SMOKE_ROLES
    ]

    tool_defs = available_tool_defs()
    tools_available_names = [t["name"] for t in tool_defs]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_heavy"
    traces_dir = HEAVY_DIR / f"{run_id}_traces"

    n_calls = len(HEAVY_TASK_KEYS) * len(HEAVY_CONDITIONS) * len(model_keys) * repeats
    print(f"\n{'═'*120}")
    print(f"  Heavy Tasks (--heavy)   run_id={run_id}")
    print(f"  Tasks: {', '.join(HEAVY_TASK_KEYS)}  (locked, established datasets — see src/heavy_tasks.py)")
    print(f"  Conditions: {', '.join(HEAVY_CONDITIONS)}  (never forced)")
    print(
        f"  {len(HEAVY_TASK_KEYS)} tasks × {len(HEAVY_CONDITIONS)} conditions × "
        f"{len(model_keys)} models × {repeats} repeats = {n_calls} rows"
    )
    print(f"  Tools offered (invited_auto only): {', '.join(tools_available_names)}")
    if not search_available():
        print(f"  !! web_search PENDING — SEARCH_API_KEY not set.")
    print(f"  reasoning_effort={reasoning_effort!r}  |  correctness is quality control, not the headline metric")
    print(f"  Versions: panel.yaml pinned dated strings (identical to --variance/--tools3). No alias.")
    print(f"{'═'*120}")

    # Force the OpenRouter route for Anthropic models — same reasoning as --variance/--tools3.
    saved_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    if saved_anthropic_key:
        print(f"  !! ANTHROPIC_API_KEY temporarily removed from env — forcing OpenRouter route for this run.")

    try:
        resolved = resolve_models(panel, model_keys)
        print_resolution_table(resolved)
        assert_no_silent_direct_route(panel, model_keys, allow_direct)
        print(
            f"\n  NOTE: catalog mismatches above are not a gate by themselves — dated snapshot"
            f" slugs can be absent from OpenRouter's /models listing while still callable."
            f" The real protection is assert_model_pin_honored() checking request_model_id"
            f" (the request variable itself, NOT resp.model — see adapters/base.py) before"
            f" every call. A retired-pin silent-reroute incident on 2026-07-14 is why this"
            f" exists; resp.model was proven that day to be identical whether the dated pin"
            f" or the undated slug is sent, so it cannot serve as this check.\n"
        )

        has_failure = False
        agg: list[dict] = []

        col_hdr = (
            f"  {'Model':<22} {'Cond':<13} {'Pass':>4} {'Inp':>6} {'Reas':>7} {'Out':>6}  "
            f"{'Cost($)':>10}  {'Tools called':<20} {'Correct':<8} Status"
        )

        for task_key in HEAVY_TASK_KEYS:
            task = tasks_safe[task_key]
            assert "facit_grading" not in task, (
                f"CRITICAL SECURITY VIOLATION: facit_grading in request-path object for {task_key}"
            )
            base_prompt: str = task["prompt"]
            domain = task["domain"]
            task_id = task["task_id"]
            facit_grading = tasks_facit[task_key]["facit_grading"]

            for condition in HEAVY_CONDITIONS:
                prompt_text = base_prompt + (HEAVY_INVITATION if condition == "invited_auto" else "")

                print(f"\n{'═'*120}")
                print(f"  [{task_key}]  {domain}  ({task_id})  condition={condition}")
                print(f"  {prompt_text[:110].strip()!r}")
                print(f"{'═'*120}")
                print(col_hdr)
                print("  " + "─" * 100)

                for model_key in model_keys:
                    cfg = panel[model_key]
                    provider = cfg["provider"]
                    thinking_budget: int = cfg.get("thinking_budget", 4096)

                    cls = PROVIDER_MAP.get(provider)
                    if cls is None:
                        print(f"  {model_key:<22} SKIPPED (unknown provider: {provider})")
                        continue

                    for pass_index in range(1, repeats + 1):
                        try:
                            adapter = cls(model_key, cfg)
                        except CredentialMissingError as e:
                            print(f"  {model_key:<22} {condition:<13} {pass_index:>4}  SKIPPED — {e}")
                            continue

                        try:
                            if condition == "baseline":
                                resp = adapter.call(
                                    prompt_text, thinking_budget=thinking_budget,
                                    reasoning_effort=reasoning_effort,
                                )
                            else:
                                resp = adapter.call_with_tools(
                                    prompt_text, thinking_budget=thinking_budget,
                                    reasoning_effort=reasoning_effort, tool_choice="auto",
                                )
                        except ToolsNotSupportedError as e:
                            save_heavy_result(
                                run_id=run_id, task_id=task_id, domain=domain, model_key=model_key,
                                condition=condition, pass_index=pass_index, status="n/a_no_tool_support",
                                response=None, account=None, cost_usd=None, pricing_snapshot_date=None,
                                thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
                                tools_available=tools_available_names if condition == "invited_auto" else [],
                                correct=None, extracted_answer=None, grading_detail=None,
                                extra={"error": str(e)}, results_dir=HEAVY_DIR,
                            )
                            print(f"  {model_key:<22} {condition:<13} {pass_index:>4}  n/a — no tool support")
                            continue
                        except AdapterError as e:
                            save_heavy_result(
                                run_id=run_id, task_id=task_id, domain=domain, model_key=model_key,
                                condition=condition, pass_index=pass_index, status="error",
                                response=None, account=None, cost_usd=None, pricing_snapshot_date=None,
                                thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
                                tools_available=tools_available_names if condition == "invited_auto" else [],
                                correct=None, extracted_answer=None, grading_detail=None,
                                extra={"error": str(e)}, results_dir=HEAVY_DIR,
                            )
                            print(f"  {model_key:<22} {condition:<13} {pass_index:>4}  ERROR — {e}")
                            has_failure = True
                            continue
                        except Exception as e:
                            print(f"  {model_key:<22} {condition:<13} {pass_index:>4}  ERROR — unexpected: {e}")
                            has_failure = True
                            continue

                        account = build_account(resp)
                        cost_usd, snapshot_date = compute_cost(model_key, account)
                        gr = grade_heavy(domain, resp.answer_text, facit_grading)
                        tools_used = _tool_names_used(resp)

                        save_heavy_result(
                            run_id=run_id, task_id=task_id, domain=domain, model_key=model_key,
                            condition=condition, pass_index=pass_index, status="ok",
                            response=resp, account=account, cost_usd=cost_usd,
                            pricing_snapshot_date=snapshot_date, thinking_budget=thinking_budget,
                            reasoning_effort=reasoning_effort,
                            tools_available=tools_available_names if condition == "invited_auto" else [],
                            correct=gr.correct, extracted_answer=gr.extracted_answer,
                            grading_detail=gr.detail, results_dir=HEAVY_DIR,
                        )
                        save_heavy_trace(
                            traces_dir=traces_dir, model_key=model_key, domain=domain,
                            condition=condition, pass_index=pass_index, prompt_text=prompt_text,
                            response=resp, status="ok", correct=gr.correct,
                            extracted_answer=gr.extracted_answer, grading_detail=gr.detail,
                        )

                        agg.append({
                            "task_key": task_key, "domain": domain, "model_key": model_key,
                            "condition": condition, "pass_index": pass_index,
                            "input": account.input_tokens, "reasoning": account.reasoning_tokens,
                            "output": account.output_tokens, "cost_usd": cost_usd,
                            "tools_used": tools_used, "correct": gr.correct,
                            "grading_detail": gr.detail,
                        })

                        tools_str = ", ".join(tools_used) if tools_used else "—"
                        correct_str = "OK" if gr.correct else "FAIL"
                        print(
                            f"  {model_key:<22} {condition:<13} {pass_index:>4} {account.input_tokens:>6}"
                            f" {account.reasoning_tokens:>7} {account.output_tokens:>6}  ${cost_usd:>9.5f}  "
                            f"{tools_str:<20} {correct_str:<8} ok"
                        )

        # ═════════════════════════════════════════════════════════
        # REPORT
        # ═════════════════════════════════════════════════════════
        import statistics as _stats

        def _median(vals):
            return _stats.median(vals) if vals else None

        ok_rows = [r for r in agg if r.get("cost_usd") is not None]

        # --- 1. HOW — median tokens/cost/reasoning + tool grab rate + correctness-per-krone ---
        print(f"\n{'═'*130}")
        print(f"  1. HOW — median tokens/cost/reasoning, tool-grab rate, correctness-per-krone")
        print(f"  Per (task, condition, model)")
        print(f"{'═'*130}")
        print(
            f"  {'Task':<15} {'Cond':<13} {'Model':<20} {'n':>3}  {'MedInp':>7}  {'MedReas':>8}  "
            f"{'MedOut':>7}  {'MedCost($)':>11}  {'ToolRate':>9}  {'Correct%':>9}  {'Corr/$':>9}"
        )
        print("  " + "─" * 122)
        for task_key in HEAVY_TASK_KEYS:
            for condition in HEAVY_CONDITIONS:
                for model_key in model_keys:
                    rows = [
                        r for r in ok_rows
                        if r["task_key"] == task_key and r["condition"] == condition and r["model_key"] == model_key
                    ]
                    if not rows:
                        continue
                    med_cost = _median([r["cost_usd"] for r in rows])
                    n_correct = sum(1 for r in rows if r["correct"])
                    correct_pct = 100 * n_correct / len(rows)
                    corr_per_dollar = correct_per_dollar(rows)
                    tool_rate = (
                        sum(1 for r in rows if r["tools_used"]) / len(rows)
                        if condition == "invited_auto" else None
                    )
                    tool_rate_str = f"{tool_rate:.0%}" if tool_rate is not None else "—"
                    corr_per_dollar_str = f"{corr_per_dollar:.1f}" if corr_per_dollar is not None else "—"
                    print(
                        f"  {task_key:<15} {condition:<13} {model_key:<20} {len(rows):>3}  "
                        f"{_median([r['input'] for r in rows]):>7.0f}  {_median([r['reasoning'] for r in rows]):>8.0f}  "
                        f"{_median([r['output'] for r in rows]):>7.0f}  {med_cost:>11.5f}  {tool_rate_str:>9}  "
                        f"{correct_pct:>8.0f}%  {corr_per_dollar_str:>9}"
                    )

        # --- 2. TOOL EFFECT — baseline vs invited_auto per (model, task) ---
        print(f"\n{'═'*130}")
        print(f"  2. TOOL EFFECT — baseline vs invited_auto, per (model, task)")
        print(f"  Δ = invited_auto − baseline (median). Tested against the --variance noise floor.")
        print(f"{'═'*130}")
        try:
            noise_floor = _relative_noise_floor(_load_pooled_baseline())
        except Exception:
            noise_floor = {}
        print(
            f"  {'Model':<20} {'Task':<15} {'ΔCost($)':>10}  {'ΔReas':>8}  {'ΔReas%':>8}  "
            f"{'ΔCorrect%':>10}  {'ToolUse%':>9}  {'Decision':<28}"
        )
        print("  " + "─" * 116)
        for model_key in model_keys:
            for task_key in HEAVY_TASK_KEYS:
                base_rows = [
                    r for r in ok_rows
                    if r["model_key"] == model_key and r["task_key"] == task_key and r["condition"] == "baseline"
                ]
                inv_rows = [
                    r for r in ok_rows
                    if r["model_key"] == model_key and r["task_key"] == task_key and r["condition"] == "invited_auto"
                ]
                if not base_rows or not inv_rows:
                    continue
                base_reas = _median([r["reasoning"] for r in base_rows])
                inv_reas = _median([r["reasoning"] for r in inv_rows])
                d_reas = inv_reas - base_reas
                d_reas_pct = (d_reas / base_reas) if base_reas > 0 else None
                d_cost = _median([r["cost_usd"] for r in inv_rows]) - _median([r["cost_usd"] for r in base_rows])
                base_corr_pct = 100 * sum(1 for r in base_rows if r["correct"]) / len(base_rows)
                inv_corr_pct = 100 * sum(1 for r in inv_rows if r["correct"]) / len(inv_rows)
                tool_use_pct = 100 * sum(1 for r in inv_rows if r["tools_used"]) / len(inv_rows)
                nf = noise_floor.get(model_key)
                if nf is not None and d_reas_pct is not None:
                    decision = (
                        f"EXCEEDS ({abs(d_reas_pct):.0%} > {nf:.0%})" if abs(d_reas_pct) > nf
                        else f"within noise (<= {nf:.0%})"
                    )
                else:
                    decision = "no noise floor data"
                pct_str = f"{d_reas_pct:+.0%}" if d_reas_pct is not None else "—"
                print(
                    f"  {model_key:<20} {task_key:<15} {d_cost:>+10.5f}  {d_reas:>+8.1f}  {pct_str:>8}  "
                    f"{(inv_corr_pct - base_corr_pct):>+9.0f}%  {tool_use_pct:>8.0f}%  {decision:<28}"
                )

        # --- 3. Domain contrast ---
        print(f"\n{'═'*100}")
        print(f"  3. DOMAIN CONTRAST — tool-grab rate & correctness by domain (invited_auto, all models pooled)")
        print(f"{'═'*100}")
        print(f"  {'Domain':<16} {'n':>4}  {'ToolRate':>9}  {'Correct%':>9}")
        for task_key in HEAVY_TASK_KEYS:
            rows = [r for r in ok_rows if r["task_key"] == task_key and r["condition"] == "invited_auto"]
            if not rows:
                continue
            tool_rate = sum(1 for r in rows if r["tools_used"]) / len(rows)
            correct_pct = 100 * sum(1 for r in rows if r["correct"]) / len(rows)
            print(f"  {task_key:<16} {len(rows):>4}  {tool_rate:>8.0%}  {correct_pct:>8.0f}%")

        # --- 4. OVER-REACH — web_search on self-contained tasks ---
        print(f"\n{'═'*100}")
        print(f"  4. OVER-REACH — web_search calls on tasks that are self-contained (no lookup needed)")
        print(f"  All 3 locked tasks carry every fact needed in the prompt — any web_search here is over-reach.")
        print(f"{'═'*100}")
        overreach = [r for r in ok_rows if r["condition"] == "invited_auto" and "web_search" in r["tools_used"]]
        if overreach:
            for r in overreach:
                print(f"    {r['model_key']:<22} {r['task_key']:<15} pass={r['pass_index']}  tools={r['tools_used']}")
        else:
            print(f"  None — no model called web_search on any of the 3 tasks.")

        # --- 5. DATA QUALITY — uncertain extraction ---
        print(f"\n{'═'*100}")
        print(f"  5. DATA QUALITY — uncertain extracted_answer parsing")
        print(f"{'═'*100}")
        uncertain = []
        for r in ok_rows:
            gd = r.get("grading_detail") or {}
            if r["domain"] == "code":
                if gd.get("extraction_method") != "fenced_with_entry_point":
                    uncertain.append((r, gd.get("extraction_method")))
            else:
                if gd.get("any_number_in_text_matches") != gd.get("primary_matched"):
                    uncertain.append((r, "primary_vs_any_match_disagree"))
        if uncertain:
            for r, reason in uncertain:
                print(f"    {r['model_key']:<22} {r['task_key']:<15} {r['condition']:<13} pass={r['pass_index']}  {reason}")
        else:
            print(f"  None — extraction was unambiguous on every row.")

        status_line = "ALL CALLS COMPLETED" if not has_failure else "ERRORS DETECTED — see rows above"
        print(f"\n{'═'*100}")
        print(f"  {status_line}")
        print(f"  Structured records → results/heavy/{run_id}.jsonl")
        print(f"  Raw traces         → results/heavy/{run_id}_traces/<model>_<domain>_<condition>_pass<N>.txt")
        print()
        return 1 if has_failure else 0

    finally:
        if saved_anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_anthropic_key


# ---------------------------------------------------------------------------
# --heavy-recap — cap-artifact check for the two models --heavy's 4608-token
# completion cap (thinking_budget 4096 + 512) actually hit: mistral_medium_3_5
# (10/30 rows) and gemma_4 (1/30 rows). Only mistral_medium_3_5, gemma_4, and
# a code-only opus_4_8 sanity sample are re-run, at thinking_budget=16384
# (override — panel.yaml's 4096 is untouched, so 20260709T093542_heavy stays
# reproducible). Everything else — pinned dated model strings, reasoning_effort,
# prompts, the (unfixed) Danish HEAVY_INVITATION on English heavy tasks — is
# byte-identical to 20260709T093542_heavy, so the completion cap is the only
# variable that moved.
# ---------------------------------------------------------------------------

HEAVY_RECAP_THINKING_BUDGET = 16384

# task_key scope per model — opus is a stability sample (code only); it never
# neared the old 4096 budget (18% used, see docs), so finance domains would
# add cost without adding signal for the cap-artifact question.
HEAVY_RECAP_SCOPE: dict[str, tuple[str, ...]] = {
    "mistral_medium_3_5": ("code", "finance_calc", "finance_interp"),
    "gemma_4": ("code", "finance_calc", "finance_interp"),
    "opus_4_8": ("code",),
}

# Source run being re-examined — for the drift control and old-vs-new report.
HEAVY_RECAP_BASELINE_RUN_ID = "20260709T093542_heavy_corrected"


def _load_heavy_jsonl(run_id: str) -> list[dict]:
    path = HEAVY_DIR / f"{run_id}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_heavy_recap(repeats: int = 5, allow_direct: bool = False) -> int:
    """
    Re-run mistral_medium_3_5 + gemma_4 (all 3 tasks) and opus_4_8 (code only)
    at thinking_budget=16384 to determine whether --heavy's 0% Mistral
    HumanEval/94 result was a completion-cap artifact (4096+512=4608, hit on
    all 10 Mistral code rows and 1/30 Gemma rows) rather than a genuine
    correctness finding. 70 calls total: 30 + 30 + 10.
    """
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")

    tasks_safe = load_heavy_tasks(with_facit=False)
    tasks_facit = load_heavy_tasks(with_facit=True)

    model_keys = [
        k for k in HEAVY_RECAP_SCOPE
        if k in panel and panel[k].get("role") in SMOKE_ROLES
    ]

    tool_defs = available_tool_defs()
    tools_available_names = [t["name"] for t in tool_defs]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_heavy_recap"
    traces_dir = HEAVY_DIR / f"{run_id}_traces"

    n_calls = sum(
        len(HEAVY_RECAP_SCOPE[k]) * len(HEAVY_CONDITIONS) * repeats
        for k in model_keys
    )
    print(f"\n{'═'*120}")
    print(f"  Heavy Recap (--heavy-recap)   run_id={run_id}")
    print(f"  Cap-artifact check: thinking_budget={HEAVY_RECAP_THINKING_BUDGET} (panel.yaml's 4096 untouched)")
    print(f"  Baseline for comparison: results/heavy/{HEAVY_RECAP_BASELINE_RUN_ID}.jsonl")
    for k in model_keys:
        print(f"    {k:<22} tasks={', '.join(HEAVY_RECAP_SCOPE[k])}")
    print(f"  {n_calls} rows total (expected 70)")
    print(f"  reasoning_effort={reasoning_effort!r}  |  HEAVY_INVITATION unchanged (Danish, known defect, not fixed here)")
    print(f"{'═'*120}")

    saved_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    if saved_anthropic_key:
        print(f"  !! ANTHROPIC_API_KEY temporarily removed from env — forcing OpenRouter route for this run.")

    try:
        resolved = resolve_models(panel, model_keys)
        print_resolution_table(resolved)
        assert_no_silent_direct_route(panel, model_keys, allow_direct)

        has_failure = False
        agg: list[dict] = []

        col_hdr = (
            f"  {'Model':<22} {'Task':<15} {'Cond':<13} {'Pass':>4} {'Inp':>6} {'Reas':>7} {'Out':>6}  "
            f"{'Cost($)':>10}  {'Tools called':<20} {'Correct':<8} {'Trunc':<6} Status"
        )
        print(col_hdr)
        print("  " + "─" * 118)

        for task_key in HEAVY_TASK_KEYS:
            task = tasks_safe[task_key]
            assert "facit_grading" not in task, (
                f"CRITICAL SECURITY VIOLATION: facit_grading in request-path object for {task_key}"
            )
            base_prompt: str = task["prompt"]
            domain = task["domain"]
            task_id = task["task_id"]
            facit_grading = tasks_facit[task_key]["facit_grading"]

            for model_key in model_keys:
                if task_key not in HEAVY_RECAP_SCOPE[model_key]:
                    continue

                cfg = panel[model_key]
                provider = cfg["provider"]
                thinking_budget = HEAVY_RECAP_THINKING_BUDGET  # override — NOT cfg.get(...)

                cls = PROVIDER_MAP.get(provider)
                if cls is None:
                    print(f"  {model_key:<22} SKIPPED (unknown provider: {provider})")
                    continue

                for condition in HEAVY_CONDITIONS:
                    prompt_text = base_prompt + (HEAVY_INVITATION if condition == "invited_auto" else "")

                    for pass_index in range(1, repeats + 1):
                        try:
                            adapter = cls(model_key, cfg)
                        except CredentialMissingError as e:
                            print(f"  {model_key:<22} {task_key:<15} {condition:<13} {pass_index:>4}  SKIPPED — {e}")
                            continue

                        try:
                            if condition == "baseline":
                                resp = adapter.call(
                                    prompt_text, thinking_budget=thinking_budget,
                                    reasoning_effort=reasoning_effort,
                                )
                            else:
                                resp = adapter.call_with_tools(
                                    prompt_text, thinking_budget=thinking_budget,
                                    reasoning_effort=reasoning_effort, tool_choice="auto",
                                )
                        except ToolsNotSupportedError as e:
                            save_heavy_result(
                                run_id=run_id, task_id=task_id, domain=domain, model_key=model_key,
                                condition=condition, pass_index=pass_index, status="n/a_no_tool_support",
                                response=None, account=None, cost_usd=None, pricing_snapshot_date=None,
                                thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
                                tools_available=tools_available_names if condition == "invited_auto" else [],
                                correct=None, extracted_answer=None, grading_detail=None,
                                extra={"error": str(e)}, results_dir=HEAVY_DIR,
                            )
                            print(f"  {model_key:<22} {task_key:<15} {condition:<13} {pass_index:>4}  n/a — no tool support")
                            continue
                        except AdapterError as e:
                            save_heavy_result(
                                run_id=run_id, task_id=task_id, domain=domain, model_key=model_key,
                                condition=condition, pass_index=pass_index, status="error",
                                response=None, account=None, cost_usd=None, pricing_snapshot_date=None,
                                thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
                                tools_available=tools_available_names if condition == "invited_auto" else [],
                                correct=None, extracted_answer=None, grading_detail=None,
                                extra={"error": str(e)}, results_dir=HEAVY_DIR,
                            )
                            print(f"  {model_key:<22} {task_key:<15} {condition:<13} {pass_index:>4}  ERROR — {e}")
                            has_failure = True
                            continue
                        except Exception as e:
                            print(f"  {model_key:<22} {task_key:<15} {condition:<13} {pass_index:>4}  ERROR — unexpected: {e}")
                            has_failure = True
                            continue

                        account = build_account(resp)
                        cost_usd, snapshot_date = compute_cost(model_key, account)
                        gr = grade_heavy(domain, resp.answer_text, facit_grading)
                        tools_used = _tool_names_used(resp)

                        row = save_heavy_result(
                            run_id=run_id, task_id=task_id, domain=domain, model_key=model_key,
                            condition=condition, pass_index=pass_index, status="ok",
                            response=resp, account=account, cost_usd=cost_usd,
                            pricing_snapshot_date=snapshot_date, thinking_budget=thinking_budget,
                            reasoning_effort=reasoning_effort,
                            tools_available=tools_available_names if condition == "invited_auto" else [],
                            correct=gr.correct, extracted_answer=gr.extracted_answer,
                            grading_detail=gr.detail, results_dir=HEAVY_DIR,
                        )
                        save_heavy_trace(
                            traces_dir=traces_dir, model_key=model_key, domain=domain,
                            condition=condition, pass_index=pass_index, prompt_text=prompt_text,
                            response=resp, status="ok", correct=gr.correct,
                            extracted_answer=gr.extracted_answer, grading_detail=gr.detail,
                        )

                        agg.append({
                            "task_key": task_key, "domain": domain, "model_key": model_key,
                            "condition": condition, "pass_index": pass_index,
                            "input": account.input_tokens, "reasoning": account.reasoning_tokens,
                            "output": account.output_tokens, "cost_usd": cost_usd,
                            "tools_used": tools_used, "correct": gr.correct,
                            "grading_detail": gr.detail, "answer_text": resp.answer_text,
                            "finish_reason": row.get("finish_reason"),
                            "native_finish_reason": row.get("native_finish_reason"),
                            "truncated": row.get("truncated"),
                        })

                        tools_str = ", ".join(tools_used) if tools_used else "—"
                        correct_str = "OK" if gr.correct else "FAIL"
                        trunc_str = "YES" if row.get("truncated") else "no"
                        print(
                            f"  {model_key:<22} {task_key:<15} {condition:<13} {pass_index:>4} {account.input_tokens:>6}"
                            f" {account.reasoning_tokens:>7} {account.output_tokens:>6}  ${cost_usd:>9.5f}  "
                            f"{tools_str:<20} {correct_str:<8} {trunc_str:<6} ok"
                        )

        # ═════════════════════════════════════════════════════════
        # REPORT — 7 sections, answering the recap's actual questions
        # ═════════════════════════════════════════════════════════
        import statistics as _stats

        def _median(vals):
            return _stats.median(vals) if vals else None

        ok_rows = [r for r in agg if r.get("cost_usd") is not None]
        recap_cap = HEAVY_RECAP_THINKING_BUDGET + 512

        # --- 1. Truncated flag per row ---
        print(f"\n{'═'*120}")
        print(f"  1. TRUNCATED — per row, computed (completion_tokens >= {recap_cap}), not inferred")
        print(f"{'═'*120}")
        any_truncated = [r for r in ok_rows if r["truncated"]]
        if any_truncated:
            for r in any_truncated:
                print(
                    f"    !! {r['model_key']:<22} {r['task_key']:<15} {r['condition']:<13} pass={r['pass_index']}"
                    f"  reas+out={r['reasoning']+r['output']}  finish_reason={r['finish_reason']!r}"
                    f"  native_finish_reason={r['native_finish_reason']!r}"
                )
        else:
            print(f"  None — no row hit {recap_cap} at thinking_budget={HEAVY_RECAP_THINKING_BUDGET}.")

        # --- 2. Mistral / code ---
        print(f"\n{'═'*120}")
        print(f"  2. MISTRAL / CODE — does it deliver code now, at 16384?")
        print(f"{'═'*120}")
        mistral_code = [r for r in ok_rows if r["model_key"] == "mistral_medium_3_5" and r["task_key"] == "code"]
        if mistral_code:
            delivered = sum(1 for r in mistral_code if r["answer_text"].strip())
            correct = sum(1 for r in mistral_code if r["correct"])
            called_tools = sum(1 for r in mistral_code if r["tools_used"])
            reas_vals = [r["reasoning"] for r in mistral_code]
            print(f"  n={len(mistral_code)}  delivered_nonempty_answer={delivered}/{len(mistral_code)}  correct={correct}/{len(mistral_code)}")
            print(f"  reasoning_tokens: median={_median(reas_vals):.0f}  min={min(reas_vals)}  max={max(reas_vals)}  (cap={recap_cap})")
            print(f"  python_exec called (invited_auto): {called_tools}/{sum(1 for r in mistral_code if r['condition']=='invited_auto')}")
            still_capped = [r for r in mistral_code if r["truncated"]]
            if still_capped:
                print(f"  !! STILL TRUNCATED at 16384 on {len(still_capped)}/{len(mistral_code)} rows — genuine non-termination, not a cap artifact.")
            else:
                print(f"  Not truncated at 16384 — the 0% result at 4608 was a cap artifact." if delivered else "  Delivers no code even uncapped — inspect traces, this is NOT what the cap hypothesis predicted.")
        else:
            print("  No mistral_medium_3_5 / code rows recorded.")

        # --- 3. Gemma — does the raised cap change the code result? ---
        print(f"\n{'═'*120}")
        print(f"  3. GEMMA / CODE — effect of raised cap")
        print(f"{'═'*120}")
        gemma_code = [r for r in ok_rows if r["model_key"] == "gemma_4" and r["task_key"] == "code"]
        if gemma_code:
            correct = sum(1 for r in gemma_code if r["correct"])
            print(f"  n={len(gemma_code)}  correct={correct}/{len(gemma_code)}  truncated={sum(1 for r in gemma_code if r['truncated'])}/{len(gemma_code)}")
        else:
            print("  No gemma_4 / code rows recorded.")

        # --- 4. Opus sample — stability check ---
        print(f"\n{'═'*120}")
        print(f"  4. OPUS / CODE — stability sample (expected: no change)")
        print(f"{'═'*120}")
        opus_code = [r for r in ok_rows if r["model_key"] == "opus_4_8" and r["task_key"] == "code"]
        try:
            old_rows_for_stability = _load_heavy_jsonl(HEAVY_RECAP_BASELINE_RUN_ID)
        except FileNotFoundError:
            old_rows_for_stability = []
        old_opus_code = [
            r for r in old_rows_for_stability
            if r.get("model_key") == "opus_4_8" and r.get("domain") == "code" and r.get("status") == "ok"
        ]
        if opus_code and old_opus_code:
            new_reas = _median([r["reasoning"] for r in opus_code])
            old_reas = _median([(r.get("tokens") or {}).get("reasoning", 0) for r in old_opus_code])
            new_correct = sum(1 for r in opus_code if r["correct"])
            old_correct = sum(1 for r in old_opus_code if r.get("correct"))
            print(f"  reasoning_tokens median: old={old_reas:.0f}  new={new_reas:.0f}  (budget went 4096 -> 16384)")
            print(f"  correct: old={old_correct}/{len(old_opus_code)}  new={new_correct}/{len(opus_code)}")
            print(
                "  UNCHANGED — thinking_budget behaves as an adaptive allocation for Opus, not a hard ceiling, as expected."
                if abs(new_reas - old_reas) < max(50, 0.5 * old_reas) and new_correct == old_correct
                else "  !! CHANGED more than expected — re-examine the adaptive-thinking assumption for Opus."
            )
        else:
            print("  Insufficient data for comparison (missing new or old opus_4_8/code rows).")

        # --- 5. Drift control — rows the cap NEVER touched must reproduce ---
        print(f"\n{'═'*120}")
        print(f"  5. DRIFT CONTROL — rows the old 4608 cap never touched must reproduce within variance")
        print(f"  (Mistral finance_calc/finance_interp; Gemma code/finance rows excluding the one old capped row)")
        print(f"{'═'*120}")
        drift_failed = False
        old_by_cell: dict[tuple, list[dict]] = {}
        for r in old_rows_for_stability:
            if r.get("status") != "ok":
                continue
            key = (r.get("model_key"), r.get("domain"), r.get("condition"))
            old_by_cell.setdefault(key, []).append(r)

        drift_targets = [
            ("mistral_medium_3_5", "finance_calc"),
            ("mistral_medium_3_5", "finance_interp"),
            ("gemma_4", "finance_calc"),
            ("gemma_4", "finance_interp"),
            ("gemma_4", "code"),
        ]
        for model_key, task_key in drift_targets:
            for condition in HEAVY_CONDITIONS:
                old_cell = old_by_cell.get((model_key, task_key, condition), [])
                old_cell_uncapped = [r for r in old_cell if ((r.get("tokens") or {}).get("reasoning", 0) + (r.get("tokens") or {}).get("output", 0)) < 4608]
                new_cell = [r for r in ok_rows if r["model_key"] == model_key and r["task_key"] == task_key and r["condition"] == condition]
                if not old_cell_uncapped or not new_cell:
                    continue
                old_correct_rate = sum(1 for r in old_cell_uncapped if r.get("correct")) / len(old_cell_uncapped)
                new_correct_rate = sum(1 for r in new_cell if r["correct"]) / len(new_cell)
                verdict = "OK" if old_correct_rate == new_correct_rate else "DRIFT"
                if verdict == "DRIFT":
                    drift_failed = True
                print(
                    f"  {model_key:<22} {task_key:<15} {condition:<13} old_correct={old_correct_rate:.0%} (n={len(old_cell_uncapped)})"
                    f"  new_correct={new_correct_rate:.0%} (n={len(new_cell)})  [{verdict}]"
                )
        if drift_failed:
            print(f"\n  !!!! DRIFT DETECTED on a row the cap never touched — this is serverside drift, not a cap effect.")
            print(f"  !!!! STOP: do not interpret sections 2-4 as cap-artifact findings until drift is explained.")
        else:
            print(f"\n  No drift — uncapped rows reproduced. The cap is isolated as the sole variable.")

        # --- 6. Correctness / correctness-per-krone ---
        print(f"\n{'═'*120}")
        print(f"  6. CORRECTNESS & CORRECTNESS-PER-KRONE (this recap, all cells)")
        print(f"{'═'*120}")
        print(f"  {'Model':<22} {'Task':<15} {'Cond':<13} {'n':>3}  {'Correct%':>9}  {'MedCost($)':>11}  {'Corr/$':>8}")
        for model_key in model_keys:
            for task_key in HEAVY_RECAP_SCOPE[model_key]:
                for condition in HEAVY_CONDITIONS:
                    rows = [r for r in ok_rows if r["model_key"] == model_key and r["task_key"] == task_key and r["condition"] == condition]
                    if not rows:
                        continue
                    med_cost = _median([r["cost_usd"] for r in rows])
                    n_correct = sum(1 for r in rows if r["correct"])
                    correct_pct = 100 * n_correct / len(rows)
                    corr_per_dollar = correct_per_dollar(rows)
                    corr_per_dollar_str = f"{corr_per_dollar:.1f}" if corr_per_dollar is not None else "—"
                    print(
                        f"  {model_key:<22} {task_key:<15} {condition:<13} {len(rows):>3}  "
                        f"{correct_pct:>8.0f}%  {med_cost:>11.5f}  {corr_per_dollar_str:>8}"
                    )

        # --- 7. Raw cost ---
        print(f"\n{'═'*120}")
        print(f"  7. RAW COST")
        print(f"{'═'*120}")
        total_cost = sum(r["cost_usd"] for r in ok_rows)
        print(f"  Total ({len(ok_rows)} rows): ${total_cost:.5f}")
        for model_key in model_keys:
            model_rows = [r for r in ok_rows if r["model_key"] == model_key]
            if model_rows:
                print(f"    {model_key:<22} ${sum(r['cost_usd'] for r in model_rows):.5f}  ({len(model_rows)} rows)")

        status_line = "ALL CALLS COMPLETED" if not has_failure else "ERRORS DETECTED — see rows above"
        print(f"\n{'═'*100}")
        print(f"  {status_line}")
        print(f"  Structured records → results/heavy/{run_id}.jsonl")
        print(f"  Raw traces         → results/heavy/{run_id}_traces/<model>_<domain>_<condition>_pass<N>.txt")
        print()
        return 1 if (has_failure or drift_failed) else 0

    finally:
        if saved_anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_anthropic_key


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reasoning Benchmark runner")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke test across all available scored models and the Gemma anchor",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_KEY",
        default=None,
        help="Limit smoke test to a single model key (e.g. claude_sonnet_4_6)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the Phase 1 pilot across all scored+anchor models",
    )
    parser.add_argument(
        "--prompts",
        metavar="P3,P5",
        default=",".join(PILOT_DEFAULT_PROMPTS),
        help=f"Comma-separated prompt IDs for the pilot (default: {','.join(PILOT_DEFAULT_PROMPTS)})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run Phase 1 full economy run: all 10 prompts × all scored+anchor models",
    )
    parser.add_argument(
        "--models",
        metavar="MODEL1,MODEL2",
        default=None,
        help="Restrict --full to these model keys only (e.g. opus_4_8,mistral_medium_3_5)",
    )
    parser.add_argument(
        "--validate-judges",
        action="store_true",
        help=(
            "Phase 2 gate: run both judges on Gemma traces only (English) "
            "for human verification before proceeding to Chinese traces"
        ),
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Phase 2 full legibility scoring: both judges on deepseek/glm/kimi + gemma. "
            "Run --validate-judges first and confirm results."
        ),
    )
    parser.add_argument(
        "--source-run-id",
        metavar="RUN_ID",
        default=None,
        help=(
            "Run ID to use as source. For --validate-judges/--judge: most recent "
            "results/full/*.jsonl. For --langcost-report: most recent langcost full JSONL."
        ),
    )
    # ── Language-cost experiment (new track — does not touch main benchmark data) ──
    parser.add_argument(
        "--langcost",
        action="store_true",
        help=(
            "Language-cost pilot: M1 × da/en/zh × 5 open models (15 calls). "
            "Prints cost projection for the full 90-call grid and stops. "
            "Confirm, then run --langcost-full."
        ),
    )
    parser.add_argument(
        "--langcost-full",
        action="store_true",
        help=(
            "Language-cost full grid: M1–M6 × da/en/zh × 5 open models (90 calls). "
            "Run after reviewing the pilot projection."
        ),
    )
    parser.add_argument(
        "--langcost-report",
        action="store_true",
        help=(
            "Generate results/syntese/sprogets_pris_data.md from a completed "
            "langcost full run. Use --source-run-id to specify a run; "
            "defaults to the most recent *_langcost_full*.jsonl."
        ),
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help=(
            "Tool-offload experiment: two arms (baseline, tools) per (model, prompt) "
            "across all 10 prompts and the 8 scored+anchor models. Requires no extra "
            "flags — python_exec always runs; web_search needs SEARCH_API_KEY in .env."
        ),
    )
    parser.add_argument(
        "--variance",
        action="store_true",
        help=(
            "Variance repro run: baseline only, no tools, all 8 models pinned to "
            "fully dated openrouter_model_id strings (identical to June). Repeats "
            "the full 10-prompt set --passes times to measure run-to-run variance."
        ),
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        metavar="N",
        help="Number of full passes for --variance (default: 2).",
    )
    parser.add_argument(
        "--tools3",
        action="store_true",
        help=(
            "Replaces --tools: invited_auto (tool_choice=auto, explicit invitation) "
            "and forced (tool_choice=required) conditions on pinned versions, "
            "--repeats times per cell. Baseline is REUSED from --variance/June, not "
            "re-run. Gated by a mandatory Gemma/Mistral required-compliance smoke test."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        metavar="N",
        help="Number of repeats per cell for --tools3 (default: 5).",
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help=(
            "Heavy-task correctness+economy run: 3 locked tasks (HumanEval/94, "
            "FinQA CDNS calc, FinQA AMAT interp) x baseline/invited_auto x 8 "
            "models x --repeats passes. Correctness graded as quality control."
        ),
    )
    parser.add_argument(
        "--heavy-recap",
        action="store_true",
        help=(
            "Cap-artifact check for --heavy: re-runs mistral_medium_3_5 + gemma_4 "
            "(all 3 tasks) and opus_4_8 (code only) at thinking_budget=16384 "
            "(override; panel.yaml's 4096 is untouched). 70 calls. Only run this "
            "for models --heavy's 4608-token completion cap actually hit."
        ),
    )
    parser.add_argument(
        "--juni-recap-mistral",
        action="store_true",
        help=(
            "Cap-artifact check for June Phase 1 (--full): re-runs "
            "mistral_medium_3_5 on P1-P10 (1 pass each) at thinking_budget=16384 "
            "(override; panel.yaml's 4096 is untouched). 10 calls. De-censors "
            "the light-task reasoning median that June's completion cap undercut."
        ),
    )
    parser.add_argument(
        "--steer",
        action="store_true",
        help=(
            "SCAFFOLD (Step 4) — prepend a language instruction to each prompt, "
            "e.g. 'Tænk og ræsonnér på dansk.' Use with --langcost or --langcost-full. "
            "Keep steered and unsteered runs in separate run_ids (handled automatically)."
        ),
    )
    parser.add_argument(
        "--allow-direct",
        action="store_true",
        help=(
            "Override Defect 2's guard: permit a direct provider key (e.g. "
            "ANTHROPIC_API_KEY) to silently route a pinned model straight to "
            "that provider's own API, bypassing OpenRouter and openrouter_model_id "
            "entirely. Without this flag, any pinned model with a direct key "
            "present in the environment stops the run before the first call."
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        code = run_smoke(model_filter=args.model)
        sys.exit(code)
    elif args.pilot:
        pid_list = [p.strip() for p in args.prompts.split(",") if p.strip()]
        code = run_pilot(prompt_ids=pid_list)
        sys.exit(code)
    elif args.full:
        mfilter = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
        code = run_full(model_filter=mfilter, allow_direct=args.allow_direct)
        sys.exit(code)
    elif args.validate_judges:
        code = run_validate_judges(source_run_id=args.source_run_id)
        sys.exit(code)
    elif args.judge:
        code = run_judge(source_run_id=args.source_run_id)
        sys.exit(code)
    elif args.langcost or args.langcost_full:
        code = run_langcost(full=args.langcost_full, steer=args.steer, allow_direct=args.allow_direct)
        sys.exit(code)
    elif args.langcost_report:
        code = run_langcost_report(source_run_id=args.source_run_id)
        sys.exit(code)
    elif args.tools:
        code = run_tools(allow_direct=args.allow_direct)
        sys.exit(code)
    elif args.variance:
        code = run_variance(passes=args.passes, allow_direct=args.allow_direct)
        sys.exit(code)
    elif args.tools3:
        code = run_tools3(repeats=args.repeats, allow_direct=args.allow_direct)
        sys.exit(code)
    elif args.heavy:
        code = run_heavy(repeats=args.repeats, allow_direct=args.allow_direct)
        sys.exit(code)
    elif args.heavy_recap:
        code = run_heavy_recap(repeats=args.repeats, allow_direct=args.allow_direct)
        sys.exit(code)
    elif args.juni_recap_mistral:
        code = run_juni_recap_mistral(allow_direct=args.allow_direct)
        sys.exit(code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
