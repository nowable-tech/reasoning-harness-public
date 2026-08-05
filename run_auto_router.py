"""
Standalone candidate test: OpenRouter's Auto Router (`openrouter/auto`)
against our facit — NOT a panel addition.

Auto is not a model. It is a routing POLICY sitting on top of the panel's
models (and possibly others): it picks which underlying model answers each
request. This script asks the same question the rest of the series asks
("which model should you pick?") of the market's own router, and checks
whether it lands on the economically correct answer (cheapest model, since
correctness is ~flat across the panel) or overpays for tasks any model in
the panel solves.

Runs the exact same 16 task-cells as the panel:
  - 10 light prompts P1-P10 (data/prompts.yaml), baseline only
  - 3 heavy tasks H1-H3 (code / finance_calc / finance_interp,
    src/heavy_tasks.py), x 2 conditions (baseline, invited_auto)
5 passes per cell = 80 calls total. thinking_budget=16384,
reasoning_effort="high" — same as the rest of the series.

Critical logging on every row (Auto hides its choice behind one slug —
without this the run is worthless):
  - model_version: the actual routed model, as reported by OpenRouter
    (resp.model)
  - served_by: OpenRouter's raw "provider" field (backend that served it)
  - request_model_id: always "openrouter/auto" — the literal string sent
  - tokens (input/reasoning/output), cost_usd (from OpenRouter's own live
    per-call `usage.cost`, NOT config/pricing.yaml — Auto and whatever it
    routes to are not necessarily in that panel-only price list)
  - correct, latency_s, raw_reasoning_trace, trace_status, tool_calls
  - finish_reason / native_finish_reason / provider_warnings /
    param_dropped flag (marked, not failed, when reasoning params look
    like they were dropped by the routed model)

Writes to results/auto/<run_id>.jsonl — a new file. Never touches
results/heavy, results/full, reasoning-data.json, or
docs/reasoning_findings.md.

Cost guard: run stops (not crashes) if cumulative cost_usd across all rows
reaches PRICE_CAP_USD. Partial results are still valid data.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import os

from openai import OpenAI

from src.accounting import build_account
from src.adapters.base import (
    AdapterError,
    ModelResponse,
    OPENROUTER_BASE_URL,
    extract_finish_reasons,
    extract_served_by,
    extract_think_tags,
    extract_warnings,
    split_token_estimate,
)
from src.config_loader import load_pricing, load_prompts
from src.grader import PROGRAMMATIC_PROMPTS, grade_programmatic
from src.heavy_grader import grade as grade_heavy
from src.heavy_tasks import TASK_KEYS, load_heavy_tasks
from src.storage import save_heavy_result
from src.tool_loop import ToolsNotSupportedError

AUTO_SLUG = "openrouter/auto"
REASONING_EFFORT = "high"
THINKING_BUDGET = 16384
N_PASSES = 5
PRICE_CAP_USD = 10.0
RESULTS_DIR = Path(__file__).parent / "results" / "auto"

HEAVY_CONDITIONS: tuple[str, ...] = ("baseline", "invited_auto")

# Verbatim from run.py's TOOLS3_INVITATION / HEAVY_INVITATION — same source,
# unchanged, per the brief's "samme prompts, uændrede" requirement.
HEAVY_INVITATION = (
    "\n\nDu har adgang til to værktøjer: python_exec (kør Python for eksakt "
    "beregning) og web_search (slå fakta op). Brug dem hvis de hjælper med at "
    "svare korrekt."
)

LIGHT_PROMPT_IDS = [f"P{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# Auto Router adapter — same OpenAI-compatible-dialect convention as the
# rest of the repo's OpenRouter-only adapters (see src/adapters/thinkingmachines.py),
# but with no panel.yaml entry, no dated pin (Auto has none to check), and a
# model string fixed to "openrouter/auto" regardless of what actually answers.
# ---------------------------------------------------------------------------


def _extract_cost(raw_usage: Optional[dict]) -> Optional[float]:
    """OpenRouter's usage object carries its own live 'cost' field (USD),
    independent of config/pricing.yaml. Handles both the single-call shape
    and the tool-loop {"call_1":..., "call_2":...} shape."""
    if not raw_usage:
        return None
    if "cost" in raw_usage:
        return raw_usage.get("cost")
    total = 0.0
    found = False
    for v in raw_usage.values():
        if isinstance(v, dict) and "cost" in v:
            total += v.get("cost") or 0.0
            found = True
    return total if found else None


class OpenRouterAutoAdapter:
    def __init__(self) -> None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set — required for openrouter/auto")
        self.client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)

    def _base_extra_body(self) -> dict:
        return {
            "include_reasoning": True,
            "reasoning": {"effort": REASONING_EFFORT},
            "usage": {"include": True},  # force cost accounting on every call
        }

    def call(self, prompt: str) -> ModelResponse:
        try:
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=AUTO_SLUG,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=THINKING_BUDGET + 512,
                extra_body=self._base_extra_body(),
            )
            latency = time.perf_counter() - t0
        except Exception as exc:
            raise AdapterError(f"openrouter_auto API error: {exc}") from exc
        response = self._to_model_response(resp, latency)
        response.warnings = extract_warnings(resp)
        return response

    def _to_model_response(self, resp, latency: float) -> ModelResponse:
        msg = resp.choices[0].message
        raw_content = msg.content or ""
        finish_reason, native_finish_reason = extract_finish_reasons(resp)

        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        if reasoning:
            answer = raw_content
        else:
            reasoning, answer = extract_think_tags(raw_content)

        usage = resp.usage
        raw = usage.model_dump() if hasattr(usage, "model_dump") else {}

        total_completion = usage.completion_tokens
        comp_details = getattr(usage, "completion_tokens_details", None)
        api_reasoning = getattr(comp_details, "reasoning_tokens", None) if comp_details else None
        if api_reasoning is not None and api_reasoning > 0:
            reasoning_tokens = api_reasoning
            output_tokens = max(0, total_completion - reasoning_tokens)
            reasoning_source = "api"
        else:
            reasoning_tokens, output_tokens = split_token_estimate(reasoning, answer, total_completion)
            reasoning_source = "text_estimate"

        if reasoning:
            trace_status = "raw"
        elif reasoning_tokens > 0:
            trace_status = "count_only"
        else:
            trace_status = "absent"

        return ModelResponse(
            answer_text=answer,
            input_tokens=usage.prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            raw_reasoning_trace=reasoning,
            trace_status=trace_status,
            reasoning_source=reasoning_source,
            latency_s=latency,
            model_version=resp.model,
            raw_usage=raw,
            served_by=extract_served_by(resp),
            finish_reason=finish_reason,
            native_finish_reason=native_finish_reason,
            request_model_id=AUTO_SLUG,
            via_openrouter=True,
        )

    def call_with_tools(self, prompt: str, tool_choice: str = "auto") -> ModelResponse:
        from src.tool_loop import call_with_tools_openai_style

        return call_with_tools_openai_style(
            model_key="openrouter_auto",
            client=self.client,
            model_id=AUTO_SLUG,
            prompt=prompt,
            max_tokens=THINKING_BUDGET + 512,
            tool_choice=tool_choice,
            base_extra_body=self._base_extra_body(),
            via_openrouter=True,
        )


def _param_dropped(warnings: Optional[list]) -> bool:
    return bool(warnings)


def _call_with_retry(fn, *args, attempts: int = 3, backoff_s: float = 5.0, **kwargs):
    """Retry transient AdapterErrors (e.g. HTTP 429) up to `attempts` times.
    ToolsNotSupportedError is never retried — it's raised unchanged for the
    caller to record as a distinct status, same convention as run_heavy()."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except ToolsNotSupportedError:
            raise
        except AdapterError as exc:
            last_exc = exc
            if attempt < attempts:
                print(f"    retry {attempt}/{attempts - 1} after: {exc}")
                time.sleep(backoff_s * attempt)
    raise last_exc


def _build_cells() -> list[dict]:
    cells: list[dict] = []
    for pid in LIGHT_PROMPT_IDS:
        cells.append({"kind": "light", "pid": pid, "condition": "baseline"})
    for domain in TASK_KEYS:
        for condition in HEAVY_CONDITIONS:
            cells.append({"kind": "heavy", "domain": domain, "condition": condition})
    assert len(cells) == 16, len(cells)
    return cells


def run() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_auto"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = OpenRouterAutoAdapter()
    prompts = load_prompts()
    heavy_tasks = load_heavy_tasks(with_facit=True)
    panel_pricing = load_pricing()
    panel_pricing_snapshot_date = panel_pricing["snapshot_date"]

    cells = _build_cells()
    total_cost = 0.0
    rows_written = 0
    stopped_reason: Optional[str] = None

    print(f"run_id={run_id}  cells={len(cells)}  passes={N_PASSES}  total_calls={len(cells) * N_PASSES}")
    print(f"price_cap_usd={PRICE_CAP_USD}  panel_pricing_snapshot_date={panel_pricing_snapshot_date}")
    print("-" * 100)

    for cell in cells:
        for pass_index in range(1, N_PASSES + 1):
            if total_cost >= PRICE_CAP_USD:
                stopped_reason = (
                    f"PRICE CAP REACHED: cumulative cost_usd={total_cost:.4f} >= "
                    f"{PRICE_CAP_USD} before cell={cell} pass={pass_index}"
                )
                print(f"\n!!! {stopped_reason}")
                break

            if cell["kind"] == "light":
                pid = cell["pid"]
                p = prompts[pid]
                prompt_text = p["prompt"]
                task_id = pid
                domain = "light"
                condition = "baseline"
                tools_available: list[str] = []
                extra = {
                    "prompt_type": p.get("type"),
                    "language_probe": p.get("language_probe"),
                    "reasoning_load": p.get("reasoning_load"),
                    "carries_correctness": p.get("carries_correctness"),
                }
            else:
                domain = cell["domain"]
                condition = cell["condition"]
                task = heavy_tasks[domain]
                task_id = task["task_id"]
                base_prompt = task["prompt"]
                prompt_text = base_prompt + (HEAVY_INVITATION if condition == "invited_auto" else "")
                tools_available = ["python_exec", "web_search"] if condition == "invited_auto" else []
                extra = {}

            row_cost = 0.0
            try:
                if condition == "invited_auto":
                    response = _call_with_retry(adapter.call_with_tools, prompt_text, tool_choice="auto")
                else:
                    response = _call_with_retry(adapter.call, prompt_text)
                warnings = response.warnings

                account = build_account(response)
                cost_usd = _extract_cost(response.raw_usage)
                row_cost = cost_usd or 0.0
                row_ts_date = datetime.now(timezone.utc).date().isoformat()

                if cell["kind"] == "light":
                    pid = cell["pid"]
                    if pid in PROGRAMMATIC_PROMPTS:
                        gr = grade_programmatic(pid, response.answer_text)
                        correct = gr.verdict == "correct"
                        extracted_answer = str(gr.extracted)
                        grading_detail = {"verdict": gr.verdict, "grading_method": "programmatic"}
                    elif p.get("carries_correctness"):
                        correct = None
                        extracted_answer = None
                        grading_detail = {
                            "grading_method": "not_run",
                            "note": (
                                "P3/P4/P8 correctness requires the offline LLM judge "
                                "(src/grader.py grade_llm); not invoked by this script "
                                "to keep its own cost/request-path scope minimal."
                            ),
                        }
                    else:
                        correct = None
                        extracted_answer = None
                        grading_detail = {"grading_method": "none", "note": "carries_correctness=false"}
                else:
                    facit_grading = heavy_tasks[domain]["facit_grading"]
                    gr = grade_heavy(domain, response.answer_text, facit_grading)
                    correct = gr.correct
                    extracted_answer = gr.extracted_answer
                    grading_detail = gr.detail

                extra = dict(extra)
                extra.update({
                    "param_dropped": _param_dropped(warnings),
                    "provider_warnings": warnings,
                    "cost_source": "openrouter_live_per_call_usage.cost",
                    "auto_call_date": row_ts_date,
                    "panel_pricing_snapshot_date": panel_pricing_snapshot_date,
                    "request_model_id_is_auto_slug": response.request_model_id == AUTO_SLUG,
                })

                save_heavy_result(
                    run_id=run_id,
                    task_id=task_id,
                    domain=domain,
                    model_key="openrouter_auto",
                    condition=condition,
                    pass_index=pass_index,
                    status="ok",
                    response=response,
                    account=account,
                    cost_usd=cost_usd,
                    pricing_snapshot_date=f"live_per_call:{row_ts_date}",
                    thinking_budget=THINKING_BUDGET,
                    reasoning_effort=REASONING_EFFORT,
                    tools_available=tools_available,
                    correct=correct,
                    extracted_answer=extracted_answer,
                    grading_detail=grading_detail,
                    extra=extra,
                    results_dir=RESULTS_DIR,
                )
                print(
                    f"  ok    {domain:<14} {condition:<12} pass={pass_index} "
                    f"routed={response.model_version:<40} served_by={str(response.served_by):<14} "
                    f"cost=${row_cost:.5f} correct={correct} trace={response.trace_status}"
                )
            except (AdapterError, ToolsNotSupportedError) as exc:
                status = "n/a_no_tool_support" if isinstance(exc, ToolsNotSupportedError) else "error"
                save_heavy_result(
                    run_id=run_id,
                    task_id=task_id,
                    domain=domain,
                    model_key="openrouter_auto",
                    condition=condition,
                    pass_index=pass_index,
                    status=status,
                    response=None,
                    account=None,
                    cost_usd=None,
                    pricing_snapshot_date=None,
                    thinking_budget=THINKING_BUDGET,
                    reasoning_effort=REASONING_EFFORT,
                    tools_available=tools_available,
                    correct=None,
                    extracted_answer=None,
                    grading_detail=None,
                    extra={**extra, "error": str(exc)},
                    results_dir=RESULTS_DIR,
                )
                print(f"  {status.upper():<8} {domain:<14} {condition:<12} pass={pass_index}  {exc}")

            total_cost += row_cost
            rows_written += 1

        if stopped_reason:
            break

    print("-" * 100)
    print(f"rows_written={rows_written}/{len(cells) * N_PASSES}  total_cost_usd={total_cost:.4f}")
    if stopped_reason:
        print(f"STOPPED EARLY: {stopped_reason}")
    print(f"results: {RESULTS_DIR / (run_id + '.jsonl')}")

    meta = {
        "run_id": run_id,
        "purpose": "Standalone candidate test of openrouter/auto against panel facit — not a panel member.",
        "request_model_id": AUTO_SLUG,
        "reasoning_effort": REASONING_EFFORT,
        "thinking_budget": THINKING_BUDGET,
        "n_passes": N_PASSES,
        "cells": len(cells),
        "planned_calls": len(cells) * N_PASSES,
        "rows_written": rows_written,
        "total_cost_usd": round(total_cost, 4),
        "price_cap_usd": PRICE_CAP_USD,
        "stopped_early": stopped_reason,
        "panel_pricing_snapshot_date": panel_pricing_snapshot_date,
        "cost_source": "openrouter_live_per_call_usage.cost (not config/pricing.yaml)",
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
    }
    import json

    meta_path = RESULTS_DIR / f"{run_id}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    run()
