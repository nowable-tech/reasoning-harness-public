"""
Standalone panel addition: Claude Opus 5, fourth version-jump entry
(Opus 4.8 -> Opus 5), added the same way fable_5/kimi_k3/gpt_5_6_sol/inkling
were — panel.yaml/pricing.yaml/model_resolver.py document the model, but this
script (not run.py's --full/--heavy, whose hardcoded FULL_MODEL_ORDER/
HEAVY_MODELS lists were never extended for those four models either) performs
the actual run, reusing every shared helper (AnthropicAdapter, save_result,
save_heavy_result, grade_heavy, compute_cost, ...) so the output schema is
byte-identical to what run_full()/run_heavy() themselves would have produced.

Same conventions as the rest of the series:
  - 10 light prompts P1-P10 (data/prompts.yaml), baseline only, 1 pass each
  - 3 heavy tasks H1-H3 (code / finance_calc / finance_interp,
    src/heavy_tasks.py), x 2 conditions (baseline, invited_auto), 5 passes
    each = 30 heavy calls
  40 calls total. thinking_budget=16384, reasoning_effort="high" (panel.yaml
  default — NOT "max", for comparability with the rest of the panel).

Forces the OpenRouter route (pops ANTHROPIC_API_KEY for the duration of the
run) for BOTH phases. This matches fable_5's actual measured channel: fable_5's
own results/full/*.jsonl rows show via_openrouter=true, not direct, even
though only --heavy's docstring claims to force that route — so "match
fable_5's channel" means both phases, not just the heavy one.

Writes results/full/<run_id>_full.jsonl + traces (save_result/save_trace,
identical schema to run_full()) and results/heavy/<run_id>_heavy.jsonl +
traces (save_heavy_result/save_heavy_trace, identical schema to run_heavy()).
Never touches run.py, any existing results file, or docs/reasoning_findings.md.

Cost guard: stops (not crashes) if cumulative cost_usd reaches PRICE_CAP_USD.
Partial results remain valid data.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.accounting import build_account
from src.adapters import PROVIDER_MAP, CredentialMissingError
from src.adapters.base import AdapterError
from src.config_loader import load_panel, load_prompts
from src.cost import compute_cost
from src.heavy_grader import grade as grade_heavy
from src.heavy_tasks import TASK_KEYS as HEAVY_TASK_KEYS
from src.heavy_tasks import load_heavy_tasks
from src.language_metric import measure_trace_language
from src.model_resolver import assert_no_silent_direct_route, print_resolution_table, resolve_models
from src.storage import save_heavy_result, save_heavy_trace, save_result, save_trace
from src.tool_loop import ToolsNotSupportedError
from src.tools import available_tool_defs

MODEL_KEY = "claude_opus_5"
LIGHT_PROMPT_IDS = [f"P{i}" for i in range(1, 11)]
HEAVY_CONDITIONS: tuple[str, ...] = ("baseline", "invited_auto")
HEAVY_REPEATS = 5
REASONING_EFFORT = "high"
PRICE_CAP_USD = 5.0

RESULTS_DIR = Path(__file__).parent / "results"

# Verbatim from run.py's TOOLS3_INVITATION / HEAVY_INVITATION — same source,
# unchanged.
HEAVY_INVITATION = (
    "\n\nDu har adgang til to værktøjer: python_exec (kør Python for eksakt "
    "beregning) og web_search (slå fakta op). Brug dem hvis de hjælper med at "
    "svare korrekt."
)


def _tool_names_used(response) -> tuple[str, ...]:
    return tuple(sorted({tc["name"] for tc in response.tool_calls}))


def main() -> int:
    panel = load_panel()
    cfg = panel[MODEL_KEY]
    assert cfg["provider"] == "anthropic"
    thinking_budget = cfg.get("thinking_budget", 16384)

    total_cost = 0.0
    stopped_reason: str | None = None
    has_failure = False

    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    if saved_key:
        print("!! ANTHROPIC_API_KEY temporarily removed — forcing OpenRouter route for both phases (matches fable_5's measured channel).")

    try:
        resolved = resolve_models(panel, [MODEL_KEY])
        hard_errors = print_resolution_table(resolved)
        assert_no_silent_direct_route(panel, [MODEL_KEY], allow_direct=False)
        if hard_errors > 0:
            print(f"\n!! {hard_errors} model(s) not resolved. Fix panel.yaml.\n")
            return 1

        cls = PROVIDER_MAP[cfg["provider"]]
        try:
            adapter = cls(MODEL_KEY, cfg)
        except CredentialMissingError as e:
            print(f"SKIPPED — {e}")
            return 1

        # ================= LIGHT PHASE (10 prompts x 1 pass) =================
        light_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_full"
        full_dir = RESULTS_DIR / "full"
        full_traces_dir = full_dir / f"{light_run_id}_traces"
        prompts = load_prompts()

        print(f"\n{'='*100}")
        print(f"  LIGHT PHASE  run_id={light_run_id}  model={MODEL_KEY}  {len(LIGHT_PROMPT_IDS)} calls")
        print(f"{'='*100}")

        for pid in LIGHT_PROMPT_IDS:
            if total_cost >= PRICE_CAP_USD:
                stopped_reason = f"PRICE CAP reached before light {pid}: cum_cost=${total_cost:.4f} >= ${PRICE_CAP_USD}"
                print(f"\n!!! {stopped_reason}")
                break

            p = prompts[pid]
            assert "facit" not in p, f"CRITICAL SECURITY VIOLATION: facit in request-path object for {pid}"
            prompt_text: str = p["prompt"]

            try:
                response = adapter.call(prompt_text, thinking_budget=thinking_budget, reasoning_effort=REASONING_EFFORT)
            except AdapterError as e:
                print(f"  {pid:<5} ERROR — {e}")
                has_failure = True
                continue

            account = build_account(response)
            cost_usd, snapshot_date = compute_cost(MODEL_KEY, account)
            total_cost += cost_usd

            lm = measure_trace_language(response.raw_reasoning_trace) if response.raw_reasoning_trace else measure_trace_language(None)

            save_result(
                run_id=light_run_id,
                model_key=MODEL_KEY,
                prompt=prompt_text,
                response=response,
                account=account,
                cost_usd=cost_usd,
                pricing_snapshot_date=snapshot_date,
                thinking_budget=thinking_budget,
                reasoning_effort=REASONING_EFFORT,
                results_dir=full_dir,
                extra={
                    "prompt_id": pid,
                    "prompt_type": p.get("type"),
                    "language_probe": p.get("language_probe"),
                    "reasoning_load": p.get("reasoning_load"),
                    "regime": cfg.get("trace_exposure"),
                    "language_metric": lm,
                },
            )
            save_trace(
                traces_dir=full_traces_dir,
                run_id=light_run_id,
                model_key=MODEL_KEY,
                prompt_id=pid,
                prompt_meta=p,
                prompt_text=prompt_text,
                answer_text=response.answer_text,
                reasoning_trace=response.raw_reasoning_trace,
                trace_status=response.trace_status,
                reasoning_tokens=account.reasoning_tokens,
                reasoning_source=response.reasoning_source,
            )
            print(
                f"  {pid:<5} in={account.input_tokens:>5} reas={account.reasoning_tokens:>5} "
                f"out={account.output_tokens:>5}  cost=${cost_usd:.5f}  {response.latency_s:>6.2f}s  "
                f"trace={response.trace_status:<11} finish={response.finish_reason}"
            )

        # ================= HEAVY PHASE (3 tasks x 2 conditions x 5 passes) =================
        heavy_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_heavy"
        heavy_dir = RESULTS_DIR / "heavy"
        heavy_traces_dir = heavy_dir / f"{heavy_run_id}_traces"
        tasks_safe = load_heavy_tasks(with_facit=False)
        tasks_facit = load_heavy_tasks(with_facit=True)

        tool_defs = available_tool_defs()
        tools_available_names = [t["name"] for t in tool_defs]

        n_heavy_calls = len(HEAVY_TASK_KEYS) * len(HEAVY_CONDITIONS) * HEAVY_REPEATS
        print(f"\n{'='*100}")
        print(f"  HEAVY PHASE  run_id={heavy_run_id}  model={MODEL_KEY}  {n_heavy_calls} calls")
        print(f"  Tools offered (invited_auto only): {', '.join(tools_available_names)}")
        print(f"{'='*100}")

        if not stopped_reason:
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

                    for pass_index in range(1, HEAVY_REPEATS + 1):
                        if total_cost >= PRICE_CAP_USD:
                            stopped_reason = (
                                f"PRICE CAP reached before {task_key}/{condition}/pass{pass_index}: "
                                f"cum_cost=${total_cost:.4f} >= ${PRICE_CAP_USD}"
                            )
                            print(f"\n!!! {stopped_reason}")
                            break

                        try:
                            if condition == "baseline":
                                resp = adapter.call(
                                    prompt_text, thinking_budget=thinking_budget, reasoning_effort=REASONING_EFFORT,
                                )
                            else:
                                resp = adapter.call_with_tools(
                                    prompt_text, thinking_budget=thinking_budget, reasoning_effort=REASONING_EFFORT,
                                    tool_choice="auto",
                                )
                        except ToolsNotSupportedError as e:
                            save_heavy_result(
                                run_id=heavy_run_id, task_id=task_id, domain=domain, model_key=MODEL_KEY,
                                condition=condition, pass_index=pass_index, status="n/a_no_tool_support",
                                response=None, account=None, cost_usd=None, pricing_snapshot_date=None,
                                thinking_budget=thinking_budget, reasoning_effort=REASONING_EFFORT,
                                tools_available=tools_available_names if condition == "invited_auto" else [],
                                correct=None, extracted_answer=None, grading_detail=None,
                                extra={"error": str(e)}, results_dir=heavy_dir,
                            )
                            print(f"  {task_key:<15} {condition:<13} pass{pass_index}  n/a — no tool support")
                            continue
                        except AdapterError as e:
                            save_heavy_result(
                                run_id=heavy_run_id, task_id=task_id, domain=domain, model_key=MODEL_KEY,
                                condition=condition, pass_index=pass_index, status="error",
                                response=None, account=None, cost_usd=None, pricing_snapshot_date=None,
                                thinking_budget=thinking_budget, reasoning_effort=REASONING_EFFORT,
                                tools_available=tools_available_names if condition == "invited_auto" else [],
                                correct=None, extracted_answer=None, grading_detail=None,
                                extra={"error": str(e)}, results_dir=heavy_dir,
                            )
                            print(f"  {task_key:<15} {condition:<13} pass{pass_index}  ERROR — {e}")
                            has_failure = True
                            continue

                        account = build_account(resp)
                        cost_usd, snapshot_date = compute_cost(MODEL_KEY, account)
                        total_cost += cost_usd
                        gr = grade_heavy(domain, resp.answer_text, facit_grading)
                        tools_used = _tool_names_used(resp)

                        save_heavy_result(
                            run_id=heavy_run_id, task_id=task_id, domain=domain, model_key=MODEL_KEY,
                            condition=condition, pass_index=pass_index, status="ok",
                            response=resp, account=account, cost_usd=cost_usd,
                            pricing_snapshot_date=snapshot_date, thinking_budget=thinking_budget,
                            reasoning_effort=REASONING_EFFORT,
                            tools_available=tools_available_names if condition == "invited_auto" else [],
                            correct=gr.correct, extracted_answer=gr.extracted_answer,
                            grading_detail=gr.detail, results_dir=heavy_dir,
                        )
                        save_heavy_trace(
                            traces_dir=heavy_traces_dir, model_key=MODEL_KEY, domain=domain,
                            condition=condition, pass_index=pass_index, prompt_text=prompt_text,
                            response=resp, status="ok", correct=gr.correct,
                            extracted_answer=gr.extracted_answer, grading_detail=gr.detail,
                        )

                        tools_str = ", ".join(tools_used) if tools_used else "—"
                        correct_str = "OK" if gr.correct else "FAIL"
                        print(
                            f"  {task_key:<15} {condition:<13} pass{pass_index} in={account.input_tokens:>5} "
                            f"reas={account.reasoning_tokens:>5} out={account.output_tokens:>5}  cost=${cost_usd:.5f}  "
                            f"tools={tools_str:<20} correct={correct_str}  finish={resp.finish_reason}"
                        )

                    if stopped_reason:
                        break
                if stopped_reason:
                    break

        print(f"\n{'='*100}")
        print(f"  TOTAL COST: ${total_cost:.4f}  (cap ${PRICE_CAP_USD})")
        if stopped_reason:
            print(f"  STOPPED EARLY: {stopped_reason}")
        print(f"  light results: results/full/{light_run_id}.jsonl")
        print(f"  heavy results: results/heavy/{heavy_run_id}.jsonl")
        print(f"{'='*100}")

        return 1 if has_failure else 0
    finally:
        if saved_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_key


if __name__ == "__main__":
    sys.exit(main())
