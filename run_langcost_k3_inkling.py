"""
Standalone supplement: language-cost experiment (docs/forsoegsspec_sprog_omkostning_reasoning.md)
for the two models missing from the original run -- kimi_k3 and inkling.
Both were added to the panel after 20260626T162923_langcost_full ran, so
LANGCOST_MODELS in run.py (still just the original 5) never covered them.

Same design as the original run: 6 culture-neutral tasks (M1-M6,
data/prompts_multilang.yaml) x 3 language variants (da/en/zh) x 1 pass =
18 calls per model, 36 total. thinking_budget=16384, reasoning_effort="high"
-- identical to the original run and to the rest of the panel. Both models
have trace_exposure=raw, satisfying the experiment's raw-trace requirement
(same reason Opus/Sonnet/GPT were excluded from the original 5).

Reuses save_langcost_result/save_langcost_trace unchanged, so the output
schema is byte-identical to the original run's. Writes to a NEW directory,
results/sprog/<run_id>.jsonl (not results/langcost/, per instruction) --
never touches the original 90-call file or any other existing results.

Cost guard: stops (not crashes) if cumulative cost_usd reaches PRICE_CAP_USD.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.accounting import build_account
from src.adapters import PROVIDER_MAP, CredentialMissingError
from src.adapters.base import AdapterError
from src.config_loader import load_experiment, load_multilang_prompts, load_panel
from src.cost import compute_cost
from src.language_metric import measure_trace_language
from src.model_resolver import assert_no_silent_direct_route, print_resolution_table, resolve_models
from src.storage import save_langcost_result, save_langcost_trace

MODEL_KEYS = ["kimi_k3", "inkling"]
LANGS = ["da", "en", "zh"]
THINKING_BUDGET = 16384
PRICE_CAP_USD = 3.0

RESULTS_DIR = Path(__file__).parent / "results" / "sprog"


def main() -> int:
    panel = load_panel()
    experiment = load_experiment()
    reasoning_effort: str = experiment.get("reasoning_effort", "high")
    prompts = load_multilang_prompts()
    all_task_ids = sorted(prompts.keys(), key=lambda t: int(t[1:]))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_sprog_k3_inkling"
    traces_dir = RESULTS_DIR / f"{run_id}_traces"

    n_calls = len(all_task_ids) * len(LANGS) * len(MODEL_KEYS)
    print(f"\n{'='*110}")
    print(f"  Language-cost supplement (kimi_k3 + inkling)   run_id={run_id}")
    print(f"  Grid: {len(all_task_ids)} tasks x {len(LANGS)} langs x {len(MODEL_KEYS)} models = {n_calls} calls")
    print(f"  thinking_budget={THINKING_BUDGET}  reasoning_effort={reasoning_effort!r}")
    print(f"{'='*110}")

    resolved = resolve_models(panel, MODEL_KEYS)
    print_resolution_table(resolved)
    assert_no_silent_direct_route(panel, MODEL_KEYS, allow_direct=False)
    print(
        "\n  NOTE: catalog mismatches above are not a gate by themselves -- dated snapshot"
        " slugs (kimi_k3, inkling) can be absent from OpenRouter's /models listing while"
        " still callable. Same convention as run_heavy() in run.py: the real protection is"
        " assert_model_pin_honored() checking request_model_id before every call.\n"
    )

    total_cost = 0.0
    stopped_reason: str | None = None
    has_failure = False

    col_hdr = (
        f"  {'Model':<14} {'Task':<5} {'Lang':<4} {'Inp':>6} {'Reas':>7} {'Out':>6}  "
        f"{'ReasChars':>10}  {'Cost($)':>10}  {'Lat':>7}  {'TraceLang':<10} Status"
    )

    for task_id in all_task_ids:
        p = prompts[task_id]
        assert "facit" not in p, f"CRITICAL SECURITY VIOLATION: facit in request-path object for {task_id}"
        p_type = p.get("type", "?")
        p_load = p.get("reasoning_load", "?")

        for lang in LANGS:
            prompt_text: str = p["variants"][lang]

            print(f"\n{'='*110}")
            print(f"  [{task_id}]  lang={lang}  type={p_type}  load={p_load}")
            print(f"  {prompt_text[:100].strip()!r}")
            print(f"{'='*110}")
            print(col_hdr)

            for key in MODEL_KEYS:
                if total_cost >= PRICE_CAP_USD:
                    stopped_reason = (
                        f"PRICE CAP reached before {key}/{task_id}/{lang}: "
                        f"cum_cost=${total_cost:.4f} >= ${PRICE_CAP_USD}"
                    )
                    print(f"\n!!! {stopped_reason}")
                    break

                cfg = panel[key]
                provider = cfg["provider"]
                regime = cfg.get("trace_exposure", "raw")
                thinking_budget = cfg.get("thinking_budget", THINKING_BUDGET)

                cls = PROVIDER_MAP[provider]
                try:
                    adapter = cls(key, cfg)
                except CredentialMissingError as e:
                    print(f"  {key:<14} SKIPPED — {e}")
                    continue

                try:
                    response = adapter.call(
                        prompt_text, thinking_budget=thinking_budget, reasoning_effort=reasoning_effort,
                    )
                except AdapterError as e:
                    print(f"  {key:<14} {task_id:<5} {lang:<4} ERROR — {e}")
                    has_failure = True
                    continue

                account = build_account(response)
                cost_usd, snapshot_date = compute_cost(key, account)
                total_cost += cost_usd

                reasoning_chars = len(response.raw_reasoning_trace or "")
                lm = measure_trace_language(response.raw_reasoning_trace)

                save_langcost_result(
                    run_id=run_id, model_key=key, task_id=task_id, prompt_lang=lang,
                    prompt_text=prompt_text, response=response, account=account, cost_usd=cost_usd,
                    pricing_snapshot_date=snapshot_date, thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort, reasoning_chars=reasoning_chars,
                    output_chars=len(response.answer_text or ""), regime=regime,
                    language_metric=lm, results_dir=RESULTS_DIR,
                )
                save_langcost_trace(
                    traces_dir=traces_dir, model_key=key, task_id=task_id, prompt_lang=lang,
                    prompt_text=prompt_text, answer_text=response.answer_text,
                    reasoning_trace=response.raw_reasoning_trace, trace_status=response.trace_status,
                    reasoning_tokens=account.reasoning_tokens, reasoning_source=response.reasoning_source,
                )

                lang_str = lm.get("primary_trace_language") or "?"
                print(
                    f"  {key:<14} {task_id:<5} {lang:<4} {account.input_tokens:>6} {account.reasoning_tokens:>7}"
                    f" {account.output_tokens:>6}  {reasoning_chars:>10}  ${cost_usd:>9.5f}  "
                    f"{response.latency_s:>6.2f}s  {lang_str:<10} {response.trace_status}"
                )

            if stopped_reason:
                break
        if stopped_reason:
            break

    print(f"\n{'='*110}")
    print(f"  TOTAL COST: ${total_cost:.4f}  (cap ${PRICE_CAP_USD})")
    if stopped_reason:
        print(f"  STOPPED EARLY: {stopped_reason}")
    print(f"  results: results/sprog/{run_id}.jsonl")
    print(f"{'='*110}")

    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(main())
