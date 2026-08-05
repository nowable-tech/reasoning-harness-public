from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .adapters.base import ModelResponse
    from .accounting import TokenAccount
    from .judge import JudgeResponse

RESULTS_DIR = Path(__file__).parent.parent / "results"
PHASE2_DIR = RESULTS_DIR / "phase2"
TOOLS_DIR = RESULTS_DIR / "tools"
TOOLS3_DIR = RESULTS_DIR / "tools3"
VARIANCE_DIR = RESULTS_DIR / "variance"
HEAVY_DIR = RESULTS_DIR / "heavy"


def save_result(
    *,
    run_id: str,
    model_key: str,
    prompt: str,
    response: "ModelResponse",
    account: "TokenAccount",
    cost_usd: float,
    pricing_snapshot_date: str,
    thinking_budget: int,
    reasoning_effort: str,
    results_dir: Optional[Path] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Persist one (model, prompt, run) record to results/<run_id>.jsonl.
    Each line is a self-contained JSON object.
    Correctness fields (score, correct) are reserved for Phase 3 — left absent here.
    """
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "model_version": response.model_version,
        "prompt": prompt,
        "thinking_budget": thinking_budget,
        "reasoning_effort": reasoning_effort,
        "answer_text": response.answer_text,
        "raw_reasoning_trace": response.raw_reasoning_trace,
        "trace_status": response.trace_status,
        "tokens": {
            "input": account.input_tokens,
            "reasoning": account.reasoning_tokens,
            "reasoning_source": response.reasoning_source,
            "output": account.output_tokens,
            "cache_read": account.cache_read_tokens,
            "cache_write": account.cache_write_tokens,
            "reasoning_share": round(account.reasoning_share, 4),
        },
        "cost_usd": cost_usd,
        "pricing_snapshot_date": pricing_snapshot_date,
        "latency_s": round(response.latency_s, 3),
        "raw_usage": response.raw_usage,
        # request_model_id: the exact string sent in the API call (Defect 1) —
        # model_version above is resp.model (the response label), which cannot
        # prove what was requested; see adapters/base.py:assert_model_pin_honored.
        "request_model_id": response.request_model_id,
        "via_openrouter": response.via_openrouter,
        # Phase 3 placeholders — not populated here:
        # "correct": null,
        # "score": null,
    }
    if extra:
        record.update(extra)

    out_dir = results_dir if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def save_trace(
    *,
    traces_dir: Path,
    run_id: str,
    model_key: str,
    prompt_id: str,
    prompt_meta: dict,
    prompt_text: str,
    answer_text: str,
    reasoning_trace: Optional[str],
    trace_status: str,
    reasoning_tokens: int,
    reasoning_source: str,
) -> Path:
    """
    Write a human-readable plain-text file with the raw reasoning trace.
    Saved to traces_dir/<model_key>_<prompt_id>.txt so a reader can eyeball
    the language probe without parsing JSONL.
    """
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{model_key}_{prompt_id}.txt"

    p_type = prompt_meta.get("type", "")
    p_probe = prompt_meta.get("language_probe", "")
    sep = "=" * 72

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"run_id:           {run_id}\n")
        f.write(f"model:            {model_key}\n")
        f.write(f"prompt_id:        {prompt_id}  ({p_type} / {p_probe})\n")
        f.write(f"trace_status:     {trace_status}\n")
        f.write(f"reasoning_tokens: {reasoning_tokens}  [{reasoning_source}]\n")
        f.write(f"\n{sep}\nPROMPT\n{sep}\n")
        f.write(prompt_text.strip() + "\n")
        f.write(f"\n{sep}\nREASONING TRACE\n{sep}\n")
        if reasoning_trace:
            f.write(reasoning_trace.strip() + "\n")
        else:
            f.write(f"[{trace_status} — reasoning text not exposed]\n")
        f.write(f"\n{sep}\nANSWER\n{sep}\n")
        f.write(answer_text.strip() + "\n")

    return path


def save_langcost_result(
    *,
    run_id: str,
    model_key: str,
    task_id: str,
    prompt_lang: str,
    prompt_text: str,
    response: "ModelResponse",
    account: "TokenAccount",
    cost_usd: float,
    pricing_snapshot_date: str,
    thinking_budget: int,
    reasoning_effort: str,
    reasoning_chars: int,
    output_chars: int,
    regime: str,
    language_metric: dict,
    results_dir: Path,
) -> dict:
    """
    Persist one (model, task, lang) langcost record to results/langcost/<run_id>.jsonl.
    Fields follow the Phase 1 structure, extended with per-language dimensions.
    """
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "model_version": response.model_version,
        "task_id": task_id,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
        "thinking_budget": thinking_budget,
        "reasoning_effort": reasoning_effort,
        "answer_text": response.answer_text,
        "raw_reasoning_trace": response.raw_reasoning_trace,
        "trace_status": response.trace_status,
        "tokens": {
            "input": account.input_tokens,
            "reasoning": account.reasoning_tokens,
            "reasoning_source": response.reasoning_source,
            "output": account.output_tokens,
            "cache_read": account.cache_read_tokens,
            "cache_write": account.cache_write_tokens,
            "reasoning_share": round(account.reasoning_share, 4),
        },
        "reasoning_chars": reasoning_chars,
        "output_chars": output_chars,
        "cost_usd": cost_usd,
        "pricing_snapshot_date": pricing_snapshot_date,
        "latency_s": round(response.latency_s, 3),
        "regime": regime,
        "language_metric": language_metric,
        "raw_usage": response.raw_usage,
        "request_model_id": response.request_model_id,
        "via_openrouter": response.via_openrouter,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_id}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def save_langcost_trace(
    *,
    traces_dir: Path,
    model_key: str,
    task_id: str,
    prompt_lang: str,
    prompt_text: str,
    answer_text: str,
    reasoning_trace: Optional[str],
    trace_status: str,
    reasoning_tokens: int,
    reasoning_source: str,
) -> Path:
    """
    Write human-readable trace file to traces_dir/<model>_<task>_<lang>.txt.
    """
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{model_key}_{task_id}_{prompt_lang}.txt"

    sep = "=" * 72
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"model:            {model_key}\n")
        f.write(f"task_id:          {task_id}\n")
        f.write(f"prompt_lang:      {prompt_lang}\n")
        f.write(f"trace_status:     {trace_status}\n")
        f.write(f"reasoning_tokens: {reasoning_tokens}  [{reasoning_source}]\n")
        f.write(f"\n{sep}\nPROMPT\n{sep}\n")
        f.write(prompt_text.strip() + "\n")
        f.write(f"\n{sep}\nREASONING TRACE\n{sep}\n")
        if reasoning_trace:
            f.write(reasoning_trace.strip() + "\n")
        else:
            f.write(f"[{trace_status} — reasoning text not exposed]\n")
        f.write(f"\n{sep}\nANSWER\n{sep}\n")
        f.write(answer_text.strip() + "\n")

    return path


def save_phase2_result(
    *,
    run_id: str,
    source_run_id: str,
    model_key: str,
    prompt_id: str,
    prompt_type: str,
    language_probe: str,
    reasoning_load: str,
    judge_key: str,
    judge_response: "JudgeResponse",
    agreement: Optional[dict] = None,
    phase1_language_metric: Optional[dict] = None,
    trace_status: str,
    results_dir: Optional[Path] = None,
) -> dict:
    """
    Persist one (model, prompt, judge) Phase 2 legibility record.

    FIREWALL: no correctness or faithfulness fields are stored here.
    Legibility scores are kept strictly separate from Phase 1 economy records.
    """
    record = {
        "phase": 2,
        "run_id": run_id,
        "source_run_id": source_run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "prompt_id": prompt_id,
        "prompt_type": prompt_type,
        "language_probe": language_probe,
        "reasoning_load": reasoning_load,
        "trace_status": trace_status,
        "judge": judge_key,
        "judge_model_version": judge_response.model_version,
        # Legibility scores — do not mix into economy or correctness tables
        "scores": judge_response.scores,
        "justifications": judge_response.justifications,
        "parse_ok": judge_response.parse_ok,
        "parse_error": judge_response.parse_error,
        "agreement": agreement,
        # Phase 1 language metric carried for reference only — not re-scored here
        "phase1_language_metric": phase1_language_metric,
        "judge_tokens": {
            "input": judge_response.input_tokens,
            "output": judge_response.output_tokens,
        },
        "cost_usd": judge_response.cost_usd,
        "latency_s": round(judge_response.latency_s, 3),
    }

    out_dir = results_dir if results_dir is not None else PHASE2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


# ---------------------------------------------------------------------------
# --tools phase (tool-offload experiment: arm=baseline | arm=tools)
# ---------------------------------------------------------------------------

def save_tools_result(
    *,
    run_id: str,
    model_key: str,
    prompt_id: str,
    arm: str,
    status: str,
    response: Optional["ModelResponse"],
    account: Optional["TokenAccount"],
    cost_usd: Optional[float],
    pricing_snapshot_date: Optional[str],
    thinking_budget: int,
    reasoning_effort: str,
    tools_available: list[str],
    extra: Optional[dict] = None,
    results_dir: Optional[Path] = None,
) -> dict:
    """
    Persist one (model, prompt, arm) row to results/tools/<run_id>_tools.jsonl.

    status: "ok" | "n/a_no_tool_support" | "error". When status != "ok" the
    response/account/cost fields may be None — the row still gets written so
    the n/a is visible as data, not silently dropped.
    """
    record: dict = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "prompt_id": prompt_id,
        "arm": arm,
        "status": status,
        "thinking_budget": thinking_budget,
        "reasoning_effort": reasoning_effort,
        "tools_available": tools_available,
    }

    if response is not None and account is not None:
        record.update({
            "model_version": response.model_version,
            "answer_text": response.answer_text,
            "raw_reasoning_trace": response.raw_reasoning_trace,
            "trace_status": response.trace_status,
            "n_api_calls": response.n_api_calls,
            "tokens": {
                "input": account.input_tokens,
                "reasoning": account.reasoning_tokens,
                "reasoning_source": response.reasoning_source,
                "output": account.output_tokens,
                "cache_read": account.cache_read_tokens,
                "cache_write": account.cache_write_tokens,
                "reasoning_share": round(account.reasoning_share, 4),
            },
            "cost_usd": cost_usd,
            "pricing_snapshot_date": pricing_snapshot_date,
            "latency_s": round(response.latency_s, 3),
            "raw_usage": response.raw_usage,
            "tool_calls": response.tool_calls,
            "raw_tool_events": response.raw_tool_events,
            "request_model_id": response.request_model_id,
            "via_openrouter": response.via_openrouter,
        })
    else:
        record.update({
            "model_version": None,
            "answer_text": None,
            "raw_reasoning_trace": None,
            "trace_status": None,
            "n_api_calls": 0,
            "tokens": None,
            "cost_usd": None,
            "pricing_snapshot_date": pricing_snapshot_date,
            "latency_s": None,
            "raw_usage": None,
            "tool_calls": [],
            "raw_tool_events": [],
            "request_model_id": None,
            "via_openrouter": None,
        })

    if extra:
        record.update(extra)

    out_dir = results_dir if results_dir is not None else TOOLS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def save_tools_trace(
    *,
    traces_dir: Path,
    model_key: str,
    prompt_id: str,
    arm: str,
    prompt_text: str,
    response: Optional["ModelResponse"],
    status: str,
    pass_index: Optional[int] = None,
) -> Path:
    """Write a human-readable trace to traces_dir/<model>_<prompt>_<arm>[_pass<N>].txt."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_pass{pass_index}" if pass_index is not None else ""
    path = traces_dir / f"{model_key}_{prompt_id}_{arm}{suffix}.txt"
    sep = "=" * 72

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"model:      {model_key}\n")
        f.write(f"prompt_id:  {prompt_id}\n")
        f.write(f"arm:        {arm}\n")
        f.write(f"status:     {status}\n")
        f.write(f"\n{sep}\nPROMPT\n{sep}\n")
        f.write(prompt_text.strip() + "\n")

        if response is None:
            f.write(f"\n{sep}\n[{status} — no response]\n{sep}\n")
            return path

        f.write(f"\n{sep}\nREASONING TRACE\n{sep}\n")
        if response.raw_reasoning_trace:
            f.write(response.raw_reasoning_trace.strip() + "\n")
        else:
            f.write(f"[{response.trace_status} — reasoning text not exposed]\n")

        f.write(f"\n{sep}\nTOOL CALLS (executed)\n{sep}\n")
        if response.tool_calls:
            for tc in response.tool_calls:
                f.write(f"- {tc['name']}({tc['args']!r})\n")
                f.write(f"  result_char_len={tc['result_char_len']}  result_token_est={tc['result_token_est']}\n")
        else:
            f.write("(none)\n")

        f.write(f"\n{sep}\nRAW TOOL EVENTS (as emitted, incl. unknown/serverside)\n{sep}\n")
        if response.raw_tool_events:
            for ev in response.raw_tool_events:
                f.write(f"- {ev}\n")
        else:
            f.write("(none)\n")

        f.write(f"\n{sep}\nANSWER\n{sep}\n")
        f.write(response.answer_text.strip() + "\n")

    return path


# ---------------------------------------------------------------------------
# --variance phase (pinned-version repro run for cross-run variance analysis)
# ---------------------------------------------------------------------------

def save_variance_result(
    *,
    run_id: str,
    model_key: str,
    prompt_id: str,
    pass_index: int,
    pinned_model_id: str,
    status: str,
    response: Optional["ModelResponse"],
    account: Optional["TokenAccount"],
    cost_usd: Optional[float],
    pricing_snapshot_date: Optional[str],
    thinking_budget: int,
    reasoning_effort: str,
    extra: Optional[dict] = None,
    results_dir: Optional[Path] = None,
) -> dict:
    """
    Persist one (model, prompt, pass) row to results/variance/<run_id>.jsonl.

    status: "ok" | "dead_pin" | "error". pinned_model_id is the exact string we
    requested. response.model_version is what the provider echoed back — NOT
    proof of what was requested (diagnostics on 2026-07-14 showed OpenRouter
    echoes the same canonical label whether the dated pin or the undated slug
    is sent). request_model_id is the field that actually proves what was
    sent — it and pinned_model_id must match; see
    adapters/base.py:assert_model_pin_honored(), which hard-stops on mismatch
    before this row would ever be written. served_by is OpenRouter's raw
    backend fingerprint.
    """
    record: dict = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "prompt_id": prompt_id,
        "pass_index": pass_index,
        "pinned_model_id": pinned_model_id,
        "status": status,
        "thinking_budget": thinking_budget,
        "reasoning_effort": reasoning_effort,
    }

    if response is not None and account is not None:
        record.update({
            "model_version": response.model_version,
            "served_by": response.served_by,
            "answer_text": response.answer_text,
            "raw_reasoning_trace": response.raw_reasoning_trace,
            "trace_status": response.trace_status,
            "tokens": {
                "input": account.input_tokens,
                "reasoning": account.reasoning_tokens,
                "reasoning_source": response.reasoning_source,
                "output": account.output_tokens,
                "cache_read": account.cache_read_tokens,
                "cache_write": account.cache_write_tokens,
                "reasoning_share": round(account.reasoning_share, 4),
            },
            "cost_usd": cost_usd,
            "pricing_snapshot_date": pricing_snapshot_date,
            "latency_s": round(response.latency_s, 3),
            "raw_usage": response.raw_usage,
            "request_model_id": response.request_model_id,
            "via_openrouter": response.via_openrouter,
        })
    else:
        record.update({
            "model_version": None,
            "served_by": None,
            "answer_text": None,
            "raw_reasoning_trace": None,
            "trace_status": None,
            "tokens": None,
            "cost_usd": None,
            "pricing_snapshot_date": pricing_snapshot_date,
            "latency_s": None,
            "raw_usage": None,
            "request_model_id": None,
            "via_openrouter": None,
        })

    if extra:
        record.update(extra)

    out_dir = results_dir if results_dir is not None else VARIANCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


# ---------------------------------------------------------------------------
# --heavy phase (heavy tasks: code + finance_calc + finance_interp;
# baseline | invited_auto; correctness as quality control)
# ---------------------------------------------------------------------------

def save_heavy_result(
    *,
    run_id: str,
    task_id: str,
    domain: str,
    model_key: str,
    condition: str,
    pass_index: int,
    status: str,
    response: Optional["ModelResponse"],
    account: Optional["TokenAccount"],
    cost_usd: Optional[float],
    pricing_snapshot_date: Optional[str],
    thinking_budget: int,
    reasoning_effort: str,
    tools_available: list[str],
    correct: Optional[bool],
    extracted_answer: Optional[str],
    grading_detail: Optional[dict],
    extra: Optional[dict] = None,
    results_dir: Optional[Path] = None,
) -> dict:
    """
    Persist one (task, model, condition, pass) row to results/heavy/<run_id>.jsonl.

    status: "ok" | "n/a_no_tool_support" | "error". When status != "ok",
    response/account/cost/correct fields are None — the row is still written
    so the failure is visible as data, not silently dropped.

    finish_reason / native_finish_reason: raw stop signal from the provider for
    the final call in the chain (see adapters/base.py:extract_finish_reasons).
    truncated: computed, not provider-asserted — True when the completion hit
    the request's max_tokens cap. Every adapter exercised by --heavy/--heavy-recap
    routes through OpenRouter with max_tokens=thinking_budget+512 (the direct
    Anthropic adaptive-thinking budget is never used here, since ANTHROPIC_API_KEY
    is popped for this phase — see run_heavy()), so that offset is fixed here
    rather than threaded through as a parameter.
    """
    record: dict = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "domain": domain,
        "model_key": model_key,
        "condition": condition,
        "pass_index": pass_index,
        "status": status,
        "thinking_budget": thinking_budget,
        "reasoning_effort": reasoning_effort,
        "tools_available": tools_available,
        "correct": correct,
        "extracted_answer": extracted_answer,
        "grading_detail": grading_detail,
    }

    if response is not None and account is not None:
        completion_tokens = account.output_tokens + account.reasoning_tokens
        request_max_tokens = thinking_budget + 512
        record.update({
            "model_version": response.model_version,
            "served_by": response.served_by,
            "answer_text": response.answer_text,
            "raw_reasoning_trace": response.raw_reasoning_trace,
            "trace_status": response.trace_status,
            "n_api_calls": response.n_api_calls,
            "tokens": {
                "input": account.input_tokens,
                "reasoning": account.reasoning_tokens,
                "reasoning_source": response.reasoning_source,
                "output": account.output_tokens,
                "cache_read": account.cache_read_tokens,
                "cache_write": account.cache_write_tokens,
                "reasoning_share": round(account.reasoning_share, 4),
            },
            "cost_usd": cost_usd,
            "pricing_snapshot_date": pricing_snapshot_date,
            "latency_s": round(response.latency_s, 3),
            "raw_usage": response.raw_usage,
            "tool_calls": response.tool_calls,
            "raw_tool_events": response.raw_tool_events,
            "finish_reason": response.finish_reason,
            "native_finish_reason": response.native_finish_reason,
            "request_max_tokens": request_max_tokens,
            "truncated": completion_tokens >= request_max_tokens,
            "request_model_id": response.request_model_id,
            "via_openrouter": response.via_openrouter,
        })
    else:
        record.update({
            "model_version": None,
            "served_by": None,
            "answer_text": None,
            "raw_reasoning_trace": None,
            "trace_status": None,
            "n_api_calls": 0,
            "tokens": None,
            "cost_usd": None,
            "pricing_snapshot_date": pricing_snapshot_date,
            "latency_s": None,
            "raw_usage": None,
            "request_model_id": None,
            "via_openrouter": None,
            "tool_calls": [],
            "raw_tool_events": [],
            "finish_reason": None,
            "native_finish_reason": None,
            "request_max_tokens": None,
            "truncated": None,
        })

    if extra:
        record.update(extra)

    out_dir = results_dir if results_dir is not None else HEAVY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def save_heavy_trace(
    *,
    traces_dir: Path,
    model_key: str,
    domain: str,
    condition: str,
    pass_index: int,
    prompt_text: str,
    response: Optional["ModelResponse"],
    status: str,
    correct: Optional[bool],
    extracted_answer: Optional[str],
    grading_detail: Optional[dict],
) -> Path:
    """Write a human-readable trace to traces_dir/<model>_<domain>_<condition>_pass<N>.txt."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{model_key}_{domain}_{condition}_pass{pass_index}.txt"
    sep = "=" * 72

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"model:      {model_key}\n")
        f.write(f"domain:     {domain}\n")
        f.write(f"condition:  {condition}\n")
        f.write(f"pass_index: {pass_index}\n")
        f.write(f"status:     {status}\n")
        f.write(f"correct:    {correct}\n")
        f.write(f"extracted_answer: {extracted_answer!r}\n")
        f.write(f"grading_detail:   {grading_detail!r}\n")
        f.write(f"\n{sep}\nPROMPT\n{sep}\n")
        f.write(prompt_text.strip() + "\n")

        if response is None:
            f.write(f"\n{sep}\n[{status} — no response]\n{sep}\n")
            return path

        f.write(f"\n{sep}\nREASONING TRACE\n{sep}\n")
        if response.raw_reasoning_trace:
            f.write(response.raw_reasoning_trace.strip() + "\n")
        else:
            f.write(f"[{response.trace_status} — reasoning text not exposed]\n")

        f.write(f"\n{sep}\nTOOL CALLS (executed)\n{sep}\n")
        if response.tool_calls:
            for tc in response.tool_calls:
                f.write(f"- {tc['name']}({tc['args']!r})\n")
                f.write(f"  result_char_len={tc['result_char_len']}  result_token_est={tc['result_token_est']}\n")
        else:
            f.write("(none)\n")

        f.write(f"\n{sep}\nRAW TOOL EVENTS (as emitted, incl. unknown/serverside)\n{sep}\n")
        if response.raw_tool_events:
            for ev in response.raw_tool_events:
                f.write(f"- {ev}\n")
        else:
            f.write("(none)\n")

        f.write(f"\n{sep}\nANSWER\n{sep}\n")
        f.write(response.answer_text.strip() + "\n")

    return path
