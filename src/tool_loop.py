"""
Shared harness loop for OpenAI-compatible chat-completions adapters
(deepseek, zai, moonshot, openai, gemma, mistral, and the Anthropic
OpenRouter fallback path all speak this dialect).

One round = one tool-calling assistant turn, forced closed by a continuation
call made WITHOUT the tools parameter. If the model does not call a tool at
all, the first call's response IS the final answer — that's a valid row too.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from .adapters.base import (
    AdapterError,
    ModelResponse,
    estimate_tokens,
    extract_finish_reasons,
    extract_served_by,
    extract_think_tags,
    extract_warnings,
    split_token_estimate,
)
from .tools import available_tool_defs, execute_tool, to_openai_tools

# Substring match against the raw API error message — used to distinguish
# "this model provably cannot tool-call" from a generic transient failure.
TOOL_UNSUPPORTED_MARKERS = (
    "does not support tool",
    "does not support function",
    "tool use is not supported",
    "function calling is not supported",
    "tools is not supported",
    "no endpoints found that support tool use",
    "not support tools",
)


class ToolsNotSupportedError(Exception):
    """Raised when the API itself rejects the request because this model/endpoint cannot tool-call."""


@dataclass
class ToolLoopResult:
    answer_text: str
    reasoning_text: Optional[str]
    input_tokens: int
    reasoning_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_source: str
    trace_status: str
    tool_calls: list = field(default_factory=list)
    raw_tool_events: list = field(default_factory=list)
    n_api_calls: int = 1
    latency_s: float = 0.0
    model_version: str = ""
    raw_usage: dict = field(default_factory=dict)
    served_by: Optional[str] = None
    finish_reason: Optional[str] = None
    native_finish_reason: Optional[str] = None
    request_model_id: str = ""
    via_openrouter: bool = False
    warnings: Optional[list] = None


def _extract_reasoning(msg, raw_content: str) -> tuple[Optional[str], str]:
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        return reasoning, raw_content
    return extract_think_tags(raw_content)


def _call_tokens(usage) -> tuple[int, Optional[int], int, int, int, dict]:
    """
    Returns (input, api_reasoning_or_None, total_completion, cache_read, cache_write, raw_dict).
    Mirrors the per-call extraction logic already used by every single-call adapter.
    """
    raw = usage.model_dump() if hasattr(usage, "model_dump") else {}
    comp_details = getattr(usage, "completion_tokens_details", None)
    api_reasoning = getattr(comp_details, "reasoning_tokens", None) if comp_details else None
    cache_read = getattr(
        getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
    ) or getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    cache_write = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
    return (
        usage.prompt_tokens,
        api_reasoning if (api_reasoning is not None and api_reasoning > 0) else None,
        usage.completion_tokens,
        cache_read,
        cache_write,
        raw,
    )


def _raise_if_tool_unsupported(exc: Exception, model_key: str) -> None:
    msg = str(exc).lower()
    if any(marker in msg for marker in TOOL_UNSUPPORTED_MARKERS):
        raise ToolsNotSupportedError(f"{model_key}: tool-calling not supported — {exc}") from exc


def run_openai_tool_loop(
    *,
    model_key: str,
    client,
    model_id: str,
    prompt: str,
    openai_tools: list[dict],
    max_tokens: int,
    base_extra_body: Optional[dict] = None,
    tool_choice: Optional[str] = None,
    via_openrouter: bool = False,
) -> ToolLoopResult:
    """
    tool_choice: None (provider default, effectively "auto"), "auto", or
    "required" (model MUST call >=1 tool). Applies to the first call only —
    the continuation call never receives `tools`, so tool_choice is moot there.
    A "required" call that comes back with no tool_calls means the route did
    NOT enforce it — the caller (run_tools3) treats that as a possible
    adapter/routing defect, not a valid "chose not to call".

    via_openrouter: pure metadata, carried onto the result's via_openrouter
    field. The model-pin assert itself runs in the calling adapter, before
    model_id is even handed to this function — see adapters/base.py's
    assert_model_pin_honored(), which checks the request variable, not any
    response field (resp.model cannot prove what was requested; see Defect 1).
    """
    base_extra_body = dict(base_extra_body or {})
    messages: list[dict] = [{"role": "user", "content": prompt}]

    first_call_kwargs: dict = dict(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        tools=openai_tools,
        extra_body=base_extra_body or None,
    )
    if tool_choice is not None:
        first_call_kwargs["tool_choice"] = tool_choice

    t0 = time.perf_counter()
    try:
        resp1 = client.chat.completions.create(**first_call_kwargs)
    except Exception as exc:
        _raise_if_tool_unsupported(exc, model_key)
        raise

    msg1 = resp1.choices[0].message
    raw_content1 = msg1.content or ""
    reasoning1, answer1 = _extract_reasoning(msg1, raw_content1)
    finish_reason1, native_finish_reason1 = extract_finish_reasons(resp1)

    in1, api_r1, comp1, cread1, cwrite1, rawusage1 = _call_tokens(resp1.usage)
    if api_r1 is not None:
        reas1, out1 = api_r1, max(0, comp1 - api_r1)
        src1 = "api"
    else:
        reas1, out1 = split_token_estimate(reasoning1, answer1, comp1)
        src1 = "text_estimate"

    raw_tool_events: list[dict] = []
    tool_calls_log: list[dict] = []

    tool_calls = getattr(msg1, "tool_calls", None) or []

    if not tool_calls:
        latency = time.perf_counter() - t0
        trace_status = "raw" if reasoning1 else ("count_only" if reas1 > 0 else "absent")
        return ToolLoopResult(
            answer_text=answer1,
            reasoning_text=reasoning1,
            input_tokens=in1,
            reasoning_tokens=reas1,
            output_tokens=out1,
            cache_read_tokens=cread1,
            cache_write_tokens=cwrite1,
            reasoning_source=src1,
            trace_status=trace_status,
            tool_calls=[],
            raw_tool_events=[],
            n_api_calls=1,
            latency_s=latency,
            model_version=resp1.model,
            raw_usage={"call_1": rawusage1},
            served_by=extract_served_by(resp1),
            finish_reason=finish_reason1,
            native_finish_reason=native_finish_reason1,
            request_model_id=model_id,
            via_openrouter=via_openrouter,
            warnings=extract_warnings(resp1),
        )

    # --- Execute every tool call the model emitted, then force a final answer ---
    assistant_msg = {
        "role": "assistant",
        "content": msg1.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ],
    }
    messages.append(assistant_msg)

    for tc in tool_calls:
        name = tc.function.name
        raw_args = tc.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
        raw_tool_events.append({"name": name, "args_raw": raw_args, "id": tc.id})

        result = execute_tool(name, args)
        result_text = result["text"]

        tool_calls_log.append({
            "name": name,
            "args": args,
            "result_char_len": len(result_text),
            "result_token_est": estimate_tokens(result_text),
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result_text,
        })

    try:
        resp2 = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=base_extra_body or None,
            # tools omitted entirely — forces a final answer, no second round.
        )
    except Exception as exc:
        _raise_if_tool_unsupported(exc, model_key)
        raise

    msg2 = resp2.choices[0].message
    raw_content2 = msg2.content or ""
    reasoning2, answer2 = _extract_reasoning(msg2, raw_content2)
    finish_reason2, native_finish_reason2 = extract_finish_reasons(resp2)

    in2, api_r2, comp2, cread2, cwrite2, rawusage2 = _call_tokens(resp2.usage)
    if api_r2 is not None:
        reas2, out2 = api_r2, max(0, comp2 - api_r2)
        src2 = "api"
    else:
        reas2, out2 = split_token_estimate(reasoning2, answer2, comp2)
        src2 = "text_estimate"

    latency = time.perf_counter() - t0

    combined_reasoning = "\n\n".join(t for t in (reasoning1, reasoning2) if t) or None
    reasoning_source = "api" if "api" in (src1, src2) else "text_estimate"
    trace_status = "raw" if combined_reasoning else ("count_only" if (reas1 + reas2) > 0 else "absent")

    return ToolLoopResult(
        answer_text=answer2,
        reasoning_text=combined_reasoning,
        input_tokens=in1 + in2,
        reasoning_tokens=reas1 + reas2,
        output_tokens=out1 + out2,
        cache_read_tokens=cread1 + cread2,
        cache_write_tokens=cwrite1 + cwrite2,
        reasoning_source=reasoning_source,
        trace_status=trace_status,
        tool_calls=tool_calls_log,
        raw_tool_events=raw_tool_events,
        n_api_calls=2,
        latency_s=latency,
        model_version=resp2.model,
        raw_usage={"call_1": rawusage1, "call_2": rawusage2},
        served_by=extract_served_by(resp2),
        finish_reason=finish_reason2,
        native_finish_reason=native_finish_reason2,
        request_model_id=model_id,
        via_openrouter=via_openrouter,
        warnings=(extract_warnings(resp1) or []) + (extract_warnings(resp2) or []) or None,
    )


def _to_model_response(result: ToolLoopResult) -> ModelResponse:
    return ModelResponse(
        answer_text=result.answer_text,
        input_tokens=result.input_tokens,
        reasoning_tokens=result.reasoning_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_write_tokens=result.cache_write_tokens,
        raw_reasoning_trace=result.reasoning_text,
        trace_status=result.trace_status,
        reasoning_source=result.reasoning_source,
        latency_s=result.latency_s,
        model_version=result.model_version,
        raw_usage=result.raw_usage,
        tool_calls=result.tool_calls,
        raw_tool_events=result.raw_tool_events,
        n_api_calls=result.n_api_calls,
        served_by=result.served_by,
        finish_reason=result.finish_reason,
        native_finish_reason=result.native_finish_reason,
        request_model_id=result.request_model_id,
        via_openrouter=result.via_openrouter,
        warnings=result.warnings,
    )


def call_with_tools_openai_style(
    *,
    model_key: str,
    client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    base_extra_body: Optional[dict] = None,
    tool_choice: Optional[str] = None,
    via_openrouter: bool = False,
) -> ModelResponse:
    """
    One-call convenience wrapper for the (many) adapters that speak the
    OpenAI chat-completions dialect natively or via OpenRouter. Raises
    ToolsNotSupportedError unchanged (caller marks the row n/a); any other
    failure is wrapped in AdapterError, matching the existing call() contract.

    via_openrouter: pure metadata carried onto ModelResponse.via_openrouter.
    The caller must run assert_model_pin_honored() itself, before calling this
    function, when model_id is a pinned openrouter_model_id — see the adapter
    call_with_tools() methods.
    """
    openai_tools = to_openai_tools(available_tool_defs())
    try:
        result = run_openai_tool_loop(
            model_key=model_key,
            client=client,
            model_id=model_id,
            prompt=prompt,
            openai_tools=openai_tools,
            max_tokens=max_tokens,
            base_extra_body=base_extra_body,
            tool_choice=tool_choice,
            via_openrouter=via_openrouter,
        )
    except ToolsNotSupportedError:
        raise
    except Exception as exc:
        raise AdapterError(f"{model_key} tools call error: {exc}") from exc
    return _to_model_response(result)
