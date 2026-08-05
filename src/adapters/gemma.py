"""Gemma 4 adapter via OpenRouter — trace_exposure: raw.

Runs google/gemma-4-31b-it through the OpenRouter OpenAI-compatible gateway.
include_reasoning=True is permanent so OpenRouter forwards the thinking trace.
Gemma 4 exposes thinking via <think>...</think> tags in the content stream;
OpenRouter may also surface it under message.reasoning or message.reasoning_content.
All three paths are checked.

Cost: $0.12/MTok input, $0.35/MTok output (confirmed 2026-06-25).
"""
from __future__ import annotations

import os
import time

from .base import (
    AdapterError,
    BaseAdapter,
    CredentialMissingError,
    ModelResponse,
    assert_model_pin_honored,
    extract_finish_reasons,
    extract_served_by,
    extract_think_tags,
    split_token_estimate,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GemmaAdapter(BaseAdapter):
    required_env = ["OPENROUTER_API_KEY"]

    def _check_credentials(self) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise CredentialMissingError(
                "gemma_4: missing OPENROUTER_API_KEY (Gemma 4 runs via OpenRouter)"
            )

    def call(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high") -> ModelResponse:
        from openai import OpenAI

        or_key = os.environ["OPENROUTER_API_KEY"]
        model_id = self.config.get("openrouter_model_id", self.config["model_id"])
        assert_model_pin_honored(self.model_key, self.config, model_id)
        client = OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL)

        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=thinking_budget + 512,
                # reasoning.default_enabled=false for Gemma 4 — must request explicitly.
                # include_reasoning ensures OpenRouter forwards the content.
                extra_body={
                    "include_reasoning": True,
                    "reasoning": {"effort": reasoning_effort},
                },
            )
            latency = time.perf_counter() - t0
        except Exception as exc:
            raise AdapterError(f"gemma_4 API error: {exc}") from exc

        msg = resp.choices[0].message
        raw_content = msg.content or ""
        finish_reason, native_finish_reason = extract_finish_reasons(resp)

        # OpenRouter may surface thinking in a separate field after include_reasoning
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        if reasoning:
            answer = raw_content
        else:
            # Gemma's native thinking format: <think>...</think> in content stream
            reasoning, answer = extract_think_tags(raw_content)

        usage = resp.usage
        raw = usage.model_dump() if hasattr(usage, "model_dump") else {}

        total_completion = usage.completion_tokens
        comp_details = getattr(usage, "completion_tokens_details", None)
        api_reasoning = (
            getattr(comp_details, "reasoning_tokens", None) if comp_details else None
        )
        # Use api_reasoning only when strictly positive — a zero means the gateway
        # doesn't track thinking tokens for this model, not that there are none.
        if api_reasoning is not None and api_reasoning > 0:
            reasoning_tokens = api_reasoning
            output_tokens = max(0, total_completion - reasoning_tokens)
            reasoning_source = "api"
        else:
            reasoning_tokens, output_tokens = split_token_estimate(
                reasoning, answer, total_completion
            )
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
            request_model_id=model_id,
            via_openrouter=True,
        )

    def call_with_tools(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high", tool_choice: str | None = None) -> ModelResponse:
        from openai import OpenAI

        from ..tool_loop import call_with_tools_openai_style

        or_key = os.environ["OPENROUTER_API_KEY"]
        model_id = self.config.get("openrouter_model_id", self.config["model_id"])
        assert_model_pin_honored(self.model_key, self.config, model_id)
        client = OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL)

        return call_with_tools_openai_style(
            model_key=self.model_key,
            client=client,
            model_id=model_id,
            prompt=prompt,
            max_tokens=thinking_budget + 512,
            tool_choice=tool_choice,
            base_extra_body={
                "include_reasoning": True,
                "reasoning": {"effort": reasoning_effort},
            },
            via_openrouter=True,
        )
