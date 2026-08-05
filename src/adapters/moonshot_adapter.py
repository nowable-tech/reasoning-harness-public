"""Moonshot AI Kimi K2.7 adapter — trace_exposure: raw.

Uses the OpenAI-compatible endpoint at api.moonshot.cn. Falls back to OpenRouter
when MOONSHOT_API_KEY is absent. Kimi reasoning models expose raw CoT via
reasoning_content; falls back to <think> tag parsing.
Verify model_id and field names against Moonshot AI docs at run time.
"""
from __future__ import annotations

import time

from .base import (
    AdapterError,
    BaseAdapter,
    ModelResponse,
    assert_model_pin_honored,
    extract_finish_reasons,
    extract_served_by,
    extract_think_tags,
    split_token_estimate,
)


class MoonshotAdapter(BaseAdapter):
    required_env = ["MOONSHOT_API_KEY"]

    def call(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high") -> ModelResponse:
        from openai import OpenAI

        api_key, base_url, model_id, via_or = self._resolve_openai_creds()
        if via_or:
            assert_model_pin_honored(self.model_key, self.config, model_id)
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)

        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=thinking_budget + 512,
                # include_reasoning ensures reasoning_content passthrough.
                # Kimi K2.7 has reasoning.mandatory=true so effort is advisory here,
                # but we set it explicitly for experiment consistency.
                extra_body={
                    "include_reasoning": True,
                    "reasoning": {"effort": reasoning_effort},
                },
            )
            latency = time.perf_counter() - t0
        except Exception as exc:
            raise AdapterError(f"kimi_k2_7 API error: {exc}") from exc

        msg = resp.choices[0].message
        answer = msg.content or ""
        finish_reason, native_finish_reason = extract_finish_reasons(resp)

        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(msg, "reasoning", None)
        if reasoning is None:
            reasoning, answer = extract_think_tags(answer)

        usage = resp.usage
        raw = usage.model_dump() if hasattr(usage, "model_dump") else {}

        total_completion = usage.completion_tokens
        comp_details = getattr(usage, "completion_tokens_details", None)
        api_reasoning = (
            getattr(comp_details, "reasoning_tokens", None) if comp_details else None
        )
        if api_reasoning is not None and api_reasoning > 0:
            reasoning_tokens = api_reasoning
            output_tokens = max(0, total_completion - reasoning_tokens)
            reasoning_source = "api"
        else:
            reasoning_tokens, output_tokens = split_token_estimate(
                reasoning, answer, total_completion
            )
            reasoning_source = "text_estimate"

        cache_read = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_write = getattr(usage, "prompt_cache_miss_tokens", 0) or 0

        if reasoning:
            trace_status = "raw"
        elif reasoning_tokens > 0:
            trace_status = "count_only"  # text stripped (e.g. via OpenRouter)
        else:
            trace_status = "absent"

        return ModelResponse(
            answer_text=answer,
            input_tokens=usage.prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
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
            via_openrouter=via_or,
        )

    def call_with_tools(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high", tool_choice: str | None = None) -> ModelResponse:
        from openai import OpenAI

        from ..tool_loop import call_with_tools_openai_style

        api_key, base_url, model_id, via_or = self._resolve_openai_creds()
        if via_or:
            assert_model_pin_honored(self.model_key, self.config, model_id)
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)

        return call_with_tools_openai_style(
            model_key=self.model_key,
            client=client,
            model_id=model_id,
            prompt=prompt,
            max_tokens=thinking_budget + 512,
            base_extra_body={
                "include_reasoning": True,
                "reasoning": {"effort": reasoning_effort},
            },
            tool_choice=tool_choice,
            via_openrouter=via_or,
        )
