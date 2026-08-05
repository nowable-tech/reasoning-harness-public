"""OpenAI GPT-5.5 adapter — trace_exposure: count_only.

Raw CoT is hidden. The reasoning token COUNT is captured from the usage object.
Falls back to OpenRouter when OPENAI_API_KEY is absent (openrouter_model_id is
used in that case, which may be a different model than gpt-5.5 if unavailable).
Note: when routing via OpenRouter, reasoning token counts depend on OpenRouter's
passthrough — the smoke test will verify what is actually reported.
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
)


class OpenAIAdapter(BaseAdapter):
    required_env = ["OPENAI_API_KEY"]

    def call(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high") -> ModelResponse:
        from openai import OpenAI

        api_key, base_url, model_id, via_openrouter = self._resolve_openai_creds()
        if via_openrouter:
            assert_model_pin_honored(self.model_key, self.config, model_id)
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)

        # Pass reasoning effort through OpenRouter; for direct OpenAI o-series, reasoning
        # is always active — extra_body is a no-op but keeps the call uniform.
        extra: dict = {}
        if via_openrouter:
            extra["reasoning"] = {"effort": reasoning_effort}

        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=thinking_budget + 512,
                extra_body=extra or None,
            )
            latency = time.perf_counter() - t0
        except Exception as exc:
            raise AdapterError(f"gpt_5_5 API error: {exc}") from exc

        answer = resp.choices[0].message.content or ""
        finish_reason, native_finish_reason = extract_finish_reasons(resp)
        usage = resp.usage
        raw = usage.model_dump() if hasattr(usage, "model_dump") else {}

        # Reasoning token count — hidden CoT, count only.
        comp_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = (
            getattr(comp_details, "reasoning_tokens", 0) or 0
            if comp_details else 0
        )
        # Also try Responses API field layout
        if reasoning_tokens == 0:
            out_details = getattr(usage, "output_tokens_details", None)
            reasoning_tokens = (
                getattr(out_details, "reasoning_tokens", 0) or 0
                if out_details else 0
            )

        total_completion = usage.completion_tokens
        output_tokens = max(0, total_completion - reasoning_tokens)

        cache_read = (
            getattr(
                getattr(usage, "prompt_tokens_details", None),
                "cached_tokens",
                0,
            ) or 0
        )

        return ModelResponse(
            answer_text=answer,
            input_tokens=usage.prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=0,
            raw_reasoning_trace=None,   # raw CoT hidden by OpenAI
            trace_status="count_only" if reasoning_tokens > 0 else "absent",
            reasoning_source="api",     # reasoning_tokens always from usage fields
            latency_s=latency,
            model_version=resp.model,
            raw_usage=raw,
            served_by=extract_served_by(resp),
            finish_reason=finish_reason,
            native_finish_reason=native_finish_reason,
            request_model_id=model_id,
            via_openrouter=via_openrouter,
        )

    def call_with_tools(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high", tool_choice: str | None = None) -> ModelResponse:
        from openai import OpenAI

        from ..tool_loop import call_with_tools_openai_style

        api_key, base_url, model_id, via_openrouter = self._resolve_openai_creds()
        if via_openrouter:
            assert_model_pin_honored(self.model_key, self.config, model_id)
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)

        extra: dict = {}
        if via_openrouter:
            extra["reasoning"] = {"effort": reasoning_effort}

        return call_with_tools_openai_style(
            model_key=self.model_key,
            client=client,
            model_id=model_id,
            prompt=prompt,
            max_tokens=thinking_budget + 512,
            base_extra_body=extra,
            tool_choice=tool_choice,
            via_openrouter=via_openrouter,
        )
