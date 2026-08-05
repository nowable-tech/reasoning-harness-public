"""Mistral Medium 3.5 adapter — trace_exposure: raw (expected).

Runs mistralai/mistral-medium-3.5 through the OpenRouter OpenAI-compatible
gateway. The smoke test MUST confirm that OpenRouter actually forwards the raw
reasoning trace. If only a summary or nothing is returned, stop and report to
Lars — Mistral's legibility participation depends on raw trace availability.

Reasoning is passed via reasoning.effort=high in extra_body (OpenRouter maps
this to Mistral's configurable reasoning_effort). include_reasoning=True
ensures OpenRouter forwards the thinking content.

The reasoning text arrives either:
  - As msg.reasoning or msg.reasoning_content (OpenRouter native field)
  - Inside <think>...</think> tags in msg.content (Mistral native format)
Both paths are checked.
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


class MistralAdapter(BaseAdapter):
    required_env = ["OPENROUTER_API_KEY"]

    def _check_credentials(self) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise CredentialMissingError(
                "mistral_medium_3_5: missing OPENROUTER_API_KEY (Mistral runs via OpenRouter)"
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
                extra_body={
                    "include_reasoning": True,
                    "reasoning": {"effort": reasoning_effort},
                },
            )
            latency = time.perf_counter() - t0
        except Exception as exc:
            raise AdapterError(f"mistral_medium_3_5 API error: {exc}") from exc

        msg = resp.choices[0].message
        raw_content = msg.content or ""
        finish_reason, native_finish_reason = extract_finish_reasons(resp)

        # Check native OpenRouter reasoning field first (set when include_reasoning=True
        # and the upstream model exposes the thinking block).
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        if reasoning:
            answer = raw_content
        else:
            # Mistral's native thinking format: <think>...</think> in content stream
            reasoning, answer = extract_think_tags(raw_content)

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

        if reasoning:
            trace_status = "raw"
        elif reasoning_tokens > 0:
            # OpenRouter did not forward text — counts only (MISMATCH vs expected raw)
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
