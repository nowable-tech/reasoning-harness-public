"""Anthropic Claude Sonnet 4.6 adapter — trace_exposure: summarized.

Direct path (ANTHROPIC_API_KEY): uses the Anthropic SDK with extended thinking
enabled. The thinking block contains Anthropic's summarized CoT.

OpenRouter fallback path (OPENROUTER_API_KEY): uses the OpenAI SDK with
extra_body to pass the thinking parameter. OpenRouter's handling of thinking
blocks varies — the smoke test will show what actually comes through and report
a PASS or MISMATCH accordingly.

Thinking token count sources (tried in order):
  1. usage.thinking_tokens
  2. usage.output_tokens_details.thinking_tokens
  3. Proportional estimate from thinking text vs total output_tokens
"""
from __future__ import annotations

import time

from .base import (
    AdapterError,
    BaseAdapter,
    ModelResponse,
    assert_model_pin_honored,
    estimate_tokens,
    extract_finish_reasons,
    extract_served_by,
    extract_think_tags,
    split_token_estimate,
)


class AnthropicAdapter(BaseAdapter):
    required_env = ["ANTHROPIC_API_KEY"]

    def call(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high") -> ModelResponse:
        import os
        if os.environ.get("ANTHROPIC_API_KEY"):
            return self._call_direct(prompt, thinking_budget)
        return self._call_via_openrouter(prompt, thinking_budget)

    def call_with_tools(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high", tool_choice: str | None = None) -> ModelResponse:
        import os
        if os.environ.get("ANTHROPIC_API_KEY"):
            return self._call_with_tools_direct(prompt, thinking_budget, tool_choice)
        return self._call_with_tools_via_openrouter(prompt, thinking_budget, tool_choice)

    # ------------------------------------------------------------------
    # Direct Anthropic SDK path
    # ------------------------------------------------------------------

    def _call_direct(self, prompt: str, thinking_budget: int) -> ModelResponse:
        import anthropic

        client = anthropic.Anthropic(api_key=__import__("os").environ["ANTHROPIC_API_KEY"])
        model_id = self.config["model_id"]

        # Claude 4.X models (opus-4, sonnet-4 ≥ 4.8) use the adaptive thinking API.
        # Older models (3.7 Sonnet) use the legacy enabled+budget_tokens format.
        use_adaptive = "opus-4" in model_id or "sonnet-4" in model_id

        try:
            t0 = time.perf_counter()
            if use_adaptive:
                resp = client.messages.create(
                    model=model_id,
                    max_tokens=thinking_budget + 4096,
                    thinking={"type": "adaptive"},
                    output_config={"effort": "high"},
                    messages=[{"role": "user", "content": prompt}],
                )
            else:
                resp = client.messages.create(
                    model=model_id,
                    max_tokens=thinking_budget + 512,
                    thinking={"type": "enabled", "budget_tokens": thinking_budget},
                    messages=[{"role": "user", "content": prompt}],
                )
            latency = time.perf_counter() - t0
        except Exception as exc:
            raise AdapterError(f"claude_sonnet_4_6 API error: {exc}") from exc

        thinking_texts: list[str] = []
        answer_parts: list[str] = []
        for block in resp.content:
            if block.type == "thinking":
                thinking_texts.append(getattr(block, "thinking", "") or "")
            elif block.type == "text":
                answer_parts.append(getattr(block, "text", "") or "")

        thinking_text = "\n\n".join(thinking_texts) if thinking_texts else None
        answer_text = "\n\n".join(answer_parts)

        usage = resp.usage
        raw = usage.model_dump() if hasattr(usage, "model_dump") else {}

        reasoning_tokens: int | None = getattr(usage, "thinking_tokens", None)
        if reasoning_tokens is None:
            out_details = getattr(usage, "output_tokens_details", None)
            if out_details is not None:
                reasoning_tokens = getattr(out_details, "thinking_tokens", None)
        if reasoning_tokens is not None:
            reasoning_source = "api"
        else:
            reasoning_tokens, _ = split_token_estimate(
                thinking_text, answer_text, usage.output_tokens
            )
            reasoning_source = "text_estimate"

        output_tokens = max(0, usage.output_tokens - reasoning_tokens)

        return ModelResponse(
            answer_text=answer_text,
            input_tokens=usage.input_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            raw_reasoning_trace=thinking_text,
            trace_status="summarized" if thinking_text else "absent",
            reasoning_source=reasoning_source,
            latency_s=latency,
            model_version=resp.model,
            raw_usage=raw,
            finish_reason=resp.stop_reason,
            request_model_id=model_id,
            via_openrouter=False,
        )

    # ------------------------------------------------------------------
    # OpenRouter fallback path (OpenAI-compatible SDK)
    # ------------------------------------------------------------------

    def _call_via_openrouter(self, prompt: str, thinking_budget: int) -> ModelResponse:
        import os
        from openai import OpenAI
        from .base import OPENROUTER_BASE_URL

        or_key = os.environ.get("OPENROUTER_API_KEY")
        if not or_key:
            raise AdapterError("claude_sonnet_4_6: no ANTHROPIC_API_KEY or OPENROUTER_API_KEY")

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
                    "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
                },
            )
            latency = time.perf_counter() - t0
        except Exception as exc:
            raise AdapterError(
                f"claude_sonnet_4_6 via OpenRouter API error: {exc}"
            ) from exc

        msg = resp.choices[0].message
        raw_content = msg.content or ""
        finish_reason, native_finish_reason = extract_finish_reasons(resp)

        # OpenRouter may pass thinking as <thinking> tags, a separate field, or strip it.
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "thinking", None)
        if reasoning is None:
            import re
            m = re.search(r"<thinking>(.*?)</thinking>\s*", raw_content, re.DOTALL)
            if m:
                reasoning = m.group(1).strip()
                answer = raw_content[m.end():].strip()
            else:
                reasoning, answer = extract_think_tags(raw_content)
                if reasoning is None:
                    answer = raw_content
        else:
            answer = raw_content

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

        return ModelResponse(
            answer_text=answer,
            input_tokens=usage.prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            raw_reasoning_trace=reasoning,
            trace_status="summarized" if reasoning else "absent",
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

    # ------------------------------------------------------------------
    # Tools phase — direct Anthropic SDK (native tool_use/tool_result blocks)
    # ------------------------------------------------------------------

    def _call_with_tools_direct(self, prompt: str, thinking_budget: int, tool_choice: str | None = None) -> ModelResponse:
        import anthropic
        import os

        from ..tool_loop import ToolsNotSupportedError
        from ..tools import available_tool_defs, execute_tool, to_anthropic_tools

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        model_id = self.config["model_id"]
        use_adaptive = "opus-4" in model_id or "sonnet-4" in model_id
        anthropic_tools = to_anthropic_tools(available_tool_defs())
        # Anthropic's native tool_choice format differs from the OpenAI string enum.
        anthropic_tool_choice = {"required": {"type": "any"}, "auto": {"type": "auto"}}.get(tool_choice)

        def _make_call(messages: list, with_tools: bool):
            kwargs: dict = {"model": model_id, "messages": messages}
            if use_adaptive:
                kwargs.update(
                    max_tokens=thinking_budget + 4096,
                    thinking={"type": "adaptive"},
                    output_config={"effort": "high"},
                )
            else:
                kwargs.update(
                    max_tokens=thinking_budget + 512,
                    thinking={"type": "enabled", "budget_tokens": thinking_budget},
                )
            if with_tools:
                kwargs["tools"] = anthropic_tools
                if anthropic_tool_choice is not None:
                    kwargs["tool_choice"] = anthropic_tool_choice
            return client.messages.create(**kwargs)

        def _tokens_for(usage, thinking_text, answer_text):
            raw = usage.model_dump() if hasattr(usage, "model_dump") else {}
            r: int | None = getattr(usage, "thinking_tokens", None)
            if r is None:
                out_details = getattr(usage, "output_tokens_details", None)
                if out_details is not None:
                    r = getattr(out_details, "thinking_tokens", None)
            if r is not None:
                out = max(0, usage.output_tokens - r)
                src = "api"
            else:
                r, out = split_token_estimate(thinking_text, answer_text, usage.output_tokens)
                src = "text_estimate"
            return usage.input_tokens, r, out, src, raw

        t0 = time.perf_counter()
        messages: list = [{"role": "user", "content": prompt}]

        try:
            resp1 = _make_call(messages, with_tools=True)
        except Exception as exc:
            msg = str(exc).lower()
            if "tool" in msg and ("not support" in msg or "unsupported" in msg):
                raise ToolsNotSupportedError(f"{self.model_key}: tool-calling not supported — {exc}") from exc
            raise AdapterError(f"{self.model_key} tools call error: {exc}") from exc

        thinking1, answer_parts1, tool_use_blocks = [], [], []
        for block in resp1.content:
            if block.type == "thinking":
                thinking1.append(getattr(block, "thinking", "") or "")
            elif block.type == "text":
                answer_parts1.append(getattr(block, "text", "") or "")
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
        thinking_text1 = "\n\n".join(thinking1) if thinking1 else None
        answer_text1 = "\n\n".join(answer_parts1)

        in1, reas1, out1, src1, raw1 = _tokens_for(resp1.usage, thinking_text1, answer_text1)

        raw_tool_events: list[dict] = []
        tool_calls_log: list[dict] = []

        if not tool_use_blocks:
            latency = time.perf_counter() - t0
            trace_status = "summarized" if thinking_text1 else "absent"
            return ModelResponse(
                answer_text=answer_text1,
                input_tokens=in1,
                reasoning_tokens=reas1,
                output_tokens=out1,
                cache_read_tokens=getattr(resp1.usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(resp1.usage, "cache_creation_input_tokens", 0) or 0,
                raw_reasoning_trace=thinking_text1,
                trace_status=trace_status,
                reasoning_source=src1,
                latency_s=latency,
                model_version=resp1.model,
                raw_usage={"call_1": raw1},
                tool_calls=[],
                raw_tool_events=[],
                n_api_calls=1,
                finish_reason=resp1.stop_reason,
                request_model_id=model_id,
                via_openrouter=False,
            )

        assistant_content = [
            block.model_dump() if hasattr(block, "model_dump") else block
            for block in resp1.content
        ]
        messages.append({"role": "assistant", "content": assistant_content})

        tool_result_blocks = []
        for tb in tool_use_blocks:
            name = tb.name
            args = tb.input if isinstance(tb.input, dict) else {}
            raw_tool_events.append({"name": name, "args": args, "id": tb.id})

            result = execute_tool(name, args)
            result_text = result["text"]
            tool_calls_log.append({
                "name": name,
                "args": args,
                "result_char_len": len(result_text),
                "result_token_est": estimate_tokens(result_text),
            })
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": result_text,
            })
        messages.append({"role": "user", "content": tool_result_blocks})

        try:
            resp2 = _make_call(messages, with_tools=False)
        except Exception as exc:
            raise AdapterError(f"{self.model_key} tools continuation error: {exc}") from exc

        thinking2, answer_parts2 = [], []
        for block in resp2.content:
            if block.type == "thinking":
                thinking2.append(getattr(block, "thinking", "") or "")
            elif block.type == "text":
                answer_parts2.append(getattr(block, "text", "") or "")
        thinking_text2 = "\n\n".join(thinking2) if thinking2 else None
        answer_text2 = "\n\n".join(answer_parts2)

        in2, reas2, out2, src2, raw2 = _tokens_for(resp2.usage, thinking_text2, answer_text2)

        latency = time.perf_counter() - t0
        combined_thinking = "\n\n".join(t for t in (thinking_text1, thinking_text2) if t) or None
        reasoning_source = "api" if "api" in (src1, src2) else "text_estimate"
        trace_status = "summarized" if combined_thinking else "absent"

        return ModelResponse(
            answer_text=answer_text2,
            input_tokens=in1 + in2,
            reasoning_tokens=reas1 + reas2,
            output_tokens=out1 + out2,
            cache_read_tokens=(
                (getattr(resp1.usage, "cache_read_input_tokens", 0) or 0)
                + (getattr(resp2.usage, "cache_read_input_tokens", 0) or 0)
            ),
            cache_write_tokens=(
                (getattr(resp1.usage, "cache_creation_input_tokens", 0) or 0)
                + (getattr(resp2.usage, "cache_creation_input_tokens", 0) or 0)
            ),
            raw_reasoning_trace=combined_thinking,
            trace_status=trace_status,
            reasoning_source=reasoning_source,
            latency_s=latency,
            model_version=resp2.model,
            raw_usage={"call_1": raw1, "call_2": raw2},
            tool_calls=tool_calls_log,
            raw_tool_events=raw_tool_events,
            n_api_calls=2,
            finish_reason=resp2.stop_reason,
            request_model_id=model_id,
            via_openrouter=False,
        )

    # ------------------------------------------------------------------
    # Tools phase — OpenRouter fallback (OpenAI-compatible dialect)
    # ------------------------------------------------------------------

    def _call_with_tools_via_openrouter(self, prompt: str, thinking_budget: int, tool_choice: str | None = None) -> ModelResponse:
        import os
        from openai import OpenAI
        from .base import OPENROUTER_BASE_URL
        from ..tool_loop import call_with_tools_openai_style

        or_key = os.environ.get("OPENROUTER_API_KEY")
        if not or_key:
            raise AdapterError(f"{self.model_key}: no ANTHROPIC_API_KEY or OPENROUTER_API_KEY")

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
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
            },
            via_openrouter=True,
        )
