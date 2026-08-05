"""Gemini 3.1 Pro adapter — role: judge (Phase 2 only).

Cross-lineage (Western) second judge. Strong multilingual; reads Chinese and
Danish traces in their original language.

The scored-model call() raises AdapterError as a safety net. Judging is done
via call_judge_openrouter() in src/judge.py.
"""
from __future__ import annotations

from .base import AdapterError, BaseAdapter, ModelResponse


class GoogleAdapter(BaseAdapter):
    required_env: list[str] = []

    def call(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high") -> ModelResponse:
        raise AdapterError(
            "gemini_3_1_pro is a Phase 2 judge — it must not appear on the scored-model path"
        )
