"""MiniMax M3 adapter — role: judge (Phase 2 only).

MiniMax is a Chinese-native model. It reads Chinese traces in their original
language without translation. Used as primary judge for legibility scoring.

The scored-model call() raises AdapterError as a safety net — MiniMax must
never appear on the scored-model path. Judging is done via call_judge_openrouter()
in src/judge.py, called directly from the Phase 2 orchestration in run.py.
"""
from __future__ import annotations

from .base import AdapterError, BaseAdapter, ModelResponse


class MinimaxAdapter(BaseAdapter):
    required_env: list[str] = []

    def call(self, prompt: str, thinking_budget: int = 4096, reasoning_effort: str = "high") -> ModelResponse:
        raise AdapterError(
            "minimax is a Phase 2 judge — it must not appear on the scored-model path"
        )
