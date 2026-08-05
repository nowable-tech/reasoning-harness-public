# Economy axis — token-phase accounting.
# Firewall: no quality judgment here. Numbers only.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapters.base import ModelResponse


@dataclass
class TokenAccount:
    input_tokens: int
    reasoning_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_share: float  # reasoning / (reasoning + output), or 0 if no output

    @property
    def total_billed_output(self) -> int:
        """Reasoning + output are both billed at the output rate."""
        return self.reasoning_tokens + self.output_tokens


def build_account(response: "ModelResponse") -> TokenAccount:
    total_out = response.reasoning_tokens + response.output_tokens
    share = response.reasoning_tokens / total_out if total_out > 0 else 0.0
    return TokenAccount(
        input_tokens=response.input_tokens,
        reasoning_tokens=response.reasoning_tokens,
        output_tokens=response.output_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
        reasoning_share=share,
    )
