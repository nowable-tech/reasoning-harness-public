from __future__ import annotations

from typing import TYPE_CHECKING

from .config_loader import load_pricing

if TYPE_CHECKING:
    from .accounting import TokenAccount


def compute_cost(model_key: str, account: "TokenAccount") -> tuple[float, str]:
    """
    Returns (cost_usd, pricing_snapshot_date).

    Formula:
        cost = input*p_in
             + cache_read*p_cache_read
             + cache_write*p_cache_write
             + (reasoning + output)*p_out

    Prices are read exclusively from config/pricing.yaml — never hardcoded here.

    Raises KeyError if model_key has no pricing.yaml entry. This used to
    return (0.0, snapshot_date) silently — a missing entry produced a
    valid-looking zero cost with no signal anywhere, which would corrupt
    every dollar-based aggregate downstream (correct_per_dollar, sum-cost
    totals) without warning. Failing loudly here is the fix.
    """
    pricing = load_pricing()
    snapshot_date: str = pricing["snapshot_date"]
    model_prices: dict | None = pricing["models"].get(model_key)

    if model_prices is None:
        raise KeyError(
            f"compute_cost: no pricing.yaml entry for model_key={model_key!r}. "
            f"Add a models.{model_key} block to config/pricing.yaml before running "
            f"this model — a missing entry must not silently produce cost_usd=0.0."
        )

    def mtok(tokens: int, price_per_mtok: float) -> float:
        return (tokens / 1_000_000) * price_per_mtok

    cost = (
        mtok(account.input_tokens, model_prices["input_per_mtok"])
        + mtok(account.cache_read_tokens, model_prices["cache_read_per_mtok"])
        + mtok(account.cache_write_tokens, model_prices["cache_write_per_mtok"])
        + mtok(account.total_billed_output, model_prices["output_per_mtok"])
    )
    return cost, snapshot_date
