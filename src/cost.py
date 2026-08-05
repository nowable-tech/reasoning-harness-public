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
    """
    pricing = load_pricing()
    snapshot_date: str = pricing["snapshot_date"]
    model_prices: dict | None = pricing["models"].get(model_key)

    if model_prices is None:
        return 0.0, snapshot_date

    def mtok(tokens: int, price_per_mtok: float) -> float:
        return (tokens / 1_000_000) * price_per_mtok

    cost = (
        mtok(account.input_tokens, model_prices["input_per_mtok"])
        + mtok(account.cache_read_tokens, model_prices["cache_read_per_mtok"])
        + mtok(account.cache_write_tokens, model_prices["cache_write_per_mtok"])
        + mtok(account.total_billed_output, model_prices["output_per_mtok"])
    )
    return cost, snapshot_date
