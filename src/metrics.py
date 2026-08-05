"""Shared cross-report metric definitions.

Single source of truth so a metric's definition cannot silently diverge
between run.py's different report sections (or between those and ad-hoc
analysis scripts) again.
"""
from __future__ import annotations


def correct_per_dollar(rows: list[dict]) -> float | None:
    """
    Correctness per dollar = total correct rows / total dollars actually spent
    (sum of cost_usd across rows), NOT median-cost-times-row-count.

    Why sum, not median: cost distributions are right-skewed (a handful of
    expensive outlier rows pull the total well above what median*n predicts).
    Confirmed empirically 2026-07-19 on mistral_medium_3_5: median*n predicted
    $0.56 in total spend against an actual $0.84 — a 34% error. Sum-of-actuals
    is what was really paid; that's the only number this metric should reflect.

    rows: any iterable of dicts with "cost_usd" and "correct" keys — works on
    both raw jsonl rows loaded from results/heavy/*.jsonl and on run.py's
    internal `agg` accumulator, since both use these same key names. Rows
    without a cost (status != "ok", cost_usd is None) are excluded, matching
    how error/no-tool-support rows are already excluded from every other
    aggregate in these reports.
    """
    priced = [r for r in rows if r.get("cost_usd") is not None]
    total_cost = sum(r["cost_usd"] for r in priced)
    if not priced or total_cost == 0:
        return None
    n_correct = sum(1 for r in priced if r.get("correct"))
    return n_correct / total_cost
