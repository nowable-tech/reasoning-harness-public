#!/usr/bin/env python3
"""
Recompute the register's headline figures directly from a results tree —
deterministic, no API calls, no dependency on datasite/reasoning-data.json.

Works against either this repo's own results/ (all phases present) or a
packaged scripts/make_data_release.py bundle (typically only full/, heavy/,
auto/, and optionally sprog/ — tools/tools3/variance sections will report
"no data" there, not crash).

Usage:
    python3 scripts/compute_findings.py <root_dir>

<root_dir> must contain zero or more of these subdirectories, each holding
*.jsonl files in this repo's native row schema:
    full/      -- light suite (--full), n=1 per (model, prompt)
    heavy/     -- heavy suite (--heavy), 5 passes per (model, domain, condition)
    tools/     -- --tools experiment (arm: baseline | tools)
    tools3/    -- --tools3 experiment (arm + tool_choice_sent)
    variance/  -- --variance repro (repeated full-suite passes)
    auto/      -- openrouter/auto[-beta] candidate runs
    sprog/     -- language-cost supplement (if present)

CONVENTIONS APPLIED (must match README.md's Conventions section exactly —
see that file, and docs/reasoning_findings.md's TEMA 2 declaration, for the
source of truth this script implements):
  - Pass-level dedup: latest run wins per exact cell. "Cell" = (model, domain,
    condition, pass_index) for heavy/tools3/variance; (model, prompt_id[,
    arm]) for full/tools. "Latest" = highest run_id (sortable — run_ids are
    UTC timestamps: YYYYMMDDTHHMMSS_suffix).
  - Heavy headline numbers (reasoning medians, reasoning share) = median of
    per-task medians, BASELINE condition only. Each task's own median is
    taken across its (deduped) repeated passes first; the reported number is
    the median of those three per-task medians.
  - correct/actual-$ (heavy) = total correct / total actual cost_usd, BOTH
    conditions pooled (matches docs/reasoning_findings.md §4.3's stated
    method — a deliberately different pooling rule than the headline
    reasoning numbers above). Never median-cost x row-count.
  - Light-suite numbers are read directly (n=1 per prompt after dedup) — no
    median needed, but multiple prompts are combined with a median across
    prompts when a single per-model figure is reported.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

HEAVY_DOMAINS = ("code", "finance_calc", "finance_interp")


# ---------------------------------------------------------------------------
# Loading + dedup
# ---------------------------------------------------------------------------

def _load_jsonl_dir(root: Path, subdir: str) -> list[dict]:
    d = root / subdir
    if not d.is_dir():
        return []
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _dedup(rows: list[dict], key_fn) -> list[dict]:
    """Latest run_id wins per key_fn(row). run_id is a sortable UTC timestamp
    prefix (YYYYMMDDTHHMMSS_...), so string comparison is correct."""
    best: dict[tuple, dict] = {}
    for r in rows:
        k = key_fn(r)
        cur = best.get(k)
        if cur is None or r.get("run_id", "") >= cur.get("run_id", ""):
            best[k] = r
    return list(best.values())


def _median(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _quartiles(values: list[float]) -> dict:
    if not values:
        return {"min": None, "q1": None, "median": None, "q3": None, "max": None, "n": 0}
    s = sorted(values)
    if len(s) < 2:
        q1 = q3 = s[0]
    else:
        q1, _, q3 = statistics.quantiles(s, n=4, method="inclusive")
    return {
        "min": s[0], "q1": q1, "median": statistics.median(s), "q3": q3, "max": s[-1], "n": len(s),
    }


# ---------------------------------------------------------------------------
# Light suite (full/)
# ---------------------------------------------------------------------------

def compute_light(root: Path) -> dict:
    rows = _load_jsonl_dir(root, "full")
    if not rows:
        return {}
    rows = _dedup(rows, lambda r: (r["model_key"], r["prompt_id"]))

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model_key"]].append(r)

    out: dict[str, dict] = {}
    for model, mrows in by_model.items():
        reas = [r["tokens"]["reasoning"] for r in mrows]
        out_tok = [r["tokens"]["output"] for r in mrows]
        shares = [
            r["tokens"]["reasoning"] / (r["tokens"]["reasoning"] + r["tokens"]["output"])
            for r in mrows
            if (r["tokens"]["reasoning"] + r["tokens"]["output"]) > 0
        ]
        out[model] = {
            "n_prompts": len(mrows),
            "reasoning_median": _median(reas),
            "reasoning_share_median": _median(shares),
        }
    return out


# ---------------------------------------------------------------------------
# Heavy suite (heavy/)
# ---------------------------------------------------------------------------

def compute_heavy(root: Path) -> dict:
    rows = _load_jsonl_dir(root, "heavy")
    if not rows:
        return {}
    rows = [r for r in rows if r.get("status") == "ok"]
    rows = _dedup(rows, lambda r: (r["model_key"], r["domain"], r["condition"], r["pass_index"]))

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model_key"]].append(r)

    out: dict[str, dict] = {}
    for model, mrows in by_model.items():
        # Headline numbers: baseline only, median-of-per-task-medians.
        baseline = [r for r in mrows if r["condition"] == "baseline"]
        per_task_reas_medians = []
        per_task_share_medians = []
        for domain in HEAVY_DOMAINS:
            drows = [r for r in baseline if r["domain"] == domain]
            if not drows:
                continue
            reas = [r["tokens"]["reasoning"] for r in drows]
            shares = [
                r["tokens"]["reasoning"] / (r["tokens"]["reasoning"] + r["tokens"]["output"])
                for r in drows
                if (r["tokens"]["reasoning"] + r["tokens"]["output"]) > 0
            ]
            if reas:
                per_task_reas_medians.append(statistics.median(reas))
            if shares:
                per_task_share_medians.append(statistics.median(shares))

        # correct/actual-$ : BOTH conditions pooled, per docs/reasoning_findings.md §4.3.
        priced = [r for r in mrows if r.get("cost_usd") is not None]
        total_cost = sum(r["cost_usd"] for r in priced)
        n_correct = sum(1 for r in priced if r.get("correct") is True)
        correct_per_dollar = (n_correct / total_cost) if total_cost > 0 else None

        out[model] = {
            "n_rows_baseline": len(baseline),
            "n_rows_total": len(mrows),
            "reasoning_median_of_task_medians": _median(per_task_reas_medians),
            "reasoning_share_median_of_task_medians": _median(per_task_share_medians),
            "n_correct": n_correct,
            "n_priced": len(priced),
            "total_cost_usd": round(total_cost, 6),
            "correct_per_dollar": correct_per_dollar,
        }
    return out


# ---------------------------------------------------------------------------
# Tool-offload (tools/, tools3/)
# ---------------------------------------------------------------------------

def compute_tools(root: Path) -> dict:
    rows = _load_jsonl_dir(root, "tools") + _load_jsonl_dir(root, "tools3")
    if not rows:
        return {}
    rows = [r for r in rows if r.get("status") == "ok"]
    # Dedup: (model, prompt_id, arm, pass_index-or-0) — tools/ has no pass_index (n=1).
    rows = _dedup(
        rows,
        lambda r: (r["model_key"], r["prompt_id"], r.get("arm"), r.get("pass_index", 0)),
    )

    invited = [r for r in rows if r.get("arm") not in (None, "baseline") and not r.get("is_control")]

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in invited:
        by_model[r["model_key"]].append(r)

    out: dict[str, dict] = {}
    for model, mrows in by_model.items():
        grabbed = [r for r in mrows if r.get("tool_calls")]
        grab_rate = len(grabbed) / len(mrows) if mrows else None
        # "tool-ratio for graspers": among rows that DID grab, how many tool
        # calls per row on average (intensity of use, not just whether).
        tool_ratio = (
            statistics.mean(len(r["tool_calls"]) for r in grabbed) if grabbed else None
        )
        out[model] = {
            "n_invited_rows": len(mrows),
            "n_grabbed_rows": len(grabbed),
            "grab_rate": grab_rate,
            "tool_calls_per_grabbing_row": tool_ratio,
        }
    return out


# ---------------------------------------------------------------------------
# Variance (variance/)
# ---------------------------------------------------------------------------

def compute_variance(root: Path) -> dict:
    rows = _load_jsonl_dir(root, "variance")
    if not rows:
        return {}
    rows = [r for r in rows if r.get("status") == "ok"]
    rows = _dedup(rows, lambda r: (r["model_key"], r["prompt_id"], r.get("pass_index", 0)))

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["model_key"], r["prompt_id"])].append(r)

    out: dict[str, dict] = {}
    for (model, prompt_id), cell_rows in by_cell.items():
        reas = [r["tokens"]["reasoning"] for r in cell_rows]
        costs = [r["cost_usd"] for r in cell_rows if r.get("cost_usd") is not None]
        out.setdefault(model, {})[prompt_id] = {
            "reasoning_quartiles": _quartiles(reas),
            "cost_quartiles": _quartiles(costs),
        }
    return out


# ---------------------------------------------------------------------------
# trace_status inventory (across every available phase)
# ---------------------------------------------------------------------------

def compute_trace_inventory(root: Path) -> dict:
    all_rows: list[dict] = []
    for sub in ("full", "heavy", "tools", "tools3", "variance", "auto", "sprog"):
        all_rows.extend(_load_jsonl_dir(root, sub))

    by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        model = r.get("model_key") or r.get("model_version") or "unknown"
        ts = r.get("trace_status") or "unknown"
        by_model[model][ts] += 1
    return {m: dict(counts) for m, counts in by_model.items()}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, nd=1):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/compute_findings.py <root_dir>", file=sys.stderr)
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    light = compute_light(root)
    heavy = compute_heavy(root)
    tools = compute_tools(root)
    variance = compute_variance(root)
    trace_inv = compute_trace_inventory(root)

    print(f"{'='*100}\nreasoning-harness findings — recomputed from {root}\n{'='*100}\n")

    print("--- Reasoning-token medians (light suite, full/) ---")
    if light:
        for model, d in sorted(light.items()):
            print(f"  {model:<22} n={d['n_prompts']:<3} reasoning_median={_fmt(d['reasoning_median'])}  "
                  f"reasoning_share_median={_fmt(d['reasoning_share_median'], 3)}")
    else:
        print("  (no full/ data in this dataset)")

    print("\n--- Reasoning-token medians (heavy suite, baseline, median-of-task-medians) ---")
    if heavy:
        for model, d in sorted(heavy.items()):
            print(f"  {model:<22} n_baseline={d['n_rows_baseline']:<4} "
                  f"reasoning_median={_fmt(d['reasoning_median_of_task_medians'])}  "
                  f"reasoning_share_median={_fmt(d['reasoning_share_median_of_task_medians'], 3)}")
    else:
        print("  (no heavy/ data in this dataset)")

    print("\n--- Correct per actual dollar (heavy, both conditions pooled — docs/reasoning_findings.md §4.3 method) ---")
    if heavy:
        ranked = sorted(heavy.items(), key=lambda kv: (kv[1]["correct_per_dollar"] or -1), reverse=True)
        for model, d in ranked:
            print(f"  {model:<22} {d['n_correct']}/{d['n_priced']} correct   "
                  f"${d['total_cost_usd']:.4f} spent   {_fmt(d['correct_per_dollar'])} correct/$")
    else:
        print("  (no heavy/ data in this dataset)")

    print("\n--- Tool-offload: grab-rate and tool-ratio for graspers (tools/, tools3/) ---")
    if tools:
        for model, d in sorted(tools.items()):
            print(f"  {model:<22} n_invited={d['n_invited_rows']:<4} grab_rate={_fmt(d['grab_rate'], 3)}  "
                  f"tool_calls_per_grabbing_row={_fmt(d['tool_calls_per_grabbing_row'], 2)}")
    else:
        print("  (no tools/ or tools3/ data in this dataset)")

    print("\n--- Variance quartiles per (model, prompt) cell, reasoning tokens (variance/) ---")
    if variance:
        for model, cells in sorted(variance.items()):
            for pid, d in sorted(cells.items()):
                q = d["reasoning_quartiles"]
                print(f"  {model:<22} {pid:<5} n={q['n']:<3} min={_fmt(q['min'])} q1={_fmt(q['q1'])} "
                      f"median={_fmt(q['median'])} q3={_fmt(q['q3'])} max={_fmt(q['max'])}")
    else:
        print("  (no variance/ data in this dataset)")

    print("\n--- trace_status inventory (all phases combined) ---")
    for model, counts in sorted(trace_inv.items()):
        total = sum(counts.values())
        parts = ", ".join(f"{k}={v} ({v/total*100:.0f}%)" for k, v in sorted(counts.items()))
        print(f"  {model:<30} n={total:<4} {parts}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
