"""
HTML report generator for Phase 2 judge validation results.
"""
from __future__ import annotations

import html
import json
import math
import pathlib
from datetime import datetime, timezone
from typing import Optional


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px; background: #f8f9fa; color: #212529; }
h1 { border-bottom: 2px solid #dee2e6; padding-bottom: 8px; }
h2 { color: #495057; font-size: 1rem; text-transform: uppercase; letter-spacing: .05em;
     margin: 32px 0 4px; }
.prompt-block { background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
                padding: 20px; margin: 16px 0; }
.prompt-block h3 { margin: 0 0 12px; font-size: 1rem; }
.pid { display: inline-block; background: #0d6efd; color: #fff; border-radius: 4px;
       padding: 2px 8px; font-size: .85rem; margin-right: 8px; }
.meta { color: #6c757d; font-size: .85rem; margin-bottom: 4px; }
.excerpt { background: #f1f3f5; border-left: 3px solid #adb5bd; padding: 8px 12px;
           font-size: .82rem; font-family: monospace; white-space: pre-wrap;
           margin: 8px 0; border-radius: 0 4px 4px 0; max-height: 120px; overflow-y: auto; }
details.trace-wrap { margin: 8px 0; }
details.trace-wrap summary { cursor: pointer; color: #495057; font-size: .85rem;
                              padding: 4px 0; user-select: none; }
details.trace-wrap summary:hover { color: #0d6efd; }
.trace-full { background: #f1f3f5; border-left: 3px solid #adb5bd; padding: 8px 12px;
              font-size: .82rem; font-family: monospace; white-space: pre-wrap;
              margin: 4px 0; border-radius: 0 4px 4px 0; max-height: 520px; overflow-y: auto; }
.judges { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
.judge-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 12px; }
.judge-card h4 { margin: 0 0 8px; font-size: .9rem; color: #343a40; }
.score-line { font-weight: bold; margin: 4px 0; }
.just { color: #495057; font-size: .82rem; margin: 2px 0 8px; }
.parse-error { color: #dc3545; font-size: .82rem; font-style: italic; }
.agreement { margin-top: 10px; padding: 6px 10px; border-radius: 4px; font-size: .85rem; }
.agr-ok   { background: #d1e7dd; color: #0f5132; }
.agr-warn { background: #fff3cd; color: #664d03; }
.agr-na   { background: #e2e3e5; color: #41464b; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th { background: #343a40; color: #fff; padding: 8px 12px; text-align: left; font-size: .85rem; }
td { padding: 7px 12px; border-bottom: 1px solid #dee2e6; font-size: .85rem; }
tr:nth-child(even) td { background: #f8f9fa; }
.num { text-align: right; }
"""


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _stddev(values: list[float]) -> float:
    return math.sqrt(_variance(values))


def generate_validation_html(
    jsonl_path: pathlib.Path,
    phase1_jsonl: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """
    Generate a human-readable HTML review from a Phase 2 validation JSONL.
    Returns the path to the written HTML file.
    """
    records = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Load Phase 1 data for trace excerpts
    trace_map: dict[str, str] = {}
    prompt_text_map: dict[str, str] = {}
    if phase1_jsonl and phase1_jsonl.exists():
        for line in phase1_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model_key") == "gemma_4":
                pid = r["prompt_id"]
                trace_map[pid] = r.get("raw_reasoning_trace") or ""
                prompt_text_map[pid] = r.get("prompt") or ""

    # Group records by prompt_id
    by_prompt: dict[str, dict] = {}
    for rec in records:
        pid = rec["prompt_id"]
        judge = rec["judge"]
        if pid not in by_prompt:
            by_prompt[pid] = {
                "prompt_type": rec.get("prompt_type", ""),
                "language_probe": rec.get("language_probe", ""),
                "reasoning_load": rec.get("reasoning_load", ""),
                "judges": {},
            }
        by_prompt[pid]["judges"][judge] = rec

    run_id = records[0]["run_id"] if records else "unknown"
    source_run_id = records[0].get("source_run_id", "unknown") if records else "unknown"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    blocks = []
    for pid in sorted(by_prompt.keys(), key=lambda p: int(p[1:])):
        info = by_prompt[pid]
        judges_data = info["judges"]
        prompt_excerpt = html.escape((prompt_text_map.get(pid) or "")[:200])
        trace_full = html.escape(trace_map.get(pid) or "")

        judge_cards = []
        agr_record = None
        for judge_key, rec in judges_data.items():
            if rec.get("agreement") is not None:
                agr_record = rec["agreement"]

            if rec.get("parse_ok"):
                r_score = rec["scores"].get("redundancy", "?")
                c_score = rec["scores"].get("coherence", "?")
                r_just = html.escape(rec.get("justifications", {}).get("redundancy", ""))
                c_just = html.escape(rec.get("justifications", {}).get("coherence", ""))
                cost = rec.get("cost_usd", 0)
                lat = rec.get("latency_s", 0)
                body = (
                    f'<div class="score-line">Redundancy = {r_score} &nbsp; Coherence = {c_score}'
                    f' &nbsp; <span style="color:#6c757d;font-weight:normal">${cost:.5f} / {lat:.1f}s</span></div>'
                    f'<div class="just"><b>R:</b> {r_just}</div>'
                    f'<div class="just"><b>C:</b> {c_just}</div>'
                )
            else:
                err = html.escape(rec.get("parse_error", "")[:200])
                body = f'<div class="parse-error">PARSE ERROR — {err}</div>'

            judge_cards.append(
                f'<div class="judge-card">'
                f'<h4>{html.escape(judge_key)}</h4>{body}</div>'
            )

        if agr_record:
            rd = agr_record.get("dim_diffs", {}).get("redundancy", "?")
            cd = agr_record.get("dim_diffs", {}).get("coherence", "?")
            md = agr_record.get("mean_diff", "?")
            hi = agr_record.get("high_disagreement", False)
            cls = "agr-warn" if hi else "agr-ok"
            flag = " ← HIGH DISAGREEMENT" if hi else ""
            agr_html = (
                f'<div class="agreement {cls}">'
                f'Agreement: redundancy=Δ{rd} coherence=Δ{cd} mean_diff={md}{flag}</div>'
            )
        else:
            failed = [j for j, r in judges_data.items() if not r.get("parse_ok")]
            if failed:
                note = html.escape(", ".join(failed) + " parse failed")
                agr_html = f'<div class="agreement agr-na">Agreement: n/a ({note})</div>'
            else:
                agr_html = ""

        trace_char_count = len(trace_map.get(pid) or "")
        blocks.append(
            f'<div class="prompt-block">'
            f'<h3><span class="pid">{html.escape(pid)}</span>'
            f'{html.escape(info["prompt_type"])} / {html.escape(info["language_probe"])}'
            f' &nbsp; <span style="color:#6c757d">load: {html.escape(info["reasoning_load"])}</span></h3>'
            f'<div class="meta">Prompt:</div>'
            f'<div class="excerpt">{prompt_excerpt}</div>'
            f'<details class="trace-wrap">'
            f'<summary>Reasoning trace ({trace_char_count:,} chars) — click to expand</summary>'
            f'<div class="trace-full">{trace_full}</div>'
            f'</details>'
            f'<div class="judges">{"".join(judge_cards)}</div>'
            f'{agr_html}'
            f'</div>'
        )

    html_out = [
        "<!DOCTYPE html><html lang='en'><head>",
        "<meta charset='utf-8'>",
        f"<title>Phase 2 Judge Validation — {run_id}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        "<h1>Phase 2 — Judge Validation Review</h1>",
        f"<p><b>Run:</b> {html.escape(run_id)} &nbsp; "
        f"<b>Source:</b> {html.escape(source_run_id)} &nbsp; "
        f"<b>Generated:</b> {generated_at}</p>",
        "<p><em>Scoring: gemma_4 only (10 traces). "
        "Both judges read Gemma traces (English) — Lars can verify before trusting on Chinese.</em></p>",
        "<hr>",
        *blocks,
    ]

    # Variance table
    scores_by_judge: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        if not rec.get("parse_ok"):
            continue
        j = rec["judge"]
        if j not in scores_by_judge:
            scores_by_judge[j] = {"redundancy": [], "coherence": []}
        for dim in ("redundancy", "coherence"):
            v = rec.get("scores", {}).get(dim)
            if isinstance(v, (int, float)) and v > 0:
                scores_by_judge[j][dim].append(float(v))

    var_rows = []
    for j_key in sorted(scores_by_judge.keys()):
        for dim in ("redundancy", "coherence"):
            vals = scores_by_judge[j_key][dim]
            if not vals:
                continue
            mean_v = sum(vals) / len(vals)
            var_v = _variance(vals)
            sd_v = _stddev(vals)
            mn, mx = min(vals), max(vals)
            n = len(vals)
            secondary_note = ""
            if dim == "coherence" and var_v < 0.1:
                secondary_note = " <span style='color:#856404;font-size:.8rem'>SECONDARY — non-discriminating (var≈0)</span>"
            var_rows.append(
                f"<tr><td>{html.escape(j_key)}</td><td>{dim}{secondary_note}</td>"
                f"<td class='num'>{n}</td>"
                f"<td class='num'>{mean_v:.2f}</td>"
                f"<td class='num'>{var_v:.2f}</td>"
                f"<td class='num'>{sd_v:.2f}</td>"
                f"<td class='num'>{int(mn)}–{int(mx)}</td></tr>"
            )

    html_out += [
        "<h2>Score Variance per Judge per Dimension</h2>",
        "<p><strong>PRIMARY signal: redundancy.</strong> "
        "Coherence is <em>secondary — currently non-discriminating on the anchor set "
        "(Gemini variance 0); do not read as a primary finding.</em></p>",
        "<table>",
        "<tr><th>Judge</th><th>Dimension</th><th>N</th>"
        "<th>Mean</th><th>Variance</th><th>SD</th><th>Range</th></tr>",
        *var_rows,
        "</table>",
        "</body></html>",
    ]

    out_path = jsonl_path.with_suffix(".html")
    out_path.write_text("\n".join(html_out), encoding="utf-8")
    return out_path


def print_variance_table(jsonl_path: pathlib.Path) -> None:
    """Print a compact variance table to stdout."""
    records = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    scores_by_judge: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        if not rec.get("parse_ok"):
            continue
        j = rec["judge"]
        if j not in scores_by_judge:
            scores_by_judge[j] = {"redundancy": [], "coherence": []}
        for dim in ("redundancy", "coherence"):
            v = rec.get("scores", {}).get(dim)
            if isinstance(v, (int, float)) and v > 0:
                scores_by_judge[j][dim].append(float(v))

    W = 100
    print(f"\n{'═' * W}")
    print("  JUDGE SCORE VARIANCE (parsed scores only)")
    print(f"{'═' * W}")
    hdr = f"  {'Judge':<22}  {'Dim':<12}  {'N':>3}  {'Mean':>5}  {'Var':>5}  {'SD':>5}  {'Min':>4}  {'Max':>4}"
    print(hdr)
    print(f"  {'-' * (W - 2)}")

    for j_key in sorted(scores_by_judge.keys()):
        for dim in ("redundancy", "coherence"):
            vals = scores_by_judge[j_key][dim]
            if not vals:
                print(f"  {j_key:<22}  {dim:<12}  {'—':>3}")
                continue
            n = len(vals)
            mean_v = sum(vals) / n
            var_v = _variance(vals)
            sd_v = _stddev(vals)
            flag = ""
            if dim == "coherence" and var_v < 0.1:
                flag = "  ← SECONDARY — non-discriminating (var≈0)"
            print(
                f"  {j_key:<22}  {dim:<12}  {n:>3}  {mean_v:>5.2f}  "
                f"{var_v:>5.2f}  {sd_v:>5.2f}  {int(min(vals)):>4}  {int(max(vals)):>4}"
                f"{flag}"
            )
    print(f"{'═' * W}")
    print("  PRIMARY signal: redundancy. Coherence is SECONDARY — currently non-discriminating")
    print("  on the anchor set (Gemini variance 0); do not read as a primary finding.")
