"""
Phase 3 grader — correctness scoring of saved Phase 1 answers against the facit.

Two paths:
  Programmatic (exact, no LLM): P5 (math=168), P6 (logic=David,Anna,Bo,Clara), P7 (JSON)
  LLM grader (blind to model):  P3 (legal timely?), P4 (two rules + ugyldig?), P8 (code bug)

The grader is BLIND to model identity — model key is never included in the LLM prompt.

FIREWALL: this module measures CORRECTNESS only. It does not import or use any
economy (Phase 1) or legibility (Phase 2) data.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from .config_loader import load_pricing

CORRECTNESS_PROMPTS: list[str] = ["P3", "P4", "P5", "P6", "P7", "P8"]
PROGRAMMATIC_PROMPTS: list[str] = ["P5", "P6", "P7"]
LLM_GRADER_PROMPTS: list[str] = ["P3", "P4", "P8"]
VERDICTS: tuple[str, ...] = ("correct", "partial", "incorrect")


@dataclass
class GradeResult:
    prompt_id: str
    model_key: str          # filled in by caller after construction
    verdict: str            # "correct" | "partial" | "incorrect"
    extracted: str          # extracted value (programmatic) or justification (LLM)
    grading_method: str     # "programmatic" | "llm_grader"
    grader_model: str       # model ID string, or "programmatic"
    cost_usd: float
    parse_ok: bool
    parse_error: str


# ─────────────────────────────────────────────────────────────────────────────
# Programmatic graders — exact, no LLM, no cost
# ─────────────────────────────────────────────────────────────────────────────

def _grade_p5_math(answer: str) -> tuple[str, str]:
    """P5: correct iff answer contains 168 (total minutes)."""
    if re.search(r'\b168\b', answer):
        return "correct", "Found 168"
    nums = re.findall(r'\b\d+\b', answer)
    return "incorrect", f"168 not found; numbers seen: {nums[:8]}"


def _grade_p6_logic(answer: str) -> tuple[str, str]:
    """P6: correct iff ordering is David(1), Anna(2), Bo(3), Clara(4)."""
    correct = ["david", "anna", "bo", "clara"]

    # Primary: numbered list "1. Name" / "1) Name" with optional markdown bold
    numbered: list[str] = []
    for i in range(1, 5):
        m = re.search(rf'{i}[.)]\s*\**\s*(\w+)', answer, re.IGNORECASE)
        if m:
            numbered.append(m.group(1).lower())
    if len(numbered) == 4:
        ok = numbered == correct
        return ("correct" if ok else "incorrect"), f"Numbered list: {', '.join(numbered)}"

    # Fallback: first occurrence of each name in the answer text
    positions: dict[str, int] = {}
    for name in correct:
        m = re.search(rf'\b{name}\b', answer, re.IGNORECASE)
        if m:
            positions[name] = m.start()
    if len(positions) == 4:
        by_pos = sorted(positions, key=lambda n: positions[n])
        ok = by_pos == correct
        return ("correct" if ok else "incorrect"), f"Name order: {', '.join(by_pos)}"

    return "incorrect", "Could not extract 4-name ordering"


def _grade_p7_json(answer: str) -> tuple[str, str]:
    """P7: correct iff JSON matches facit field-by-field."""
    cleaned = re.sub(r'```(?:json)?\s*', '', answer).strip().rstrip('`').strip()
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not m:
        return "incorrect", "No JSON object found"
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError as e:
        return "incorrect", f"JSON parse error: {e}"

    expected: dict = {
        "navn": "Mette Sørensen",
        "alder": 34,
        "by": "Odense",
        "abonnement": "premium",
    }
    errors: list[str] = []
    for k, v in expected.items():
        if k not in obj:
            errors.append(f"missing '{k}'")
        elif obj[k] != v:
            errors.append(f"'{k}': expected {v!r}, got {obj[k]!r}")

    if errors:
        return "incorrect", f"Mismatch: {'; '.join(errors)}"
    return "correct", f"JSON correct: {json.dumps(obj, ensure_ascii=False)}"


def grade_programmatic(prompt_id: str, answer: str) -> GradeResult:
    """Run the appropriate programmatic grader for prompt_id."""
    if prompt_id == "P5":
        verdict, detail = _grade_p5_math(answer)
    elif prompt_id == "P6":
        verdict, detail = _grade_p6_logic(answer)
    elif prompt_id == "P7":
        verdict, detail = _grade_p7_json(answer)
    else:
        raise ValueError(f"No programmatic grader for {prompt_id}")

    return GradeResult(
        prompt_id=prompt_id,
        model_key="",
        verdict=verdict,
        extracted=detail,
        grading_method="programmatic",
        grader_model="programmatic",
        cost_usd=0.0,
        parse_ok=True,
        parse_error="",
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM grader — blind (model identity stripped from prompt)
# ─────────────────────────────────────────────────────────────────────────────

# Per-prompt grading criteria sent to the LLM grader
_GRADER_CRITERIA: dict[str, str] = {
    "P3": (
        "Correct iff the answer concludes the appeal IS timely (rettidig / ja) "
        "based on the 4-weeks = 28 days → 3 April reasoning. "
        "Incorrect if it reaches the wrong conclusion or makes a date error. "
        "This prompt has no 'partial' — it is binary correct/incorrect."
    ),
    "P4": (
        "Correct iff the answer identifies BOTH violated rules — "
        "(1) partshøring / partshørselspligten (§ 19) AND "
        "(2) begrundelsespligten (§§ 22-24) — "
        "AND concludes the decision is as-a-starting-point invalid (ugyldig). "
        "Partial if it gets one rule correctly, or identifies both rules but misses "
        "the ugyldig conclusion. "
        "Incorrect if it misses both rules or reaches an entirely wrong conclusion."
    ),
    "P8": (
        "Correct iff the answer identifies the initialization bug "
        "('største = 0' fails on an all-negative list) "
        "AND provides or names the correct fix ('største = tal[0]'). "
        "Partial if it identifies the bug correctly but the fix is wrong or absent, "
        "or names the fix but misidentifies when the bug triggers. "
        "Incorrect if it misidentifies the bug entirely."
    ),
}

_GRADER_PROMPT = """\
You are grading a student answer against a grading key. You do not know which model produced this answer — that information has been intentionally withheld.

GRADING KEY:
<facit>
{facit}
</facit>

STUDENT ANSWER:
<answer>
{answer}
</answer>

GRADING CRITERIA:
{criteria}

Respond ONLY with valid JSON in exactly this structure. No other text before or after.

{{
  "verdict": "<correct or partial or incorrect>",
  "justification": "<one sentence explaining the verdict>"
}}"""


def _grader_cost(pricing_key: str, input_tokens: int, output_tokens: int) -> float:
    pricing = load_pricing()
    mp = pricing["models"].get(pricing_key, {})
    return (
        (input_tokens / 1_000_000) * mp.get("input_per_mtok", 0.0)
        + (output_tokens / 1_000_000) * mp.get("output_per_mtok", 0.0)
    )


def grade_llm(
    prompt_id: str,
    answer: str,
    facit: str,
    grader_openrouter_id: str,
    grader_pricing_key: str,
) -> GradeResult:
    """
    Send a (prompt_id, answer, facit) triple to the LLM grader.
    Model identity is NOT included in the grader prompt.
    Retries up to 3× on empty content or parse failure.
    """
    from openai import OpenAI
    from .adapters.base import OPENROUTER_BASE_URL

    or_key = os.environ.get("OPENROUTER_API_KEY")
    if not or_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL)
    criteria = _GRADER_CRITERIA[prompt_id]
    grader_prompt = _GRADER_PROMPT.format(
        facit=facit.strip(),
        answer=answer.strip(),
        criteria=criteria,
    )

    _MAX_RETRIES = 3
    t0 = time.perf_counter()
    last_resp = None
    raw_text = ""
    verdict = "incorrect"
    justification = ""
    parse_ok = False
    parse_error = "no attempt completed"
    # Partial-parse fallback: if JSON is truncated but verdict is visible via regex,
    # preserve the best partial result across all retries.
    partial_verdict: str | None = None
    partial_error: str = ""

    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            time.sleep(2)
        resp = client.chat.completions.create(
            model=grader_openrouter_id,
            messages=[{"role": "user", "content": grader_prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        last_resp = resp
        raw_text = (resp.choices[0].message.content or "").strip()
        if not raw_text:
            parse_error = f"empty response (attempt {attempt + 1}/{_MAX_RETRIES})"
            continue

        # Strip markdown fences then try JSON parse
        cleaned = re.sub(r'```(?:json)?\s*', '', raw_text).strip()
        parsed = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            # Regex fallback: extract verdict from truncated JSON.
            # Covers: {"verdict": "correct", "justification": "The stu... (cut off)
            v_m = re.search(
                r'"verdict"\s*:\s*"(correct|partial|incorrect)"', cleaned, re.IGNORECASE
            )
            if v_m and partial_verdict is None:
                partial_verdict = v_m.group(1).lower()
                # Try to grab any justification text that made it through
                j_m = re.search(r'"justification"\s*:\s*"([^"]{10,})', cleaned)
                partial_error = (
                    f"partial parse: verdict='{partial_verdict}' extracted from "
                    f"truncated JSON; raw: {raw_text[:200]!r}"
                )
                _ = j_m  # justification fragment unused — full text wasn't delivered
            else:
                parse_error = f"JSON parse failed; raw: {raw_text[:200]!r}"
            continue

        raw_verdict = str(parsed.get("verdict", "")).lower().strip()
        if raw_verdict not in VERDICTS:
            parse_error = f"invalid verdict {raw_verdict!r}; raw: {raw_text[:200]!r}"
            continue

        verdict = raw_verdict
        justification = str(parsed.get("justification", ""))
        parse_ok = True
        break

    # After exhausting retries: if clean parse failed but we extracted a verdict via regex
    if not parse_ok and partial_verdict is not None:
        verdict = partial_verdict
        justification = ""
        parse_error = partial_error

    latency = time.perf_counter() - t0  # noqa: F841 — available for future logging
    usage = last_resp.usage if last_resp else None
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    cost = _grader_cost(grader_pricing_key, in_tok, out_tok)

    return GradeResult(
        prompt_id=prompt_id,
        model_key="",
        verdict=verdict,
        extracted=justification if parse_ok else parse_error,
        grading_method="llm_grader",
        grader_model=last_resp.model if last_resp else grader_openrouter_id,
        cost_usd=cost,
        parse_ok=parse_ok,
        parse_error="" if parse_ok else parse_error,
    )
