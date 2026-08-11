"""
--heavy phase correctness graders. Quality control, not the primary metric —
the phase measures HOW models solve heavy tasks (tokens, cost, tool behavior);
correctness is here so a cheap/fast row isn't mistaken for a good one.

code: extract the candidate function from the model's answer, execute it
against the official HumanEval test suite inside the SAME sandbox tools.py
uses for python_exec (network disabled, no persistent writes, 5s timeout) —
reusing the harness's own sandbox rather than a second one.

finance_calc / finance_interp: extract a numeric answer from free text and
compare against the FinQA answer_key with 1% relative tolerance. Free-text
extraction is inherently imperfect — every row logs raw_extracted_numbers so
the regex can be revised without re-running the experiment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .tools import execute_tool

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")
_FINAL_ANSWER_LABEL_RE = re.compile(r"final\s+answer\s*:?", re.IGNORECASE)

RELATIVE_TOLERANCE = 0.01


@dataclass
class GradeResult:
    correct: bool
    extracted_answer: str
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------

def _extract_python_code(answer_text: str, entry_point: str) -> tuple[str, str]:
    """Returns (code, extraction_method) — method is logged for data-quality review."""
    blocks = _CODE_FENCE_RE.findall(answer_text)
    with_entry = [b for b in blocks if f"def {entry_point}" in b]
    if with_entry:
        return with_entry[-1], "fenced_with_entry_point"
    if blocks:
        return blocks[-1], "fenced_fallback_no_entry_point_match"
    if f"def {entry_point}" in answer_text:
        return answer_text, "raw_text_fallback"
    return answer_text, "raw_text_no_def_found"


def grade_code(answer_text: str, entry_point: str, test_code: str) -> GradeResult:
    code, extraction_method = _extract_python_code(answer_text, entry_point)
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    result = execute_tool("python_exec", {"code": full_code})
    correct = result.get("error") is None
    return GradeResult(
        correct=correct,
        extracted_answer=code.strip()[:2000],
        detail={
            "extraction_method": extraction_method,
            "sandbox_error": result.get("error"),
            "sandbox_stdout": (result.get("stdout") or "")[:1000],
        },
    )


# ---------------------------------------------------------------------------
# finance_calc / finance_interp
# ---------------------------------------------------------------------------

def _parse_number(raw: str) -> Optional[float]:
    cleaned = raw.replace("$", "").replace(",", "").strip()
    cleaned = cleaned.rstrip("%")
    if not cleaned or cleaned in ("-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        v = _parse_number(m.group())
        if v is not None:
            out.append(v)
    return out


def _relative_close(a: float, b: float, tol: float = RELATIVE_TOLERANCE) -> bool:
    if b == 0:
        return abs(a - b) < 1e-9
    return abs(a - b) / abs(b) <= tol


def grade_finance(answer_text: str, answer_key_answers) -> GradeResult:
    """
    answer_key_answers: a single answer_key string, or a list of acceptable answer_key
    strings when the question itself is ambiguous (finance_interp accepts
    two readings — see heavy_tasks.py). A single string is treated as a
    one-element list; behavior for single-answer_key tasks (finance_calc) is
    unchanged.

    Primary heuristic (2026-07-19 fix): the FIRST number after the LAST
    occurrence of the literal label "Final answer:" (the phrase the prompt
    itself instructs models to use — see heavy_tasks.py). Falls back to the
    old last-number-in-the-whole-text heuristic ONLY when the label is
    entirely absent from the response.

    Why: the old last-number-in-the-whole-text heuristic silently mismatched
    correct answers whenever a model appended a unit conversion after its own
    "Final answer:" line (e.g. "Final answer: 90.62% (0.9062)" — the model's
    stated answer is exactly right, but the trailing parenthetical decimal
    restatement was what got graded). Confirmed on fable_5 2026-07-19: all 6
    of its finance failures in that run had a correct "Final answer:" label,
    zero were real computation errors — a grading defect, not a model defect,
    the third instance of this pattern in the project (loft/Mistral,
    tool-loop/Inkling, now extraction/fable_5).

    Using the LAST occurrence of the label (not the first) handles models
    that restate "Final answer:" multiple times before settling — the most
    recent statement is treated as authoritative, same principle a human
    grader would apply.

    Also checks whether ANY number in the text matches ANY accepted answer_key,
    for data-quality review when the primary heuristic and the tolerance
    check disagree.
    """
    if isinstance(answer_key_answers, str):
        answer_key_answers = [answer_key_answers]
    answer_key_numerics = [n for n in (_parse_number(f) for f in answer_key_answers) if n is not None]

    numbers = _extract_numbers(answer_text)

    label_matches = list(_FINAL_ANSWER_LABEL_RE.finditer(answer_text))
    if label_matches:
        after_label = answer_text[label_matches[-1].end():]
        numbers_after_label = _extract_numbers(after_label)
        primary = numbers_after_label[0] if numbers_after_label else None
        extraction_method = "final_answer_label" if primary is not None else "label_present_no_number_after"
    else:
        primary = numbers[-1] if numbers else None
        extraction_method = "last_number_no_label"

    matched_answer_key = None
    if primary is not None:
        for fn in answer_key_numerics:
            if _relative_close(primary, fn):
                matched_answer_key = fn
                break
    correct = matched_answer_key is not None

    any_match = any(
        _relative_close(n, fn) for n in numbers for fn in answer_key_numerics
    )
    return GradeResult(
        correct=correct,
        extracted_answer=str(primary) if primary is not None else "",
        detail={
            "answer_key_numeric": answer_key_numerics[0] if len(answer_key_numerics) == 1 else answer_key_numerics,
            "accepted_answer_key_numerics": answer_key_numerics,
            "matched_answer_key": matched_answer_key,
            "raw_extracted_numbers": numbers[:20],
            "any_number_in_text_matches": any_match,
            "primary_matched": correct,
            "extraction_method": extraction_method,
        },
    )


def grade(domain: str, answer_text: str, answer_key_grading: dict) -> GradeResult:
    """Dispatch by domain — code vs the two finance domains."""
    if domain == "code":
        return grade_code(answer_text, answer_key_grading["entry_point"], answer_key_grading["test"])
    answer_key = answer_key_grading.get("accepted_answers") or answer_key_grading["answer"]
    return grade_finance(answer_text, answer_key)
