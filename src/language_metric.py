"""
Segment-aware language detection for reasoning traces.

Traces contain code fences, math expressions, and natural-language passages.
Naive line-by-line detection is unreliable on short or symbol-heavy segments;
this module filters to natural-language paragraphs only.

Pipeline:
  1. Strip ``` code fences, inline `code`, $...$ / $$...$$ math blocks
  2. Split into paragraphs (prefer double-newline; fall back to single-newline
     for traces that use one newline per thought, common in Chinese CoTs)
  3. Per paragraph:
       – skip if alpha ratio < MIN_ALPHA_RATIO  (symbol/math heavy)
       – skip if alpha char count < MIN_ALPHA_CHARS  (too short; unreliable)
       – run langdetect; accept only if top-language probability >= PROB_THRESHOLD
       – record the language with Nordic-family normalisation (da/no/nb/nn/sv → "da")
  4. primary_trace_language = plurality winner across classified paragraphs
  5. language_switch_count = consecutive pairs with DIFFERENT normalised languages
  6. switch_count_confidence = "low" when code/math paragraph fraction
     exceeds CODE_MATH_HEAVY_THRESHOLD; "normal" otherwise

Nordic normalisation rationale: langdetect cannot reliably distinguish Danish
from Norwegian and Swedish on short passages. Since this benchmark looks for
Danish specifically, the Nordic cluster (da/no/nb/nn/sv) is collapsed to "da"
for switch counting. The raw detected label is stored separately.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect_langs  # type: ignore[import]

# Reproducible detection across runs.
DetectorFactory.seed = 42

# --- Tuning knobs -----------------------------------------------------------
MIN_ALPHA_CHARS: int = 50      # min alphabet chars in a paragraph for classification
MIN_ALPHA_RATIO: float = 0.35  # paragraphs below this ratio treated as code/math
PROB_THRESHOLD: float = 0.70   # min langdetect confidence to accept a label
CODE_MATH_HEAVY_THRESHOLD: float = 0.50  # fraction above which switch_count = "low"

# --- Language family normalisation ------------------------------------------
# langdetect confuses Nordic languages on short text. Collapse for switch counting.
_NORDIC = frozenset({"da", "no", "nb", "nn", "sv"})
_CHINESE = frozenset({"zh-cn", "zh-tw"})


def _normalise_lang(lang: str) -> str:
    """Collapse near-identical families to prevent false switches."""
    if lang in _NORDIC:
        return "da"       # Nordic cluster; report canonical as "da" for this benchmark
    if lang in _CHINESE:
        return "zh-cn"
    return lang


# --- Stripping patterns -----------------------------------------------------
_CODE_FENCE = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_LATEX_DISPLAY = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_LATEX_INLINE = re.compile(r"\$[^\$\n]+\$")
_MARKDOWN_DECOR = re.compile(r"^[#>\s*\-]{1,4}", re.MULTILINE)


def _strip_non_natural(text: str) -> str:
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _LATEX_DISPLAY.sub(" ", text)
    text = _LATEX_INLINE.sub(" ", text)
    text = _MARKDOWN_DECOR.sub("", text)
    return text


def _alpha_ratio(s: str) -> float:
    """Fraction of non-whitespace characters that are Unicode letters."""
    non_ws = [c for c in s if not c.isspace()]
    if not non_ws:
        return 0.0
    return sum(1 for c in non_ws if c.isalpha()) / len(non_ws)


def _detect_lang(text: str) -> Optional[str]:
    """Return the top ISO 639-1 label if its probability >= PROB_THRESHOLD."""
    try:
        langs = detect_langs(text)
        if langs and langs[0].prob >= PROB_THRESHOLD:
            return langs[0].lang
    except LangDetectException:
        pass
    return None


def _split_paragraphs(text: str) -> list[str]:
    """
    Split into paragraphs. Prefer double-newline boundaries; if the result
    has fewer than 4 segments (common in Chinese CoTs that use single newlines),
    fall back to single-newline splitting and merge very-short fragments.
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paras) >= 4:
        return paras
    # Fall back: single-newline split, then merge lines < MIN_ALPHA_CHARS
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    merged: list[str] = []
    buf = ""
    for ln in lines:
        buf = (buf + " " + ln).strip() if buf else ln
        if sum(1 for c in buf if c.isalpha()) >= MIN_ALPHA_CHARS:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged if merged else paras


def measure_trace_language(trace: Optional[str]) -> dict:
    """
    Classify the language content of a reasoning trace.

    Returns a dict with:
        primary_trace_language   – canonical language code (normalised) or None
        language_switch_count    – int, consecutive pair transitions between families
        switch_count_confidence  – "normal" | "low" | "no_trace" | "no_text"
        code_math_fraction       – float
        classified_segments      – int, paragraphs that received a language label
        measurement              – tool version tag (for firewall traceability)

    All None values mean "not applicable", not "unknown" or "zero".
    """
    if not trace or not trace.strip():
        return {
            "primary_trace_language": None,
            "language_switch_count": 0,
            "switch_count_confidence": "no_trace",
            "code_math_fraction": 0.0,
            "classified_segments": 0,
            "measurement": "langdetect-1.0.9/segment-aware",
        }

    cleaned = _strip_non_natural(trace)
    paras = _split_paragraphs(cleaned)

    total_segs = len(paras)
    code_math_segs = 0
    raw_langs: list[str] = []          # detected before normalisation
    norm_langs: list[str] = []         # after normalisation, for switch counting

    for para in paras:
        ratio = _alpha_ratio(para)
        if ratio < MIN_ALPHA_RATIO:
            code_math_segs += 1
            continue

        alpha_count = sum(1 for c in para if c.isalpha())
        if alpha_count < MIN_ALPHA_CHARS:
            continue

        lang = _detect_lang(para)
        if lang:
            raw_langs.append(lang)
            norm_langs.append(_normalise_lang(lang))

    code_math_frac = code_math_segs / max(total_segs, 1)

    if not norm_langs:
        confidence = "low" if code_math_frac > CODE_MATH_HEAVY_THRESHOLD else "no_text"
        return {
            "primary_trace_language": None,
            "language_switch_count": 0,
            "switch_count_confidence": confidence,
            "code_math_fraction": round(code_math_frac, 3),
            "classified_segments": 0,
            "measurement": "langdetect-1.0.9/segment-aware",
        }

    # Primary language from the normalised sequence (the plurality winner).
    primary = Counter(norm_langs).most_common(1)[0][0]
    switches = sum(1 for a, b in zip(norm_langs, norm_langs[1:]) if a != b)
    confidence = "low" if code_math_frac > CODE_MATH_HEAVY_THRESHOLD else "normal"

    return {
        "primary_trace_language": primary,
        "language_switch_count": switches,
        "switch_count_confidence": confidence,
        "code_math_fraction": round(code_math_frac, 3),
        "classified_segments": len(norm_langs),
        "measurement": "langdetect-1.0.9/segment-aware",
    }
