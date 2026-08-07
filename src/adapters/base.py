from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class CredentialMissingError(Exception):
    pass


class AdapterError(Exception):
    pass


@dataclass
class ModelResponse:
    answer_text: str
    input_tokens: int
    reasoning_tokens: int        # billed thinking tokens (count); 0 when not reported
    output_tokens: int           # visible answer tokens
    cache_read_tokens: int
    cache_write_tokens: int
    raw_reasoning_trace: Optional[str]   # CoT text, or None if not exposed
    trace_status: str            # "raw" | "summarized" | "count_only" | "absent"
    reasoning_source: str        # "api" — from usage fields; "text_estimate" — split_token_estimate()
    latency_s: float
    model_version: str           # pinned snapshot id reported by the provider
    raw_usage: dict = field(default_factory=dict)
    # Raw stop signal from the provider for the FINAL call in the chain (the one
    # whose answer_text is reported). finish_reason is the OpenAI-compatible
    # normalized value (e.g. "length", "stop", "tool_calls"); native_finish_reason
    # is OpenRouter's passthrough of the upstream provider's own value. Both None
    # on the direct Anthropic SDK path (finish_reason there is set from stop_reason).
    finish_reason: Optional[str] = None
    native_finish_reason: Optional[str] = None
    # --- --tools phase only; empty/default for every other phase ---
    tool_calls: list = field(default_factory=list)       # executed: {name, args, result_char_len, result_token_est}
    raw_tool_events: list = field(default_factory=list)  # every tool_call block as emitted, incl. unknown/serverside
    n_api_calls: int = 1                                 # 1 = model answered directly; 2 = one tool round + continuation
    served_by: Optional[str] = None                      # OpenRouter's raw "provider" field — which backend served the call
    # request_model_id: the exact string placed in the API call's `model=` param
    # — set from the SAME variable the adapter sends, not derived from the
    # response. model_version (above) is resp.model — the provider's response
    # label — and diagnostics on 2026-07-14 proved that label is identical
    # whether the dated pin or the canonical undated slug is requested, so it
    # can never prove what was sent. request_model_id is the only field that can.
    request_model_id: str = ""
    via_openrouter: bool = False  # False = direct provider API (undated cfg["model_id"]), bypassing the pin entirely
    warnings: Optional[list] = None  # OpenRouter's dropped/altered-parameter notices, see extract_warnings()


class BaseAdapter:
    required_env: list[str] = []

    def __init__(self, model_key: str, config: dict) -> None:
        self.model_key = model_key
        self.config = config
        self._check_credentials()

    def _check_credentials(self) -> None:
        """
        Accept if (a) all required_env vars are present, OR (b) OPENROUTER_API_KEY
        is set (allows OpenAI-compatible adapters to fall back to OpenRouter).
        Subclasses that cannot use OpenRouter (e.g. local) override this.
        """
        direct_ok = all(os.environ.get(v) for v in self.required_env)
        openrouter_ok = bool(os.environ.get("OPENROUTER_API_KEY"))
        if not direct_ok and not openrouter_ok:
            raise CredentialMissingError(
                f"{self.model_key}: missing env var(s): {', '.join(self.required_env)}"
                " (and no OPENROUTER_API_KEY fallback)"
            )

    def _resolve_openai_creds(self) -> tuple[str, Optional[str], str, bool]:
        """
        Returns (api_key, base_url, model_id, via_openrouter).
        Prefers direct provider key; falls back to OpenRouter when absent.
        """
        direct_key_var = self.required_env[0] if self.required_env else None
        direct_key = os.environ.get(direct_key_var) if direct_key_var else None

        if direct_key:
            return (
                direct_key,
                self.config.get("base_url"),
                self.config["model_id"],
                False,
            )

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            model_id = self.config.get(
                "openrouter_model_id", self.config["model_id"]
            )
            return openrouter_key, OPENROUTER_BASE_URL, model_id, True

        raise CredentialMissingError(
            f"{self.model_key}: no usable credential (need "
            f"{', '.join(self.required_env)} or OPENROUTER_API_KEY)"
        )

    def call(
        self,
        prompt: str,
        thinking_budget: int = 4096,
        reasoning_effort: str = "high",
    ) -> ModelResponse:
        raise NotImplementedError

    def call_with_tools(
        self,
        prompt: str,
        thinking_budget: int = 4096,
        reasoning_effort: str = "high",
        tool_choice: Optional[str] = None,
    ) -> ModelResponse:
        """--tools/--tools3 phase. tool_choice: None/"auto" (model routes itself)
        or "required" (model MUST call >=1 tool). Raise ToolsNotSupportedError
        (see tool_loop.py) when the API itself confirms this model/endpoint
        cannot tool-call."""
        raise NotImplementedError


# Shared utility for providers that embed reasoning in <think>...</think> tags.
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def extract_think_tags(text: str) -> tuple[Optional[str], str]:
    """
    Returns (reasoning_text, answer_text).
    reasoning_text is None when no <think> block is found.
    """
    m = _THINK_PATTERN.search(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text


def extract_served_by(resp) -> Optional[str]:
    """
    OpenRouter attaches a top-level "provider" field to chat-completion responses
    naming the backend that actually served the call (e.g. "DeepInfra", "Together").
    The openai SDK keeps unknown top-level fields on `.model_extra`. Direct
    provider APIs (no OpenRouter hop) have no such field — returns None there.
    """
    extra = getattr(resp, "model_extra", None)
    if extra:
        return extra.get("provider")
    return None


def extract_warnings(resp) -> Optional[list]:
    """
    OpenRouter attaches a top-level "warnings" field when it drops or alters a
    request parameter the routed model/endpoint doesn't support (e.g. an
    unsupported reasoning.effort level). Same model_extra mechanism as
    extract_served_by(). None when absent — that is the common case and is
    not itself evidence of anything.
    """
    extra = getattr(resp, "model_extra", None)
    if extra:
        return extra.get("warnings")
    return None


def classify_provider_regime(resolved_model_id: Optional[str]) -> str:
    """
    Best-effort provider-family trace-exposure classification from a RESOLVED
    model id (e.g. resp.model — the model that actually answered, not the
    one requested). Used where the caller cannot assume a single fixed
    panel.yaml trace_exposure ahead of time (a shared tool-calling loop
    serving many adapters; a router call like openrouter/auto[-beta] whose
    target is only known after the response comes back).

    Returns "summarized" | "count_only" | "raw":
      - "summarized": Anthropic's API never returns the raw CoT, only a
        summary, regardless of how much reasoning text is present. A naive
        "non-empty reasoning text => raw" check mislabels this — see
        docs/reasoning_findings.md §4.5 (tool_loop.py used to do exactly
        this for every Anthropic-family call routed through the shared
        OpenAI-dialect tool loop).
      - "count_only": OpenAI's reasoning-model family (o-series, GPT-5.x)
        reports a reasoning token count but never reasoning text.
      - "raw": everything else — open-weight models and, per direct
        observation, Google Gemini — where "reasoning text present" already
        correctly implies a genuine raw trace. This is the historical
        default behavior and is unchanged for these providers.

    None/empty input classifies as "raw" (preserves old behavior when the
    resolved model id isn't available for some reason — better to fall back
    to the presence-based heuristic than to silently mislabel as summarized).
    """
    if not resolved_model_id:
        return "raw"
    mid = resolved_model_id.lower()
    if "anthropic" in mid or "claude" in mid:
        return "summarized"
    if "openai" in mid or mid.startswith("gpt-") or "/gpt-" in mid:
        return "count_only"
    return "raw"


def assert_model_pin_honored(model_key: str, cfg: dict, request_model_id: str) -> None:
    """
    Hard stop — NOT a warning, NOT an AdapterError (which every run loop's
    `except AdapterError`/`except Exception: ... continue` would silently
    swallow and move on to the next row). Call this right after computing the
    `model_id` variable that is about to go into (or just went into) the
    OpenRouter request — before spending an API call, when possible.

    IMPORTANT — this checks request_model_id against cfg['openrouter_model_id'],
    NOT model_version (resp.model). An earlier version of this guard compared
    resp.model instead, on the theory that OpenRouter's response echoes back
    what it actually served. Diagnostics on 2026-07-14 disproved that: sending
    the dated pin (mistralai/mistral-medium-3.5-20260430) and the canonical
    undated slug (mistralai/mistral-medium-3-5) to OpenRouter both returned
    resp.model == 'mistralai/mistral-medium-3-5' — identical either way. So
    resp.model encodes OpenRouter's response-labeling convention, not the
    request, and can never prove what was sent. This function checks the one
    thing that CAN prove it: the literal variable placed in the request.

    In practice this can only fail from a config-authoring mistake (e.g. a
    typo'd or removed openrouter_model_id key silently falling back to
    cfg['model_id']) — but that mistake is real and would otherwise route an
    unpinned string through OpenRouter with zero error or warning, since
    OpenRouter accepts arbitrary model strings without validation (see Defect
    1 diagnostics). No row is worth recording once the pin the run is
    supposed to test isn't the string actually sent.
    """
    expected = cfg.get("openrouter_model_id")
    if expected is not None and request_model_id != expected:
        import sys
        print(f"\n{'!'*100}", file=sys.stderr)
        print(f"  MODEL PIN VIOLATED — {model_key}", file=sys.stderr)
        print(f"  cfg['openrouter_model_id']: {expected!r}", file=sys.stderr)
        print(f"  Actually about to send / sent (request_model_id): {request_model_id!r}", file=sys.stderr)
        print(
            "  The adapter did not use panel.yaml's pinned string. This run cannot "
            "isolate any experimental variable while the requested model differs "
            "from the pin.",
            file=sys.stderr,
        )
        print(f"  Stopping immediately. No further calls will be made.", file=sys.stderr)
        print(f"{'!'*100}\n", file=sys.stderr)
        sys.exit(1)


def extract_finish_reasons(resp) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (finish_reason, native_finish_reason) from an OpenAI-compatible
    chat-completion response's first choice. finish_reason is the normalized
    value; native_finish_reason is OpenRouter's raw passthrough of whatever the
    upstream provider actually returned (absent for direct, non-OpenRouter
    endpoints — returns None there, not a bug).
    """
    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    native_finish_reason = getattr(choice, "native_finish_reason", None)
    return finish_reason, native_finish_reason


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 UTF-8 bytes per token (good enough for CJK/EN mix)."""
    return max(0, len(text.encode("utf-8")) // 4)


def split_token_estimate(
    reasoning_text: Optional[str],
    answer_text: str,
    total_completion: int,
) -> tuple[int, int]:
    """
    Proportionally split total_completion into (reasoning_tokens, output_tokens)
    based on character length when the provider does not report them separately.
    """
    if not reasoning_text:
        return 0, total_completion
    r_len = len(reasoning_text.encode("utf-8"))
    a_len = len(answer_text.encode("utf-8"))
    total_len = r_len + a_len or 1
    reasoning = round(total_completion * r_len / total_len)
    output = total_completion - reasoning
    return reasoning, output
