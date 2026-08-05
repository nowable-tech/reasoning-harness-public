from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def _raw_panel() -> dict:
    with open(CONFIG_DIR / "panel.yaml") as f:
        return yaml.safe_load(f)


def load_panel() -> dict:
    return _raw_panel()["models"]


def load_experiment() -> dict:
    """Return the experiment-level constants from panel.yaml (reasoning_effort, etc.)."""
    return _raw_panel().get("experiment", {})


@lru_cache(maxsize=1)
def load_pricing() -> dict:
    with open(CONFIG_DIR / "pricing.yaml") as f:
        return yaml.safe_load(f)


def load_prompts() -> dict[str, dict]:
    """
    Load prompts from data/prompts.yaml, keyed by prompt ID.

    SECURITY INVARIANT: facit is stripped before returning. It must never
    appear in any outgoing model request. The function asserts this on every
    call — if the strip ever fails, execution stops with a hard error.
    """
    with open(DATA_DIR / "prompts.yaml") as f:
        raw = yaml.safe_load(f)

    result: dict[str, dict] = {}
    for entry in raw["prompts"]:
        pid = entry["id"]
        # Require facit to exist in source so omissions are caught early.
        assert "facit" in entry, (
            f"prompt {pid} is missing the 'facit' field — "
            "add it (null is acceptable if there is no known answer)"
        )
        # Strip facit — this is the guard on the request path.
        send_obj = {k: v for k, v in entry.items() if k != "facit"}
        # Hard assertion: facit must not survive the strip.
        assert "facit" not in send_obj, (
            f"BUG: facit leaked into the request-path object for prompt {pid}"
        )
        result[pid] = send_obj
    return result


def load_multilang_prompts() -> dict[str, dict]:
    """
    Load multilingual prompts from data/prompts_multilang.yaml, keyed by prompt ID.

    SECURITY INVARIANT: identical to load_prompts() — facit is stripped before
    returning. The variants dict (da/en/zh) is safe to send; facit is not.
    """
    with open(DATA_DIR / "prompts_multilang.yaml") as f:
        raw = yaml.safe_load(f)

    result: dict[str, dict] = {}
    for entry in raw["prompts"]:
        pid = entry["id"]
        assert "facit" in entry, (
            f"multilang prompt {pid} is missing the 'facit' field — "
            "add it (null is acceptable if there is no known answer)"
        )
        send_obj = {k: v for k, v in entry.items() if k != "facit"}
        assert "facit" not in send_obj, (
            f"BUG: facit leaked into the request-path object for multilang prompt {pid}"
        )
        result[pid] = send_obj
    return result
