"""
Harness-executed client-side tools for the --tools phase.

Both tools are defined ONCE here and executed by the harness itself — never by
the provider's own server-side tool infrastructure. This guarantees all eight
models see the exact same tool surface, which is the point of the experiment
(measuring whether reasoning tokens move to tool-mediated input tokens).

Tools:
  python_exec(code)  — exec (not just eval) in a sandboxed subprocess.
  web_search(query)  — Brave or Tavily, parametrized via SEARCH_PROVIDER/SEARCH_API_KEY.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

MAX_RESULT_CHARS = 20_000
EXEC_TIMEOUT_S = 5

# ---------------------------------------------------------------------------
# Tool schema — provider-neutral. Translated per-adapter format below.
# ---------------------------------------------------------------------------

TOOL_DEFS: list[dict] = [
    {
        "name": "python_exec",
        "description": (
            "Execute Python code in a sandboxed subprocess (5s timeout, no network, "
            "no persistent filesystem writes, no environment variables). The code can "
            "define and call functions, not just evaluate a single expression. Returns "
            "captured stdout plus the value of the final expression, if there is one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code to execute."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web. Returns up to 5 results, each with a title, url, and snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
]

_SEARCH_PENDING_LOGGED = False


def search_available() -> bool:
    return bool(os.environ.get("SEARCH_API_KEY"))


def available_tool_defs() -> list[dict]:
    """
    python_exec is always available. web_search is dropped when SEARCH_API_KEY
    is missing — the repl measurement must not be blocked by a missing search key.
    """
    global _SEARCH_PENDING_LOGGED
    if search_available():
        return list(TOOL_DEFS)
    if not _SEARCH_PENDING_LOGGED:
        print(
            "  !! SEARCH_API_KEY not set — web_search omitted from the tool list. "
            "Search-offload measurement is PENDING; repl-offload proceeds unaffected."
        )
        _SEARCH_PENDING_LOGGED = True
    return [t for t in TOOL_DEFS if t["name"] != "web_search"]


def to_anthropic_tools(defs: list[dict]) -> list[dict]:
    return [
        {"name": d["name"], "description": d["description"], "input_schema": d["parameters"]}
        for d in defs
    ]


def to_openai_tools(defs: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d["description"],
                "parameters": d["parameters"],
            },
        }
        for d in defs
    ]


# ---------------------------------------------------------------------------
# Tool execution — the harness runs these itself, on behalf of every model.
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict) -> dict:
    """
    Returns {"text": <string sent back to the model as the tool result>, ...}.
    Unknown tool names (e.g. a provider server-side tool we never declared)
    are NOT silently dropped — they get a clear error result so the run can
    still complete, and the raw call is logged separately by the caller.
    """
    if name == "python_exec":
        return _python_exec_impl(args.get("code", "") if isinstance(args, dict) else "")
    if name == "web_search":
        return _web_search_impl(args.get("query", "") if isinstance(args, dict) else "")
    return {
        "text": f"Error: tool {name!r} is not defined in this harness — not executed.",
        "error": "unknown_tool",
    }


# ---------------------------------------------------------------------------
# python_exec — sandboxed subprocess
# ---------------------------------------------------------------------------

# Fixed wrapper source (never contains model-authored text — the model's code
# is piped in via stdin, not interpolated into this string).
_SANDBOX_WRAPPER_SRC = r"""
import ast, builtins, contextlib, io, socket, sys

def _blocked(*a, **k):
    raise PermissionError("network access is disabled in this sandbox")

socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked

_orig_open = builtins.open
def _guarded_open(file, mode="r", *args, **kwargs):
    if any(c in mode for c in ("w", "a", "x", "+")):
        raise PermissionError("file writes are disabled in this sandbox")
    return _orig_open(file, mode, *args, **kwargs)
builtins.open = _guarded_open

code = sys.stdin.read()
buf = io.StringIO()
try:
    tree = ast.parse(code, mode="exec")
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    ns = {"__name__": "__sandbox__"}
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<tool_code>", "exec"), ns)
        result = None
        has_result = False
        if last_expr is not None:
            result = eval(compile(last_expr, "<tool_code>", "eval"), ns)
            has_result = True
except Exception as e:
    sys.stdout.write(buf.getvalue())
    print(f"__SANDBOX_ERROR__:{e!r}")
    sys.exit(1)

sys.stdout.write(buf.getvalue())
if has_result and result is not None:
    print(f"__SANDBOX_RESULT__:{result!r}")
"""


def _preexec_limits():
    """Best-effort resource caps (Unix only) — belt-and-suspenders on top of timeout."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT_S + 1, EXEC_TIMEOUT_S + 1))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass


def _python_exec_impl(code: str) -> dict:
    if not code.strip():
        return {"text": "Error: no code provided.", "error": "empty_code"}

    with tempfile.TemporaryDirectory(prefix="reasoning_sandbox_") as tmpdir:
        restricted_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _SANDBOX_WRAPPER_SRC],
                input=code,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_S,
                cwd=tmpdir,
                env=restricted_env,
                preexec_fn=_preexec_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {"text": f"Error: execution timed out after {EXEC_TIMEOUT_S}s.", "error": "timeout"}

    stdout = proc.stdout or ""
    error_line = None
    result_line = None
    kept_lines = []
    for line in stdout.splitlines():
        if line.startswith("__SANDBOX_ERROR__:"):
            error_line = line[len("__SANDBOX_ERROR__:"):]
        elif line.startswith("__SANDBOX_RESULT__:"):
            result_line = line[len("__SANDBOX_RESULT__:"):]
        else:
            kept_lines.append(line)
    captured_stdout = "\n".join(kept_lines).strip()

    parts = []
    if captured_stdout:
        parts.append(f"stdout:\n{captured_stdout}")
    if result_line is not None:
        parts.append(f"result: {result_line}")
    if error_line is not None:
        parts.append(f"error: {error_line}")
    if proc.stderr and proc.stderr.strip():
        parts.append(f"stderr:\n{proc.stderr.strip()}")
    if not parts:
        parts.append("(no output)")

    text = "\n\n".join(parts)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n...[truncated, {len(text)} chars total]"

    return {"text": text, "error": error_line, "stdout": captured_stdout, "result": result_line}


# ---------------------------------------------------------------------------
# web_search — Brave or Tavily, parametrized by env
# ---------------------------------------------------------------------------

def _web_search_impl(query: str) -> dict:
    import requests

    if not query.strip():
        return {"text": "Error: no query provided.", "error": "empty_query"}

    provider = os.environ.get("SEARCH_PROVIDER", "brave").lower()
    api_key = os.environ.get("SEARCH_API_KEY")
    if not api_key:
        return {
            "text": "web_search is unavailable: SEARCH_API_KEY is not configured.",
            "error": "search_unavailable",
        }

    try:
        if provider == "brave":
            r = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                params={"q": query, "count": 5},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            results = [
                {
                    "title": w.get("title", ""),
                    "url": w.get("url", ""),
                    "snippet": w.get("description", ""),
                }
                for w in data.get("web", {}).get("results", [])[:5]
            ]
        elif provider == "tavily":
            r = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": 5},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            results = [
                {
                    "title": x.get("title", ""),
                    "url": x.get("url", ""),
                    "snippet": x.get("content", ""),
                }
                for x in data.get("results", [])[:5]
            ]
        else:
            return {
                "text": f"web_search error: unknown SEARCH_PROVIDER={provider!r} (expected brave|tavily).",
                "error": f"unknown_provider:{provider}",
            }
    except Exception as e:
        return {"text": f"web_search error: {e}", "error": str(e)}

    text = json.dumps(results, ensure_ascii=False, indent=2)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n...[truncated, {len(text)} chars total]"
    return {"text": text, "results": results}
