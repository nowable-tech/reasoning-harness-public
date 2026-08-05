"""
--heavy phase: three locked tasks from established, external datasets.

  code:            HumanEval task_id "HumanEval/94"
  finance_calc:    FinQA id "CDNS/2015/page_30.pdf-3"
  finance_interp:  FinQA id "AMAT/2013/page_37.pdf-2"

Datasets are downloaded once and cached under data/heavy/ (gitignored — these
are third-party redistributions, not our own curated content, unlike
data/prompts.yaml).

SECURITY INVARIANT: load_heavy_tasks() defaults to with_facit=False and never
includes grading data in that path — mirrors config_loader.load_prompts()'s
strip-before-return contract. with_facit=True is valid ONLY on the grading
path (never sent to a model).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
HEAVY_CACHE_DIR = DATA_DIR / "heavy"

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/test.json"

HUMANEVAL_TASK_ID = "HumanEval/94"
FINQA_CALC_ID = "CDNS/2015/page_30.pdf-3"
FINQA_INTERP_ID = "AMAT/2013/page_37.pdf-2"

# Task-order-stable keys used throughout run_heavy() and the OUTPUT records' "domain" field.
TASK_KEYS: tuple[str, ...] = ("code", "finance_calc", "finance_interp")

CODE_INSTRUCTION = (
    "\n\nComplete this Python function. Respond with the complete function "
    "definition (signature, docstring, and body) in a single ```python code "
    "block, and nothing else outside the block."
)

FINANCE_INSTRUCTION = (
    "\n\nAnswer the question using only the table and text above. State your "
    "final numeric answer clearly, e.g. \"Final answer: X\"."
)


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _load_humaneval_record(task_id: str) -> dict:
    path = _download(HUMANEVAL_URL, HEAVY_CACHE_DIR / "HumanEval.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["task_id"] == task_id:
                return rec
    raise KeyError(f"HumanEval task_id not found in cached dataset: {task_id}")


def _load_finqa_record(record_id: str) -> dict:
    path = _download(FINQA_URL, HEAVY_CACHE_DIR / "FinQA_test.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for rec in data:
        if rec.get("id") == record_id:
            return rec
    raise KeyError(f"FinQA id not found in cached dataset: {record_id}")


def _format_table(table: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row) for row in table)


def _build_finqa_prompt(rec: dict) -> str:
    """RELEVANT table + pre_text only — never post_text, never the whole document."""
    pre_text = "\n".join(rec.get("pre_text", []))
    table_text = _format_table(rec.get("table", []))
    question = rec["qa"]["question"]
    return (
        f"{pre_text}\n\nTable:\n{table_text}\n\nQuestion: {question}"
        f"{FINANCE_INSTRUCTION}"
    )


def load_heavy_tasks(with_facit: bool = False) -> dict[str, dict]:
    """
    Returns {task_key: {"task_id", "domain", "prompt", ["facit_grading"]}}.

    with_facit=False (default): request-path safe — no grading data present.
    with_facit=True: adds "facit_grading" — grading path ONLY, never sent to a model.
    """
    he = _load_humaneval_record(HUMANEVAL_TASK_ID)
    fq_calc = _load_finqa_record(FINQA_CALC_ID)
    fq_interp = _load_finqa_record(FINQA_INTERP_ID)

    tasks: dict[str, dict] = {
        "code": {
            "task_id": HUMANEVAL_TASK_ID,
            "domain": "code",
            "prompt": he["prompt"] + CODE_INSTRUCTION,
        },
        "finance_calc": {
            "task_id": FINQA_CALC_ID,
            "domain": "finance_calc",
            "prompt": _build_finqa_prompt(fq_calc),
        },
        "finance_interp": {
            "task_id": FINQA_INTERP_ID,
            "domain": "finance_interp",
            "prompt": _build_finqa_prompt(fq_interp),
        },
    }

    if not with_facit:
        return tasks

    tasks["code"]["facit_grading"] = {
        "entry_point": he["entry_point"],
        "test": he["test"],
    }
    tasks["finance_calc"]["facit_grading"] = {"answer": fq_calc["qa"]["answer"]}
    # finance_interp: FinQA's qa.answer string ("3829") is a dataset annotation
    # bug — it's missing a trailing zero. qa.exe_ans is 38290.0, and the FinQA
    # program (subtract(138.29, const_100), divide(100000, const_100),
    # multiply) computes 38290 directly. Separately, the question itself
    # ("what is the total return if 100000 are invested...") is genuinely
    # ambiguous between the profit (38290) and the ending portfolio value
    # (138290 = 100000 + 38290) — both are defensible readings of "total
    # return", so both are accepted, each with the usual 1% relative
    # tolerance (see grade_finance in heavy_grader.py).
    assert fq_interp["qa"]["answer"] == "3829", (
        "finance_interp source facit changed upstream — re-verify the "
        "38290/138290 correction still applies before trusting it silently."
    )
    tasks["finance_interp"]["facit_grading"] = {
        "answer": "38290",
        "accepted_answers": ["38290", "138290"],
    }
    return tasks
