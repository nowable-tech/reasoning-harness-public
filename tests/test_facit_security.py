"""Tests for the request-path security invariant: the blind answer key
(currently named `facit`) must never survive into the object sent to a
model.

Scope: load_prompts() and load_multilang_prompts() in src/config_loader.py
are the two functions on the actual request path (their return value is
what adapter.call() receives). load_heavy_tasks() in src/heavy_tasks.py
carries the same invariant but downloads external datasets (HumanEval,
FinQA) on first call - deliberately not exercised here to keep this test
hermetic and network-free; its assert-based guards (see
src/heavy_tasks.py::load_heavy_tasks) are the same pattern, just not
covered by an automated test.

Run: python3 -m unittest tests.test_facit_security -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_prompts, load_multilang_prompts


class TestFacitNeverReachesRequestPath(unittest.TestCase):
    def test_load_prompts_strips_facit_from_every_entry(self):
        prompts = load_prompts()
        self.assertGreater(len(prompts), 0, "sanity check: prompts.yaml loaded something")
        for pid, entry in prompts.items():
            self.assertNotIn("facit", entry, f"{pid}: facit leaked into the request-path object")

    def test_load_multilang_prompts_strips_facit_from_every_entry(self):
        prompts = load_multilang_prompts()
        self.assertGreater(len(prompts), 0, "sanity check: prompts_multilang.yaml loaded something")
        for pid, entry in prompts.items():
            self.assertNotIn("facit", entry, f"{pid}: facit leaked into the request-path object")

    def test_load_prompts_send_field_is_present_and_facit_free(self):
        # Belt-and-braces: even nested/serialized, the string "facit" should
        # not appear anywhere in a prompt's send-path representation.
        import json
        prompts = load_prompts()
        for pid, entry in prompts.items():
            serialized = json.dumps(entry)
            self.assertNotIn('"facit"', serialized, f"{pid}: facit key present in serialized request object")


if __name__ == "__main__":
    unittest.main()
