"""Tests for scripts/compute_findings.py's dedup tie-break rule.

Run: python3 -m unittest tests.test_compute_findings -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compute_findings import _dedup, _dedup_priority  # noqa: E402


def _row(run_id: str, source_file: str, marker: str) -> dict:
    return {"run_id": run_id, "_source_file": source_file, "marker": marker}


class TestDedupPriority(unittest.TestCase):
    def test_higher_run_id_always_wins_regardless_of_source_file(self):
        older_corrected = _row("20260709T093542_heavy", "20260709T093542_heavy_corrected.jsonl", "older")
        newer_plain = _row("20260714T091621_heavy_recap", "20260714T091621_heavy_recap.jsonl", "newer")
        self.assertGreater(_dedup_priority(newer_plain), _dedup_priority(older_corrected))

    def test_tie_on_run_id_prefers_corrected_over_plain_original(self):
        original = _row("20260709T093542_heavy", "20260709T093542_heavy.jsonl", "original")
        corrected = _row("20260709T093542_heavy", "20260709T093542_heavy_corrected.jsonl", "corrected")
        self.assertGreater(_dedup_priority(corrected), _dedup_priority(original))

    def test_tie_on_run_id_prefers_recap_over_plain_original(self):
        original = _row("20260719T171948_heavy", "20260719T171948_heavy.jsonl", "original")
        recap = _row("20260719T171948_heavy", "20260719T171948_heavy_recap.jsonl", "recap")
        self.assertGreater(_dedup_priority(recap), _dedup_priority(original))


class TestDedup(unittest.TestCase):
    def test_correction_wins_regardless_of_iteration_order(self):
        original = _row("20260709T093542_heavy", "20260709T093542_heavy.jsonl", "original")
        corrected = _row("20260709T093542_heavy", "20260709T093542_heavy_corrected.jsonl", "corrected")

        result_a = _dedup([original, corrected], key_fn=lambda r: "k")
        result_b = _dedup([corrected, original], key_fn=lambda r: "k")

        self.assertEqual(result_a[0]["marker"], "corrected")
        self.assertEqual(result_b[0]["marker"], "corrected")

    def test_source_file_key_is_stripped_from_output(self):
        original = _row("20260709T093542_heavy", "20260709T093542_heavy.jsonl", "original")
        result = _dedup([original], key_fn=lambda r: "k")
        self.assertNotIn("_source_file", result[0])

    def test_genuinely_later_run_wins_over_an_earlier_corrected_run(self):
        # A *_corrected file from an OLDER run must not beat a plain file
        # from a genuinely newer run — run_id is primary, the marker is
        # only a tie-break for an exact run_id match.
        old_corrected = _row("20260709T093542_heavy", "20260709T093542_heavy_corrected.jsonl", "old_corrected")
        new_plain = _row("20260719T171948_heavy", "20260719T171948_heavy.jsonl", "new_plain")

        result = _dedup([old_corrected, new_plain], key_fn=lambda r: "k")
        self.assertEqual(result[0]["marker"], "new_plain")

    def test_no_collision_passes_through_unchanged(self):
        a = _row("20260709T093542_heavy", "20260709T093542_heavy.jsonl", "a")
        b = _row("20260719T171948_heavy", "20260719T171948_heavy.jsonl", "b")

        result = _dedup([a, b], key_fn=lambda r: r["marker"])
        self.assertEqual({r["marker"] for r in result}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
