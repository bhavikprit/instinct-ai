"""
Unit tests for DecisionBench automated evaluation and leaderboard generator.
"""

import os
import tempfile
import unittest
from reflex.eval import evaluate_backend, generate_leaderboard
from reflex.client import Reflex


class TestDecisionBenchEval(unittest.TestCase):

    def test_evaluate_local_backend(self):
        rx = Reflex(backend="local")
        metrics = evaluate_backend(rx)
        self.assertIn("accuracy", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)
        self.assertIn("p50_latency_ms", metrics)
        self.assertGreater(metrics["accuracy"], 0.0)
        self.assertLess(metrics["brier_score"], 1.0)

    def test_generate_leaderboard_markdown(self):
        report = generate_leaderboard()
        self.assertIn("# 🏆 DecisionBench Official Leaderboard", report)
        self.assertIn("Reflex Local", report)
        self.assertIn("Reflex PureSemantic", report)
        self.assertIn("Claude 3.5 Sonnet", report)
        self.assertIn("|", report)

    def test_generate_leaderboard_file_export(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            generate_leaderboard(output_path=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Official Leaderboard", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
