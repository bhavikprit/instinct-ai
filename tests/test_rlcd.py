"""
Unit tests for Reflex OpenRLCD Calibration metrics and Dataset generator.
"""

import os
import tempfile
import unittest
from reflex.rlcd import (
    brier_score,
    expected_calibration_error,
    epistemic_entropy,
    reliability_diagram_data,
    generate_decision_dataset,
    save_dataset_jsonl,
)


class TestOpenRLCDMetrics(unittest.TestCase):

    def test_brier_score_perfect(self):
        # Perfect predictions: BS = 0.0
        preds = [1.0, 0.0, 1.0, 0.0]
        targets = [1, 0, 1, 0]
        self.assertAlmostEqual(brier_score(preds, targets), 0.0)

    def test_brier_score_worst(self):
        # Completely inverted predictions: BS = 1.0
        preds = [0.0, 1.0]
        targets = [1, 0]
        self.assertAlmostEqual(brier_score(preds, targets), 1.0)

    def test_ece_perfect(self):
        preds = [1.0, 0.0]
        targets = [1, 0]
        self.assertAlmostEqual(expected_calibration_error(preds, targets, num_bins=5), 0.0)

    def test_epistemic_entropy(self):
        # Maximal entropy at p = 0.5
        ent_mid = epistemic_entropy(0.5)
        ent_low = epistemic_entropy(0.01)
        ent_high = epistemic_entropy(0.99)
        self.assertGreater(ent_mid, ent_low)
        self.assertGreater(ent_mid, ent_high)
        self.assertAlmostEqual(ent_mid, 0.6931, places=3)

    def test_reliability_diagram_data(self):
        preds = [0.1, 0.2, 0.8, 0.9]
        targets = [0, 0, 1, 1]
        data = reliability_diagram_data(preds, targets, num_bins=5)
        self.assertTrue(len(data) > 0)
        # Each entry is (avg_conf, accuracy, count)
        for conf, acc, count in data:
            self.assertTrue(0.0 <= conf <= 1.0)
            self.assertTrue(0.0 <= acc <= 1.0)
            self.assertGreater(count, 0)


class TestOpenRLCDDataset(unittest.TestCase):

    def test_generate_samples(self):
        samples = generate_decision_dataset(num_samples=25, seed=123)
        self.assertEqual(len(samples), 25)

        first = samples[0]
        self.assertIn("id", first)
        self.assertIn("category", first)
        self.assertIn("type", first)
        self.assertIn("state", first)
        self.assertIn("question", first)

    def test_save_jsonl(self):
        samples = generate_decision_dataset(num_samples=10, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_dataset_jsonl(samples, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 10)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
