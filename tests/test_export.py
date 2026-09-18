"""
Unit tests for Reflex Export and Temperature Calibration Tooling.
"""

import unittest
from reflex.export import (
    calibrate_temperature,
    evaluate_calibration,
    quantize_onnx_model,
)


class TestExportTooling(unittest.TestCase):

    def test_calibrate_temperature_optimization(self):
        # Over-confident uncalibrated logits
        # When model is overconfident, optimal temperature T should be > 1.0 to soften logits
        logits = [5.0, 4.0, 3.0, -3.0, -4.0, -5.0]
        labels = [1, 1, 0, 1, 0, 0]  # Notice some mistakes (label 0 for logit 3.0, label 1 for logit -3.0)

        calibrated_t = calibrate_temperature(logits, labels, init_temp=1.0, max_iters=50)
        self.assertGreater(calibrated_t, 0.1)

        eval_uncalibrated = evaluate_calibration(logits, labels, temperature=1.0)
        eval_calibrated = evaluate_calibration(logits, labels, temperature=calibrated_t)

        self.assertIn("brier_score", eval_calibrated)
        self.assertIn("ece", eval_calibrated)
        self.assertEqual(eval_calibrated["temperature"], calibrated_t)

    def test_invalid_calibration_inputs(self):
        with self.assertRaises(ValueError):
            calibrate_temperature([], [])
        with self.assertRaises(ValueError):
            calibrate_temperature([1.0], [1, 0])

    def test_quantize_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            quantize_onnx_model("nonexistent_model.onnx")


if __name__ == "__main__":
    unittest.main()
