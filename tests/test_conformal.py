"""
Unit tests for Reflex Phase 33: Distribution-Free Conformal Prediction & Calibration Bounds.
Validates finite-sample coverage guarantees, Mondrian class-conditional calibration,
singleton routing, certified ambiguity escalation, binary persistence with CRC32 integrity,
and Reflex client integration.
Uses Python standard library unittest only (strict zero external dependencies).
"""

import math
import os
import random
import tempfile
import unittest

from reflex.conformal import (
    ConformalConfig,
    ConformalPredictor,
    ConformalNoulResult,
    ConformalChoiceResult,
    CONFORMAL_MAGIC,
)
from reflex.primitives import Noul, Choice
from reflex.client import Reflex


class TestConformalConfig(unittest.TestCase):
    def test_config_validation(self):
        cfg = ConformalConfig(alpha=0.05, mondrian=True, min_calibration_samples=30)
        self.assertEqual(cfg.alpha, 0.05)
        self.assertAlmostEqual(cfg.coverage_guarantee, 0.95)
        self.assertTrue(cfg.mondrian)

        with self.assertRaises(ValueError):
            ConformalConfig(alpha=0.0)
        with self.assertRaises(ValueError):
            ConformalConfig(alpha=1.0)
        with self.assertRaises(ValueError):
            ConformalConfig(alpha=-0.1)
        with self.assertRaises(ValueError):
            ConformalConfig(min_calibration_samples=2)


class TestConformalPredictor(unittest.TestCase):
    def setUp(self):
        self.cfg = ConformalConfig(alpha=0.05, mondrian=True, min_calibration_samples=20, seed=42)
        self.predictor = ConformalPredictor(self.cfg)

        # Generate realistic calibration data:
        # True class: mostly high probability (mean ~0.85)
        # False class: mostly low probability (mean ~0.15)
        rng = random.Random(42)
        for _ in range(300):
            self.predictor.add_calibration_noul(rng.betavariate(5, 1.2), true_label=True)
            self.predictor.add_calibration_noul(rng.betavariate(1.2, 5), true_label=False)

        # Choice calibration data (3 options)
        for _ in range(100):
            # Option A
            self.predictor.add_calibration_choice({"A": 0.85, "B": 0.10, "C": 0.05}, true_label="A")
            # Option B
            self.predictor.add_calibration_choice({"A": 0.08, "B": 0.82, "C": 0.10}, true_label="B")
            # Option C
            self.predictor.add_calibration_choice({"A": 0.05, "B": 0.15, "C": 0.80}, true_label="C")

    def test_uncalibrated_predictor_raises(self):
        uncal = ConformalPredictor()
        with self.assertRaises(RuntimeError):
            uncal.predict_noul(Noul("test", probability=0.9))
        with self.assertRaises(RuntimeError):
            uncal.predict_choice(Choice("test", options=["A", "B"]))

    def test_empty_calibration_raises(self):
        empty_cp = ConformalPredictor()
        with self.assertRaises(ValueError):
            empty_cp.calibrate()

    def test_noul_singleton_true(self):
        self.predictor.calibrate()
        self.assertTrue(self.predictor.is_calibrated)

        noul = Noul("Is user authorized?", probability=0.98)
        res = self.predictor.predict_noul(noul)

        self.assertEqual(res.prediction_set, {True})
        self.assertTrue(res.is_singleton)
        self.assertFalse(res.is_ambiguous)
        self.assertFalse(res.is_empty)
        self.assertFalse(res.should_escalate)
        self.assertEqual(res.certified_value, True)
        self.assertGreater(res.p_value_true, 0.05)
        self.assertLessEqual(res.p_value_false, 0.05)

    def test_noul_singleton_false(self):
        self.predictor.calibrate()
        noul = Noul("Is user authorized?", probability=0.02)
        res = self.predictor.predict_noul(noul)

        self.assertEqual(res.prediction_set, {False})
        self.assertTrue(res.is_singleton)
        self.assertFalse(res.is_ambiguous)
        self.assertFalse(res.should_escalate)
        self.assertEqual(res.certified_value, False)
        self.assertGreater(res.p_value_false, 0.05)
        self.assertLessEqual(res.p_value_true, 0.05)

    def test_noul_ambiguous_escalation(self):
        # Borderline input with uncertainty: prediction set contains {True, False}
        cfg = ConformalConfig(alpha=0.01, min_calibration_samples=20, seed=42)
        cp = ConformalPredictor(cfg)
        rng = random.Random(42)
        for _ in range(200):
            cp.add_calibration_noul(rng.betavariate(4, 1.5), true_label=True)
            cp.add_calibration_noul(rng.betavariate(1.5, 4), true_label=False)
        cp.calibrate()

        noul = Noul("Is user authorized?", probability=0.50)
        res = cp.predict_noul(noul)

        self.assertEqual(res.prediction_set, {True, False})
        self.assertFalse(res.is_singleton)
        self.assertTrue(res.is_ambiguous)
        self.assertFalse(res.is_empty)
        self.assertTrue(res.should_escalate)
        self.assertIsNone(res.certified_value)

    def test_noul_ood_empty_set_escalation(self):
        # Out-of-distribution probability yields empty prediction set, triggering escalation
        cp = ConformalPredictor(ConformalConfig(alpha=0.20, min_calibration_samples=20, seed=42))
        for _ in range(50):
            cp.add_calibration_noul(0.99, true_label=True)
            cp.add_calibration_noul(0.01, true_label=False)
        cp.calibrate()

        # Input probability 0.50 is completely out-of-distribution relative to extreme confident calibration
        noul = Noul("OOD query", probability=0.50)
        res = cp.predict_noul(noul)
        self.assertEqual(res.prediction_set, set())
        self.assertTrue(res.is_empty)
        self.assertFalse(res.is_singleton)
        self.assertTrue(res.should_escalate)
        self.assertIsNone(res.certified_value)

    def test_choice_singleton_selection(self):
        self.predictor.calibrate()
        choice = Choice(
            "Select operations queue",
            options=["A", "B", "C"],
            distribution={"A": 0.96, "B": 0.03, "C": 0.01},
        )
        res = self.predictor.predict_choice(choice)

        self.assertEqual(res.prediction_set, ["A"])
        self.assertTrue(res.is_singleton)
        self.assertFalse(res.is_ambiguous)
        self.assertFalse(res.should_escalate)
        self.assertEqual(res.certified_selection, "A")
        self.assertGreater(res.p_values["A"], 0.05)

    def test_choice_ambiguous_multi_label(self):
        self.predictor.calibrate()
        choice = Choice(
            "Select operations queue",
            options=["A", "B", "C"],
            distribution={"A": 0.48, "B": 0.47, "C": 0.05},
        )
        res = self.predictor.predict_choice(choice)

        self.assertIn("A", res.prediction_set)
        self.assertIn("B", res.prediction_set)
        self.assertTrue(res.is_ambiguous)
        self.assertTrue(res.should_escalate)
        self.assertIsNone(res.certified_selection)

    def test_mondrian_class_conditional_coverage(self):
        # Asymmetric noise
        cp = ConformalPredictor(ConformalConfig(alpha=0.05, mondrian=True, min_calibration_samples=10))
        for _ in range(50):
            cp.add_calibration_noul(0.99, true_label=True)  # True class very clean
            cp.add_calibration_noul(0.40, true_label=False) # False class noisy
        cp.calibrate()

        # Mondrian thresholds must differ between classes
        self.assertNotEqual(cp.noul_thresholds["true"], cp.noul_thresholds["false"])
        self.assertLess(cp.noul_thresholds["true"], cp.noul_thresholds["false"])

    def test_global_calibration_mode(self):
        cp = ConformalPredictor(ConformalConfig(alpha=0.05, mondrian=False))
        for _ in range(50):
            cp.add_calibration_noul(0.90, true_label=True)
            cp.add_calibration_noul(0.10, true_label=False)
        cp.calibrate()

        self.assertEqual(cp.noul_thresholds["true"], cp.noul_thresholds["false"])
        self.assertIn("global", cp.noul_thresholds)

    def test_p_values_properties(self):
        self.predictor.calibrate()
        noul = Noul("test", probability=0.75)
        res = self.predictor.predict_noul(noul)

        self.assertGreaterEqual(res.p_value_true, 0.0)
        self.assertLessEqual(res.p_value_true, 1.0)
        self.assertGreaterEqual(res.p_value_false, 0.0)
        self.assertLessEqual(res.p_value_false, 1.0)

    def test_empirical_coverage_guarantee_95_percent(self):
        self.predictor.calibrate()
        rng = random.Random(99)
        test_samples = []
        for _ in range(1000):
            true_label = (rng.random() > 0.5)
            if true_label:
                p = rng.betavariate(5, 1.2)
            else:
                p = rng.betavariate(1.2, 5)
            test_samples.append((p, true_label))

        metrics = self.predictor.evaluate_coverage_noul(test_samples)
        self.assertGreaterEqual(metrics["empirical_coverage"], 0.94)  # Within finite sample tolerance of 95%
        self.assertGreater(metrics["singleton_ratio"], 0.50)

    def test_empirical_coverage_guarantee_99_percent(self):
        cfg = ConformalConfig(alpha=0.01, mondrian=True, min_calibration_samples=20, seed=42)
        cp = ConformalPredictor(cfg)
        rng = random.Random(42)
        for _ in range(200):
            cp.add_calibration_noul(rng.betavariate(6, 1.1), true_label=True)
            cp.add_calibration_noul(rng.betavariate(1.1, 6), true_label=False)
        cp.calibrate()

        test_samples = []
        for _ in range(500):
            true_label = (rng.random() > 0.5)
            if true_label:
                p = rng.betavariate(6, 1.1)
            else:
                p = rng.betavariate(1.1, 6)
            test_samples.append((p, true_label))

        metrics = cp.evaluate_coverage_noul(test_samples)
        self.assertGreaterEqual(metrics["empirical_coverage"], 0.985)  # 99% coverage guarantee

    def test_binary_serialization_roundtrip(self):
        self.predictor.calibrate()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.reflex-conformal")
            self.predictor.save(path)
            self.assertTrue(os.path.exists(path))

            loaded = ConformalPredictor.load(path)
            self.assertTrue(loaded.is_calibrated)
            self.assertEqual(loaded.config.alpha, self.predictor.config.alpha)
            self.assertEqual(loaded.noul_thresholds, self.predictor.noul_thresholds)
            self.assertEqual(loaded.choice_thresholds, self.predictor.choice_thresholds)

            # Verification of exact prediction parity
            noul = Noul("test", probability=0.88)
            res1 = self.predictor.predict_noul(noul)
            res2 = loaded.predict_noul(noul)
            self.assertEqual(res1.prediction_set, res2.prediction_set)
            self.assertAlmostEqual(res1.p_value_true, res2.p_value_true)

    def test_crc32_tamper_detection(self):
        self.predictor.calibrate()
        raw_bytes = bytearray(self.predictor.save_to_bytes())
        # Corrupt single byte in payload
        raw_bytes[15] ^= 0xAA
        with self.assertRaises(ValueError) as ctx:
            ConformalPredictor.load_from_bytes(bytes(raw_bytes))
        self.assertIn("CRC32", str(ctx.exception))

    def test_reflex_client_conformal_integration(self):
        self.predictor.calibrate()
        rx = Reflex(backend="semantic", conformal=self.predictor)

        res = rx.evaluate(
            state="Can I reset my password please?",
            questions={"auth": Noul("Is user asking for password reset?")},
        )

        self.assertIsNotNone(res.conformal)
        self.assertIn("auth", res.conformal)
        c_auth = res.conformal["auth"]
        self.assertIsInstance(c_auth, ConformalNoulResult)
        # Verify should_escalate boolean property
        self.assertIsInstance(res.should_escalate, bool)


if __name__ == "__main__":
    unittest.main()
