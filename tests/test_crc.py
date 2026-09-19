"""
Unit tests for Reflex Phase 34: Conformal Risk Control (CRC) & Expected Loss Bounding.
Validates finite-sample risk guarantees E[L] <= alpha, continuous Score intervals,
false-negative risk control, binary persistence (.reflex-crc) with CRC32 verification,
and Reflex client integration.
Uses Python standard library unittest only (strict zero external dependencies).
"""

import math
import os
import random
import tempfile
import unittest

from reflex.crc import (
    CRCConfig,
    ConformalRiskController,
    ScoreRiskBound,
    DecisionRiskBound,
    CRC_MAGIC,
)
from reflex.primitives import Noul, Choice, Score, DecisionResult
from reflex.client import Reflex


class TestCRCConfig(unittest.TestCase):
    def test_config_validation(self):
        cfg = CRCConfig(alpha=0.05, max_loss=1.0, min_calibration_samples=30)
        self.assertEqual(cfg.alpha, 0.05)
        self.assertEqual(cfg.max_loss, 1.0)
        self.assertEqual(cfg.min_calibration_samples, 30)

        with self.assertRaises(ValueError):
            CRCConfig(alpha=0.0)
        with self.assertRaises(ValueError):
            CRCConfig(alpha=1.0)
        with self.assertRaises(ValueError):
            CRCConfig(alpha=-0.05)
        with self.assertRaises(ValueError):
            CRCConfig(max_loss=0.0)
        with self.assertRaises(ValueError):
            CRCConfig(max_loss=-1.0)
        with self.assertRaises(ValueError):
            CRCConfig(min_calibration_samples=2)


class TestConformalRiskController(unittest.TestCase):
    def setUp(self):
        self.cfg = CRCConfig(alpha=0.05, max_loss=1.0, min_calibration_samples=20, seed=42)
        self.controller = ConformalRiskController(self.cfg)

        # Generate realistic continuous score calibration data:
        # True score in [1.0, 10.0], model prediction with Gaussian noise sigma=0.6
        rng = random.Random(42)
        for _ in range(300):
            true_s = rng.uniform(1.0, 10.0)
            noise = rng.gauss(0, 0.6)
            pred_s = max(1.0, min(10.0, true_s + noise))
            self.controller.add_calibration_score(pred_s, true_s, 1.0, 10.0)

        # Ingest binary decision calibration data (Noul)
        for _ in range(300):
            label = (rng.random() > 0.5)
            p = rng.betavariate(5, 1.2) if label else rng.betavariate(1.2, 5)
            self.controller.add_calibration_noul(p, label)

    def test_uncalibrated_controller_raises(self):
        uncal = ConformalRiskController()
        with self.assertRaises(RuntimeError):
            uncal.predict_score(Score("test", score=7.5))
        with self.assertRaises(RuntimeError):
            uncal.predict_decision_threshold(Noul("test", probability=0.8))

    def test_empty_calibration_raises(self):
        empty_ctrl = ConformalRiskController()
        with self.assertRaises(ValueError):
            empty_ctrl.calibrate()

    def test_insufficient_sample_size_for_alpha(self):
        # When sample size is too small to mathematically guarantee alpha:
        # (n + 1) * alpha <= B => for alpha=0.01, B=1.0, requires n >= 100
        ctrl = ConformalRiskController(CRCConfig(alpha=0.01, max_loss=1.0, min_calibration_samples=20))
        for _ in range(30):
            ctrl.add_calibration_score(5.0, 5.2, 1.0, 10.0)
        with self.assertRaises(ValueError) as ctx:
            ctrl.calibrate()
        self.assertIn("too small to guarantee risk budget", str(ctx.exception))

    def test_score_risk_bound_miscoverage(self):
        self.controller.calibrate()
        self.assertTrue(self.controller.is_calibrated)
        self.assertIsNotNone(self.controller.score_lambda)
        self.assertGreater(self.controller.score_lambda, 0.0)

        score = Score("Toxicity evaluation", score=6.5, min_val=1.0, max_val=10.0)
        res = self.controller.predict_score(score)

        self.assertIsInstance(res, ScoreRiskBound)
        self.assertEqual(res.point_estimate, 6.5)
        self.assertAlmostEqual(res.interval[0], 6.5 - self.controller.score_lambda)
        self.assertAlmostEqual(res.interval[1], 6.5 + self.controller.score_lambda)
        self.assertTrue(res.is_safe)
        self.assertFalse(res.should_escalate)
        self.assertLessEqual(res.empirical_risk, self.controller.config.alpha)

    def test_score_interval_clamping(self):
        self.controller.calibrate()
        # Near boundary score (9.8 on [1.0, 10.0])
        score = Score("Credit score rating", score=9.8, min_val=1.0, max_val=10.0)
        res = self.controller.predict_score(score)

        # Upper bound must be clamped to max_val 10.0
        self.assertEqual(res.interval[1], 10.0)
        self.assertGreaterEqual(res.interval[0], 1.0)

    def test_score_margin_tolerance_escalation(self):
        self.controller.calibrate()
        score = Score("High-stakes medical score", score=5.0, min_val=1.0, max_val=10.0)

        # Strict tolerance: margin must not exceed 0.2 units
        res_strict = self.controller.predict_score(score, max_margin_tolerance=0.2)
        self.assertFalse(res_strict.is_safe)
        self.assertTrue(res_strict.should_escalate)

        # Generous tolerance: margin allowed up to 5.0 units
        res_generous = self.controller.predict_score(score, max_margin_tolerance=5.0)
        self.assertTrue(res_generous.is_safe)
        self.assertFalse(res_generous.should_escalate)

    def test_score_risk_bound_excess_error(self):
        ctrl = ConformalRiskController(CRCConfig(alpha=0.02, loss_type="excess_error", min_calibration_samples=20))
        rng = random.Random(42)
        for _ in range(300):
            true_s = rng.uniform(1.0, 10.0)
            pred_s = max(1.0, min(10.0, true_s + rng.gauss(0, 1.0)))
            ctrl.add_calibration_score(pred_s, true_s, 1.0, 10.0)
        ctrl.calibrate()

        res = ctrl.predict_score(Score("Test", score=5.0))
        self.assertIsNotNone(res.margin)
        self.assertGreater(res.margin, 0.0)

    def test_decision_threshold_risk_control(self):
        self.controller.calibrate()
        self.assertIsNotNone(self.controller.decision_threshold)
        self.assertGreaterEqual(self.controller.decision_threshold, 0.0)
        self.assertLessEqual(self.controller.decision_threshold, 1.0)

        noul = Noul("Is query malicious?", probability=0.95)
        res = self.controller.predict_decision_threshold(noul)

        self.assertIsInstance(res, DecisionRiskBound)
        self.assertEqual(res.optimal_threshold, self.controller.decision_threshold)
        self.assertLessEqual(res.empirical_risk, self.controller.config.alpha)

    def test_empirical_risk_guarantee_95_percent(self):
        self.controller.calibrate()
        rng = random.Random(99)
        test_samples = []
        for _ in range(1000):
            true_s = rng.uniform(1.0, 10.0)
            noise = rng.gauss(0, 0.6)
            pred_s = max(1.0, min(10.0, true_s + noise))
            test_samples.append((pred_s, true_s, 1.0, 10.0))

        metrics = self.controller.evaluate_score_risk(test_samples)
        # Empirical test loss must not significantly exceed nominal alpha (0.05)
        self.assertLessEqual(metrics["empirical_risk"], self.controller.config.alpha + 0.01)
        self.assertEqual(metrics["passed_guarantee"], 1.0)

    def test_empirical_risk_guarantee_99_percent(self):
        ctrl = ConformalRiskController(CRCConfig(alpha=0.01, min_calibration_samples=20))
        rng = random.Random(42)
        for _ in range(500):
            true_s = rng.uniform(1.0, 10.0)
            pred_s = max(1.0, min(10.0, true_s + rng.gauss(0, 0.4)))
            ctrl.add_calibration_score(pred_s, true_s, 1.0, 10.0)
        ctrl.calibrate()

        test_samples = []
        for _ in range(1000):
            true_s = rng.uniform(1.0, 10.0)
            pred_s = max(1.0, min(10.0, true_s + rng.gauss(0, 0.4)))
            test_samples.append((pred_s, true_s, 1.0, 10.0))

        metrics = ctrl.evaluate_score_risk(test_samples)
        self.assertLessEqual(metrics["empirical_risk"], 0.01 + 0.005)

    def test_binary_serialization_roundtrip(self):
        self.controller.calibrate()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.reflex-crc")
            self.controller.save(path)
            self.assertTrue(os.path.exists(path))

            loaded = ConformalRiskController.load(path)
            self.assertTrue(loaded.is_calibrated)
            self.assertEqual(loaded.config.alpha, self.controller.config.alpha)
            self.assertEqual(loaded.score_lambda, self.controller.score_lambda)
            self.assertEqual(loaded.decision_threshold, self.controller.decision_threshold)

            # Test inference parity
            score = Score("Toxicity", score=7.0)
            r1 = self.controller.predict_score(score)
            r2 = loaded.predict_score(score)
            self.assertEqual(r1.interval, r2.interval)
            self.assertEqual(r1.margin, r2.margin)

    def test_crc32_tamper_detection(self):
        self.controller.calibrate()
        raw_bytes = bytearray(self.controller.save_to_bytes())
        # Corrupt header byte
        raw_bytes[10] ^= 0xFF
        with self.assertRaises(ValueError) as ctx:
            ConformalRiskController.load_from_bytes(bytes(raw_bytes))
        self.assertIn("CRC32", str(ctx.exception))

    def test_decision_result_get_score(self):
        res = DecisionResult(
            decisions={
                "tox": Score("Toxicity", score=2.5),
                "auth": Noul("Authorized", probability=0.9),
            },
            latency_ms=1.5,
            backend="test",
        )
        s = res.get_score("tox")
        self.assertIsInstance(s, Score)
        self.assertEqual(s.score, 2.5)

        with self.assertRaises(KeyError):
            res.get_score("auth")  # Is a Noul, not a Score
        with self.assertRaises(KeyError):
            res.get_score("nonexistent")

    def test_reflex_client_crc_integration(self):
        self.controller.calibrate()
        rx = Reflex(backend="semantic", crc=self.controller)

        res = rx.evaluate(
            state="How do I reset my API key safely?",
            questions={"rating": Score("Assess query quality from 1 to 10")},
        )

        self.assertIsNotNone(res.risk_bounds)
        self.assertIn("rating", res.risk_bounds)
        bound = res.risk_bounds["rating"]
        self.assertIsInstance(bound, ScoreRiskBound)
        self.assertIsNotNone(bound.interval)
        self.assertIsInstance(res.should_escalate, bool)

    def test_reflex_client_crc_escalation(self):
        # Controller configured with very strict tolerance
        ctrl = ConformalRiskController(CRCConfig(alpha=0.05, max_margin_tolerance=0.05, min_calibration_samples=20))
        rng = random.Random(42)
        for _ in range(200):
            ctrl.add_calibration_score(5.0, 5.0 + rng.gauss(0, 1.0), 1.0, 10.0)
        ctrl.calibrate()

        rx = Reflex(backend="semantic", crc=ctrl)
        res = rx.evaluate(
            state="Analyze market crash risk",
            questions={"risk": Score("Assess portfolio risk from 1 to 10")},
        )

        self.assertTrue(res.should_escalate)
        self.assertIn("risk", res.risk_bounds)
        self.assertTrue(res.risk_bounds["risk"].should_escalate)

    def test_score_risk_bound_dict_export(self):
        self.controller.calibrate()
        score = Score("Toxicity", score=5.5)
        res = self.controller.predict_score(score)
        d = res.to_dict()
        self.assertIn("point_estimate", d)
        self.assertIn("interval", d)
        self.assertIn("interval_width", d)
        self.assertIn("margin", d)
        self.assertIn("is_safe", d)
        self.assertIn("should_escalate", d)

    def test_decision_risk_bound_dict_export(self):
        self.controller.calibrate()
        noul = Noul("Malicious", probability=0.7)
        res = self.controller.predict_decision_threshold(noul)
        d = res.to_dict()
        self.assertIn("optimal_threshold", d)
        self.assertIn("empirical_risk", d)
        self.assertIn("alpha", d)
        self.assertIn("loss_type", d)
        self.assertIn("is_safe", d)


if __name__ == "__main__":
    unittest.main()
