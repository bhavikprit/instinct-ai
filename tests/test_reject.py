"""
Tests for Selective Classification & Risk-Controlled Rejection (reflex.reject).
Phase 39: Unit tests for finite-sample risk guarantees, optimal rejection thresholds,
target risk vs target coverage, multi-metric scoring, .reflex-reject persistence,
thread-safety, and Reflex runtime integration.
"""

from collections import deque
import math
import os
import random
import tempfile
import threading
import unittest

from reflex.reject import (
    REJECT_MAGIC,
    REJECT_MIN_FILE_SIZE,
    SelectiveClassifier,
    SelectiveDecision,
    SelectiveRejectConfig,
    RiskCoveragePoint,
    _inv_norm_cdf,
    binomial_risk_upper_bound,
)
from reflex.client import Reflex
from reflex.primitives import Choice, DecisionResult, Noul


class TestInvNormCdfAndRiskBound(unittest.TestCase):
    """Tests for inverse normal CDF and Wilson score risk upper bound."""

    def test_inv_norm_cdf_standard_values(self):
        # Phi^-1(0.5) = 0.0
        self.assertAlmostEqual(_inv_norm_cdf(0.5), 0.0, places=3)
        # Phi^-1(0.8413) ~ 1.0
        self.assertAlmostEqual(_inv_norm_cdf(0.8413), 1.0, places=2)
        # Phi^-1(0.95) ~ 1.645
        self.assertAlmostEqual(_inv_norm_cdf(0.95), 1.645, places=2)
        # Phi^-1(0.975) ~ 1.960
        self.assertAlmostEqual(_inv_norm_cdf(0.975), 1.960, places=2)

    def test_inv_norm_cdf_symmetry(self):
        for p in [0.01, 0.05, 0.1, 0.2, 0.35]:
            lower = _inv_norm_cdf(p)
            upper = _inv_norm_cdf(1.0 - p)
            self.assertAlmostEqual(lower, -upper, places=3)

    def test_inv_norm_cdf_monotonicity(self):
        probs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantiles = [_inv_norm_cdf(p) for p in probs]
        for i in range(len(quantiles) - 1):
            self.assertLess(quantiles[i], quantiles[i + 1])

    def test_inv_norm_cdf_clamping(self):
        # Should not raise or return NaN at extreme values
        val_0 = _inv_norm_cdf(0.0)
        val_1 = _inv_norm_cdf(1.0)
        self.assertTrue(math.isfinite(val_0))
        self.assertTrue(math.isfinite(val_1))
        self.assertLess(val_0, -4.0)
        self.assertGreater(val_1, 4.0)

    def test_binomial_risk_upper_bound_zero_total(self):
        # If total accepted is 0, risk is unconstrained (1.0)
        self.assertEqual(binomial_risk_upper_bound(0, 0), 1.0)
        self.assertEqual(binomial_risk_upper_bound(5, 0), 1.0)

    def test_binomial_risk_upper_bound_zero_errors(self):
        # 0 errors in 100 trials: rule of three ~ -log(0.05) / 100 ~ 2.996 / 100 ~ 0.03
        bnd = binomial_risk_upper_bound(0, 100, delta=0.05)
        self.assertGreater(bnd, 0.0)
        self.assertLess(bnd, 0.05)
        # More samples with 0 errors should strictly reduce upper bound
        bnd_500 = binomial_risk_upper_bound(0, 500, delta=0.05)
        self.assertLess(bnd_500, bnd)

    def test_binomial_risk_upper_bound_all_errors(self):
        # If all decisions are errors, upper bound should be close to 1.0
        bnd = binomial_risk_upper_bound(50, 50, delta=0.05)
        self.assertGreaterEqual(bnd, 0.90)
        self.assertLessEqual(bnd, 1.0)

    def test_binomial_risk_upper_bound_monotonicity(self):
        # Holding total constant, more errors must strictly increase the upper bound
        bounds = [binomial_risk_upper_bound(k, 100, delta=0.05) for k in range(0, 50, 5)]
        for i in range(len(bounds) - 1):
            self.assertLess(bounds[i], bounds[i + 1])


class TestSelectiveRejectConfig(unittest.TestCase):
    """Tests for SelectiveRejectConfig validation and serialization."""

    def test_default_config(self):
        cfg = SelectiveRejectConfig()
        self.assertEqual(cfg.target_risk, 0.02)
        self.assertIsNone(cfg.target_coverage)
        self.assertEqual(cfg.confidence_bound, 0.05)
        self.assertEqual(cfg.min_calibration_samples, 30)
        self.assertEqual(cfg.scoring_method, "confidence")
        self.assertEqual(cfg.window_size, 1000)

    def test_valid_custom_config(self):
        cfg = SelectiveRejectConfig(
            target_risk=None,
            target_coverage=0.85,
            confidence_bound=0.01,
            min_calibration_samples=50,
            scoring_method="margin",
            window_size=2000,
        )
        self.assertIsNone(cfg.target_risk)
        self.assertEqual(cfg.target_coverage, 0.85)
        self.assertEqual(cfg.confidence_bound, 0.01)
        self.assertEqual(cfg.scoring_method, "margin")

    def test_invalid_target_risk(self):
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(target_risk=-0.1)
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(target_risk=1.5)

    def test_invalid_target_coverage(self):
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(target_coverage=0.0)
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(target_coverage=1.1)

    def test_invalid_confidence_bound(self):
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(confidence_bound=0.0)
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(confidence_bound=1.0)

    def test_invalid_min_samples(self):
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(min_calibration_samples=2)

    def test_invalid_scoring_method(self):
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(scoring_method="unsupported")

    def test_invalid_window_size(self):
        with self.assertRaises(ValueError):
            SelectiveRejectConfig(min_calibration_samples=50, window_size=20)

    def test_serialization_roundtrip(self):
        cfg = SelectiveRejectConfig(
            target_risk=0.015,
            target_coverage=None,
            confidence_bound=0.02,
            min_calibration_samples=40,
            scoring_method="entropy",
            window_size=500,
        )
        d = cfg.to_dict()
        restored = SelectiveRejectConfig.from_dict(d)
        self.assertEqual(cfg.target_risk, restored.target_risk)
        self.assertEqual(cfg.target_coverage, restored.target_coverage)
        self.assertEqual(cfg.confidence_bound, restored.confidence_bound)
        self.assertEqual(cfg.min_calibration_samples, restored.min_calibration_samples)
        self.assertEqual(cfg.scoring_method, restored.scoring_method)
        self.assertEqual(cfg.window_size, restored.window_size)


class TestSelectiveDecision(unittest.TestCase):
    """Tests for SelectiveDecision dataclass and dict/item access."""

    def test_decision_fields_and_to_dict(self):
        dec = SelectiveDecision(
            accepted=True,
            score=0.92,
            threshold=0.85,
            empirical_risk=0.012,
            upper_bound_risk=0.018,
            coverage=0.78,
            should_escalate=False,
            reason="Autonomous: score 0.9200 >= threshold 0.8500",
        )
        self.assertTrue(dec.accepted)
        self.assertFalse(dec.should_escalate)
        d = dec.to_dict()
        self.assertEqual(d["accepted"], True)
        self.assertEqual(d["score"], 0.92)
        self.assertEqual(d["threshold"], 0.85)
        self.assertEqual(d["should_escalate"], False)

    def test_decision_contains_and_getitem(self):
        dec = SelectiveDecision(
            accepted=False,
            score=0.62,
            threshold=0.85,
            empirical_risk=0.012,
            upper_bound_risk=0.018,
            coverage=0.78,
            should_escalate=True,
            reason="Abstained: score 0.6200 < threshold 0.8500",
        )
        # Subscript access
        self.assertEqual(dec["accepted"], False)
        self.assertEqual(dec["should_escalate"], True)
        self.assertEqual(dec["score"], 0.62)
        # in operator
        self.assertIn("accepted", dec)
        self.assertIn("should_escalate", dec)
        self.assertIn("reason", dec)
        self.assertNotIn("non_existent_field", dec)


class TestSelectiveClassifierCalibration(unittest.TestCase):
    """Tests for SelectiveClassifier calibration and risk control guarantees."""

    def test_initial_uncalibrated_state(self):
        sc = SelectiveClassifier()
        self.assertFalse(sc.is_calibrated)
        self.assertEqual(sc.num_calibration_samples, 0)
        self.assertEqual(sc.total_samples, 0)
        self.assertEqual(sc.threshold, 0.5)

    def test_uncalibrated_evaluation(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=30))
        noul = Noul(instructions="test", probability=0.7)
        dec = sc.evaluate_noul(noul)
        # Uncalibrated warm-up: accepts by default
        self.assertTrue(dec.accepted)
        self.assertFalse(dec.should_escalate)
        self.assertIn("warmup", dec.reason.lower())

    def test_target_risk_calibration(self):
        # Create dataset where high confidence has 0 errors, low confidence has errors
        sc = SelectiveClassifier(SelectiveRejectConfig(target_risk=0.05, min_calibration_samples=50))
        for _ in range(50):
            # High confidence [0.90, 0.99] with 0 errors
            sc.add_sample(0.95, 0)
        for _ in range(50):
            # Low confidence [0.55, 0.65] with many errors
            sc.add_sample(0.60, 1)

        th = sc.calibrate()
        self.assertTrue(sc.is_calibrated)
        # Threshold should isolate the high-confidence slice
        self.assertGreaterEqual(th, 0.90)
        self.assertEqual(sc.calibrated_empirical_risk, 0.0)
        # Upper bound should be <= target risk or optimal
        self.assertLessEqual(sc.calibrated_upper_risk, 0.10)
        self.assertAlmostEqual(sc.calibrated_coverage, 0.50, places=2)

    def test_target_coverage_calibration(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(
            target_risk=None,
            target_coverage=0.70,
            min_calibration_samples=50,
        ))
        rng = random.Random(42)
        for _ in range(100):
            conf = 0.5 + 0.5 * rng.random()
            err = 1 if conf < 0.7 else 0
            sc.add_sample(conf, err)

        th = sc.calibrate()
        self.assertTrue(sc.is_calibrated)
        self.assertGreaterEqual(sc.calibrated_coverage, 0.70)

    def test_fit_method(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=20))
        scores = [0.6 + 0.01 * i for i in range(30)]
        errors = [1 if s < 0.75 else 0 for s in scores]

        sc.fit(scores, errors)
        self.assertTrue(sc.is_calibrated)
        self.assertEqual(sc.num_calibration_samples, 30)

    def test_fit_mismatched_lengths(self):
        sc = SelectiveClassifier()
        with self.assertRaises(ValueError):
            sc.fit([0.5, 0.6], [0])

    def test_rolling_window_eviction(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=10, window_size=20))
        for i in range(35):
            sc.add_sample(0.8, 0)
        self.assertEqual(sc.num_calibration_samples, 20)
        self.assertEqual(sc.total_samples, 35)


class TestSelectiveClassifierEvaluation(unittest.TestCase):
    """Tests for Noul and Choice evaluation across scoring methods."""

    def test_extract_noul_score(self):
        sc = SelectiveClassifier()
        # p = 0.9 -> score 0.9
        self.assertAlmostEqual(sc.extract_noul_score(Noul(instructions="test", probability=0.9)), 0.9)
        # p = 0.1 -> score 0.9 (confident False)
        self.assertAlmostEqual(sc.extract_noul_score(Noul(instructions="test", probability=0.1)), 0.9)
        # p = 0.5 -> score 0.5 (maximum ambiguity)
        self.assertAlmostEqual(sc.extract_noul_score(Noul(instructions="test", probability=0.5)), 0.5)

    def test_extract_choice_score_confidence(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(scoring_method="confidence"))
        c = Choice(instructions="test", options=["A", "B"], selected="A", distribution={"A": 0.85, "B": 0.15})
        self.assertAlmostEqual(sc.extract_choice_score(c), 0.85)

    def test_extract_choice_score_margin(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(scoring_method="margin"))
        c = Choice(instructions="test", options=["A", "B", "C"], selected="A", distribution={"A": 0.7, "B": 0.2, "C": 0.1})
        # margin = 0.7 - 0.2 = 0.5
        self.assertAlmostEqual(sc.extract_choice_score(c), 0.5)

    def test_extract_choice_score_entropy(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(scoring_method="entropy"))
        # Uniform distribution (highest entropy -> lowest score)
        c_unif = Choice(instructions="test", options=["A", "B"], selected="A", distribution={"A": 0.5, "B": 0.5})
        score_unif = sc.extract_choice_score(c_unif)
        # Sharp distribution (lowest entropy -> highest score)
        c_sharp = Choice(instructions="test", options=["A", "B"], selected="A", distribution={"A": 0.99, "B": 0.01})
        score_sharp = sc.extract_choice_score(c_sharp)
        self.assertGreater(score_sharp, score_unif)
        self.assertAlmostEqual(score_unif, 0.0, places=3)
        self.assertGreater(score_sharp, 0.90)
        self.assertLessEqual(score_sharp, 1.0)

    def test_evaluate_noul_accepted_and_rejected(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=30))
        for _ in range(30):
            sc.add_sample(0.9, 0)
        sc.calibrate()
        sc._threshold = 0.80

        # Confident Noul (score 0.92 >= 0.80) -> accepted
        noul_hi = Noul(instructions="test", probability=0.92)
        dec_hi = sc.evaluate_noul(noul_hi)
        self.assertTrue(dec_hi.accepted)
        self.assertFalse(dec_hi.should_escalate)

        # Ambiguous Noul (score 0.60 < 0.80) -> rejected
        noul_lo = Noul(instructions="test", probability=0.60)
        dec_lo = sc.evaluate_noul(noul_lo)
        self.assertFalse(dec_lo.accepted)
        self.assertTrue(dec_lo.should_escalate)
        self.assertIn("Abstained", dec_lo.reason)

    def test_evaluate_choice_accepted_and_rejected(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=30))
        for _ in range(30):
            sc.add_sample(0.9, 0)
        sc.calibrate()
        sc._threshold = 0.80

        c_hi = Choice(instructions="test", options=["refund", "escalate"], selected="refund", distribution={"refund": 0.88, "escalate": 0.12})
        dec_hi = sc.evaluate_choice(c_hi)
        self.assertTrue(dec_hi.accepted)
        self.assertFalse(dec_hi.should_escalate)

        c_lo = Choice(instructions="test", options=["refund", "escalate"], selected="refund", distribution={"refund": 0.65, "escalate": 0.35})
        dec_lo = sc.evaluate_choice(c_lo)
        self.assertFalse(dec_lo.accepted)
        self.assertTrue(dec_lo.should_escalate)


class TestRiskCoverageCurveAndAURC(unittest.TestCase):
    """Tests for Risk-Coverage curve generation, AURC, and ASCII visualization."""

    def test_risk_coverage_curve_points(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=10))
        for i in range(50):
            conf = 0.5 + 0.01 * i
            err = 1 if conf < 0.75 else 0
            sc.add_sample(conf, err)

        curve = sc.risk_coverage_curve(num_points=10)
        self.assertEqual(len(curve), 10)
        # As threshold increases, coverage should be monotonically non-increasing
        for i in range(len(curve) - 1):
            self.assertGreaterEqual(curve[i].coverage, curve[i + 1].coverage)
            self.assertGreaterEqual(curve[i].empirical_risk, curve[i + 1].empirical_risk)

    def test_aurc_calculation(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=10))
        # Perfect model: zero errors everywhere
        for i in range(50):
            sc.add_sample(0.5 + 0.01 * i, 0)
        aurc_zero = sc.aurc(num_points=20)
        self.assertAlmostEqual(aurc_zero, 0.0, places=5)

        # Noisy model: lower confidence has high errors
        sc2 = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=10))
        for i in range(50):
            conf = 0.5 + 0.01 * i
            err = 1 if conf < 0.75 else 0
            sc2.add_sample(conf, err)
        aurc_noisy = sc2.aurc(num_points=20)
        self.assertGreater(aurc_noisy, 0.0)

    def test_ascii_curve_rendering(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=10))
        # Empty curve
        self.assertIn("No calibration data", sc.ascii_risk_coverage_curve())

        for i in range(30):
            sc.add_sample(0.5 + 0.01 * i, 1 if i < 10 else 0)
        txt = sc.ascii_risk_coverage_curve(num_points=5)
        self.assertIn("Threshold", txt)
        self.assertIn("Coverage", txt)
        self.assertIn("Empirical Risk", txt)
        self.assertIn("|", txt)


class TestBinaryPersistence(unittest.TestCase):
    """Tests for .reflex-reject binary persistence, header format, and CRC32 tamper detection."""

    def test_save_and_load_roundtrip(self):
        cfg = SelectiveRejectConfig(
            target_risk=0.03,
            confidence_bound=0.01,
            min_calibration_samples=30,
            scoring_method="margin",
            window_size=500,
        )
        sc = SelectiveClassifier(config=cfg)
        for i in range(50):
            sc.add_sample(0.6 + 0.005 * i, 1 if i < 15 else 0)
        sc.calibrate()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "model.reflex-reject")
            sc.save(file_path)
            self.assertTrue(os.path.exists(file_path))

            # Inspect raw file
            with open(file_path, "rb") as f:
                raw = f.read()
            self.assertEqual(raw[:4], REJECT_MAGIC)
            self.assertGreaterEqual(len(raw), REJECT_MIN_FILE_SIZE)

            # Load back
            loaded = SelectiveClassifier.load(file_path)
            self.assertEqual(loaded.config.target_risk, 0.03)
            self.assertEqual(loaded.config.confidence_bound, 0.01)
            self.assertEqual(loaded.config.scoring_method, "margin")
            self.assertEqual(loaded.num_calibration_samples, 50)
            self.assertAlmostEqual(loaded.threshold, sc.threshold, places=4)
            self.assertAlmostEqual(loaded.calibrated_empirical_risk, sc.calibrated_empirical_risk, places=4)
            self.assertAlmostEqual(loaded.calibrated_upper_risk, sc.calibrated_upper_risk, places=4)
            self.assertAlmostEqual(loaded.calibrated_coverage, sc.calibrated_coverage, places=4)

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            SelectiveClassifier.load("/path/to/nonexistent/file.reflex-reject")

    def test_load_truncated_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.reject")
            with open(path, "wb") as f:
                f.write(b"RFRJshort")
            with self.assertRaises(ValueError) as ctx:
                SelectiveClassifier.load(path)
            self.assertIn("below minimum", str(ctx.exception))

    def test_load_corrupted_magic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad_magic.reject")
            with open(path, "wb") as f:
                f.write(b"XXXX" + b"\x00" * 100)
            with self.assertRaises(ValueError) as ctx:
                SelectiveClassifier.load(path)
            self.assertIn("Invalid magic", str(ctx.exception))

    def test_load_corrupted_checksum(self):
        cfg = SelectiveRejectConfig(min_calibration_samples=10)
        sc = SelectiveClassifier(config=cfg)
        for _ in range(15):
            sc.add_sample(0.85, 0)
        sc.calibrate()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrupt.reject")
            sc.save(path)

            with open(path, "rb") as f:
                data = bytearray(f.read())
            # Corrupt byte in payload
            data[55] ^= 0xFF
            with open(path, "wb") as f:
                f.write(data)

            with self.assertRaises(ValueError) as ctx:
                SelectiveClassifier.load(path)
            self.assertIn("checksum mismatch", str(ctx.exception))


class TestReflexClientIntegration(unittest.TestCase):
    """Tests for Reflex runtime client integration with selective classification."""

    def test_client_initialization_with_reject_obj(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=20))
        client = Reflex(selective_reject=sc)
        self.assertIs(client.selective_reject, sc)

    def test_client_initialization_with_reject_bool(self):
        client = Reflex(selective_reject=True)
        self.assertIsNotNone(client.selective_reject)
        self.assertIsInstance(client.selective_reject, SelectiveClassifier)

    def test_client_evaluate_populates_rejection(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=20))
        for _ in range(25):
            sc.add_sample(0.9, 0)
        sc.calibrate()

        # Permissive threshold -> accepted
        sc._threshold = 0.01
        client = Reflex(selective_reject=sc)

        res_hi = client.evaluate(
            "User refund verification",
            {"is_approved": Noul(instructions="Is refund approved?", threshold=0.5)},
        )
        self.assertIsNotNone(res_hi.rejection)
        self.assertIn("is_approved", res_hi.rejection)
        self.assertTrue(res_hi.rejection["is_approved"]["accepted"])
        self.assertFalse(res_hi.rejection["is_approved"]["should_escalate"])

        # Stringent threshold -> rejected and escalated
        sc._threshold = 0.999
        res_lo = client.evaluate(
            "User refund verification",
            {"is_approved": Noul(instructions="Is refund approved?", threshold=0.5)},
        )
        self.assertIsNotNone(res_lo.rejection)
        self.assertIn("is_approved", res_lo.rejection)
        self.assertFalse(res_lo.rejection["is_approved"]["accepted"])
        self.assertTrue(res_lo.rejection["is_approved"]["should_escalate"])
        self.assertTrue(res_lo.should_escalate)

    def test_client_record_selective_feedback(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=10))
        client = Reflex(selective_reject=sc)
        self.assertEqual(sc.num_calibration_samples, 0)

        client.record_selective_feedback(0.88, is_error=0)
        self.assertEqual(sc.num_calibration_samples, 1)
        client.record_selective_feedback(0.62, is_error=1)
        self.assertEqual(sc.num_calibration_samples, 2)


class TestThreadSafety(unittest.TestCase):
    """Tests for thread safety under concurrent updates and evaluations."""

    def test_concurrent_add_and_evaluate(self):
        sc = SelectiveClassifier(SelectiveRejectConfig(min_calibration_samples=20))
        for _ in range(25):
            sc.add_sample(0.8, 0)
        sc.calibrate()

        errors = []

        def worker_writer():
            try:
                for i in range(100):
                    sc.add_sample(0.7 + 0.002 * (i % 50), is_error=i % 10 == 0)
            except Exception as e:
                errors.append(e)

        def worker_evaluator():
            try:
                for _ in range(100):
                    noul = Noul(instructions="test", probability=0.85)
                    dec = sc.evaluate_noul(noul)
                    self.assertIsInstance(dec.accepted, bool)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker_writer),
            threading.Thread(target=worker_evaluator),
            threading.Thread(target=worker_writer),
            threading.Thread(target=worker_evaluator),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
