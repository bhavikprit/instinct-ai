"""
Tests for Venn-Abers Multi-Class Conformal Predictors (reflex.venn_abers).
Phase 38: Unit tests for PAVA isotonic regression, certified multi-probabilistic intervals [p0, p1],
epistemic uncertainty bounds, multi-class IVAP, binary persistence (.reflex-va), and Reflex runtime integration.
"""

from collections import deque
import math
import os
import random
import tempfile
import threading
import unittest

from reflex.venn_abers import (
    VA_MAGIC,
    VA_MIN_FILE_SIZE,
    VennAbersConfig,
    VennAbersChoiceResult,
    VennAbersNoulResult,
    VennAbersPredictor,
    pava_isotonic_regression,
)
from reflex.client import Reflex
from reflex.primitives import Choice, DecisionResult, Noul


class TestPAVA(unittest.TestCase):
    """Tests for the Pool Adjacent Violators Algorithm (PAVA)."""

    def test_empty_and_single(self):
        self.assertEqual(pava_isotonic_regression([], []), [])
        self.assertEqual(pava_isotonic_regression([0.5], [1.0]), [1.0])

    def test_already_monotonic(self):
        scores = [0.1, 0.2, 0.3, 0.4]
        labels = [0.1, 0.3, 0.7, 0.9]
        fitted = pava_isotonic_regression(scores, labels)
        for expected, actual in zip(labels, fitted):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_complete_inversion(self):
        scores = [0.1, 0.2, 0.3, 0.4]
        labels = [1.0, 1.0, 0.0, 0.0]
        fitted = pava_isotonic_regression(scores, labels)
        # Average is (1 + 1 + 0 + 0) / 4 = 0.5
        for val in fitted:
            self.assertAlmostEqual(val, 0.5, places=5)

    def test_partial_pooling(self):
        # [1, 0, 1] -> first two pool to 0.5, third is 1.0 -> [0.5, 0.5, 1.0]
        scores = [0.1, 0.2, 0.3]
        labels = [1.0, 0.0, 1.0]
        fitted = pava_isotonic_regression(scores, labels)
        self.assertAlmostEqual(fitted[0], 0.5, places=5)
        self.assertAlmostEqual(fitted[1], 0.5, places=5)
        self.assertAlmostEqual(fitted[2], 1.0, places=5)

    def test_monotonicity_guarantee(self):
        rng = random.Random(42)
        scores = sorted([rng.random() for _ in range(50)])
        labels = [rng.choice([0.0, 1.0]) for _ in range(50)]
        fitted = pava_isotonic_regression(scores, labels)
        self.assertEqual(len(fitted), 50)
        for i in range(len(fitted) - 1):
            self.assertLessEqual(fitted[i], fitted[i + 1] + 1e-9)


class TestVennAbersConfig(unittest.TestCase):
    """Tests for VennAbersConfig validation and serialization."""

    def test_default_config(self):
        cfg = VennAbersConfig()
        self.assertAlmostEqual(cfg.max_uncertainty_threshold, 0.20)
        self.assertEqual(cfg.min_calibration_samples, 20)
        self.assertEqual(cfg.point_estimator, "balanced")
        self.assertEqual(cfg.window_size, 500)

    def test_validation(self):
        with self.assertRaises(ValueError):
            VennAbersConfig(max_uncertainty_threshold=-0.1)
        with self.assertRaises(ValueError):
            VennAbersConfig(max_uncertainty_threshold=1.5)
        with self.assertRaises(ValueError):
            VennAbersConfig(min_calibration_samples=1)
        with self.assertRaises(ValueError):
            VennAbersConfig(point_estimator="invalid_mode")
        with self.assertRaises(ValueError):
            VennAbersConfig(min_calibration_samples=100, window_size=50)

    def test_serialization_roundtrip(self):
        cfg = VennAbersConfig(
            max_uncertainty_threshold=0.15,
            min_calibration_samples=25,
            point_estimator="midpoint",
            window_size=300,
        )
        d = cfg.to_dict()
        loaded = VennAbersConfig.from_dict(d)
        self.assertAlmostEqual(cfg.max_uncertainty_threshold, loaded.max_uncertainty_threshold)
        self.assertEqual(cfg.min_calibration_samples, loaded.min_calibration_samples)
        self.assertEqual(cfg.point_estimator, loaded.point_estimator)
        self.assertEqual(cfg.window_size, loaded.window_size)


class TestVennAbersPredictor(unittest.TestCase):
    """Unit tests for VennAbersPredictor core behavior."""

    def setUp(self):
        self.config = VennAbersConfig(
            max_uncertainty_threshold=0.20,
            min_calibration_samples=20,
            point_estimator="balanced",
            window_size=200,
        )
        self.va = VennAbersPredictor(config=self.config)

    def test_uncalibrated_state(self):
        self.assertFalse(self.va.is_calibrated)
        self.assertEqual(self.va.num_calibration_samples, 0)
        res = self.va.predict_noul(0.65)
        self.assertFalse(res.is_calibrated)
        self.assertAlmostEqual(res.p_calibrated, 0.65)
        self.assertAlmostEqual(res.uncertainty, 0.0)
        self.assertFalse(res.should_escalate)

    def test_add_calibration_sample_and_warmup(self):
        for i in range(19):
            self.va.add_calibration_sample(0.1 + 0.04 * i, i % 2)
        self.assertFalse(self.va.is_calibrated)

        self.va.add_calibration_sample(0.9, 1)
        self.assertTrue(self.va.is_calibrated)
        self.assertEqual(self.va.num_calibration_samples, 20)

    def test_p0_le_p1_mathematical_invariant(self):
        rng = random.Random(101)
        # Populate calibration data
        for _ in range(50):
            sc = rng.random()
            y = 1 if sc > 0.4 else 0
            self.va.add_calibration_sample(sc, y)

        # Test invariant across 30 query scores
        for _ in range(30):
            q = rng.random()
            res = self.va.predict_noul(q)
            self.assertGreaterEqual(res.p0, 0.0)
            self.assertLessEqual(res.p1, 1.0)
            self.assertLessEqual(res.p0, res.p1 + 1e-9, f"p0 ({res.p0}) > p1 ({res.p1}) at query {q}")
            self.assertGreaterEqual(res.uncertainty, 0.0)
            self.assertGreaterEqual(res.p_calibrated, res.p0 - 1e-9)
            self.assertLessEqual(res.p_calibrated, res.p1 + 1e-9)

    def test_dense_vs_sparse_epistemic_uncertainty(self):
        # Calibration data clustered around score 0.50 (normal distribution)
        rng = random.Random(42)
        for _ in range(200):
            sc = min(0.95, max(0.05, rng.gauss(0.5, 0.08)))
            y = 1 if rng.random() < sc else 0
            self.va.add_calibration_sample(sc, y)

        # Query near 0.50 (in-distribution, dense calibration data)
        res_dense = self.va.predict_noul(0.50)
        # Query near 0.02 (out-of-distribution, very sparse data)
        res_sparse = self.va.predict_noul(0.01)

        # Dense query should have much tighter uncertainty interval than sparse/OOD query
        self.assertLess(res_dense.uncertainty, 0.10)
        self.assertGreater(res_sparse.uncertainty, res_dense.uncertainty)

    def test_balanced_point_estimator(self):
        # Balanced formula: p = p1 / (1 - p0 + p1)
        for i in range(25):
            self.va.add_calibration_sample(0.2, 0)
            self.va.add_calibration_sample(0.8, 1)

        res = self.va.predict_noul(0.5)
        # Check that point estimator lies between bounds
        self.assertGreaterEqual(res.p_calibrated, res.p0)
        self.assertLessEqual(res.p_calibrated, res.p1)

    def test_midpoint_point_estimator(self):
        cfg = VennAbersConfig(min_calibration_samples=10, point_estimator="midpoint")
        va = VennAbersPredictor(config=cfg)
        for i in range(20):
            va.add_calibration_sample(0.1 * i, i % 2)

        res = va.predict_noul(0.5)
        expected_mid = 0.5 * (res.p0 + res.p1)
        self.assertAlmostEqual(res.p_calibrated, expected_mid, places=5)

    def test_uncertainty_threshold_escalation(self):
        # Small dataset with sparse extremes creates wide intervals
        for i in range(20):
            self.va.add_calibration_sample(0.5, i % 2)

        # Query far from 0.5 where data is absent
        res_ood = self.va.predict_noul(0.99)
        if res_ood.uncertainty > self.config.max_uncertainty_threshold:
            self.assertTrue(res_ood.should_escalate)

    def test_fit_batch(self):
        scores = [0.1 * i for i in range(30)]
        labels = [1 if s > 1.5 else 0 for s in scores]
        self.va.fit(scores, labels)
        self.assertEqual(self.va.num_calibration_samples, 30)
        self.assertTrue(self.va.is_calibrated)

    def test_mean_uncertainty_and_brier(self):
        for i in range(30):
            self.va.add_calibration_sample(0.1 + 0.02 * i, 1 if i > 15 else 0)

        mean_u = self.va.mean_uncertainty(num_eval_points=10)
        self.assertGreater(mean_u, 0.0)
        self.assertLess(mean_u, 1.0)

        brier = self.va.compute_brier_score()
        self.assertGreaterEqual(brier, 0.0)


class TestMultiClassVennAbers(unittest.TestCase):
    """Tests for multi-class Inductive Venn-Abers Predictor (IVAP)."""

    def setUp(self):
        self.classes = ["search", "calculator", "finish"]
        self.va = VennAbersPredictor(
            config=VennAbersConfig(min_calibration_samples=25, max_uncertainty_threshold=0.25),
            classes=self.classes,
        )

    def test_uncalibrated_choice_prediction(self):
        raw_dist = {"search": 0.6, "calculator": 0.3, "finish": 0.1}
        res = self.va.predict_choice(raw_dist)
        self.assertFalse(res.is_calibrated)
        self.assertEqual(res.selected, "search")
        self.assertEqual(res.calibrated_distribution, raw_dist)

    def test_multi_class_ivap_calibration(self):
        rng = random.Random(42)
        for _ in range(60):
            chosen = rng.choice(self.classes)
            # Higher score for the chosen class
            sc = rng.uniform(0.6, 0.95)
            self.va.add_calibration_sample(sc, chosen)

        raw_dist = {"search": 0.7, "calculator": 0.2, "finish": 0.1}
        res = self.va.predict_choice(raw_dist)

        self.assertTrue(res.is_calibrated)
        self.assertEqual(len(res.intervals), 3)
        self.assertAlmostEqual(sum(res.calibrated_distribution.values()), 1.0, places=4)
        for c in self.classes:
            p0, p1 = res.intervals[c]
            self.assertLessEqual(p0, p1 + 1e-9)
            self.assertGreaterEqual(res.uncertainties[c], 0.0)

        self.assertIn(res.selected, self.classes)


class TestVennAbersPersistence(unittest.TestCase):
    """Tests for .reflex-va binary serialization and CRC32 tamper detection."""

    def test_save_load_roundtrip(self):
        config = VennAbersConfig(max_uncertainty_threshold=0.18, min_calibration_samples=20)
        va = VennAbersPredictor(config=config, classes=["fraud", "legit"])
        for i in range(30):
            va.add_calibration_sample(0.1 + 0.02 * i, 1 if i % 2 == 0 else 0)

        with tempfile.NamedTemporaryFile(suffix=".reflex-va", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            va.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreaterEqual(os.path.getsize(tmp_path), VA_MIN_FILE_SIZE)

            loaded = VennAbersPredictor.load(tmp_path)
            self.assertEqual(loaded.num_calibration_samples, va.num_calibration_samples)
            self.assertAlmostEqual(loaded.config.max_uncertainty_threshold, 0.18)
            self.assertEqual(loaded.classes, ["fraud", "legit"])

            # Verify predictions match
            res_orig = va.predict_noul(0.45)
            res_loaded = loaded.predict_noul(0.45)
            self.assertAlmostEqual(res_orig.p0, res_loaded.p0, places=5)
            self.assertAlmostEqual(res_orig.p1, res_loaded.p1, places=5)
            self.assertAlmostEqual(res_orig.p_calibrated, res_loaded.p_calibrated, places=5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_corrupted_crc32_raises_error(self):
        va = VennAbersPredictor()
        for i in range(20):
            va.add_calibration_sample(0.5, 1)

        with tempfile.NamedTemporaryFile(suffix=".reflex-va", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            va.save(tmp_path)
            # Corrupt byte 12
            with open(tmp_path, "r+b") as f:
                f.seek(12)
                b = f.read(1)
                f.seek(12)
                f.write(bytes([(b[0] ^ 0xFF)]))

            with self.assertRaises(ValueError) as ctx:
                VennAbersPredictor.load(tmp_path)
            self.assertIn("CRC32 checksum mismatch", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestReflexClientVennAbersIntegration(unittest.TestCase):
    """Tests Reflex client integration with Venn-Abers predictor."""

    def test_client_noul_recalibration(self):
        va = VennAbersPredictor(config=VennAbersConfig(min_calibration_samples=20))
        for i in range(25):
            va.add_calibration_sample(0.2 if i < 15 else 0.8, 0 if i < 15 else 1)

        rx = Reflex(venn_abers=va)
        res = rx.evaluate(
            "User payment verification",
            {"is_fraud": Noul(instructions="Is fraudulent?", threshold=0.6)},
        )
        self.assertIsNotNone(res.venn_abers)
        self.assertIn("is_fraud", res.venn_abers)
        va_res = res.venn_abers["is_fraud"]
        self.assertTrue(va_res["is_calibrated"])
        self.assertIn("interval", va_res)

    def test_client_escalation_on_high_uncertainty(self):
        va = VennAbersPredictor(
            config=VennAbersConfig(min_calibration_samples=20, max_uncertainty_threshold=0.01)
        )
        for i in range(20):
            va.add_calibration_sample(0.5, i % 2)

        rx = Reflex(venn_abers=va)
        res = rx.evaluate(
            "Ambiguous question",
            {"q": Noul(instructions="Unfamiliar question?", threshold=0.5)},
        )
        self.assertTrue(res.should_escalate)

    def test_record_venn_abers_feedback(self):
        va = VennAbersPredictor()
        rx = Reflex(venn_abers=va)

        rx.record_venn_abers_feedback(0.85, True)
        self.assertEqual(va.num_calibration_samples, 1)


class TestVennAbersThreadSafety(unittest.TestCase):
    """Tests thread-safety of VennAbersPredictor under concurrent calls."""

    def test_concurrent_predictions_and_updates(self):
        va = VennAbersPredictor(config=VennAbersConfig(min_calibration_samples=20))
        for i in range(25):
            va.add_calibration_sample(0.5, i % 2)

        errors = []

        def worker(w_id: int):
            try:
                for i in range(30):
                    q = 0.1 + 0.02 * (i % 40)
                    va.add_calibration_sample(q, 1 if q > 0.5 else 0)
                    _ = va.predict_noul(q)
                    _ = va.predict_choice({"A": q, "B": 1.0 - q})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


if __name__ == "__main__":
    unittest.main()
