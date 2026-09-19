"""
Tests for Online Calibrated ECE & Temperature-Scaling Drift Adaptation (reflex.calib).
Phase 37: Unit tests for streaming calibration metrics, online NLL gradient adaptation,
reliability diagrams, binary persistence (.reflex-calib), and Reflex runtime integration.
"""

from collections import deque
import math
import os
import tempfile
import threading
import unittest

from reflex.calib import (
    CALIB_MAGIC,
    CALIB_MIN_FILE_SIZE,
    CalibConfig,
    CalibrationStatus,
    OnlineProbabilityCalibrator,
    _logit,
    _sigmoid,
    _softmax,
)
from reflex.client import Reflex
from reflex.primitives import Choice, DecisionResult, Noul, Score


class TestCalibConfig(unittest.TestCase):
    """Tests for CalibConfig validation and serialization."""

    def test_default_config(self):
        cfg = CalibConfig()
        self.assertEqual(cfg.num_bins, 10)
        self.assertEqual(cfg.window_size, 100)
        self.assertAlmostEqual(cfg.learning_rate, 0.05)
        self.assertAlmostEqual(cfg.min_temperature, 0.05)
        self.assertAlmostEqual(cfg.max_temperature, 10.0)
        self.assertAlmostEqual(cfg.ece_threshold, 0.08)
        self.assertEqual(cfg.min_samples_for_alarm, 30)
        self.assertAlmostEqual(cfg.initial_temperature, 1.0)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            CalibConfig(num_bins=1)
        with self.assertRaises(ValueError):
            CalibConfig(window_size=5)
        with self.assertRaises(ValueError):
            CalibConfig(learning_rate=-0.01)
        with self.assertRaises(ValueError):
            CalibConfig(min_temperature=5.0, max_temperature=2.0)
        with self.assertRaises(ValueError):
            CalibConfig(ece_threshold=1.5)
        with self.assertRaises(ValueError):
            CalibConfig(min_samples_for_alarm=0)

    def test_config_serialization_roundtrip(self):
        cfg = CalibConfig(
            num_bins=15,
            window_size=200,
            learning_rate=0.02,
            min_temperature=0.1,
            max_temperature=5.0,
            ece_threshold=0.05,
            min_samples_for_alarm=50,
            initial_temperature=1.2,
        )
        d = cfg.to_dict()
        cfg_loaded = CalibConfig.from_dict(d)
        self.assertEqual(cfg.num_bins, cfg_loaded.num_bins)
        self.assertEqual(cfg.window_size, cfg_loaded.window_size)
        self.assertAlmostEqual(cfg.learning_rate, cfg_loaded.learning_rate)
        self.assertAlmostEqual(cfg.min_temperature, cfg_loaded.min_temperature)
        self.assertAlmostEqual(cfg.max_temperature, cfg_loaded.max_temperature)
        self.assertAlmostEqual(cfg.ece_threshold, cfg_loaded.ece_threshold)
        self.assertEqual(cfg.min_samples_for_alarm, cfg_loaded.min_samples_for_alarm)
        self.assertAlmostEqual(cfg.initial_temperature, cfg_loaded.initial_temperature)


class TestMathHelpers(unittest.TestCase):
    """Tests for logit, sigmoid, and softmax functions."""

    def test_logit_sigmoid_roundtrip(self):
        for p in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
            z = _logit(p)
            p_rec = _sigmoid(z)
            self.assertAlmostEqual(p, p_rec, places=5)

    def test_boundary_clamping(self):
        # Should not crash on extreme probabilities 0.0 and 1.0
        z_zero = _logit(0.0)
        z_one = _logit(1.0)
        self.assertTrue(math.isfinite(z_zero))
        self.assertTrue(math.isfinite(z_one))
        self.assertGreater(z_one, z_zero)

        p_neg = _sigmoid(-1000.0)
        p_pos = _sigmoid(1000.0)
        self.assertAlmostEqual(p_neg, 0.0, places=5)
        self.assertAlmostEqual(p_pos, 1.0, places=5)

    def test_softmax(self):
        logits = [2.0, 1.0, 0.1]
        probs = _softmax(logits, temperature=1.0)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

        # High temperature produces near-uniform distribution
        probs_high_t = _softmax(logits, temperature=100.0)
        for p in probs_high_t:
            self.assertAlmostEqual(p, 1.0 / 3.0, places=2)


class TestOnlineProbabilityCalibrator(unittest.TestCase):
    """Unit tests for OnlineProbabilityCalibrator core behavior."""

    def setUp(self):
        self.config = CalibConfig(
            num_bins=10,
            window_size=50,
            learning_rate=0.05,
            min_temperature=0.1,
            max_temperature=5.0,
            ece_threshold=0.08,
            min_samples_for_alarm=15,
        )
        self.calibrator = OnlineProbabilityCalibrator(config=self.config)

    def test_initial_state(self):
        st = self.calibrator.status()
        self.assertAlmostEqual(st.temperature, 1.0)
        self.assertEqual(st.total_samples, 0)
        self.assertEqual(st.rolling_samples, 0)
        self.assertFalse(st.is_miscalibrated)
        self.assertAlmostEqual(st.ece, 0.0)
        self.assertAlmostEqual(st.brier_score, 0.0)

    def test_calibrate_probability_identity_at_temp_1(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            cal_p = self.calibrator.calibrate_probability(p)
            self.assertAlmostEqual(p, cal_p, places=5)

    def test_calibrate_probability_softens_at_high_temp(self):
        cal = OnlineProbabilityCalibrator(initial_temperature=2.0)
        # High confidence 0.9 should soften towards 0.5
        cal_p = cal.calibrate_probability(0.9)
        self.assertLess(cal_p, 0.9)
        self.assertGreater(cal_p, 0.5)

        # Low confidence 0.1 should soften towards 0.5
        cal_low = cal.calibrate_probability(0.1)
        self.assertGreater(cal_low, 0.1)
        self.assertLess(cal_low, 0.5)

    def test_calibrate_probability_sharpens_at_low_temp(self):
        cal = OnlineProbabilityCalibrator(initial_temperature=0.5)
        # 0.8 should sharpen towards 1.0
        cal_p = cal.calibrate_probability(0.8)
        self.assertGreater(cal_p, 0.8)

        # 0.2 should sharpen towards 0.0
        cal_low = cal.calibrate_probability(0.2)
        self.assertLess(cal_low, 0.2)

    def test_calibrate_distribution(self):
        dist = {"A": 0.7, "B": 0.2, "C": 0.1}
        # T=1.0 preserves relative ranking
        cal_dist = self.calibrator.calibrate_distribution(dist)
        self.assertAlmostEqual(sum(cal_dist.values()), 1.0, places=2)
        self.assertGreater(cal_dist["A"], cal_dist["B"])
        self.assertGreater(cal_dist["B"], cal_dist["C"])

        # Empty distribution returns empty dict
        self.assertEqual(self.calibrator.calibrate_distribution({}), {})

    def test_online_update_overconfident_increases_temperature(self):
        # Raw probability is 0.95 (overconfident) but ground truth is y=0
        initial_t = self.calibrator.temperature
        for _ in range(10):
            self.calibrator.update(raw_prob=0.95, true_label=0)

        # Temperature should have increased to soften predictions
        self.assertGreater(self.calibrator.temperature, initial_t)

    def test_online_update_underconfident_decreases_temperature(self):
        # Start at T=2.0, raw prob 0.65 but ground truth is consistently 1
        cal = OnlineProbabilityCalibrator(initial_temperature=2.0)
        for _ in range(15):
            cal.update(raw_prob=0.65, true_label=1)

        # Temperature should have decreased to sharpen predictions
        self.assertLess(cal.temperature, 2.0)

    def test_temperature_clamping(self):
        cal = OnlineProbabilityCalibrator(
            config=CalibConfig(min_temperature=0.2, max_temperature=3.0, learning_rate=1.0)
        )
        # Push with extreme errors
        for _ in range(50):
            cal.update(raw_prob=0.999, true_label=0)
        self.assertLessEqual(cal.temperature, 3.0)

        for _ in range(50):
            cal.update(raw_prob=0.51, true_label=1)
        self.assertGreaterEqual(cal.temperature, 0.2)

    def test_streaming_metrics_and_ece(self):
        # Feed 20 perfectly calibrated samples: 10 with prob 0.8 and y=1 (80% 1), 10 with prob 0.2 and y=0 (80% 0)
        for _ in range(8):
            self.calibrator.update(raw_prob=0.8, true_label=1)
        for _ in range(2):
            self.calibrator.update(raw_prob=0.8, true_label=0)
        for _ in range(8):
            self.calibrator.update(raw_prob=0.2, true_label=0)
        for _ in range(2):
            self.calibrator.update(raw_prob=0.2, true_label=1)

        ece, mce, brier, bins = self.calibrator.compute_metrics()
        self.assertGreater(ece, 0.0)
        self.assertGreater(brier, 0.0)
        self.assertEqual(len(bins), self.config.num_bins)

    def test_miscalibration_alarm(self):
        # Warmup threshold is 15 samples
        for _ in range(10):
            self.calibrator.update(raw_prob=0.95, true_label=0)  # Huge gap
        # Before warmup, alarm shouldn't fire
        self.assertFalse(self.calibrator.is_miscalibrated)

        # Add 10 more severe miscalibration samples
        for _ in range(10):
            self.calibrator.update(raw_prob=0.95, true_label=0)

        st = self.calibrator.status()
        self.assertTrue(st.is_miscalibrated)
        self.assertGreater(st.ece, self.config.ece_threshold)

    def test_reset(self):
        for _ in range(25):
            self.calibrator.update(raw_prob=0.9, true_label=0)
        self.assertGreater(self.calibrator.total_samples, 0)
        self.assertNotEqual(self.calibrator.temperature, 1.0)

        self.calibrator.reset(initial_temperature=1.0)
        st = self.calibrator.status()
        self.assertEqual(st.total_samples, 0)
        self.assertEqual(st.rolling_samples, 0)
        self.assertAlmostEqual(st.temperature, 1.0)
        self.assertFalse(st.is_miscalibrated)

    def test_ascii_reliability_diagram(self):
        for _ in range(20):
            self.calibrator.update(raw_prob=0.85, true_label=1)
        diagram = self.calibrator.ascii_reliability_diagram()
        self.assertIn("Bin", diagram)
        self.assertIn("Conf Range", diagram)
        self.assertIn("Avg Conf", diagram)
        self.assertIn("Avg Acc", diagram)


class TestCalibPersistence(unittest.TestCase):
    """Tests for .reflex-calib binary serialization and CRC32 integrity checks."""

    def test_save_load_roundtrip(self):
        config = CalibConfig(num_bins=8, window_size=40, learning_rate=0.03, ece_threshold=0.07)
        cal = OnlineProbabilityCalibrator(config=config, initial_temperature=1.4)
        for i in range(25):
            cal.update(raw_prob=0.75, true_label=1 if i % 2 == 0 else 0)

        with tempfile.NamedTemporaryFile(suffix=".reflex-calib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cal.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreaterEqual(os.path.getsize(tmp_path), CALIB_MIN_FILE_SIZE)

            loaded = OnlineProbabilityCalibrator.load(tmp_path)
            self.assertEqual(loaded.config.num_bins, 8)
            self.assertEqual(loaded.config.window_size, 40)
            self.assertAlmostEqual(loaded.temperature, cal.temperature, places=5)
            self.assertEqual(loaded.total_samples, cal.total_samples)

            st_orig = cal.status()
            st_loaded = loaded.status()
            self.assertAlmostEqual(st_orig.ece, st_loaded.ece, places=4)
            self.assertAlmostEqual(st_orig.brier_score, st_loaded.brier_score, places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_corrupted_crc32_raises_error(self):
        cal = OnlineProbabilityCalibrator()
        cal.update(raw_prob=0.8, true_label=1)

        with tempfile.NamedTemporaryFile(suffix=".reflex-calib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cal.save(tmp_path)
            # Tamper with file content
            with open(tmp_path, "r+b") as f:
                f.seek(15)
                byte = f.read(1)
                f.seek(15)
                f.write(bytes([(byte[0] ^ 0xFF)]))

            with self.assertRaises(ValueError) as ctx:
                OnlineProbabilityCalibrator.load(tmp_path)
            self.assertIn("CRC32 checksum mismatch", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_magic_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".reflex-calib", delete=False) as tmp:
            tmp.write(b"BADM" + b"\x00" * 100)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                OnlineProbabilityCalibrator.load(tmp_path)
            self.assertIn("Invalid magic header", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestReflexClientCalibIntegration(unittest.TestCase):
    """Tests Reflex client integration with calibrator."""

    def test_client_noul_and_choice_calibration(self):
        cal = OnlineProbabilityCalibrator(initial_temperature=2.0)
        rx = Reflex(calibrator=cal)

        # Mock evaluate
        res = rx.evaluate(
            "User wants to reset password",
            {
                "is_auth": Noul(instructions="Is this an auth request?", threshold=0.5),
                "action": Choice(instructions="Category", options=["auth", "billing", "other"]),
            },
        )
        self.assertIsNotNone(res.calibration)
        self.assertIn("temperature", res.calibration)
        self.assertAlmostEqual(res.calibration["temperature"], 2.0, places=2)
        # Noul probability is calibrated
        noul_res = res.decisions["is_auth"]
        self.assertIsNotNone(noul_res.probability)

    def test_miscalibration_triggers_escalation(self):
        cal = OnlineProbabilityCalibrator(
            config=CalibConfig(min_samples_for_alarm=5, ece_threshold=0.05)
        )
        # Feed severe miscalibration samples
        for _ in range(10):
            cal.update(raw_prob=0.95, true_label=0)

        self.assertTrue(cal.is_miscalibrated)

        rx = Reflex(calibrator=cal)
        res = rx.evaluate(
            "Test input",
            {"q": Noul(instructions="Test question?", threshold=0.5)},
        )
        self.assertTrue(res.should_escalate)
        self.assertTrue(res.calibration["is_miscalibrated"])

    def test_client_record_calibration_feedback(self):
        cal = OnlineProbabilityCalibrator(initial_temperature=1.0)
        rx = Reflex(calibrator=cal)

        initial_samples = cal.total_samples
        rx.record_calibration_feedback(raw_prob=0.9, true_label=False)
        self.assertEqual(cal.total_samples, initial_samples + 1)
        self.assertGreater(cal.temperature, 1.0)

        # Test feedback via general record_feedback with raw_prob
        rx.record_feedback(key="q", true_value=True, raw_prob=0.6)
        self.assertEqual(cal.total_samples, initial_samples + 2)


class TestCalibThreadSafety(unittest.TestCase):
    """Tests thread-safety of OnlineProbabilityCalibrator under concurrent updates."""

    def test_concurrent_updates_and_reads(self):
        cal = OnlineProbabilityCalibrator()
        errors = []

        def worker(w_id: int):
            try:
                for i in range(50):
                    p = 0.5 + 0.3 * math.sin(w_id + i)
                    y = 1 if p > 0.5 else 0
                    cal.update(raw_prob=p, true_label=y)
                    _ = cal.calibrate_probability(p)
                    _ = cal.status()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(cal.total_samples, 400)


if __name__ == "__main__":
    unittest.main()
