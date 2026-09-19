"""
Unit tests for Phase 35: Adaptive Conformal Inference (ACI) & Online Distribution Shift Adaptation.
Tests online Gibbs & Candes controller updates, asymmetric penalties, drift alarms,
binary serialization with CRC32 tamper detection, and Reflex client integration.
"""

from collections import deque
import os
import random
import tempfile
import threading
import unittest

from reflex import (
    Reflex,
    Noul,
    Choice,
    ConformalConfig,
    ConformalPredictor,
    ACIConfig,
    AdaptiveConformalTracker,
    ACIStatus,
)


class TestAdaptiveConformalInference(unittest.TestCase):

    def test_aci_config_validation(self):
        """Validates configuration parameter ranges and error handling."""
        cfg = ACIConfig(target_alpha=0.05, gamma=0.02)
        self.assertAlmostEqual(cfg.target_alpha, 0.05)
        self.assertAlmostEqual(cfg.gamma, 0.02)

        # Invalid target_alpha <= 0 or >= 1
        with self.assertRaises(ValueError):
            ACIConfig(target_alpha=0.0)
        with self.assertRaises(ValueError):
            ACIConfig(target_alpha=1.0)

        # Invalid gamma <= 0
        with self.assertRaises(ValueError):
            ACIConfig(gamma=0.0)
        with self.assertRaises(ValueError):
            ACIConfig(gamma=-0.01)

        # alpha_min >= alpha_max
        with self.assertRaises(ValueError):
            ACIConfig(alpha_min=0.5, alpha_max=0.5)
        with self.assertRaises(ValueError):
            ACIConfig(alpha_min=0.8, alpha_max=0.2)

        # window_size <= 0
        with self.assertRaises(ValueError):
            ACIConfig(window_size=0)

        # gamma_down_multiplier <= 0
        with self.assertRaises(ValueError):
            ACIConfig(gamma_down_multiplier=0.0)

    def test_aci_config_serialization(self):
        """Verifies dictionary serialization and deserialization roundtrip."""
        cfg1 = ACIConfig(
            target_alpha=0.15,
            gamma=0.03,
            alpha_min=0.005,
            alpha_max=0.95,
            window_size=150,
            drift_threshold=0.10,
            min_samples_for_drift=40,
            gamma_down_multiplier=2.5,
        )
        d = cfg1.to_dict()
        cfg2 = ACIConfig.from_dict(d)
        self.assertEqual(cfg1.target_alpha, cfg2.target_alpha)
        self.assertEqual(cfg1.gamma, cfg2.gamma)
        self.assertEqual(cfg1.alpha_min, cfg2.alpha_min)
        self.assertEqual(cfg1.alpha_max, cfg2.alpha_max)
        self.assertEqual(cfg1.window_size, cfg2.window_size)
        self.assertEqual(cfg1.drift_threshold, cfg2.drift_threshold)
        self.assertEqual(cfg1.min_samples_for_drift, cfg2.min_samples_for_drift)
        self.assertEqual(cfg1.gamma_down_multiplier, cfg2.gamma_down_multiplier)

    def test_controller_initialization(self):
        """Verifies initial alpha state and default clamping."""
        tracker = AdaptiveConformalTracker(ACIConfig(target_alpha=0.10))
        self.assertAlmostEqual(tracker.current_alpha, 0.10)
        self.assertEqual(tracker.total_steps, 0)
        self.assertEqual(tracker.total_errors, 0)

        # Explicit initial alpha
        tracker2 = AdaptiveConformalTracker(ACIConfig(target_alpha=0.10), initial_alpha=0.05)
        self.assertAlmostEqual(tracker2.current_alpha, 0.05)

        # Initial alpha outside bounds gets clamped
        tracker3 = AdaptiveConformalTracker(ACIConfig(alpha_min=0.01, alpha_max=0.90), initial_alpha=0.99)
        self.assertAlmostEqual(tracker3.current_alpha, 0.90)

    def test_online_update_rule_covered(self):
        """When covered (err=0), alpha increases by gamma * target_alpha."""
        cfg = ACIConfig(target_alpha=0.10, gamma=0.01)
        tracker = AdaptiveConformalTracker(config=cfg)
        initial_alpha = tracker.current_alpha

        new_alpha = tracker.update(is_covered=True)
        expected_step = 0.01 * 0.10  # +0.001
        self.assertAlmostEqual(new_alpha, initial_alpha + expected_step)
        self.assertAlmostEqual(tracker.current_alpha, initial_alpha + expected_step)
        self.assertEqual(tracker.total_steps, 1)
        self.assertEqual(tracker.total_errors, 0)

    def test_online_update_rule_miscovered(self):
        """When miscovered (err=1), alpha decreases by gamma * (1 - target_alpha)."""
        cfg = ACIConfig(target_alpha=0.10, gamma=0.01)
        tracker = AdaptiveConformalTracker(config=cfg)
        initial_alpha = tracker.current_alpha

        new_alpha = tracker.update(is_covered=False)
        expected_step = 0.01 * (0.10 - 1.0)  # -0.009
        self.assertAlmostEqual(new_alpha, initial_alpha + expected_step)
        self.assertAlmostEqual(tracker.current_alpha, initial_alpha + expected_step)
        self.assertEqual(tracker.total_steps, 1)
        self.assertEqual(tracker.total_errors, 1)

    def test_stationary_expectation_equilibrium(self):
        """
        Under stationary i.i.d. stream with error probability = target_alpha,
        the expected drift E[alpha_{t+1} - alpha_t] = 0.
        """
        cfg = ACIConfig(target_alpha=0.10, gamma=0.01)
        tracker = AdaptiveConformalTracker(config=cfg)

        # 9 covered steps followed by 1 miscovered step -> net step = 9*(+0.001) + 1*(-0.009) = 0
        start_a = tracker.current_alpha
        for _ in range(9):
            tracker.update(is_covered=True)
        tracker.update(is_covered=False)

        self.assertAlmostEqual(tracker.current_alpha, start_a, places=7)
        self.assertEqual(tracker.total_steps, 10)
        self.assertEqual(tracker.total_errors, 1)

    def test_clamping_bounds(self):
        """Alpha must stay clamped strictly between alpha_min and alpha_max."""
        cfg = ACIConfig(target_alpha=0.10, gamma=0.05, alpha_min=0.02, alpha_max=0.50)
        tracker = AdaptiveConformalTracker(config=cfg)

        # Drive alpha down by consecutive miscoverages
        for _ in range(50):
            tracker.update(is_covered=False)
        self.assertGreaterEqual(tracker.current_alpha, 0.02)
        self.assertAlmostEqual(tracker.current_alpha, 0.02)

        # Drive alpha up by consecutive coverages
        for _ in range(100):
            tracker.update(is_covered=True)
        self.assertLessEqual(tracker.current_alpha, 0.50)
        self.assertAlmostEqual(tracker.current_alpha, 0.50)

    def test_asymmetric_penalty_multiplier(self):
        """gamma_down_multiplier > 1 accelerates alpha reduction on errors."""
        cfg_normal = ACIConfig(target_alpha=0.10, gamma=0.01, gamma_down_multiplier=1.0)
        cfg_asym = ACIConfig(target_alpha=0.10, gamma=0.01, gamma_down_multiplier=3.0)

        tracker_norm = AdaptiveConformalTracker(config=cfg_normal)
        tracker_asym = AdaptiveConformalTracker(config=cfg_asym)

        tracker_norm.update(is_covered=False)
        tracker_asym.update(is_covered=False)

        # Asymmetric tracker should reduce alpha 3x faster on miscoverage
        step_norm = 0.10 - tracker_norm.current_alpha
        step_asym = 0.10 - tracker_asym.current_alpha
        self.assertAlmostEqual(step_asym, step_norm * 3.0)

    def test_rolling_window_coverage(self):
        """Empirical coverage tracks the rolling window of recent steps."""
        cfg = ACIConfig(window_size=10)
        tracker = AdaptiveConformalTracker(config=cfg)

        # 8 covered, 2 miscovered
        for _ in range(8):
            tracker.update(is_covered=True)
        for _ in range(2):
            tracker.update(is_covered=False)

        self.assertAlmostEqual(tracker.empirical_coverage, 0.80)

        # Add 10 consecutive coverages to push older errors out of window
        for _ in range(10):
            tracker.update(is_covered=True)

        self.assertAlmostEqual(tracker.empirical_coverage, 1.0)
        self.assertEqual(tracker.total_steps, 20)
        self.assertEqual(tracker.total_errors, 2)
        self.assertAlmostEqual(tracker.cumulative_coverage, 18.0 / 20.0)

    def test_drift_score_and_alarm(self):
        """is_drifting triggers when rolling coverage drops below target - threshold."""
        cfg = ACIConfig(
            target_alpha=0.10,     # Target coverage = 0.90
            window_size=50,
            drift_threshold=0.08,  # Triggers if rolling cov < 0.82
            min_samples_for_drift=20,
        )
        tracker = AdaptiveConformalTracker(config=cfg)

        # Before min_samples_for_drift, is_drifting should remain False
        for _ in range(15):
            tracker.update(is_covered=False)
        self.assertFalse(tracker.is_drifting)

        # Reach min_samples (20 steps) with 100% errors
        for _ in range(5):
            tracker.update(is_covered=False)
        self.assertEqual(tracker.total_steps, 20)
        self.assertTrue(tracker.is_drifting)
        self.assertGreater(tracker.drift_score, 0.08)

    def test_drift_recovery(self):
        """Drift alarm clears once stream recovers to target coverage."""
        cfg = ACIConfig(
            target_alpha=0.10,
            window_size=30,
            drift_threshold=0.08,
            min_samples_for_drift=20,
        )
        tracker = AdaptiveConformalTracker(config=cfg)

        # Induce drift
        for _ in range(25):
            tracker.update(is_covered=False)
        self.assertTrue(tracker.is_drifting)

        # Recover stream with 30 consecutive coverages
        for _ in range(30):
            tracker.update(is_covered=True)
        self.assertFalse(tracker.is_drifting)
        self.assertAlmostEqual(tracker.empirical_coverage, 1.0)

    def test_reset(self):
        """reset() restores initial alpha and clears history."""
        tracker = AdaptiveConformalTracker(ACIConfig(target_alpha=0.10))
        for _ in range(15):
            tracker.update(is_covered=False)
        self.assertNotEqual(tracker.current_alpha, 0.10)
        self.assertGreater(tracker.total_steps, 0)

        tracker.reset(initial_alpha=0.15)
        self.assertAlmostEqual(tracker.current_alpha, 0.15)
        self.assertEqual(tracker.total_steps, 0)
        self.assertEqual(tracker.total_errors, 0)
        self.assertEqual(len(tracker._history), 0)

    def test_status_snapshot(self):
        """status() returns complete and accurate telemetry data."""
        tracker = AdaptiveConformalTracker(ACIConfig(target_alpha=0.10))
        tracker.update(is_covered=True)
        tracker.update(is_covered=False)

        st = tracker.status()
        self.assertIsInstance(st, ACIStatus)
        self.assertAlmostEqual(st.target_alpha, 0.10)
        self.assertAlmostEqual(st.target_coverage, 0.90)
        self.assertEqual(st.total_steps, 2)
        self.assertEqual(st.total_errors, 1)

        d = st.to_dict()
        self.assertIn("current_alpha", d)
        self.assertIn("nominal_coverage", d)
        self.assertIn("empirical_coverage", d)
        self.assertIn("is_drifting", d)

    def test_update_with_result(self):
        """update_with_result tests membership in prediction set."""
        tracker = AdaptiveConformalTracker(ACIConfig(target_alpha=0.10))

        # Covered: "fraud" in {"fraud", "review"}
        new_a1 = tracker.update_with_result("fraud", ["fraud", "review"])
        self.assertEqual(tracker.total_errors, 0)

        # Miscovered: "chargeback" not in {"fraud", "review"}
        new_a2 = tracker.update_with_result("chargeback", ["fraud", "review"])
        self.assertEqual(tracker.total_errors, 1)
        self.assertLess(new_a2, new_a1)

    def test_binary_serialization_roundtrip(self):
        """save() and load() faithfully persist and restore .reflex-aci state."""
        cfg = ACIConfig(target_alpha=0.08, gamma=0.015, window_size=50)
        tracker = AdaptiveConformalTracker(config=cfg)

        for i in range(40):
            tracker.update(is_covered=(i % 3 != 0))

        with tempfile.NamedTemporaryFile(suffix=".reflex-aci", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tracker.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 52)

            loaded = AdaptiveConformalTracker.load(tmp_path)
            self.assertAlmostEqual(loaded.target_alpha, tracker.target_alpha)
            self.assertAlmostEqual(loaded.current_alpha, tracker.current_alpha)
            self.assertAlmostEqual(loaded.config.gamma, tracker.config.gamma)
            self.assertEqual(loaded.total_steps, tracker.total_steps)
            self.assertEqual(loaded.total_errors, tracker.total_errors)
            self.assertAlmostEqual(loaded.empirical_coverage, tracker.empirical_coverage)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_tamper_detection_crc32(self):
        """Modifying a single byte triggers CRC32 validation failure."""
        tracker = AdaptiveConformalTracker()
        tracker.update(True)

        with tempfile.NamedTemporaryFile(suffix=".reflex-aci", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tracker.save(tmp_path)
            with open(tmp_path, "r+b") as f:
                data = bytearray(f.read())
                # Corrupt one byte in the header
                data[10] = (data[10] + 1) % 256
                f.seek(0)
                f.write(data)

            with self.assertRaises(ValueError) as ctx:
                AdaptiveConformalTracker.load(tmp_path)
            self.assertIn("CRC32", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_magic_rejection(self):
        """Files with invalid magic header are rejected immediately."""
        with tempfile.NamedTemporaryFile(suffix=".reflex-aci", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"FAIL" + b"\x00" * 60)

        try:
            with self.assertRaises(ValueError) as ctx:
                AdaptiveConformalTracker.load(tmp_path)
            self.assertIn("magic", str(ctx.exception).lower())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_client_integration_evaluate(self):
        """Reflex client integrates ACI dynamic alpha and attaches telemetry to DecisionResult."""
        cp = ConformalPredictor(ConformalConfig(alpha=0.10))
        # Seed calibration data
        for _ in range(50):
            cp.add_calibration_noul(0.92, True)
            cp.add_calibration_noul(0.08, False)
        cp.calibrate()

        tracker = AdaptiveConformalTracker(ACIConfig(target_alpha=0.10))
        rx = Reflex(backend="local", conformal=cp, aci=tracker)

        res = rx.evaluate("Transaction amount $500", {"is_fraud": Noul("Is fraud?", threshold=0.5)})
        self.assertIsNotNone(res.conformal)
        self.assertIn("is_fraud", res.conformal)
        self.assertIsNotNone(res.aci)
        self.assertIn("current_alpha", res.aci)
        self.assertFalse(res.should_escalate)

        # Trigger drift alarm on tracker and verify should_escalate triggers
        for _ in range(35):
            tracker.update(is_covered=False)
        self.assertTrue(tracker.is_drifting)

        res_drift = rx.evaluate("Transaction amount $500", {"is_fraud": Noul("Is fraud?", threshold=0.5)})
        self.assertTrue(res_drift.should_escalate)
        self.assertTrue(res_drift.aci["is_drifting"])

    def test_client_record_feedback(self):
        """Reflex.record_feedback updates online ACI tracker."""
        tracker = AdaptiveConformalTracker(ACIConfig(target_alpha=0.10))
        rx = Reflex(backend="local", aci=tracker)

        self.assertEqual(tracker.total_steps, 0)
        rx.record_feedback("auth_key", True)
        self.assertEqual(tracker.total_steps, 1)
        self.assertEqual(tracker.total_errors, 0)

        rx.record_feedback("auth_key", False)
        self.assertEqual(tracker.total_steps, 2)
        self.assertEqual(tracker.total_errors, 1)

        # With prediction set check
        rx.record_feedback("category", "fraud", prediction_set=["fraud", "spam"])
        self.assertEqual(tracker.total_steps, 3)
        self.assertEqual(tracker.total_errors, 1)

        rx.record_feedback("category", "normal", prediction_set=["fraud", "spam"])
        self.assertEqual(tracker.total_steps, 4)
        self.assertEqual(tracker.total_errors, 2)

    def test_thread_safety(self):
        """Concurrent updates across threads maintain correct totals without race conditions."""
        tracker = AdaptiveConformalTracker(ACIConfig(window_size=500))
        num_threads = 8
        updates_per_thread = 50

        def worker():
            for i in range(updates_per_thread):
                tracker.update(is_covered=(i % 2 == 0))

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_total = num_threads * updates_per_thread
        expected_errors = num_threads * (updates_per_thread // 2)
        self.assertEqual(tracker.total_steps, expected_total)
        self.assertEqual(tracker.total_errors, expected_errors)


if __name__ == "__main__":
    unittest.main()
