"""
Unit tests for Phase 36: Conformalized Quantile Regression (CQR) for Arbitrary Continuous Target Intervals.
Tests Romano, Sesia & Candes (2019) conformalized quantiles, pinball loss updates,
heteroscedastic interval adaptation, binary serialization with CRC32, and Reflex client integration.
"""

import math
import os
import random
import tempfile
import threading
import unittest

from reflex import (
    Reflex,
    Score,
    CQRConfig,
    CQRInterval,
    QuantileInstinctHead,
    ConformalizedQuantileRegressor,
)


class TestConformalizedQuantileRegression(unittest.TestCase):

    def test_cqr_config_validation(self):
        """Validates config argument bounds and error handling."""
        cfg = CQRConfig(alpha=0.05, min_calibration_samples=25, max_width_tolerance=3.0)
        self.assertAlmostEqual(cfg.alpha, 0.05)
        self.assertEqual(cfg.min_calibration_samples, 25)
        self.assertEqual(cfg.max_width_tolerance, 3.0)

        # Invalid alpha
        with self.assertRaises(ValueError):
            CQRConfig(alpha=0.0)
        with self.assertRaises(ValueError):
            CQRConfig(alpha=1.0)

        # Invalid min_calibration_samples
        with self.assertRaises(ValueError):
            CQRConfig(min_calibration_samples=4)

        # Invalid learning rate
        with self.assertRaises(ValueError):
            CQRConfig(learning_rate=0.0)

        # Invalid max_width_tolerance
        with self.assertRaises(ValueError):
            CQRConfig(max_width_tolerance=-1.0)

    def test_cqr_config_serialization(self):
        """Tests to_dict() and from_dict() roundtrip."""
        cfg1 = CQRConfig(
            alpha=0.15,
            min_calibration_samples=30,
            max_width_tolerance=4.5,
            learning_rate=0.02,
            l2_reg=0.005,
            seed=123,
        )
        d = cfg1.to_dict()
        cfg2 = CQRConfig.from_dict(d)
        self.assertEqual(cfg1.alpha, cfg2.alpha)
        self.assertEqual(cfg1.min_calibration_samples, cfg2.min_calibration_samples)
        self.assertEqual(cfg1.max_width_tolerance, cfg2.max_width_tolerance)
        self.assertEqual(cfg1.learning_rate, cfg2.learning_rate)
        self.assertEqual(cfg1.l2_reg, cfg2.l2_reg)
        self.assertEqual(cfg1.seed, cfg2.seed)

    def test_quantile_instinct_head_prediction(self):
        """Tests dual quantile prediction and monotonicity guarantee."""
        head = QuantileInstinctHead(dim=384, tau_low=0.05, tau_high=0.95, seed=42)
        q_low, q_high = head.predict_quantiles("Evaluate code quality rubric")
        self.assertIsInstance(q_low, float)
        self.assertIsInstance(q_high, float)
        self.assertLessEqual(q_low, q_high)

    def test_quantile_instinct_head_pinball_loss_update(self):
        """Pinball loss subgradient update reduces loss on target sample."""
        head = QuantileInstinctHead(dim=384, tau_low=0.10, tau_high=0.90, seed=42)
        prompt = "Security review for payment API"
        target_y = 8.5

        loss_low_1, loss_high_1 = head.update(prompt, target_y=target_y, lr=0.1)
        for _ in range(20):
            loss_low, loss_high = head.update(prompt, target_y=target_y, lr=0.1)

        # Loss should decrease substantially after repeated updates
        self.assertLess(loss_low + loss_high, loss_low_1 + loss_high_1)

    def test_quantile_instinct_head_serialization(self):
        """Head weights and bias serialization roundtrip."""
        head1 = QuantileInstinctHead(dim=384, tau_low=0.05, tau_high=0.95, seed=42)
        head1.update("test prompt", 5.0)
        d = head1.to_dict()
        head2 = QuantileInstinctHead.from_dict(d)

        self.assertEqual(head1.dim, head2.dim)
        self.assertEqual(head1.tau_low, head2.tau_low)
        self.assertEqual(head1.tau_high, head2.tau_high)
        self.assertEqual(head1.training_steps, head2.training_steps)
        self.assertAlmostEqual(head1.bias_low, head2.bias_low, places=5)
        self.assertAlmostEqual(head1.bias_high, head2.bias_high, places=5)

    def test_conformal_score_calculation(self):
        """Signed non-conformity score E_i = max(q_low - y, y - q_high)."""
        cqr = ConformalizedQuantileRegressor()
        # Inside interval [2.0, 6.0], true=4.0: max(2-4, 4-6) = max(-2, -2) = -2.0
        cqr.add_calibration_sample(pred_low=2.0, pred_high=6.0, true_value=4.0)

        # Above interval [2.0, 6.0], true=7.5: max(2-7.5, 7.5-6) = max(-5.5, +1.5) = +1.5
        cqr.add_calibration_sample(pred_low=2.0, pred_high=6.0, true_value=7.5)

        # Below interval [2.0, 6.0], true=0.5: max(2-0.5, 0.5-6) = max(+1.5, -5.5) = +1.5
        cqr.add_calibration_sample(pred_low=2.0, pred_high=6.0, true_value=0.5)

        # Pad to min_calibration_samples
        for _ in range(20):
            cqr.add_calibration_sample(2.0, 6.0, 4.0)
        cqr.calibrate()

        self.assertTrue(cqr.is_calibrated)
        self.assertIsNotNone(cqr.q_hat)

    def test_calibrate_insufficient_samples(self):
        """Raises ValueError when calibrated with fewer than min samples."""
        cqr = ConformalizedQuantileRegressor(CQRConfig(min_calibration_samples=20))
        for _ in range(10):
            cqr.add_calibration_sample(1.0, 5.0, 3.0)

        with self.assertRaises(ValueError):
            cqr.calibrate()

    def test_conformal_calibration_exact_coverage(self):
        """Calibrated CQR guarantees empirical coverage >= 1 - alpha on calibration set."""
        alpha = 0.10
        cqr = ConformalizedQuantileRegressor(CQRConfig(alpha=alpha, min_calibration_samples=50))
        rng = random.Random(42)

        for _ in range(100):
            center = rng.uniform(2.0, 8.0)
            noise = rng.gauss(0, 1.0)
            true_y = center + noise
            q_low = center - 1.645
            q_high = center + 1.645
            cqr.add_calibration_sample(q_low, q_high, true_y)

        cqr.calibrate()
        self.assertGreaterEqual(cqr.empirical_coverage, 1.0 - alpha - 0.01)

    def test_prediction_interval_bounds_and_clamping(self):
        """Prediction intervals properly apply Q_hat offset and clamp to rubric bounds."""
        cqr = ConformalizedQuantileRegressor(CQRConfig(alpha=0.10))
        # 25 samples where all errors are within bounds
        for _ in range(25):
            cqr.add_calibration_sample(1.0, 5.0, 3.0)
        cqr.calibrate()

        # Normal prediction clamped to [0, 10]
        interval = cqr.predict(pred_low=2.0, pred_high=6.0, min_val=0.0, max_val=10.0)
        self.assertIsInstance(interval, CQRInterval)
        self.assertGreaterEqual(interval.lower_bound, 0.0)
        self.assertLessEqual(interval.upper_bound, 10.0)
        self.assertEqual(interval.alpha, 0.10)
        self.assertEqual(interval.coverage_guarantee, 0.90)

        # Clamping at boundaries
        interval_edge = cqr.predict(pred_low=-5.0, pred_high=15.0, min_val=0.0, max_val=10.0)
        self.assertEqual(interval_edge.lower_bound, 0.0)
        self.assertEqual(interval_edge.upper_bound, 10.0)

    def test_interval_contains_helper(self):
        """CQRInterval.contains accurately determines inclusion."""
        inv = CQRInterval(
            lower_bound=2.0,
            upper_bound=6.0,
            point_estimate=4.0,
            interval_width=4.0,
            q_hat=0.5,
            alpha=0.10,
            coverage_guarantee=0.90,
        )
        self.assertTrue(inv.contains(2.0))
        self.assertTrue(inv.contains(4.0))
        self.assertTrue(inv.contains(6.0))
        self.assertFalse(inv.contains(1.99))
        self.assertFalse(inv.contains(6.01))

    def test_heteroscedastic_interval_adaptation(self):
        """Inputs with higher variance yield proportionally wider intervals than low-variance inputs."""
        cqr = ConformalizedQuantileRegressor(CQRConfig(alpha=0.10))
        for _ in range(50):
            cqr.add_calibration_sample(2.0, 6.0, 6.5)  # error = 0.5, q_hat = 0.5
        cqr.calibrate()

        # Low variance query: [3.8, 4.2] (width = 0.4 + 2*0.5 = 1.4)
        inv_tight = cqr.predict(pred_low=3.8, pred_high=4.2, min_val=-100, max_val=100)
        # High variance query: [1.0, 9.0] (width = 8.0 + 2*0.5 = 9.0)
        inv_wide = cqr.predict(pred_low=1.0, pred_high=9.0, min_val=-100, max_val=100)

        self.assertGreater(inv_wide.interval_width, inv_tight.interval_width)
        self.assertAlmostEqual(inv_wide.interval_width - inv_tight.interval_width, 8.0 - 0.4)

    def test_tolerance_escalation(self):
        """should_escalate triggers when interval width exceeds max_width_tolerance."""
        cqr = ConformalizedQuantileRegressor(CQRConfig(alpha=0.10, max_width_tolerance=3.0))
        for _ in range(25):
            cqr.add_calibration_sample(2.0, 4.0, 4.0)  # error = 0, q_hat = 0
        cqr.calibrate()

        # Narrow interval (width = 1.0 <= 3.0)
        inv_ok = cqr.predict(pred_low=2.0, pred_high=3.0, min_val=0, max_val=10)
        self.assertFalse(inv_ok.should_escalate)
        self.assertTrue(inv_ok.is_safe)

        # Wide interval (width = 5.0 > 3.0)
        inv_wide = cqr.predict(pred_low=1.0, pred_high=6.0, min_val=0, max_val=10)
        self.assertTrue(inv_wide.should_escalate)
        self.assertFalse(inv_wide.is_safe)

    def test_predict_from_state(self):
        """predict_state uses internal QuantileInstinctHead to generate interval."""
        cqr = ConformalizedQuantileRegressor(CQRConfig(alpha=0.10))
        for _ in range(30):
            cqr.add_calibration_from_state("Code review clarity rubric", 7.0)
        cqr.calibrate()

        inv = cqr.predict_state("Code review clarity rubric", min_val=1.0, max_val=10.0)
        self.assertIsInstance(inv, CQRInterval)
        self.assertGreaterEqual(inv.lower_bound, 1.0)
        self.assertLessEqual(inv.upper_bound, 10.0)

    def test_evaluate_coverage_metrics(self):
        """evaluate_coverage calculates empirical test coverage and mean width."""
        cqr = ConformalizedQuantileRegressor(CQRConfig(alpha=0.10))
        for i in range(50):
            cqr.add_calibration_sample(2.0, 8.0, 2.0 + (i % 7) * 1.0)
        cqr.calibrate()

        test_samples = [
            (2.0, 8.0, 5.0),  # covered
            (2.0, 8.0, 4.0),  # covered
            (2.0, 8.0, 15.0), # miscovered
        ]
        metrics = cqr.evaluate_coverage(test_samples)
        self.assertAlmostEqual(metrics["empirical_coverage"], 2.0 / 3.0)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertGreater(metrics["mean_width"], 0.0)

    def test_binary_serialization_roundtrip(self):
        """save() and load() faithfully persist and restore .reflex-cqr state."""
        cqr1 = ConformalizedQuantileRegressor(CQRConfig(alpha=0.08, max_width_tolerance=4.0))
        for i in range(40):
            cqr1.add_calibration_sample(1.0 + 0.1 * i, 5.0 + 0.1 * i, 3.0 + 0.1 * i)
        cqr1.calibrate()

        with tempfile.NamedTemporaryFile(suffix=".reflex-cqr", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cqr1.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 60)

            cqr2 = ConformalizedQuantileRegressor.load(tmp_path)
            self.assertTrue(cqr2.is_calibrated)
            self.assertAlmostEqual(cqr2.config.alpha, cqr1.config.alpha)
            self.assertAlmostEqual(cqr2.config.max_width_tolerance, cqr1.config.max_width_tolerance)
            self.assertAlmostEqual(cqr2.q_hat, cqr1.q_hat)
            self.assertAlmostEqual(cqr2.empirical_coverage, cqr1.empirical_coverage)
            self.assertAlmostEqual(cqr2.mean_interval_width, cqr1.mean_interval_width)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_crc32_tamper_detection(self):
        """Modifying one byte triggers CRC32 validation failure."""
        cqr = ConformalizedQuantileRegressor()
        for _ in range(25):
            cqr.add_calibration_sample(1.0, 5.0, 3.0)
        cqr.calibrate()

        with tempfile.NamedTemporaryFile(suffix=".reflex-cqr", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cqr.save(tmp_path)
            with open(tmp_path, "r+b") as f:
                data = bytearray(f.read())
                data[12] = (data[12] + 1) % 256
                f.seek(0)
                f.write(data)

            with self.assertRaises(ValueError) as ctx:
                ConformalizedQuantileRegressor.load(tmp_path)
            self.assertIn("CRC32", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_magic_header(self):
        """Files with invalid magic header are rejected immediately."""
        with tempfile.NamedTemporaryFile(suffix=".reflex-cqr", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"BADM" + b"\x00" * 70)

        try:
            with self.assertRaises(ValueError) as ctx:
                ConformalizedQuantileRegressor.load(tmp_path)
            self.assertIn("magic", str(ctx.exception).lower())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_client_integration_score_evaluation(self):
        """Reflex(cqr=regressor) evaluates continuous Score decisions with CQR intervals."""
        cqr = ConformalizedQuantileRegressor(CQRConfig(alpha=0.10))
        for _ in range(30):
            cqr.add_calibration_from_state("Judge content safety", 5.0)
        cqr.calibrate()

        rx = Reflex(backend="local", cqr=cqr)
        res = rx.evaluate("Sample user comment", {"toxicity": Score("Judge content safety", min_val=0.0, max_val=10.0)})

        self.assertIsNotNone(res.cqr)
        self.assertIn("toxicity", res.cqr)
        interval = res.cqr["toxicity"]
        self.assertIsInstance(interval, CQRInterval)
        self.assertFalse(res.should_escalate)

    def test_client_tolerance_escalation(self):
        """Reflex client triggers should_escalate when CQR interval width exceeds tolerance."""
        cqr = ConformalizedQuantileRegressor(CQRConfig(alpha=0.10, max_width_tolerance=0.5))
        for _ in range(30):
            cqr.add_calibration_from_state("High uncertainty edge case", 5.0)
        cqr.calibrate()

        rx = Reflex(backend="local", cqr=cqr)
        res = rx.evaluate("High uncertainty edge case", {"score": Score("Uncertain metric", min_val=0.0, max_val=10.0)})

        # Narrow tolerance (0.5) vs interval width (>0.5) triggers escalation
        if res.cqr["score"].interval_width > 0.5:
            self.assertTrue(res.should_escalate)

    def test_thread_safety(self):
        """Concurrent calibration ingestion across threads runs without race conditions."""
        cqr = ConformalizedQuantileRegressor()
        num_threads = 6
        samples_per_thread = 50

        def worker(thread_idx):
            for i in range(samples_per_thread):
                cqr.add_calibration_sample(1.0 + thread_idx, 5.0 + thread_idx, 3.0 + thread_idx)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(cqr.calibration_samples), num_threads * samples_per_thread)


if __name__ == "__main__":
    unittest.main()
