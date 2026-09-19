"""
Tests for Cost-Aware Dual-Brain Cascades & Risk-Budgeted Routing (reflex.cascade).
Phase 40: Unit tests for multi-tier cascade optimization, sequential threshold solver,
risk upper bounds, fallback resilience, binary persistence (.reflex-cascade),
thread-safety, and Reflex runtime integration.
"""

from collections import deque
import math
import os
import random
import tempfile
import threading
import unittest

from reflex.cascade import (
    CASCADE_MAGIC,
    CASCADE_MIN_FILE_SIZE,
    CascadeConfig,
    CascadeDecision,
    CascadeFrontierPoint,
    CascadeRouter,
    CascadeTier,
)
from reflex.client import Reflex
from reflex.primitives import Choice, DecisionResult, Noul


class TestCascadeTierAndConfig(unittest.TestCase):
    """Tests for CascadeTier and CascadeConfig validation and serialization."""

    def test_valid_tiers_and_ordering(self):
        t0 = CascadeTier(name="tier0", cost_per_query=0.0, expected_latency_ms=0.1, tier_index=0)
        t1 = CascadeTier(name="tier1", cost_per_query=0.001, expected_latency_ms=50.0, tier_index=1)
        router = CascadeRouter(tiers=[t0, t1])
        self.assertEqual(router.num_tiers, 2)
        self.assertEqual(router.tiers[0].name, "tier0")
        self.assertEqual(router.tiers[1].name, "tier1")

    def test_tier_serialization_roundtrip(self):
        t = CascadeTier(name="slm", cost_per_query=0.0005, expected_latency_ms=25.0, tier_index=1, description="Fast SLM")
        d = t.to_dict()
        restored = CascadeTier.from_dict(d)
        self.assertEqual(restored.name, "slm")
        self.assertEqual(restored.cost_per_query, 0.0005)
        self.assertEqual(restored.expected_latency_ms, 25.0)
        self.assertEqual(restored.tier_index, 1)

    def test_invalid_tier_count(self):
        with self.assertRaises(ValueError):
            CascadeRouter(tiers=[CascadeTier(name="single")])

    def test_invalid_tier_cost_ordering(self):
        t0 = CascadeTier(name="expensive", cost_per_query=0.05, tier_index=0)
        t1 = CascadeTier(name="cheap", cost_per_query=0.01, tier_index=1)
        with self.assertRaises(ValueError):
            CascadeRouter(tiers=[t0, t1])

    def test_default_config(self):
        cfg = CascadeConfig()
        self.assertEqual(cfg.target_risk, 0.02)
        self.assertIsNone(cfg.target_quality)
        self.assertEqual(cfg.confidence_bound, 0.05)
        self.assertEqual(cfg.min_calibration_samples, 30)
        self.assertEqual(cfg.exploration_rate, 0.02)
        self.assertTrue(cfg.fallback_on_error)
        self.assertEqual(cfg.window_size, 1000)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            CascadeConfig(target_risk=-0.1)
        with self.assertRaises(ValueError):
            CascadeConfig(target_risk=1.2)
        with self.assertRaises(ValueError):
            CascadeConfig(target_quality=0.0)
        with self.assertRaises(ValueError):
            CascadeConfig(confidence_bound=0.0)
        with self.assertRaises(ValueError):
            CascadeConfig(min_calibration_samples=2)
        with self.assertRaises(ValueError):
            CascadeConfig(exploration_rate=1.5)
        with self.assertRaises(ValueError):
            CascadeConfig(min_calibration_samples=50, window_size=20)

    def test_config_serialization_roundtrip(self):
        cfg = CascadeConfig(
            target_risk=0.03,
            target_quality=None,
            confidence_bound=0.01,
            min_calibration_samples=40,
            exploration_rate=0.05,
            fallback_on_error=False,
            window_size=500,
        )
        d = cfg.to_dict()
        restored = CascadeConfig.from_dict(d)
        self.assertEqual(restored.target_risk, 0.03)
        self.assertIsNone(restored.target_quality)
        self.assertEqual(restored.confidence_bound, 0.01)
        self.assertEqual(restored.min_calibration_samples, 40)
        self.assertEqual(restored.exploration_rate, 0.05)
        self.assertFalse(restored.fallback_on_error)
        self.assertEqual(restored.window_size, 500)


class TestCascadeDecisionAndFrontier(unittest.TestCase):
    """Tests for CascadeDecision and CascadeFrontierPoint data structures."""

    def test_decision_to_dict_and_subscript(self):
        dec = CascadeDecision(
            selected_tier="system1_instinct",
            tier_index=0,
            score=0.95,
            threshold=0.88,
            cumulative_cost=0.0,
            cumulative_latency_ms=0.05,
            cost_savings_pct=100.0,
            guaranteed_risk=0.018,
            trace=[{"tier": "system1_instinct", "action": "ACCEPT"}],
            is_exploratory=False,
            reason="Accepted at Tier 0",
        )
        self.assertEqual(dec["selected_tier"], "system1_instinct")
        self.assertEqual(dec["tier_index"], 0)
        self.assertEqual(dec["cost_savings_pct"], 100.0)
        self.assertIn("selected_tier", dec)
        self.assertIn("cost_savings_pct", dec)
        self.assertNotIn("non_existent", dec)

        d = dec.to_dict()
        self.assertEqual(d["selected_tier"], "system1_instinct")
        self.assertEqual(d["cumulative_cost"], 0.0)

    def test_frontier_point_to_dict(self):
        pt = CascadeFrontierPoint(
            thresholds=[0.85, 0.75],
            expected_cost=0.0045,
            empirical_risk=0.015,
            upper_bound_risk=0.022,
            coverage_per_tier={"tier0": 0.6, "tier1": 0.3, "tier2": 0.1},
            cost_reduction_pct=85.0,
        )
        d = pt.to_dict()
        self.assertEqual(len(d["thresholds"]), 2)
        self.assertEqual(d["expected_cost"], 0.0045)
        self.assertEqual(d["cost_reduction_pct"], 85.0)


class TestCascadeRouterCalibration(unittest.TestCase):
    """Tests for CascadeRouter threshold calibration and risk guarantees."""

    def test_initial_uncalibrated_state(self):
        router = CascadeRouter()
        self.assertFalse(router.is_calibrated)
        self.assertEqual(router.num_calibration_samples, 0)
        self.assertEqual(router.total_samples, 0)
        self.assertEqual(len(router.thresholds), 2)  # default 3 tiers -> 2 thresholds

    def test_fit_and_calibration_2_tier(self):
        t0 = CascadeTier(name="fast", cost_per_query=0.0001, expected_latency_ms=1.0, tier_index=0)
        t1 = CascadeTier(name="slow", cost_per_query=0.0100, expected_latency_ms=500.0, tier_index=1)
        router = CascadeRouter(tiers=[t0, t1], config=CascadeConfig(target_risk=0.05, min_calibration_samples=30))

        # 50 samples: high confidence in tier 0 -> 0 errors; low confidence -> errors
        samples = []
        for i in range(50):
            if i < 30:
                samples.append(({0: 0.95, 1: 0.99}, {0: 0, 1: 0}))
            else:
                samples.append(({0: 0.40, 1: 0.99}, {0: 1, 1: 0}))

        router.fit(samples)
        self.assertTrue(router.is_calibrated)
        self.assertEqual(router.num_calibration_samples, 50)
        self.assertGreaterEqual(router.thresholds[0], 0.80)
        self.assertLessEqual(router.calibrated_upper_risk, 0.10)
        self.assertLess(router.calibrated_cost, 0.0100)

    def test_calibration_3_tier_risk_control(self):
        router = CascadeRouter(config=CascadeConfig(target_risk=0.08, min_calibration_samples=40))
        rng = random.Random(42)

        for _ in range(80):
            diff = rng.random()
            s0 = max(0.1, 1.0 - 0.8 * diff)
            e0 = 1 if s0 < 0.7 else 0

            s1 = max(0.2, 1.0 - 0.4 * diff)
            e1 = 1 if s1 < 0.5 else 0

            s2 = 0.99
            e2 = 0

            router.add_sample({0: s0, 1: s1, 2: s2}, {0: e0, 1: e1, 2: e2})

        th = router.calibrate()
        self.assertTrue(router.is_calibrated)
        self.assertEqual(len(th), 2)
        self.assertLessEqual(router.calibrated_upper_risk, 0.12)
        self.assertLess(router.calibrated_cost, router.tiers[-1].cost_per_query)

    def test_rolling_window_eviction(self):
        router = CascadeRouter(config=CascadeConfig(min_calibration_samples=10, window_size=25))
        for _ in range(40):
            router.add_sample({0: 0.9, 1: 0.95}, {0: 0, 1: 0})
        self.assertEqual(router.num_calibration_samples, 25)
        self.assertEqual(router.total_samples, 40)


class TestCascadeRoutingExecution(unittest.TestCase):
    """Tests for sequential query routing execution, costs, and resilience."""

    def setUp(self):
        self.tiers = [
            CascadeTier(name="tier0", cost_per_query=0.0, expected_latency_ms=0.1, tier_index=0),
            CascadeTier(name="tier1", cost_per_query=0.001, expected_latency_ms=50.0, tier_index=1),
            CascadeTier(name="tier2", cost_per_query=0.030, expected_latency_ms=1000.0, tier_index=2),
        ]
        self.router = CascadeRouter(
            tiers=self.tiers,
            config=CascadeConfig(min_calibration_samples=20, exploration_rate=0.0),
        )
        for _ in range(25):
            self.router.add_sample({0: 0.9, 1: 0.95, 2: 0.99}, {0: 0, 1: 0, 2: 0})
        self.router.calibrate()
        self.router._thresholds = [0.85, 0.80]

    def test_route_tier0_accepted(self):
        # Query where Tier 0 has high confidence (0.92 >= 0.85)
        dec = self.router.route(
            "simple query",
            score_provider=lambda t_idx, q: 0.92 if t_idx == 0 else 0.50,
        )
        self.assertEqual(dec.selected_tier, "tier0")
        self.assertEqual(dec.tier_index, 0)
        self.assertEqual(dec.cumulative_cost, 0.0)
        self.assertEqual(dec.cumulative_latency_ms, 0.1)
        self.assertEqual(dec.cost_savings_pct, 100.0)
        self.assertEqual(len(dec.trace), 1)
        self.assertEqual(dec.trace[0]["action"], "ACCEPT")

    def test_route_tier1_accepted(self):
        # Query where Tier 0 fails threshold (0.70 < 0.85), but Tier 1 passes (0.88 >= 0.80)
        dec = self.router.route(
            "mid-tier query",
            score_provider=lambda t_idx, q: 0.70 if t_idx == 0 else 0.88,
        )
        self.assertEqual(dec.selected_tier, "tier1")
        self.assertEqual(dec.tier_index, 1)
        self.assertAlmostEqual(dec.cumulative_cost, 0.001, places=5)
        self.assertEqual(dec.cumulative_latency_ms, 50.1)
        self.assertGreater(dec.cost_savings_pct, 90.0)
        self.assertEqual(len(dec.trace), 2)
        self.assertEqual(dec.trace[0]["action"], "ESCALATE")
        self.assertEqual(dec.trace[1]["action"], "ACCEPT")

    def test_route_terminal_fallback(self):
        # Query where Tier 0 (0.50 < 0.85) and Tier 1 (0.60 < 0.80) both fail
        dec = self.router.route(
            "complex query",
            score_provider=lambda t_idx, q: 0.50 if t_idx == 0 else 0.60,
        )
        self.assertEqual(dec.selected_tier, "tier2")
        self.assertEqual(dec.tier_index, 2)
        self.assertAlmostEqual(dec.cumulative_cost, 0.031, places=5)
        self.assertEqual(dec.cumulative_latency_ms, 1050.1)
        self.assertEqual(dec.cost_savings_pct, 0.0)
        self.assertEqual(len(dec.trace), 3)
        self.assertEqual(dec.trace[2]["action"], "TERMINAL_FALLBACK")

    def test_route_fallback_on_error(self):
        # Tier 0 has error/exception, router should safely escalate to Tier 1
        dec = self.router.route(
            "query with broken tier0",
            score_provider=lambda t_idx, q: (_ for _ in ()).throw(RuntimeError("API error")) if t_idx == 0 else 0.90,
        )
        self.assertEqual(dec.selected_tier, "tier1")
        self.assertEqual(dec.trace[0]["action"], "FALLBACK_EXCEPTION")
        self.assertEqual(dec.trace[1]["action"], "ACCEPT")

    def test_route_fallback_with_failed_tier_list(self):
        # Tier 0 is marked in operational fallback_errors
        dec = self.router.route(
            "query with tier0 down",
            score_provider=lambda t_idx, q: 0.95,
            fallback_errors=[0],
        )
        self.assertEqual(dec.selected_tier, "tier1")
        self.assertEqual(dec.trace[0]["action"], "FALLBACK_ERROR")

    def test_contextual_exploration(self):
        # Set exploration rate to 100%
        self.router.config.exploration_rate = 1.0
        dec = self.router.route(
            "exploration query",
            score_provider=lambda t_idx, q: 0.99,
        )
        self.assertTrue(dec.is_exploratory)
        self.assertEqual(dec.selected_tier, "tier2")


class TestParetoFrontierAndVisualization(unittest.TestCase):
    """Tests for Pareto Cost-Risk frontier computation and ASCII rendering."""

    def test_cost_risk_frontier(self):
        router = CascadeRouter(config=CascadeConfig(min_calibration_samples=10))
        for i in range(40):
            diff = i / 40.0
            router.add_sample(
                {0: 1.0 - 0.7 * diff, 1: 1.0 - 0.3 * diff, 2: 0.99},
                {0: 1 if diff > 0.6 else 0, 1: 1 if diff > 0.8 else 0, 2: 0},
            )
        frontier = router.cost_risk_frontier(num_points=8)
        self.assertGreaterEqual(len(frontier), 1)
        for pt in frontier:
            self.assertGreaterEqual(pt.expected_cost, 0.0)
            self.assertLessEqual(pt.empirical_risk, 1.0)
            self.assertLessEqual(pt.upper_bound_risk, 1.0)

    def test_ascii_cost_risk_frontier(self):
        router = CascadeRouter(config=CascadeConfig(min_calibration_samples=10))
        # Empty curve
        self.assertIn("No calibration data", router.ascii_cost_risk_frontier())

        for i in range(30):
            diff = i / 30.0
            router.add_sample(
                {0: 1.0 - 0.7 * diff, 1: 1.0 - 0.3 * diff, 2: 0.99},
                {0: 1 if diff > 0.5 else 0, 1: 0, 2: 0},
            )
        ascii_text = router.ascii_cost_risk_frontier(num_points=5)
        self.assertIn("Avg Cost", ascii_text)
        self.assertIn("Cost Saved", ascii_text)
        self.assertIn("Emp Risk", ascii_text)
        self.assertIn("|", ascii_text)


class TestBinaryPersistence(unittest.TestCase):
    """Tests for .reflex-cascade binary persistence, CRC32 integrity, and validation."""

    def test_save_and_load_roundtrip(self):
        cfg = CascadeConfig(target_risk=0.03, confidence_bound=0.01, min_calibration_samples=20)
        tiers = [
            CascadeTier(name="t0", cost_per_query=0.0, expected_latency_ms=0.05, tier_index=0),
            CascadeTier(name="t1", cost_per_query=0.002, expected_latency_ms=30.0, tier_index=1),
            CascadeTier(name="t2", cost_per_query=0.025, expected_latency_ms=800.0, tier_index=2),
        ]
        router = CascadeRouter(tiers=tiers, config=cfg)
        for i in range(35):
            router.add_sample({0: 0.85, 1: 0.92, 2: 0.99}, {0: 0, 1: 0, 2: 0})
        router.calibrate()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.reflex-cascade")
            router.save(file_path)
            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "rb") as f:
                raw = f.read()
            self.assertEqual(raw[:4], CASCADE_MAGIC)
            self.assertGreaterEqual(len(raw), CASCADE_MIN_FILE_SIZE)

            loaded = CascadeRouter.load(file_path)
            self.assertEqual(loaded.num_tiers, 3)
            self.assertEqual(loaded.config.target_risk, 0.03)
            self.assertEqual(loaded.num_calibration_samples, 35)
            self.assertEqual(len(loaded.thresholds), 2)
            self.assertAlmostEqual(loaded.calibrated_cost, router.calibrated_cost, places=5)
            self.assertAlmostEqual(loaded.calibrated_empirical_risk, router.calibrated_empirical_risk, places=4)

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            CascadeRouter.load("/path/to/nonexistent/model.reflex-cascade")

    def test_load_truncated_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "short.cascade")
            with open(path, "wb") as f:
                f.write(b"RFCSshort")
            with self.assertRaises(ValueError) as ctx:
                CascadeRouter.load(path)
            self.assertIn("below minimum", str(ctx.exception))

    def test_load_corrupted_magic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad_magic.cascade")
            with open(path, "wb") as f:
                f.write(b"XXXX" + b"\x00" * 100)
            with self.assertRaises(ValueError) as ctx:
                CascadeRouter.load(path)
            self.assertIn("Invalid magic", str(ctx.exception))

    def test_load_corrupted_checksum(self):
        router = CascadeRouter(config=CascadeConfig(min_calibration_samples=10))
        for _ in range(15):
            router.add_sample({0: 0.9, 1: 0.95, 2: 0.99}, {0: 0, 1: 0, 2: 0})
        router.calibrate()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrupt.cascade")
            router.save(path)

            with open(path, "rb") as f:
                data = bytearray(f.read())
            data[60] ^= 0xFF  # Corrupt payload byte
            with open(path, "wb") as f:
                f.write(data)

            with self.assertRaises(ValueError) as ctx:
                CascadeRouter.load(path)
            self.assertIn("checksum mismatch", str(ctx.exception))


class TestReflexClientIntegration(unittest.TestCase):
    """Tests for Reflex runtime client integration with CascadeRouter."""

    def test_client_initialization_with_cascade_obj(self):
        router = CascadeRouter()
        client = Reflex(cascade=router)
        self.assertIs(client.cascade, router)

    def test_client_initialization_with_cascade_bool(self):
        client = Reflex(cascade=True)
        self.assertIsNotNone(client.cascade)
        self.assertIsInstance(client.cascade, CascadeRouter)

    def test_client_evaluate_populates_cascade(self):
        router = CascadeRouter(config=CascadeConfig(min_calibration_samples=20))
        for _ in range(25):
            router.add_sample({0: 0.95, 1: 0.98, 2: 0.99}, {0: 0, 1: 0, 2: 0})
        router.calibrate()
        router._thresholds = [0.80, 0.75]

        client = Reflex(cascade=router)
        res = client.evaluate(
            "User payment verification",
            {"is_fraud": Noul(instructions="Is fraudulent?", threshold=0.5, probability=0.92)},
        )
        self.assertIsNotNone(res.cascade)
        self.assertIn("selected_tier", res.cascade)
        self.assertIn("cost_savings_pct", res.cascade)
        self.assertFalse(res.should_escalate)

    def test_client_record_cascade_feedback(self):
        router = CascadeRouter(config=CascadeConfig(min_calibration_samples=10))
        client = Reflex(cascade=router)
        self.assertEqual(router.num_calibration_samples, 0)

        client.record_cascade_feedback({0: 0.88, 1: 0.95, 2: 0.99}, {0: 0, 1: 0, 2: 0})
        self.assertEqual(router.num_calibration_samples, 1)


class TestThreadSafety(unittest.TestCase):
    """Tests for thread safety under concurrent updates and routing evaluations."""

    def test_concurrent_add_and_route(self):
        router = CascadeRouter(config=CascadeConfig(min_calibration_samples=20))
        for _ in range(25):
            router.add_sample({0: 0.85, 1: 0.92, 2: 0.99}, {0: 0, 1: 0, 2: 0})
        router.calibrate()

        errors = []

        def worker_writer():
            try:
                for i in range(50):
                    s0 = 0.7 + 0.005 * (i % 40)
                    router.add_sample({0: s0, 1: 0.90, 2: 0.99}, {0: i % 10 == 0, 1: 0, 2: 0})
            except Exception as e:
                errors.append(e)

        def worker_router():
            try:
                for _ in range(50):
                    dec = router.route(
                        "concurrent query",
                        score_provider=lambda t_idx, q: 0.88 if t_idx == 0 else 0.95,
                    )
                    self.assertIsInstance(dec.selected_tier, str)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker_writer),
            threading.Thread(target=worker_router),
            threading.Thread(target=worker_writer),
            threading.Thread(target=worker_router),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
