"""
Unit tests for Reflex Autonomous Canary Deployment & Decision Shadowing (Phase 21).
Zero external dependencies (Python standard library only).
"""

import json
import socket
import time
import unittest
import urllib.request
import urllib.error

from reflex import Reflex, Noul, Choice, Score
from reflex.shadow import (
    ShadowStage,
    ShadowConfig,
    ShadowEvaluationRecord,
    DivergenceTracker,
    DecisionShadowRouter,
)
from reflex.gateway import ReflexGatewayServer, GatewayConfig


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestDivergenceTracker(unittest.TestCase):
    """Unit tests for DivergenceTracker mathematical accuracy and edge cases."""

    def test_empty_tracker(self):
        tracker = DivergenceTracker()
        self.assertEqual(tracker.concordance_rate(), 1.0)
        self.assertEqual(tracker.cohen_kappa(), 1.0)
        self.assertEqual(tracker.mean_confidence_delta(), 0.0)
        profile = tracker.latency_profile()
        self.assertEqual(profile["champion"]["p50"], 0.0)

    def test_cohen_kappa_perfect_agreement(self):
        tracker = DivergenceTracker()
        for i in range(20):
            cat = "billing" if i % 2 == 0 else "support"
            record = ShadowEvaluationRecord(
                timestamp=time.time(),
                state=f"Sample {i}",
                question_keys=["category"],
                champion_decisions={"category": cat},
                challenger_decisions={"category": cat},
                champion_confidences={"category": 0.9},
                challenger_confidences={"category": 0.95},
                concordant=True,
                champion_latency_ms=0.5,
                challenger_latency_ms=0.6,
                served_by="champion",
            )
            tracker.record(record)

        self.assertEqual(tracker.concordance_rate(), 1.0)
        self.assertEqual(tracker.cohen_kappa("category"), 1.0)
        self.assertAlmostEqual(tracker.mean_confidence_delta(), 0.05, places=3)

    def test_cohen_kappa_partial_agreement_analytical(self):
        """
        Analytical verification of Cohen's Kappa:
        Table of 50 samples:
                  Challenger: True  Challenger: False
        Champ True:         20              5           Total: 25
        Champ False:        10             15           Total: 25
        Total:              30             20           N = 50

        p_o = (20 + 15) / 50 = 0.70
        p_e = (25*30 + 25*20) / (50^2) = (750 + 500) / 2500 = 0.50
        kappa = (0.70 - 0.50) / (1 - 0.50) = 0.20 / 0.50 = 0.40
        """
        tracker = DivergenceTracker()
        
        def add_pairs(champ_val: bool, chal_val: bool, count: int):
            for _ in range(count):
                rec = ShadowEvaluationRecord(
                    timestamp=time.time(),
                    state="test",
                    question_keys=["is_urgent"],
                    champion_decisions={"is_urgent": champ_val},
                    challenger_decisions={"is_urgent": chal_val},
                    champion_confidences={"is_urgent": 0.8},
                    challenger_confidences={"is_urgent": 0.8},
                    concordant=(champ_val == chal_val),
                    champion_latency_ms=0.2,
                    challenger_latency_ms=0.2,
                    served_by="champion",
                )
                tracker.record(rec)

        add_pairs(True, True, 20)
        add_pairs(True, False, 5)
        add_pairs(False, True, 10)
        add_pairs(False, False, 15)

        self.assertEqual(tracker.total_samples, 50)
        self.assertEqual(tracker.concordance_rate(), 0.70)
        self.assertAlmostEqual(tracker.cohen_kappa("is_urgent"), 0.40, places=2)

    def test_latency_percentiles(self):
        tracker = DivergenceTracker()
        for i in range(100):
            rec = ShadowEvaluationRecord(
                timestamp=time.time(),
                state=f"test {i}",
                question_keys=["q"],
                champion_decisions={"q": "A"},
                challenger_decisions={"q": "A"},
                champion_confidences={"q": 1.0},
                challenger_confidences={"q": 1.0},
                concordant=True,
                champion_latency_ms=float(i),
                challenger_latency_ms=float(i * 2),
                served_by="champion",
            )
            tracker.record(rec)

        prof = tracker.latency_profile()
        self.assertAlmostEqual(prof["champion"]["p50"], 50.0, delta=1.0)
        self.assertAlmostEqual(prof["challenger"]["p50"], 100.0, delta=2.0)


class TestDecisionShadowRouter(unittest.TestCase):
    """Unit tests for asynchronous shadowing and canary lifecycle."""

    def test_asynchronous_non_blocking_shadowing(self):
        rx_champ = Reflex(backend="local")
        rx_chal = Reflex(backend="semantic")
        cfg = ShadowConfig(
            shadow_traffic_pct=100.0,
            canary_traffic_pct=0.0,
            auto_promote=False,
        )
        router = DecisionShadowRouter(champion=rx_champ, challenger=rx_chal, config=cfg)

        res = router.evaluate("User wants to cancel subscription", {
            "cat": Choice("Intent", options=["billing", "support"])
        })
        self.assertIsNotNone(res)
        self.assertIn("cat", res.decisions)

        # Flush pending shadow tasks
        router.flush(timeout=3.0)
        stats = router.stats()
        self.assertEqual(stats["total_shadowed_samples"], 1)
        self.assertGreaterEqual(stats["concordance_rate"], 0.0)
        router.shutdown(wait=False)

    def test_canary_stage_transitions(self):
        router = DecisionShadowRouter(
            champion=Reflex(backend="local"),
            challenger=Reflex(backend="semantic"),
            config=ShadowConfig(stage=ShadowStage.OBSERVATION),
        )
        self.assertEqual(router.config.canary_traffic_pct, 0.0)

        router.set_stage(ShadowStage.CANARY_10)
        self.assertEqual(router.config.canary_traffic_pct, 10.0)

        router.set_stage(ShadowStage.CANARY_50)
        self.assertEqual(router.config.canary_traffic_pct, 50.0)

        router.promote()
        self.assertEqual(router.config.stage, ShadowStage.PROMOTED)
        self.assertEqual(router.config.canary_traffic_pct, 100.0)

        router.rollback("Regression test")
        self.assertEqual(router.config.stage, ShadowStage.ROLLED_BACK)
        self.assertEqual(router.config.canary_traffic_pct, 0.0)
        router.shutdown(wait=False)

    def test_autonomous_auto_promotion(self):
        # Configure low threshold for fast test execution
        cfg = ShadowConfig(
            stage=ShadowStage.OBSERVATION,
            concordance_threshold=0.90,
            min_kappa=0.50,
            min_samples_for_promotion=5,
            auto_promote=True,
        )
        # Use two identical local engines so decisions match 100%
        router = DecisionShadowRouter(
            champion=Reflex(backend="local"),
            challenger=Reflex(backend="local"),
            config=cfg,
        )

        for i in range(7):
            router.evaluate(f"Refund request transaction {i}", {
                "is_refund": Noul("Is this a refund request?")
            })

        router.flush(timeout=3.0)
        # Should have stepped up from OBSERVATION to CANARY_10
        self.assertIn(router.config.stage, (ShadowStage.CANARY_10, ShadowStage.CANARY_25, ShadowStage.CANARY_50, ShadowStage.PROMOTED))
        self.assertGreater(router.config.canary_traffic_pct, 0.0)
        router.shutdown(wait=False)

    def test_autonomous_auto_rollback(self):
        cfg = ShadowConfig(
            stage=ShadowStage.CANARY_50,
            canary_traffic_pct=50.0,
            rollback_threshold=0.85,
            min_samples_for_rollback=4,
            auto_rollback=True,
        )

        # Create challenger that always disagrees
        class InvertingChallenger:
            def evaluate(self, state, questions):
                from reflex.primitives import DecisionResult, Noul
                return DecisionResult(
                    decisions={"q": Noul("q").resolve(0.01)},
                    latency_ms=0.1,
                    backend="inverter",
                )

        class TrueChampion:
            def evaluate(self, state, questions):
                from reflex.primitives import DecisionResult, Noul
                return DecisionResult(
                    decisions={"q": Noul("q").resolve(0.99)},
                    latency_ms=0.1,
                    backend="true_champ",
                )

        router = DecisionShadowRouter(
            champion=TrueChampion(),
            challenger=InvertingChallenger(),
            config=cfg,
        )

        for i in range(5):
            router.evaluate(f"State {i}", {"q": Noul("q")})

        router.flush(timeout=3.0)
        self.assertEqual(router.config.stage, ShadowStage.ROLLED_BACK)
        self.assertEqual(router.config.canary_traffic_pct, 0.0)
        stats = router.stats()
        self.assertTrue(any(inc["type"] == "rollback" for inc in stats["recent_incidents"]))
        router.shutdown(wait=False)


class TestReflexClientShadowIntegration(unittest.TestCase):
    """Unit tests verifying Reflex client integration with shadowing."""

    def test_client_with_challenger(self):
        rx = Reflex(backend="local", shadow_challenger="semantic")
        self.assertIsNotNone(rx.shadow_router)

        res = rx.evaluate("Database latency spike above 500ms", {
            "urgent": Noul("Is this incident urgent?"),
            "team": Choice("Assign team", options=["sre", "billing", "support"]),
        })
        self.assertIn("urgent", res.decisions)
        self.assertIn("team", res.decisions)

        rx.shadow_router.flush(timeout=3.0)
        stats = rx.canary_stats()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["total_shadowed_samples"], 1)
        rx.shadow_router.shutdown(wait=False)


class TestGatewayCanaryEndpoints(unittest.TestCase):
    """Integration tests for AI Envoy Gateway canary REST endpoints."""

    def setUp(self):
        self.port = get_free_port()
        self.config = GatewayConfig(
            host="127.0.0.1",
            port=self.port,
            canary_enabled=True,
            canary_traffic_pct=10.0,
            canary_challenger_backend="semantic",
        )
        self.server = ReflexGatewayServer(self.config)
        self.server.start(background=True)
        time.sleep(0.3)

    def tearDown(self):
        self.server.stop()
        time.sleep(0.2)

    def test_canary_stats_endpoint(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/canary/stats")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["stage"], "CANARY_10")
            self.assertEqual(data["canary_traffic_pct"], 10.0)

    def test_canary_stage_and_promote_endpoint(self):
        # Update stage to CANARY_50
        body = json.dumps({"stage": "CANARY_50"}).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/canary/stage", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["stats"]["stage"], "CANARY_50")
            self.assertEqual(data["stats"]["canary_traffic_pct"], 50.0)

        # Promote challenger
        req_promote = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/canary/promote", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_promote, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "promoted")
            self.assertEqual(data["stats"]["stage"], "PROMOTED")

        # Rollback challenger
        req_rb = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/canary/rollback", data=b"{\"reason\": \"Test\"}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_rb, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "rolled_back")
            self.assertEqual(data["stats"]["stage"], "ROLLED_BACK")

    def test_chat_completions_with_canary_headers(self):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Classify this email triage: credit card fraud detected"}
            ]
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("X-Reflex-Shadowed"), "TRUE")
            self.assertIn(resp.headers.get("X-Reflex-Canary-Stage"), ("CANARY_10", "OBSERVATION", "PROMOTED"))


if __name__ == "__main__":
    unittest.main()
