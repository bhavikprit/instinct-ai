"""
Unit tests for Reflex Speculative Decision Routing & Parallel Pre-Fetch (Phase 22).
Zero external dependencies (Python standard library only).
"""

import asyncio
import json
import socket
import time
import unittest
import urllib.request
import urllib.error

from reflex import Reflex, Choice
from reflex.speculative import (
    SpeculativeEngine,
    SpeculativeAction,
    SpeculativeSession,
    SpeculativeStatus,
    SpeculativeMetrics,
)
from reflex.gateway import ReflexGatewayServer, GatewayConfig


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestSpeculativeEngine(unittest.TestCase):
    """Unit tests covering speculative execution, pre-fetch, and safety guards."""

    def setUp(self):
        self.engine = SpeculativeEngine(confidence_threshold=0.70)

    def tearDown(self):
        self.engine.shutdown(wait=False)

    def test_speculative_hit_synchronous(self):
        """Pre-fetch runs in parallel; when LLM confirms predicted action, hit=True and 0ms wait."""
        def fetch_order_details(order_id: str = "ord_99"):
            time.sleep(0.05)  # Simulate 50ms database lookup
            return {"order_id": order_id, "status": "shipped", "total_usd": 129.50}

        self.engine.register_action(
            name="fetch_order_details",
            handler=fetch_order_details,
            description="Lookup order details by ID",
            idempotent=True,
            extractor=lambda s: {"order_id": "ord_99"},
        )

        session = self.engine.speculate(
            state="Where is my shipment for ord_99?",
            action_names=["fetch_order_details"],
            override_action="fetch_order_details",
        )

        self.assertTrue(session.speculated)
        self.assertEqual(session.predicted_action, "fetch_order_details")
        self.assertEqual(session.status, SpeculativeStatus.IN_FLIGHT)

        # Simulate parallel LLM reasoning (60ms)
        time.sleep(0.06)

        # LLM confirms the tool call
        result, is_hit = session.resolve("fetch_order_details")
        self.assertTrue(is_hit)
        self.assertEqual(result["status"], "shipped")
        self.assertEqual(session.status, SpeculativeStatus.HIT)

        stats = self.engine.stats()
        self.assertEqual(stats["speculative_hits"], 1)
        self.assertEqual(stats["speculative_misses"], 0)
        self.assertGreater(stats["total_latency_saved_ms"], 40.0)

    def test_speculative_miss_and_abort(self):
        """When LLM confirms a different tool, the speculative task is aborted and confirmed tool runs."""
        tool_a_executed = False
        tool_b_executed = False

        def tool_a():
            nonlocal tool_a_executed
            time.sleep(0.1)
            tool_a_executed = True
            return "Result A"

        def tool_b():
            nonlocal tool_b_executed
            tool_b_executed = True
            return "Result B"

        self.engine.register_action("tool_a", tool_a, idempotent=True)
        self.engine.register_action("tool_b", tool_b, idempotent=True)

        session = self.engine.speculate(
            state="Execute operation A",
            override_action="tool_a",
        )
        self.assertTrue(session.speculated)

        # LLM surprisingly chooses tool_b instead
        result, is_hit = session.resolve("tool_b")
        self.assertFalse(is_hit)
        self.assertEqual(result, "Result B")
        self.assertTrue(tool_b_executed)
        self.assertEqual(session.status, SpeculativeStatus.MISS)

        stats = self.engine.stats()
        self.assertEqual(stats["speculative_misses"], 1)
        self.assertEqual(stats["speculative_hits"], 0)

    def test_side_effect_safety_guard(self):
        """Non-idempotent actions (e.g. credit card charges) MUST NEVER be pre-fetched."""
        charge_called = False

        def charge_credit_card(amount: float = 100.0):
            nonlocal charge_called
            charge_called = True
            return {"charged": True, "amount": amount}

        self.engine.register_action(
            name="charge_credit_card",
            handler=charge_credit_card,
            idempotent=False,  # Mutation!
        )

        session = self.engine.speculate(
            state="Charge customer 100 dollars",
            override_action="charge_credit_card",
        )

        # Safety guard must prevent speculation
        self.assertFalse(session.speculated)
        self.assertEqual(session.status, SpeculativeStatus.SKIPPED)
        self.assertFalse(charge_called)

        # But on confirmed resolution, it executes safely
        res, is_hit = session.resolve("charge_credit_card", amount=50.0)
        self.assertFalse(is_hit)
        self.assertTrue(charge_called)
        self.assertEqual(res["amount"], 50.0)

    def test_confidence_threshold_gating(self):
        """If System-1 confidence is below threshold, speculation is skipped."""
        engine = SpeculativeEngine(confidence_threshold=0.80)
        engine.register_action("search_orders", lambda: "orders", idempotent=True)
        engine.register_action("search_users", lambda: "users", idempotent=True)

        # Ambiguous query where neither action meets 0.80 confidence
        session = engine.speculate("Maybe find some random info")
        self.assertFalse(session.speculated)
        self.assertEqual(session.status, SpeculativeStatus.SKIPPED)

        # Explicit override_confidence below threshold
        session2 = engine.speculate("search query", override_confidence=0.50)
        self.assertFalse(session2.speculated)
        self.assertEqual(session2.status, SpeculativeStatus.SKIPPED)
        engine.shutdown(wait=False)

    def test_asyncio_speculative_session(self):
        """Test async speculative resolution inside asyncio event loop."""
        async def async_fetch():
            await asyncio.sleep(0.04)
            return "async_data"

        self.engine.register_action("async_fetch", async_fetch, idempotent=True)

        async def run_test():
            session = self.engine.speculate("fetch data", override_action="async_fetch")
            self.assertTrue(session.speculated)
            await asyncio.sleep(0.05)
            res, is_hit = await session.resolve_async("async_fetch")
            self.assertTrue(is_hit)
            self.assertEqual(res, "async_data")

        asyncio.run(run_test())

    def test_context_manager_cleanup(self):
        """Exiting a context manager cleanly aborts abandoned speculative futures."""
        def slow_action():
            time.sleep(0.2)
            return "slow"

        self.engine.register_action("slow_action", slow_action, idempotent=True)

        with self.engine.speculate("state", override_action="slow_action") as spec:
            self.assertTrue(spec.speculated)
            self.assertIsNotNone(spec.future)

        # Exiting block without resolve marks session as ABORTED and sets abort_event
        self.assertEqual(spec.status, SpeculativeStatus.ABORTED)
        self.assertTrue(spec.abort_event.is_set())
        stats = self.engine.stats()
        self.assertEqual(stats["speculative_aborts"], 1)


class TestReflexClientSpeculativeIntegration(unittest.TestCase):
    """Tests speculative integration methods on Reflex client."""

    def test_client_speculate(self):
        rx = Reflex(backend="local")
        rx.register_speculative_action(
            name="kb_search",
            handler=lambda q="test": f"found: {q}",
            idempotent=True,
            extractor=lambda s: {"q": "refund policy"},
        )

        with rx.speculate("What is the refund policy?", actions=["kb_search"]) as spec:
            self.assertEqual(spec.predicted_action, "kb_search")
            time.sleep(0.02)
            res, hit = spec.resolve("kb_search")
            self.assertTrue(hit)
            self.assertEqual(res, "found: refund policy")

        stats = rx.speculative_stats()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["speculative_hits"], 1)


class TestGatewaySpeculativeEndpoints(unittest.TestCase):
    """Integration test for AI Envoy Gateway speculative endpoint."""

    def setUp(self):
        self.port = get_free_port()
        self.config = GatewayConfig(
            host="127.0.0.1",
            port=self.port,
            speculative_enabled=True,
            speculative_threshold=0.70,
        )
        self.server = ReflexGatewayServer(self.config)
        self.server.start(background=True)
        time.sleep(0.3)

    def tearDown(self):
        self.server.stop()
        time.sleep(0.2)

    def test_speculative_stats_endpoint(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/speculative/stats")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("hit_rate", data)
            self.assertIn("speculations_launched", data)


if __name__ == "__main__":
    unittest.main()
