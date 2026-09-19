"""
Unit tests for Reflex Phase 29: Continuous Autonomous Distillation & Self-Synthesizing Model Factory.
Zero external dependencies (Python standard library only).
"""

import json
import os
import shutil
import tempfile
import time
import unittest

from reflex.distill import (
    DistillationTrace,
    DistillationBuffer,
    MinedCluster,
    ClusterMiner,
    IntentSynthesizer,
    DistillationResult,
    AutonomousDistiller,
    DistillationWorker,
)
from reflex.compiler import CompiledInstinct
from reflex.shadow import DecisionShadowRouter, ShadowConfig, ShadowStage
from reflex.primitives import Choice, Noul


class TestDistillationTrace(unittest.TestCase):
    def test_trace_serialization(self):
        trace = DistillationTrace(
            trace_id="test_123",
            prompt="How do I reset my password?",
            response="Click Forgot Password on the login screen.",
            model="gpt-4o",
            latency_ms=450.2,
            timestamp=1700000000.0,
            label="auth_security",
            metadata={"user_tier": "enterprise"},
        )
        d = trace.to_dict()
        self.assertEqual(d["trace_id"], "test_123")
        self.assertEqual(d["label"], "auth_security")
        self.assertEqual(d["latency_ms"], 450.2)

        restored = DistillationTrace.from_dict(d)
        self.assertEqual(restored.trace_id, trace.trace_id)
        self.assertEqual(restored.prompt, trace.prompt)
        self.assertEqual(restored.model, trace.model)


class TestDistillationBuffer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "distill.jsonl")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_buffer_recording_and_capacity(self):
        buf = DistillationBuffer(max_size=5, storage_path=self.log_path)
        for i in range(10):
            buf.record(
                prompt=f"Query number {i}",
                response=f"Response number {i}",
                model="gpt-4o",
                latency_ms=100.0 + i,
            )
        self.assertEqual(buf.size(), 5)
        st = buf.stats()
        self.assertEqual(st["total_recorded"], 10)
        self.assertEqual(st["current_size"], 5)

        traces = buf.get_traces()
        self.assertEqual(len(traces), 5)
        self.assertEqual(traces[-1].prompt, "Query number 9")

    def test_pii_sanitization_in_buffer(self):
        buf = DistillationBuffer(max_size=10, redact_pii=True)
        # 16-digit card number and email
        trace = buf.record(
            prompt="My card is 4532 0151 1234 5678 and email is test@domain.com",
            response="Received credentials token sk-abcdef1234567890abcdef",
        )
        self.assertNotIn("4532 0151 1234 5678", trace.prompt)
        self.assertIn("[REDACTED_CARD]", trace.prompt)
        self.assertNotIn("test@domain.com", trace.prompt)
        self.assertIn("[REDACTED_EMAIL]", trace.prompt)
        self.assertNotIn("sk-abcdef1234567890abcdef", trace.response)
        self.assertIn("[REDACTED_SECRET]", trace.response)
        self.assertGreaterEqual(buf.stats()["pii_redacted_count"], 1)

    def test_buffer_persistence_reload(self):
        buf1 = DistillationBuffer(max_size=50, storage_path=self.log_path)
        buf1.record("Help with my refund", "Refund initiated", model="claude-3-5")
        buf1.record("Server error 500", "Checking server logs", model="gpt-4o")

        # Reload from storage
        buf2 = DistillationBuffer(max_size=50, storage_path=self.log_path)
        self.assertEqual(buf2.size(), 2)
        traces = buf2.get_traces()
        self.assertEqual(traces[0].prompt, "Help with my refund")
        self.assertEqual(traces[1].prompt, "Server error 500")


class TestClusterMiner(unittest.TestCase):
    def setUp(self):
        self.miner = ClusterMiner()

    def test_mine_clusters_with_distinct_intents(self):
        prompts = [
            # Billing intent
            "I need a refund for my subscription charge",
            "Why was my credit card billed twice?",
            "Can I get an invoice receipt for payment?",
            "Cancel subscription and issue refund",
            # Auth / Security intent
            "How do I reset my password and login?",
            "Unable to log into my account authentication error",
            "Forgot password reset link not working",
            "Need two factor authentication reset",
            # Outage / Tech intent
            "Production server is down with error 500",
            "API service returning 503 gateway unavailable",
            "System outage in database cluster",
            "Servers crashed during peak traffic",
        ]
        clusters = self.miner.mine_clusters(prompts, k=3, min_cluster_size=2)
        self.assertGreaterEqual(len(clusters), 2)
        total_assigned = sum(c.size for c in clusters)
        self.assertEqual(total_assigned, len(prompts))

        for c in clusters:
            self.assertGreaterEqual(len(c.exemplars), 1)
            self.assertGreater(c.coherence, 0.0)
            self.assertTrue(len(c.label) > 0)

    def test_edge_case_insufficient_prompts(self):
        self.assertEqual(self.miner.mine_clusters([]), [])
        self.assertEqual(self.miner.mine_clusters(["Single prompt"]), [])


class TestIntentSynthesizer(unittest.TestCase):
    def test_synthesizer_spec_generation(self):
        clusters = [
            MinedCluster(
                cluster_id=0,
                label="refund_billing",
                centroid=[0.1] * 384,
                size=3,
                coherence=0.85,
                exemplars=["I need a refund", "Subscription billing dispute"],
                all_prompts=["I need a refund", "Subscription billing dispute", "Cancel charge"],
            ),
            MinedCluster(
                cluster_id=1,
                label="password_reset",
                centroid=[-0.1] * 384,
                size=3,
                coherence=0.92,
                exemplars=["Forgot password", "Login link expired"],
                all_prompts=["Forgot password", "Login link expired", "Cannot login"],
            ),
        ]
        synth = IntentSynthesizer()
        spec = synth.build_spec(clusters, name="test_distilled")
        self.assertEqual(spec.name, "test_distilled")
        self.assertEqual(spec.options, ["refund_billing", "password_reset"])
        self.assertEqual(len(spec.few_shot_examples), 6)
        self.assertIn("refund_billing", spec.guidelines)


class TestAutonomousDistiller(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_distilled.reflex")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_autonomous_distillation_pipeline(self):
        buf = DistillationBuffer(max_size=50)
        # Feed sample queries
        billing_queries = [
            "Please refund my charge",
            "Why was I billed twice this month?",
            "I want to dispute an invoice charge",
            "Cancel my subscription renewal fee",
            "Payment failed on my receipt",
        ]
        auth_queries = [
            "I forgot my password to login",
            "Password reset link is broken",
            "Cannot log in authentication error",
            "Need to reset credentials",
            "Two factor login token failed",
        ]
        for q in billing_queries:
            buf.record(prompt=q, response="Billing department handled")
        for q in auth_queries:
            buf.record(prompt=q, response="Security team handled")

        distiller = AutonomousDistiller()
        result = distiller.distill_from_buffer(
            buffer=buf,
            output_path=self.model_path,
            min_samples=8,
            k=2,
            model_name="support_intent_model",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.model_name, "support_intent_model")
        self.assertGreaterEqual(result.cluster_count, 2)
        self.assertGreaterEqual(result.accuracy, 0.70)
        self.assertTrue(os.path.exists(self.model_path))

        # Verify compiled model loads and evaluates in microsecond domain
        loaded = CompiledInstinct.load(self.model_path)
        self.assertEqual(loaded.name, "support_intent_model")

        # Test inference
        res = loaded.predict("I need a refund for the duplicate charge")
        self.assertIsNotNone(res["choice"].selected)
        self.assertGreater(res["choice"].confidence, 0.40)


class TestDistillationWorker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_worker_lifecycle_and_callback(self):
        buf = DistillationBuffer(max_size=50)
        prompts = [
            "Query alpha refund request",
            "Query beta refund charge",
            "Query gamma billing receipt",
            "Query delta invoice dispute",
            "Query epsilon login failure",
            "Query zeta password reset",
            "Query eta auth code broken",
            "Query theta credential issue",
        ]
        for p in prompts:
            buf.record(prompt=p, response="OK", model="gpt-4o")

        compiled_events = []
        def _on_ready(res: DistillationResult):
            compiled_events.append(res)

        worker = DistillationWorker(
            buffer=buf,
            interval_seconds=0.1,
            min_new_samples=6,
            output_dir=self.temp_dir,
            on_candidate_ready=_on_ready,
        )

        # Trigger manually
        res = worker.trigger_cycle()
        self.assertIsNotNone(res)
        self.assertEqual(len(compiled_events), 1)

        st = worker.status()
        self.assertEqual(st["total_cycles"], 1)
        self.assertIsNotNone(st["latest_candidate"])

        # Clean shutdown
        worker.start()
        time.sleep(0.05)
        worker.stop()
        self.assertFalse(worker.status()["running"])


class TestShadowCanaryDistillationIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "challenger.reflex")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stage_distilled_candidate_into_shadow(self):
        # 1. Distill a model
        buf = DistillationBuffer(max_size=30)
        for i in range(5):
            buf.record(f"Billing query {i}", "billing handled")
            buf.record(f"Password reset query {i}", "auth handled")

        distiller = AutonomousDistiller()
        res = distiller.distill_from_buffer(buf, output_path=self.model_path, min_samples=6, k=2)
        self.assertIsNotNone(res)

        # 2. Stage into DecisionShadowRouter
        cfg = ShadowConfig(
            canary_traffic_pct=0.0,
            concordance_threshold=0.85,
            auto_promote=True,
            min_samples_for_promotion=5,
        )
        router = DecisionShadowRouter(
            champion=lambda s, q: {"choice": "billing"},
            challenger=lambda s, q: {"choice": "billing"},
            config=cfg,
        )

        # Stage newly distilled model
        candidate = CompiledInstinct.load(self.model_path)
        router.stage_candidate_model(candidate, concordance_threshold=0.85)

        self.assertEqual(router.config.stage, ShadowStage.OBSERVATION)
        self.assertEqual(router.challenger, candidate)
        self.assertEqual(router.tracker.total_samples, 0)


class TestGatewayDistillation(unittest.TestCase):
    def test_gateway_initialization_with_distillation(self):
        from reflex.gateway import GatewayConfig, GatewayRequestHandler

        cfg = GatewayConfig(
            distill_enabled=True,
            distill_buffer_size=100,
            distill_min_samples=5,
        )
        GatewayRequestHandler.initialize(cfg)

        self.assertIsNotNone(GatewayRequestHandler.distill_buffer)
        self.assertIsNotNone(GatewayRequestHandler.distill_worker)
        self.assertEqual(GatewayRequestHandler.distill_buffer.max_size, 100)

        # Record a trace
        t = GatewayRequestHandler.distill_buffer.record(
            prompt="Test prompt",
            response="Test response",
            model="gpt-4o",
            latency_ms=150.0,
        )
        self.assertEqual(GatewayRequestHandler.distill_buffer.size(), 1)
        self.assertEqual(t.prompt, "Test prompt")

        # Cleanup
        if GatewayRequestHandler.distill_worker:
            GatewayRequestHandler.distill_worker.stop()


if __name__ == "__main__":
    unittest.main()
