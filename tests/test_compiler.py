"""
Unit tests for Reflex Prompt-to-Instinct Compiler & Calibration Pipeline (Phase 24).
Zero external dependencies (Python standard library only).
"""

import json
import os
import socket
import tempfile
import time
import unittest
import urllib.request
import urllib.error

from reflex import Reflex, Choice, Noul, Score
from reflex.compiler import (
    PromptSpec,
    CompiledInstinct,
    SyntheticDataGenerator,
    InstinctCompiler,
    CalibrationMetrics,
    MAGIC_HEADER,
)
from reflex.gateway import ReflexGatewayServer, GatewayConfig


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestPromptSpec(unittest.TestCase):
    """Unit tests for PromptSpec data structure and serialization."""

    def test_prompt_spec_creation_and_dict(self):
        spec = PromptSpec(
            prompt="Classify customer support tickets into billing, technical, or general.",
            decision_type="choice",
            options=["billing", "technical", "general"],
            guidelines={
                "billing": "Invoice, refund, payment, subscription charges.",
                "technical": "Bug, crash, error, latency, API issue.",
                "general": "Feedback, compliments, inquiries, questions.",
            },
            few_shot_examples=[
                {"text": "Why was my card billed twice?", "label": "billing"},
                {"text": "Getting 500 error when uploading file", "label": "technical"},
            ],
            name="ticket_triage",
        )
        self.assertEqual(spec.name, "ticket_triage")
        self.assertEqual(spec.decision_type, "choice")
        self.assertEqual(len(spec.options), 3)

        d = spec.to_dict()
        self.assertEqual(d["name"], "ticket_triage")
        self.assertEqual(d["decision_type"], "choice")
        self.assertEqual(len(d["few_shot_examples"]), 2)

        restored = PromptSpec.from_dict(d)
        self.assertEqual(restored.name, spec.name)
        self.assertEqual(restored.options, spec.options)
        self.assertEqual(restored.guidelines, spec.guidelines)
        self.assertEqual(len(restored.few_shot_examples), 2)


class TestSyntheticDataGenerator(unittest.TestCase):
    """Unit tests for pure-Python synthetic dataset generator."""

    def test_sample_generation(self):
        gen = SyntheticDataGenerator()
        samples = gen.generate_samples_for_option(
            option="billing",
            guideline="Refund requests, invoices, overdue charges, payments",
            count=15,
        )
        self.assertEqual(len(samples), 15)
        for s in samples:
            self.assertIsInstance(s, str)
            self.assertTrue(len(s) > 10)


class TestInstinctCompiler(unittest.TestCase):
    """Unit tests for compiler training, calibration, and prediction."""

    def test_compile_choice_model(self):
        spec = PromptSpec(
            prompt="Categorize sentiment of message",
            decision_type="choice",
            options=["positive", "negative"],
            guidelines={
                "positive": "Great, awesome, love it, happy, satisfied, excellent",
                "negative": "Terrible, hate it, angry, refund, broken, failure",
            },
            name="sentiment_classifier",
        )
        compiler = InstinctCompiler()
        model = compiler.compile(spec, samples_per_class=20, epochs=25, lr=0.08)

        self.assertIsInstance(model, CompiledInstinct)
        self.assertEqual(model.name, "sentiment_classifier")
        self.assertEqual(model.decision_type, "choice")
        self.assertIn("positive", model.weights)
        self.assertIn("negative", model.weights)
        self.assertEqual(len(model.weights["positive"]), 384)
        self.assertGreater(model.metrics.accuracy, 0.70)
        self.assertLess(model.metrics.brier_score, 0.40)
        self.assertLess(model.metrics.ece, 0.40)

        # Test prediction
        res_pos = model.predict("I absolutely love this product, it is fantastic and awesome!")
        self.assertIn("choice", res_pos.decisions)
        self.assertEqual(res_pos["choice"].selected, "positive")
        self.assertGreater(res_pos["choice"].distribution["positive"], res_pos["choice"].distribution["negative"])
        self.assertLess(res_pos.latency_ms, 5.0)

        res_neg = model.predict("This is completely broken, terrible failure, I demand a refund immediately.")
        self.assertEqual(res_neg["choice"].selected, "negative")

    def test_compile_noul_model(self):
        spec = PromptSpec(
            prompt="Detect if this is an urgent security incident",
            decision_type="noul",
            options=["normal", "urgent"],
            guidelines={
                "normal": "Routine update, documentation check, general hello",
                "urgent": "Data breach, root compromise, security exploit, ransomware",
            },
            name="security_noul",
        )
        compiler = InstinctCompiler()
        model = compiler.compile(spec, samples_per_class=15, epochs=20)

        self.assertEqual(model.decision_type, "noul")
        res = model.predict("CRITICAL: Root password compromised by unauthorized external IP address!")
        self.assertIn("noul", res.decisions)
        self.assertIsInstance(res["noul"].probability, float)

    def test_compile_score_model(self):
        spec = PromptSpec(
            prompt="Rate customer satisfaction from 1 to 10",
            decision_type="score",
            options=["low", "high"],
            guidelines={
                "low": "Extremely unsatisfied, awful experience, horrific delay",
                "high": "Spectacular service, outstanding staff, seamless delight",
            },
            name="satisfaction_scorer",
        )
        compiler = InstinctCompiler()
        model = compiler.compile(spec, samples_per_class=15, epochs=20)

        self.assertEqual(model.decision_type, "score")
        res = model.predict("Spectacular service and outstanding staff, absolute delight!")
        self.assertIn("score", res.decisions)
        self.assertGreaterEqual(res["score"].score, 1.0)
        self.assertLessEqual(res["score"].score, 10.0)


class TestCompiledModelSerialization(unittest.TestCase):
    """Unit tests for binary .reflex file format, CRC32 integrity, and tampering detection."""

    def setUp(self):
        spec = PromptSpec(
            prompt="Triage tickets",
            decision_type="choice",
            options=["billing", "support"],
            guidelines={"billing": "invoice and refund", "support": "help and bug"},
            name="triage_model",
        )
        compiler = InstinctCompiler()
        self.model = compiler.compile(spec, samples_per_class=10, epochs=10)

    def test_save_and_load_reflex_artifact(self):
        with tempfile.NamedTemporaryFile(suffix=".reflex", delete=False) as f:
            temp_path = f.name

        try:
            self.model.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 500)

            # Inspect header
            with open(temp_path, "rb") as f:
                header_magic = f.read(4)
                self.assertEqual(header_magic, MAGIC_HEADER)

            # Load model
            loaded = CompiledInstinct.load(temp_path)
            self.assertEqual(loaded.name, self.model.name)
            self.assertEqual(loaded.decision_type, self.model.decision_type)
            self.assertEqual(loaded.options, self.model.options)
            self.assertEqual(len(loaded.weights["billing"]), 384)

            # Assert identical predictions
            p1 = self.model.predict("Please send me the invoice for my subscription")
            p2 = loaded.predict("Please send me the invoice for my subscription")
            self.assertEqual(p1["choice"].selected, p2["choice"].selected)
            self.assertAlmostEqual(
                p1["choice"].distribution["billing"],
                p2["choice"].distribution["billing"],
                places=3,
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_corrupt_crc32_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".reflex", delete=False) as f:
            temp_path = f.name

        try:
            self.model.save(temp_path)
            # Tamper with file contents (modify a byte in the payload)
            with open(temp_path, "r+b") as f:
                f.seek(15)
                byte = f.read(1)
                f.seek(15)
                f.write(bytes([(byte[0] ^ 0xFF)]))

            # Loading tampered file should fail with CRC32 mismatch
            with self.assertRaises(ValueError) as ctx:
                CompiledInstinct.load(temp_path)
            self.assertIn("CRC32 checksum mismatch", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_truncated_header_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".reflex", delete=False) as f:
            temp_path = f.name

        try:
            with open(temp_path, "wb") as f:
                f.write(b"RFX")  # Too short
            with self.assertRaises(ValueError) as ctx:
                CompiledInstinct.load(temp_path)
            self.assertIn("header too short", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestClientCompilerIntegration(unittest.TestCase):
    """Unit tests for Reflex client integration with compiled models."""

    def test_client_model_path_auto_load(self):
        with tempfile.NamedTemporaryFile(suffix=".reflex", delete=False) as f:
            temp_path = f.name

        try:
            spec = PromptSpec(
                prompt="Route inquiries",
                decision_type="choice",
                options=["sales", "support"],
                guidelines={"sales": "pricing and enterprise quota", "support": "help and bug"},
                name="router_model",
            )
            compiler = InstinctCompiler()
            model = compiler.compile(spec, samples_per_class=12, epochs=15)
            model.save(temp_path)

            rx = Reflex(model_path=temp_path)
            self.assertIsNotNone(rx.compiled_instinct)
            self.assertEqual(rx.compiled_instinct.name, "router_model")

            # Direct predict method
            pred = rx.predict("What is the enterprise pricing for 1 million requests?")
            self.assertEqual(pred["choice"].selected, "sales")
            self.assertTrue(pred.backend.startswith("compiled:"))

            # Typed evaluate call
            res = rx.evaluate(
                "Need help debugging this memory leak error",
                {"team": Choice("Select team", options=["sales", "support"])},
            )
            self.assertEqual(res["team"].selected, "support")
            self.assertTrue(res.backend.startswith("compiled:"))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_client_compile_method(self):
        rx = Reflex()
        model = rx.compile(
            prompt="Is this a refund request?",
            decision_type="choice",
            options=["refund", "other"],
            guidelines={"refund": "money back, return, chargeback, refund", "other": "features and docs"},
            samples_per_class=10,
            epochs=10,
        )
        self.assertIsNotNone(rx.compiled_instinct)
        self.assertEqual(rx.compiled_instinct, model)

        res = rx.predict("I want my money back for this purchase")
        self.assertEqual(res["choice"].selected, "refund")


class TestGatewayCompilerIntegration(unittest.TestCase):
    """Unit tests for Reflex Gateway serving compiled .reflex models."""

    @classmethod
    def setUpClass(cls):
        cls.temp_file = tempfile.NamedTemporaryFile(suffix=".reflex", delete=False)
        cls.model_path = cls.temp_file.name
        cls.temp_file.close()

        spec = PromptSpec(
            prompt="Classify ticket priority",
            decision_type="choice",
            options=["high_priority", "low_priority"],
            guidelines={
                "high_priority": "emergency, critical, down, outage, crash, security breach",
                "low_priority": "typo, thank you, suggestion, cosmetic, minor question",
            },
            name="gateway_classifier",
        )
        compiler = InstinctCompiler()
        model = compiler.compile(spec, samples_per_class=15, epochs=15)
        model.save(cls.model_path)

        cls.port = get_free_port()
        cls.config = GatewayConfig(
            host="127.0.0.1",
            port=cls.port,
            cache_enabled=False,
            guardrails_enabled=False,
            compiled_model_path=cls.model_path,
        )
        cls.server = ReflexGatewayServer(cls.config)
        cls.server.start(background=True)
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        if os.path.exists(cls.model_path):
            os.unlink(cls.model_path)

    def test_gateway_models_list_exposes_compiled_instinct(self):
        url = f"http://127.0.0.1:{self.port}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            model_ids = [m["id"] for m in data["data"]]
            self.assertIn("reflex-compiled:gateway_classifier", model_ids)

    def test_gateway_chat_completion_shortcircuit_compiled(self):
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        payload = {
            "model": "reflex-compiled:gateway_classifier",
            "messages": [
                {"role": "user", "content": "EMERGENCY: Production database is down and experiencing outage!"}
            ],
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=5.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("X-Reflex-Cache"), "SHORTCIRCUIT-COMPILED")
            resp_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(resp_data["model"], "reflex-compiled:gateway_classifier")
            content = json.loads(resp_data["choices"][0]["message"]["content"])
            self.assertEqual(content["selected"], "high_priority")


if __name__ == "__main__":
    unittest.main()
