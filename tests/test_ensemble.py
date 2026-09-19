"""
Unit Tests for Mixture-of-Reflexes (MoR) & Hierarchical Instinct Ensembles (Phase 26).
Verifies gating networks, Dirichlet uncertainty voting, Shannon entropy, 3-tier cascading,
serialization with CRC32 verification, and Client/Gateway integrations.
Zero external dependencies (Python standard library only).
"""

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error

from reflex import (
    Reflex,
    Choice,
    PromptSpec,
    InstinctCompiler,
    CompiledInstinct,
    ReflexGatewayServer,
    GatewayConfig,
)
from reflex.ensemble import (
    SpecialistModel,
    MoRGatingNetwork,
    EnsembleResult,
    InstinctEnsemble,
    HierarchicalCascade,
    _shannon_entropy,
)


class TestMixtureOfReflexes(unittest.TestCase):
    """Verifies MoR gating, uncertainty voting, and hierarchical cascading."""

    @classmethod
    def setUpClass(cls):
        compiler = InstinctCompiler()

        # 1. Specialist 1: Billing Specialist
        cls.spec_billing = PromptSpec(
            name="billing_specialist",
            prompt="Triage billing, payment, invoice, and subscription inquiries",
            decision_type="choice",
            options=["billing_charge", "billing_refund", "general_inquiry"],
            guidelines={
                "billing_charge": "Credit card charges, invoices, VAT tax receipts, subscription renewals",
                "billing_refund": "Refund requests, duplicate payment claims, chargeback dispute",
                "general_inquiry": "Other general non-billing questions",
            },
        )
        cls.model_billing = compiler.compile(cls.spec_billing, samples_per_class=20, epochs=25)

        # 2. Specialist 2: Security Specialist
        cls.spec_security = PromptSpec(
            name="security_specialist",
            prompt="Triage cybersecurity, auth breaches, injection, and vulnerability reports",
            decision_type="choice",
            options=["sec_breach", "sec_vulnerability", "general_inquiry"],
            guidelines={
                "sec_breach": "Compromised API tokens, root login from unknown IP, brute force attack, ransomware",
                "sec_vulnerability": "SQL injection report, XSS advisory, dependency CVE scan, buffer overflow",
                "general_inquiry": "Other general non-security issues",
            },
        )
        cls.model_security = compiler.compile(cls.spec_security, samples_per_class=20, epochs=25)

        # 3. Create InstinctEnsemble
        cls.ensemble = InstinctEnsemble(
            name="enterprise_triage_mor",
            top_k=2,
            temperature=0.8,
            entropy_attenuation=2.0,
        )
        cls.specialist_billing = SpecialistModel(
            name="billing_head",
            domain="billing",
            model=cls.model_billing,
            description="Billing, financial invoices, credit card payments, renewals, refunds",
            keywords=["invoice", "charge", "refund", "receipt", "payment", "card", "tax", "vat"],
            weight=1.0,
        )
        cls.specialist_security = SpecialistModel(
            name="security_head",
            domain="security",
            model=cls.model_security,
            description="Cybersecurity, unauthorized root access, breach alerts, vulnerabilities",
            keywords=["breach", "attack", "token", "password", "vulnerability", "injection", "root"],
            weight=1.0,
        )
        cls.ensemble.add_specialist(cls.specialist_billing)
        cls.ensemble.add_specialist(cls.specialist_security)

    def test_shannon_entropy_properties(self):
        """Verifies normalized Shannon entropy calculation."""
        # Pure certainty -> 0.0 entropy
        self.assertAlmostEqual(_shannon_entropy({"a": 1.0, "b": 0.0}), 0.0, places=3)
        # Uniform ambiguity -> 1.0 entropy
        self.assertAlmostEqual(_shannon_entropy({"a": 0.5, "b": 0.5}), 1.0, places=3)
        self.assertAlmostEqual(_shannon_entropy({"a": 0.3333, "b": 0.3333, "c": 0.3334}), 1.0, places=2)
        # Empty or single item -> 0.0
        self.assertEqual(_shannon_entropy({}), 0.0)
        self.assertEqual(_shannon_entropy({"only": 1.0}), 0.0)

    def test_gating_network_routing(self):
        """Verifies semantic gating routes queries to the most aligned specialist."""
        gating = self.ensemble.gating

        # Billing query
        gates_billing = gating.compute_gates("Why was my credit card charged twice for last month's invoice?")
        self.assertIn("billing_head", gates_billing)
        self.assertIn("security_head", gates_billing)
        self.assertGreater(gates_billing["billing_head"], gates_billing["security_head"])

        # Security query
        gates_sec = gating.compute_gates("Critical alert: root account logged in from unknown IP address via SSH")
        self.assertGreater(gates_sec["security_head"], gates_sec["billing_head"])

        # Top-K sparse filtering
        top1 = gating.top_k(gates_billing, k=1)
        self.assertEqual(len(top1), 1)
        self.assertIn("billing_head", top1)
        self.assertAlmostEqual(top1["billing_head"], 1.0)

    def test_ensemble_prediction_and_blending(self):
        """Verifies standard MoR prediction with uncertainty-weighted voting."""
        query = "Can I get an itemized VAT invoice receipt for our subscription charge?"
        res = self.ensemble.predict(query, top_k=2)

        self.assertIsInstance(res, EnsembleResult)
        self.assertEqual(res.tier, "L2_ENSEMBLE_CONSENSUS")
        self.assertFalse(res.routed_to_system2)
        self.assertIn(res.selected, ["billing_charge", "billing_refund", "general_inquiry"])
        self.assertGreater(res.confidence, 0.40)
        self.assertLess(res.entropy, 0.90)
        self.assertIn("billing_head", res.specialist_predictions)
        self.assertIn("security_head", res.specialist_predictions)
        self.assertGreater(res.voting_weights["billing_head"], res.voting_weights["security_head"])

    def test_cascade_l1_fast_path(self):
        """Verifies clear, high-confidence queries shortcircuit at Tier 1 (L1 Fast-Path)."""
        clear_billing_query = "Invoice charge receipt payment card refund credit"
        res = self.ensemble.cascade_predict(
            clear_billing_query,
            confidence_threshold=0.60,
            entropy_threshold=0.70,
        )

        self.assertEqual(res.tier, "L1_FAST_PATH")
        self.assertFalse(res.routed_to_system2)
        self.assertGreaterEqual(res.confidence, 0.60)
        self.assertLessEqual(res.entropy, 0.70)
        # Latency should be sub-millisecond
        self.assertLess(res.latency_ms, 15.0)

    def test_cascade_l3_system2_escalation(self):
        """Verifies ambiguous or out-of-distribution queries escalate to System-2 (L3 Escalation)."""
        random_query = "Blue pineapple origami quantum banana cloud sunset harmonica"
        # Strict thresholds to trigger escalation
        res = self.ensemble.cascade_predict(
            random_query,
            confidence_threshold=0.99,
            entropy_threshold=0.10,
            consensus_threshold=0.99,
            max_consensus_entropy=0.10,
        )

        self.assertEqual(res.tier, "L3_SYSTEM2_ESCALATION")
        self.assertTrue(res.routed_to_system2)

    def test_hierarchical_cascade_convenience_class(self):
        """Verifies HierarchicalCascade wrapper behavior."""
        cascade = HierarchicalCascade(
            ensemble=self.ensemble,
            confidence_threshold=0.60,
            entropy_threshold=0.70,
        )
        res = cascade.route("Credit card invoice payment charge")
        self.assertIn(res.tier, ["L1_FAST_PATH", "L2_ENSEMBLE_CONSENSUS"])
        self.assertFalse(res.routed_to_system2)

    def test_serialization_and_checksum_integrity(self):
        """Verifies bundling into .reflex-ensemble and tamper detection via CRC32."""
        with tempfile.NamedTemporaryFile(suffix=".reflex-ensemble", delete=False) as f:
            temp_path = f.name

        try:
            self.ensemble.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))

            # Verify file format
            with open(temp_path, "rb") as f:
                header = f.read(12)
                magic = header[0:4]
                self.assertEqual(magic, b"RFXE")

            # Load restored ensemble
            loaded_ensemble = InstinctEnsemble.load(temp_path)
            self.assertEqual(loaded_ensemble.name, self.ensemble.name)
            self.assertEqual(len(loaded_ensemble.specialists), 2)
            self.assertIn("billing_head", loaded_ensemble.specialists)
            self.assertIn("security_head", loaded_ensemble.specialists)

            # Test predictions on restored ensemble
            test_query = "Security breach: stolen credentials and token compromise"
            orig_res = self.ensemble.predict(test_query)
            load_res = loaded_ensemble.predict(test_query)
            self.assertEqual(orig_res.selected, load_res.selected)
            self.assertAlmostEqual(orig_res.confidence, load_res.confidence, places=3)

            # Test tamper detection: flip byte in payload
            with open(temp_path, "rb") as f:
                content = bytearray(f.read())
            content[20] ^= 0xFF
            with open(temp_path, "wb") as f:
                f.write(content)

            with self.assertRaises(ValueError) as ctx:
                InstinctEnsemble.load(temp_path)
            self.assertIn("CRC32 checksum mismatch", str(ctx.exception))

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_reflex_client_ensemble_integration(self):
        """Verifies Reflex client evaluation through an active ensemble."""
        rx = Reflex(ensemble=self.ensemble)
        self.assertIsNotNone(rx.ensemble)

        # 1. rx.ensemble_predict
        res = rx.ensemble_predict("Why was my credit card charged for the enterprise license renewal?")
        self.assertIsInstance(res, EnsembleResult)
        self.assertIn(res.selected, ["billing_charge", "billing_refund", "general_inquiry"])

        # 2. rx.cascade_predict
        c_res = rx.cascade_predict("Suspicious unauthorized SSH login on database server")
        self.assertIsInstance(c_res, EnsembleResult)

        # 3. rx.choice routing through evaluate()
        selected = rx.choice(
            "Triage request",
            ["billing_charge", "sec_breach", "general_inquiry"],
            "Unauthorized root privilege escalation detected",
        )
        self.assertIsInstance(selected, str)
        self.assertTrue(len(selected) > 0)


class TestGatewayEnsembleIntegration(unittest.TestCase):
    """Verifies AI Envoy Gateway integration with Mixture-of-Reflexes ensembles."""

    @classmethod
    def setUpClass(cls):
        compiler = InstinctCompiler()
        spec = PromptSpec(
            name="gateway_triage_head",
            prompt="Triage inquiries into billing or technical support",
            decision_type="choice",
            options=["billing", "technical"],
            guidelines={
                "billing": "Invoices, refunds, chargebacks, subscriptions",
                "technical": "Server down, 500 error, crash, latency, bug",
            },
        )
        model = compiler.compile(spec, samples_per_class=15, epochs=20)
        cls.specialist = SpecialistModel(
            name="gw_specialist",
            domain="support",
            model=model,
            keywords=["billing", "technical", "invoice", "server"],
        )
        cls.ensemble = InstinctEnsemble(name="gw_ensemble", top_k=1)
        cls.ensemble.add_specialist(cls.specialist)

        cls.port = 18095
        cls.config = GatewayConfig(
            host="127.0.0.1",
            port=cls.port,
            system1_routing_enabled=True,
            ensemble=cls.ensemble,
        )
        cls.server = ReflexGatewayServer(cls.config)
        cls.thread = threading.Thread(target=cls.server.start, kwargs={"background": False}, daemon=True)
        cls.thread.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_gateway_ensemble_stats_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/v1/ensemble/stats"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["ensemble_name"], "gw_ensemble")
            self.assertIn("gw_specialist", data["specialists"])
            self.assertIn("stats", data)

    def test_gateway_ensemble_predict_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/v1/ensemble/predict"
        payload = json.dumps({"state": "Production database crashed with 500 error", "cascade": True}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("selected", data)
            self.assertIn("tier", data)
            self.assertIn("confidence", data)

    def test_gateway_chat_completion_ensemble_shortcircuit(self):
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        payload = json.dumps({
            "model": "reflex-ensemble:gw_ensemble",
            "messages": [{"role": "user", "content": "Need invoice receipt for last month's charge"}],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("X-Reflex-Cache"), "SHORTCIRCUIT-ENSEMBLE")
            data = json.loads(resp.read().decode("utf-8"))
            content = json.loads(data["choices"][0]["message"]["content"])
            self.assertIn("selected", content)
            self.assertIn("tier", content)


if __name__ == "__main__":
    unittest.main()
