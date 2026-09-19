"""
Unit tests for Reflex Enterprise Policy-as-Code & Cryptographic Merkle Audit Trail (Phase 23).
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

from reflex import Reflex, Noul, Choice
from reflex.policy import (
    PolicyAction,
    PolicyViolationError,
    PolicyRule,
    PolicyRuleSet,
    PolicyVerdict,
    PolicyEngine,
    AuditEntry,
    MerkleTree,
    MerkleAuditLog,
)
from reflex.gateway import ReflexGatewayServer, GatewayConfig


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestPolicyEngine(unittest.TestCase):
    """Unit tests covering declarative Policy-as-Code rules, operators, and actions."""

    def test_rule_operators(self):
        rules = [
            PolicyRule(
                rule_id="r_gt",
                action=PolicyAction.DENY,
                conditions={"field": "context.amount", "op": "gt", "value": 10000},
                description="Transactions over 10k must be denied",
            ),
            PolicyRule(
                rule_id="r_contains",
                action=PolicyAction.ENFORCE_LOCAL,
                conditions={"field": "state", "op": "contains", "value": "patient_id"},
                description="HIPAA patient records must stay on local device",
                tags=["HIPAA"],
            ),
            PolicyRule(
                rule_id="r_regex",
                action=PolicyAction.DENY,
                conditions={"field": "state", "op": "regex", "value": r"ssn:\s*\d{3}-\d{2}-\d{4}"},
                description="Social Security Numbers cannot be processed",
            ),
            PolicyRule(
                rule_id="r_in",
                action=PolicyAction.DENY,
                conditions={"field": "context.country", "op": "in", "value": ["CU", "IR", "KP", "SY"]},
                description="Sanctioned jurisdictions blocked",
            ),
        ]
        engine = PolicyEngine(PolicyRuleSet(rules=rules, default_action=PolicyAction.ALLOW))

        # 1. Test gt operator trigger
        v1 = engine.evaluate(state="Transfer money", context={"amount": 15000})
        self.assertFalse(v1.allowed)
        self.assertEqual(v1.action, PolicyAction.DENY)
        self.assertIn("r_gt", v1.violations)

        # 2. Test contains operator (HIPAA)
        v2 = engine.evaluate(state="Looking up Patient_ID: 99410", context={"amount": 50})
        self.assertTrue(v2.allowed)
        self.assertEqual(v2.action, PolicyAction.ENFORCE_LOCAL)
        self.assertIn("r_contains", v2.matched_rules)
        self.assertIn("HIPAA", v2.tags)

        # 3. Test regex operator (SSN)
        v3 = engine.evaluate(state="Customer SSN: 123-45-6789", context={})
        self.assertFalse(v3.allowed)
        self.assertEqual(v3.action, PolicyAction.DENY)
        self.assertIn("r_regex", v3.violations)

        # 4. Test in operator (Sanctions)
        v4 = engine.evaluate(state="Normal query", context={"country": "KP"})
        self.assertFalse(v4.allowed)
        self.assertEqual(v4.action, PolicyAction.DENY)
        self.assertIn("r_in", v4.violations)

        # 5. Normal clean request
        v5 = engine.evaluate(state="What is the weather today?", context={"country": "US", "amount": 10})
        self.assertTrue(v5.allowed)
        self.assertEqual(v5.action, PolicyAction.ALLOW)
        self.assertEqual(len(v5.violations), 0)

    def test_rule_combinators(self):
        rules = [
            PolicyRule(
                rule_id="r_combo",
                action=PolicyAction.DENY,
                conditions={
                    "all_of": [
                        {"field": "context.role", "op": "neq", "value": "admin"},
                        {"field": "context.action", "op": "eq", "value": "delete_database"},
                    ]
                },
                description="Non-admins cannot delete database",
            )
        ]
        engine = PolicyEngine(PolicyRuleSet(rules=rules))

        # Non-admin attempting delete -> DENIED
        v1 = engine.evaluate(state="drop tables", context={"role": "intern", "action": "delete_database"})
        self.assertFalse(v1.allowed)
        self.assertEqual(v1.action, PolicyAction.DENY)

        # Admin attempting delete -> ALLOWED
        v2 = engine.evaluate(state="drop tables", context={"role": "admin", "action": "delete_database"})
        self.assertTrue(v2.allowed)

    def test_ruleset_json_serialization(self):
        rule = PolicyRule(
            rule_id="r_audit",
            action=PolicyAction.REQUIRE_HUMAN,
            description="Require human verification",
            conditions={"field": "context.risk", "op": "gt", "value": 0.8},
            tags=["EU_AI_ACT"],
        )
        ruleset = PolicyRuleSet(rules=[rule], name="risk_policy", version="2.0")

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(ruleset.to_dict(), f)
            temp_path = f.name

        try:
            loaded = PolicyRuleSet.from_json_file(temp_path)
            self.assertEqual(loaded.name, "risk_policy")
            self.assertEqual(loaded.version, "2.0")
            self.assertEqual(len(loaded.rules), 1)
            self.assertEqual(loaded.rules[0].action, PolicyAction.REQUIRE_HUMAN)
            self.assertIn("EU_AI_ACT", loaded.rules[0].tags)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestMerkleAuditTrail(unittest.TestCase):
    """Unit tests verifying cryptographic hash chaining, Merkle roots, and O(log N) proofs."""

    def test_merkle_tree_construction(self):
        # 1. Empty tree
        tree_0 = MerkleTree([])
        self.assertEqual(tree_0.root, "0" * 64)

        # 2. Single leaf
        leaf1 = "a" * 64
        tree_1 = MerkleTree([leaf1])
        self.assertEqual(tree_1.root, leaf1)

        # 3. Two leaves
        leaf2 = "b" * 64
        tree_2 = MerkleTree([leaf1, leaf2])
        self.assertNotEqual(tree_2.root, leaf1)
        self.assertEqual(len(tree_2.root), 64)

        # 4. Odd number of leaves (3 leaves)
        leaf3 = "c" * 64
        tree_3 = MerkleTree([leaf1, leaf2, leaf3])
        self.assertEqual(len(tree_3.root), 64)

    def test_merkle_proof_generation_and_independent_verification(self):
        # Create 8 leaf hashes
        leaves = [
            f"{i:064x}" for i in range(1, 9)
        ]
        tree = MerkleTree(leaves)
        root = tree.root

        # Generate and independently verify proof for every single leaf
        for idx in range(len(leaves)):
            proof = tree.get_proof(idx)
            # Log2(8) = 3 proof steps
            self.assertEqual(len(proof), 3)

            is_valid = MerkleTree.verify_proof(leaves[idx], proof, root)
            self.assertTrue(is_valid, f"Proof failed for leaf index {idx}")

        # Tampered leaf must fail verification
        tampered_leaf = "f" * 64
        self.assertFalse(MerkleTree.verify_proof(tampered_leaf, tree.get_proof(0), root))

        # Tampered proof sibling must fail verification
        tampered_proof = tree.get_proof(0)
        tampered_proof[0]["sibling"] = "0" * 64
        self.assertFalse(MerkleTree.verify_proof(leaves[0], tampered_proof, root))

    def test_merkle_audit_log_append_and_tamper_detection(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_log_path = f.name

        try:
            log = MerkleAuditLog(storage_path=temp_log_path)
            self.assertEqual(log.height(), 0)

            # Append 5 decisions
            for i in range(5):
                log.append(
                    state=f"User request {i}",
                    decisions={"decision": f"choice_{i}", "confidence": 0.95},
                    metadata={"user_id": f"usr_{i}"},
                    policy_verdict={"action": "ALLOW", "allowed": True},
                )

            self.assertEqual(log.height(), 5)
            self.assertNotEqual(log.root, "0" * 64)

            # Verify cryptographic integrity
            is_valid, broken_idx, msg = log.verify_chain()
            self.assertTrue(is_valid)
            self.assertIsNone(broken_idx)

            # Test inclusion proof
            proof_data = log.prove(2)
            self.assertEqual(proof_data["index"], 2)
            self.assertTrue(MerkleTree.verify_proof(proof_data["entry_hash"], proof_data["proof"], proof_data["merkle_root"]))

            # Reload from disk and verify persistence
            log_reloaded = MerkleAuditLog(storage_path=temp_log_path)
            self.assertEqual(log_reloaded.height(), 5)
            self.assertEqual(log_reloaded.root, log.root)
            self.assertTrue(log_reloaded.verify_chain()[0])

            # SIMULATE TAMPERING: Artificially alter historical entry #2
            log_reloaded.entries[2].decisions["confidence"] = 0.10  # Tampered!
            is_valid_tampered, broken_idx_tampered, reason = log_reloaded.verify_chain()
            self.assertFalse(is_valid_tampered)
            self.assertEqual(broken_idx_tampered, 2)
            self.assertIn("Tampered entry content", reason)

        finally:
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)


class TestReflexClientPolicyIntegration(unittest.TestCase):
    """Unit tests verifying Reflex client integration with policy enforcement and audit logging."""

    def test_client_policy_denial(self):
        rule = PolicyRule(
            rule_id="r_prohibit_wire",
            action=PolicyAction.DENY,
            conditions={"field": "state", "op": "contains", "value": "wire transfer"},
            description="Wire transfers prohibited",
        )
        ruleset = PolicyRuleSet(rules=[rule])
        audit_log = MerkleAuditLog()
        rx = Reflex(backend="local", policy=ruleset, audit_log=audit_log)

        # 1. Prohibited prompt raises PolicyViolationError
        with self.assertRaises(PolicyViolationError) as cm:
            rx.evaluate(
                state="Please initiate wire transfer of $5,000",
                questions={"is_urgent": Noul("Is this urgent?")},
            )
        self.assertEqual(cm.exception.rule_id, "r_prohibit_wire")
        self.assertEqual(cm.exception.action, PolicyAction.DENY)

        # 2. Audit log recorded the blocked attempt
        self.assertEqual(audit_log.height(), 1)
        self.assertFalse(audit_log.entries[0].policy_verdict["allowed"])

        # 3. Compliant prompt succeeds
        res = rx.evaluate(
            state="What are the branch operating hours?",
            questions={"is_urgent": Noul("Is this urgent?")},
        )
        self.assertIsNotNone(res)
        self.assertEqual(audit_log.height(), 2)
        self.assertTrue(audit_log.entries[1].policy_verdict["allowed"])

        # 4. Verify client audit helper methods
        self.assertIsNotNone(rx.audit_root())
        is_valid, _, _ = rx.verify_audit_log()
        self.assertTrue(is_valid)

        proof_bundle = rx.export_audit_proof(1)
        self.assertEqual(proof_bundle["index"], 1)


class TestGatewayPolicyAndAuditEndpoints(unittest.TestCase):
    """Integration test for AI Envoy Gateway policy evaluation and audit REST endpoints."""

    def setUp(self):
        self.port = get_free_port()
        rule = PolicyRule(
            rule_id="r_block_pii",
            action=PolicyAction.DENY,
            conditions={"field": "state", "op": "contains", "value": "secret_token_alpha"},
            description="PII leak blocked",
            tags=["SECURITY"],
        )
        ruleset = PolicyRuleSet(rules=[rule])
        policy_engine = PolicyEngine(ruleset=ruleset)
        audit_log = MerkleAuditLog()

        self.config = GatewayConfig(
            host="127.0.0.1",
            port=self.port,
            policy_engine=policy_engine,
            audit_log=audit_log,
            policy_enabled=True,
            audit_enabled=True,
        )
        self.server = ReflexGatewayServer(self.config)
        self.server.start(background=True)
        time.sleep(0.3)

    def tearDown(self):
        self.server.stop()
        time.sleep(0.2)

    def test_policy_rules_endpoint(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/policy/rules")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("rules", data)
            self.assertEqual(len(data["rules"]), 1)
            self.assertEqual(data["rules"][0]["rule_id"], "r_block_pii")

    def test_policy_evaluate_endpoint(self):
        payload = json.dumps({"state": "Attempting to send secret_token_alpha", "context": {}}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/policy/evaluate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertFalse(data["allowed"])
            self.assertEqual(data["action"], "DENY")
            self.assertIn("r_block_pii", data["violations"])

    def test_chat_completions_policy_denial_and_audit(self):
        # 1. Prohibited prompt -> 403 Forbidden
        bad_payload = json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Here is my secret_token_alpha"}]
        }).encode("utf-8")
        req_bad = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=bad_payload,
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req_bad, timeout=3.0)
        self.assertEqual(cm.exception.code, 403)

        # 2. Query audit root and verify
        req_root = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/audit/root")
        with urllib.request.urlopen(req_root, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertGreater(data["total_entries"], 0)
            self.assertNotEqual(data["merkle_root"], "0" * 64)

        req_verify = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/audit/verify")
        with urllib.request.urlopen(req_verify, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["valid"])

        # 3. Query audit proof for entry 0
        req_proof = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/audit/proof/0")
        with urllib.request.urlopen(req_proof, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["index"], 0)
            self.assertIn("merkle_root", data)
            self.assertIn("proof", data)


if __name__ == "__main__":
    unittest.main()
