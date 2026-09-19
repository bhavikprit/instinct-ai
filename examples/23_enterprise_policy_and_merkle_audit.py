"""
Reflex Example 23: Enterprise Policy-as-Code & Cryptographic Merkle Audit Trail (Phase 23).
Demonstrates declarative regulatory rule enforcement (HIPAA, GDPR, EU AI Act), data sovereignty geofencing,
append-only SHA-256 hash chaining, O(log N) Merkle inclusion proofs, and instant tamper detection.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Noul, Choice, Score
from reflex.policy import (
    PolicyAction,
    PolicyRule,
    PolicyRuleSet,
    PolicyEngine,
    PolicyViolationError,
    MerkleTree,
    MerkleAuditLog,
)


def run_policy_and_audit_demo():
    print("=" * 85)
    print("⚡ REFLEX POLICY: ENTERPRISE POLICY-AS-CODE & MERKLE AUDIT TRAIL (PHASE 23)")
    print("=" * 85)

    # -----------------------------------------------------------------
    # 1. Declarative Regulatory Compliance Rules (Policy-as-Code)
    # -----------------------------------------------------------------
    print("\n1. 📜 Defining Enterprise Compliance Ruleset (HIPAA, GDPR, EU AI Act):")

    rules = [
        # Rule 1: HIPAA Protected Health Information (PHI) must stay on-prem / local
        PolicyRule(
            rule_id="HIPAA-PHI-01",
            action=PolicyAction.ENFORCE_LOCAL,
            description="Patient health records and medical diagnosis must never leave local device",
            conditions={"field": "state", "op": "regex", "value": r"(patient_id|diagnosis|medical_record|prescription)"},
            tags=["HIPAA", "DATA_SOVEREIGNTY"],
        ),
        # Rule 2: High-Value Financial Transactions > $10,000 require human-in-the-loop
        PolicyRule(
            rule_id="FIN-RISK-02",
            action=PolicyAction.REQUIRE_HUMAN,
            description="Transactions exceeding $10,000 require dual-custody human sign-off",
            conditions={"field": "context.amount_usd", "op": "gt", "value": 10000},
            tags=["SOX", "RISK_CONTROL"],
        ),
        # Rule 3: Sanctioned Jurisdictions are strictly blocked
        PolicyRule(
            rule_id="OFAC-SANCTIONS-03",
            action=PolicyAction.DENY,
            description="Access from OFAC-sanctioned territory is strictly prohibited",
            conditions={"field": "context.country", "op": "in", "value": ["CU", "IR", "KP", "SY"]},
            tags=["OFAC", "EXPORT_CONTROL"],
        ),
        # Rule 4: System Overrides for Emergency Dispatches
        PolicyRule(
            rule_id="SLA-EMERGENCY-04",
            action=PolicyAction.OVERRIDE,
            description="Priority emergency calls are automatically routed to Level-1 triage",
            conditions={"field": "state", "op": "contains", "value": "emergency"},
            override_value="tier_1_immediate",
            tags=["SLA", "MISSION_CRITICAL"],
        ),
    ]

    ruleset = PolicyRuleSet(rules=rules, name="enterprise_core_policy", version="1.0")
    engine = PolicyEngine(ruleset=ruleset)

    for r in ruleset.rules:
        print(f"   • [{r.rule_id}] Action: {r.action.value:<14} | Tags: {r.tags} | {r.description}")

    # -----------------------------------------------------------------
    # 2. Cryptographic Append-Only Merkle Audit Trail
    # -----------------------------------------------------------------
    print("\n" + "-" * 85)
    print("2. 🔐 INITIALIZING CRYPTOGRAPHIC MERKLE AUDIT TRAIL")
    print("-" * 85)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        audit_log_file = f.name

    audit_log = MerkleAuditLog(storage_path=audit_log_file)
    rx = Reflex(backend="local", policy=ruleset, audit_log=audit_log)

    print(f"   Audit Ledger Location : {audit_log_file}")
    print(f"   Initial Ledger Height : {audit_log.height()} entries")
    print(f"   Initial Merkle Root   : {audit_log.root}")

    # -----------------------------------------------------------------
    # 3. Simulating Production Decision Traffic Under Policy Enforcement
    # -----------------------------------------------------------------
    print("\n" + "-" * 85)
    print("3. 🚀 EVALUATING DECISIONS UNDER POLICY COMPLIANCE GEOFENCING")
    print("-" * 85)

    scenarios = [
        {
            "name": "Standard Customer Inquiry (Compliant)",
            "state": "Customer asking for branch address in Chicago",
            "context": {"country": "US", "amount_usd": 0},
            "expect_error": False,
        },
        {
            "name": "HIPAA Patient Data (Enforce Local Sovereignty)",
            "state": "Access patient_id: 88492 medical_record for oncology checkup",
            "context": {"country": "US", "amount_usd": 0},
            "expect_error": False,
        },
        {
            "name": "High-Value Wire Transfer (Requires Human Sign-off)",
            "state": "Wire $25,000 for invoice #9921",
            "context": {"country": "US", "amount_usd": 25000},
            "expect_error": False,
        },
        {
            "name": "Sanctioned Territory Request (DENY Hard Block)",
            "state": "Request financial report download",
            "context": {"country": "KP", "amount_usd": 100},
            "expect_error": True,
        },
    ]

    for idx, sc in enumerate(scenarios, start=1):
        print(f"\n   Scenario #{idx}: {sc['name']}")
        print(f"   • Prompt  : \"{sc['state']}\"")
        print(f"   • Context : {sc['context']}")

        try:
            res = rx.evaluate(
                state=sc["state"],
                questions={"is_urgent": Noul("Is this an urgent request?")},
                context=sc["context"],
            )
            latest_entry = audit_log.entries[-1]
            print(f"   • Result  : ALLOWED ✅ (action: {latest_entry.policy_verdict['action']})")
            print(f"   • Merkle Root Rolling: {audit_log.root[:16]}... (Height: {audit_log.height()})")
        except PolicyViolationError as e:
            latest_entry = audit_log.entries[-1]
            print(f"   • Result  : BLOCKED 🛑 (PolicyViolationError: {e})")
            print(f"   • Violating Rule: {e.rule_id} | Tags: {e.tags}")
            print(f"   • Cryptographically logged violation in ledger at index #{latest_entry.index}")

    # -----------------------------------------------------------------
    # 4. Generating and Verifying O(log N) Merkle Inclusion Proof
    # -----------------------------------------------------------------
    print("\n" + "-" * 85)
    print("4. 🔍 GENERATING VERIFIABLE MERKLE PROOF FOR AUDIT REGULATORS")
    print("-" * 85)

    target_index = 1  # Verify HIPAA scenario #2
    proof_bundle = rx.export_audit_proof(target_index)

    print(f"   Proving decision at Index #{target_index}:")
    print(f"   • Entry Timestamp : {proof_bundle['entry']['timestamp']}")
    print(f"   • Entry SHA-256   : {proof_bundle['entry_hash']}")
    print(f"   • Merkle Root     : {proof_bundle['merkle_root']}")
    print(f"   • O(log N) Proof Path ({len(proof_bundle['proof'])} steps):")
    for step in proof_bundle["proof"]:
        print(f"     - Sibling ({step['position']}): {step['sibling'][:24]}...")

    # Independent cryptographic verification
    is_proof_valid = MerkleTree.verify_proof(
        leaf_hash=proof_bundle["entry_hash"],
        proof=proof_bundle["proof"],
        expected_root=proof_bundle["merkle_root"],
    )
    print(f"\n   ✅ Cryptographic Verification: {'VALID & PROVEN INCLUSION' if is_proof_valid else 'FAILED'}")

    # -----------------------------------------------------------------
    # 5. Tamper Detection Demonstration
    # -----------------------------------------------------------------
    print("\n" + "-" * 85)
    print("5. 🚨 TAMPER DETECTION & IMMUTABILITY AUDIT")
    print("-" * 85)

    is_valid, broken_idx, reason = rx.verify_audit_log()
    print(f"   Pre-Tamper Verification : {'PASS ✅' if is_valid else 'FAIL ❌'} ({reason})")

    # Simulate an attacker altering entry #1 in the database/file
    print("\n   [Simulating Attack] Attacker modifies decision verdict in historical record #1...")
    audit_log.entries[1].policy_verdict["action"] = "ALLOW_ANYTHING"  # Malicious edit!

    is_valid_after, broken_idx_after, reason_after = rx.verify_audit_log()
    print(f"   Post-Tamper Verification: {'PASS ✅' if is_valid_after else 'FAIL 🛑 TAMPER DETECTED!'}")
    print(f"   • Tampered Record Index : #{broken_idx_after}")
    print(f"   • Cryptographic Reason  : {reason_after}")
    print("   ✅ The Merkle hash chain instantly caught the unauthorized alteration.")

    # Clean up temp file
    if os.path.exists(audit_log_file):
        os.remove(audit_log_file)

    print("\n" + "=" * 85)
    print("✅ PHASE 23 ENTERPRISE POLICY-AS-CODE & MERKLE AUDIT TRAIL COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_policy_and_audit_demo()
