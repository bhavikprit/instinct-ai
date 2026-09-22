"""
Reflex Policy & Merkle Audit Trail: Enterprise Policy-as-Code & Cryptographic Ledger (Phase 23).
Provides declarative regulatory rule enforcement (HIPAA, GDPR, EU AI Act), data sovereignty geofencing,
append-only SHA-256 hash chaining, O(log N) Merkle inclusion proofs, and tamper detection.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class PolicyAction(str, Enum):
    """Actions resulting from a policy rule evaluation."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    ENFORCE_LOCAL = "ENFORCE_LOCAL"
    REQUIRE_HUMAN = "REQUIRE_HUMAN"
    OVERRIDE = "OVERRIDE"


class PolicyViolationError(Exception):
    """Raised when an operation violates a mandatory enterprise compliance policy."""

    def __init__(self, message: str, rule_id: str, action: PolicyAction, tags: Optional[List[str]] = None):
        super().__init__(message)
        self.rule_id = rule_id
        self.action = action
        self.tags = tags or []


@dataclass
class PolicyRule:
    """
    Declarative rule definition specifying constraints and resulting actions.
    """
    rule_id: str
    action: PolicyAction
    description: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    override_value: Optional[Any] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "action": self.action.value if isinstance(self.action, PolicyAction) else str(self.action),
            "description": self.description,
            "conditions": self.conditions,
            "override_value": self.override_value,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyRule":
        action_val = data.get("action", "ALLOW")
        action = PolicyAction(action_val) if action_val in PolicyAction._value2member_map_ else PolicyAction.ALLOW
        return cls(
            rule_id=data.get("rule_id", "rule_default"),
            action=action,
            description=data.get("description", ""),
            conditions=data.get("conditions", {}),
            override_value=data.get("override_value"),
            tags=data.get("tags", []),
        )


@dataclass
class PolicyVerdict:
    """Outcome of evaluating an entire PolicyRuleSet against an event."""
    action: PolicyAction
    allowed: bool
    matched_rules: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    reason: str = ""
    override_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value if isinstance(self.action, PolicyAction) else str(self.action),
            "allowed": self.allowed,
            "matched_rules": self.matched_rules,
            "violations": self.violations,
            "tags": self.tags,
            "reason": self.reason,
            "override_value": self.override_value,
        }


class PolicyRuleSet:
    """Collection of ordered PolicyRules with default actions."""

    def __init__(
        self,
        rules: Optional[List[PolicyRule]] = None,
        default_action: PolicyAction = PolicyAction.ALLOW,
        name: str = "default_policy",
        version: str = "1.0",
    ):
        self.rules: List[PolicyRule] = rules or []
        self.default_action = default_action
        self.name = name
        self.version = version

    def add_rule(self, rule: PolicyRule):
        self.rules.append(rule)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "default_action": self.default_action.value,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyRuleSet":
        rules = [PolicyRule.from_dict(r) for r in data.get("rules", [])]
        def_action = PolicyAction(data.get("default_action", "ALLOW"))
        return cls(
            rules=rules,
            default_action=def_action,
            name=data.get("name", "custom_policy"),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "PolicyRuleSet":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class PolicyEngine:
    """
    Sub-millisecond Policy-as-Code evaluator.
    Evaluates state, context metadata, and decision outputs against declarative compliance rules.
    """

    def __init__(self, ruleset: Optional[PolicyRuleSet] = None):
        self.ruleset = ruleset or PolicyRuleSet()

    def _get_nested_val(self, target: Dict[str, Any], path: str) -> Any:
        keys = path.split(".")
        curr = target
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            elif hasattr(curr, k):
                curr = getattr(curr, k)
            else:
                return None
        return curr

    def _eval_condition(self, cond: Dict[str, Any], eval_env: Dict[str, Any]) -> bool:
        # Combinators take precedence
        if "all_of" in cond:
            return all(self._eval_condition(c, eval_env) for c in cond["all_of"])
        if "any_of" in cond:
            return any(self._eval_condition(c, eval_env) for c in cond["any_of"])
        if "none_of" in cond:
            return not any(self._eval_condition(c, eval_env) for c in cond["none_of"])

        field_name = cond.get("field")
        op = cond.get("op", "eq").lower()
        target_val = cond.get("value")

        actual_val = self._get_nested_val(eval_env, field_name) if field_name else None

        if op == "exists":
            return actual_val is not None
        if op == "eq":
            return actual_val == target_val
        if op == "neq":
            return actual_val != target_val
        if op == "gt":
            try:
                return float(actual_val) > float(target_val)
            except (TypeError, ValueError):
                return False
        if op == "gte":
            try:
                return float(actual_val) >= float(target_val)
            except (TypeError, ValueError):
                return False
        if op == "lt":
            try:
                return float(actual_val) < float(target_val)
            except (TypeError, ValueError):
                return False
        if op == "lte":
            try:
                return float(actual_val) <= float(target_val)
            except (TypeError, ValueError):
                return False
        if op == "in":
            return actual_val in target_val if target_val is not None else False
        if op == "not_in":
            return actual_val not in target_val if target_val is not None else True
        if op == "contains":
            if isinstance(actual_val, str) and isinstance(target_val, str):
                return target_val.lower() in actual_val.lower()
            elif isinstance(actual_val, (list, tuple, set)):
                return target_val in actual_val
            return False
        if op == "regex":
            if isinstance(actual_val, str) and isinstance(target_val, str):
                return bool(re.search(target_val, actual_val, re.IGNORECASE))
            return False

        return False

    def evaluate(
        self,
        state: str,
        context: Optional[Dict[str, Any]] = None,
        decisions: Optional[Dict[str, Any]] = None,
    ) -> PolicyVerdict:
        """
        Evaluates input state, context, and decisions against all rules.
        Precedence: DENY > ENFORCE_LOCAL > REQUIRE_HUMAN > OVERRIDE > ALLOW.
        """
        eval_env = {
            "state": state,
            "context": context or {},
            "decisions": decisions or {},
        }

        matched_rules: List[str] = []
        violations: List[str] = []
        collected_tags: List[str] = []
        final_action = self.ruleset.default_action
        override_val = None
        reasons: List[str] = []

        # Action severity ranking
        ACTION_PRIORITY = {
            PolicyAction.DENY: 5,
            PolicyAction.ENFORCE_LOCAL: 4,
            PolicyAction.REQUIRE_HUMAN: 3,
            PolicyAction.OVERRIDE: 2,
            PolicyAction.ALLOW: 1,
        }

        for rule in self.ruleset.rules:
            matches = True
            if rule.conditions:
                matches = self._eval_condition(rule.conditions, eval_env)

            if matches:
                matched_rules.append(rule.rule_id)
                collected_tags.extend(rule.tags)
                if rule.description:
                    reasons.append(f"[{rule.rule_id}] {rule.description}")

                if rule.action == PolicyAction.DENY:
                    violations.append(rule.rule_id)

                # Escalate final action if priority is higher
                if ACTION_PRIORITY.get(rule.action, 0) > ACTION_PRIORITY.get(final_action, 0):
                    final_action = rule.action
                    if rule.action == PolicyAction.OVERRIDE:
                        override_val = rule.override_value

        is_allowed = final_action != PolicyAction.DENY
        verdict = PolicyVerdict(
            action=final_action,
            allowed=is_allowed,
            matched_rules=matched_rules,
            violations=violations,
            tags=list(set(collected_tags)),
            reason="; ".join(reasons) if reasons else "Default policy action applied",
            override_value=override_val,
        )
        return verdict


# -------------------------------------------------------------------------
# Cryptographic Merkle Audit Trail
# -------------------------------------------------------------------------

def sha256_hex(data: Union[str, bytes]) -> str:
    """Computes standard SHA-256 hash in hexadecimal."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> str:
    """Deterministic canonical JSON serialization for cryptographic hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class AuditEntry:
    """
    Immutable, cryptographically chained audit record for an AI decision.
    """
    index: int
    timestamp: str
    prev_hash: str
    state_hash: str
    decisions: Dict[str, Any]
    metadata: Dict[str, Any]
    policy_verdict: Dict[str, Any]
    entry_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "state_hash": self.state_hash,
            "decisions": self.decisions,
            "metadata": self.metadata,
            "policy_verdict": self.policy_verdict,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        return cls(
            index=data["index"],
            timestamp=data["timestamp"],
            prev_hash=data["prev_hash"],
            state_hash=data["state_hash"],
            decisions=data.get("decisions", {}),
            metadata=data.get("metadata", {}),
            policy_verdict=data.get("policy_verdict", {}),
            entry_hash=data["entry_hash"],
        )

    def calculate_hash(self) -> str:
        payload = {
            "index": self.index,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "state_hash": self.state_hash,
            "decisions": self.decisions,
            "metadata": self.metadata,
            "policy_verdict": self.policy_verdict,
        }
        return sha256_hex(canonical_json(payload))


class MerkleTree:
    """
    Binary Merkle Tree providing O(log N) inclusion proofs for decision audit records.
    """

    def __init__(self, leaf_hashes: Optional[List[str]] = None):
        self.leaves: List[str] = leaf_hashes or []
        self.levels: List[List[str]] = []
        self._build_tree()

    def _build_tree(self):
        if not self.leaves:
            self.levels = [["0" * 64]]
            return

        curr = list(self.leaves)
        self.levels = [curr]

        while len(curr) > 1:
            next_level: List[str] = []
            for i in range(0, len(curr), 2):
                left = curr[i]
                if i + 1 < len(curr):
                    right = curr[i + 1]
                else:
                    right = left  # Duplicate odd leaf according to standard Merkle tree specification
                parent = sha256_hex(left + right)
                next_level.append(parent)
            self.levels.append(next_level)
            curr = next_level

    @property
    def root(self) -> str:
        """Returns the 64-character SHA-256 Merkle root."""
        if not self.levels or not self.levels[-1]:
            return "0" * 64
        return self.levels[-1][0]

    def get_proof(self, index: int) -> List[Dict[str, str]]:
        """
        Generates an O(log N) Merkle audit proof path for a leaf index.
        Returns: list of {"sibling": <hex>, "position": "left" | "right"}
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of bounds (total leaves: {len(self.leaves)})")

        proof: List[Dict[str, str]] = []
        curr_idx = index

        for level in self.levels[:-1]:
            if curr_idx % 2 == 0:
                # Sibling is on the right
                sib_idx = curr_idx + 1 if curr_idx + 1 < len(level) else curr_idx
                proof.append({"sibling": level[sib_idx], "position": "right"})
            else:
                # Sibling is on the left
                sib_idx = curr_idx - 1
                proof.append({"sibling": level[sib_idx], "position": "left"})
            curr_idx = curr_idx // 2

        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[Dict[str, str]], expected_root: str) -> bool:
        """
        Verifies that a leaf hash belongs to the Merkle tree with root expected_root.
        Independent of the original tree structure.
        """
        current = leaf_hash
        for step in proof:
            sibling = step["sibling"]
            position = step.get("position", "right")
            if position == "right":
                current = sha256_hex(current + sibling)
            else:
                current = sha256_hex(sibling + current)
        return current == expected_root


class MerkleAuditLog:
    """
    Thread-safe, append-only cryptographic ledger with hash-chaining and Merkle root rolling.
    Provides verifiable proof of compliance and instant tamper detection.
    """

    GENESIS_PREV_HASH = "0" * 64

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self.entries: List[AuditEntry] = []
        self._merkle_tree: Optional[MerkleTree] = None

        if self.storage_path and os.path.exists(self.storage_path):
            self._load_from_disk()
        else:
            self._merkle_tree = MerkleTree([])

    def _load_from_disk(self):
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self.entries.append(AuditEntry.from_dict(data))
        leaf_hashes = [e.entry_hash for e in self.entries]
        self._merkle_tree = MerkleTree(leaf_hashes)

    def append(
        self,
        state: str,
        decisions: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        policy_verdict: Optional[Union[PolicyVerdict, Dict[str, Any]]] = None,
    ) -> AuditEntry:
        """
        Appends a new decision entry to the cryptographic hash chain.
        """
        with self._lock:
            idx = len(self.entries)
            prev_hash = self.entries[-1].entry_hash if self.entries else self.GENESIS_PREV_HASH
            state_h = sha256_hex(state)
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            if isinstance(policy_verdict, PolicyVerdict):
                verdict_dict = policy_verdict.to_dict()
            elif isinstance(policy_verdict, dict):
                verdict_dict = policy_verdict
            else:
                verdict_dict = {"action": "ALLOW", "allowed": True}

            entry_payload = {
                "index": idx,
                "timestamp": ts,
                "prev_hash": prev_hash,
                "state_hash": state_h,
                "decisions": decisions,
                "metadata": metadata or {},
                "policy_verdict": verdict_dict,
            }
            entry_h = sha256_hex(canonical_json(entry_payload))

            entry = AuditEntry(
                index=idx,
                timestamp=ts,
                prev_hash=prev_hash,
                state_hash=state_h,
                decisions=decisions,
                metadata=metadata or {},
                policy_verdict=verdict_dict,
                entry_hash=entry_h,
            )

            self.entries.append(entry)
            # Rebuild Merkle Tree
            self._merkle_tree = MerkleTree([e.entry_hash for e in self.entries])

            if self.storage_path:
                with open(self.storage_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

            return entry

    @property
    def root(self) -> str:
        """Returns the current Merkle root of all logged decisions."""
        with self._lock:
            return self._merkle_tree.root if self._merkle_tree else self.GENESIS_PREV_HASH

    def height(self) -> int:
        """Returns the total number of logged audit entries."""
        with self._lock:
            return len(self.entries)

    def prove(self, index: int) -> Dict[str, Any]:
        """
        Generates an audit bundle proving that entry at index was recorded and unmodified.
        """
        with self._lock:
            if index < 0 or index >= len(self.entries):
                raise IndexError(f"Audit entry index {index} out of range [0, {len(self.entries)})")

            entry = self.entries[index]
            proof = self._merkle_tree.get_proof(index)
            return {
                "index": index,
                "entry": entry.to_dict(),
                "entry_hash": entry.entry_hash,
                "merkle_root": self._merkle_tree.root,
                "proof": proof,
            }

    def verify_chain(self) -> Tuple[bool, Optional[int], str]:
        """
        Verifies cryptographic integrity of the entire ledger.
        Returns: (is_valid: bool, broken_index: Optional[int], reason: str)
        """
        with self._lock:
            expected_prev = self.GENESIS_PREV_HASH
            for i, entry in enumerate(self.entries):
                if entry.index != i:
                    return False, i, f"Index sequence mismatch: expected {i}, got {entry.index}"

                if entry.prev_hash != expected_prev:
                    return False, i, f"Hash chain broken at index {i}: prev_hash {entry.prev_hash} != expected {expected_prev}"

                calculated_hash = entry.calculate_hash()
                if entry.entry_hash != calculated_hash:
                    return False, i, f"Tampered entry content at index {i}: hash {entry.entry_hash} != calculated {calculated_hash}"

                expected_prev = entry.entry_hash

            return True, None, "Ledger integrity 100% verified"

    def to_list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self.entries]
