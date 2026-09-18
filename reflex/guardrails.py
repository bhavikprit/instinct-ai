"""
Instant Local Guardrails Suite for Reflex.
Zero-dependency, sub-1ms security, PII, and injection protection for LLM loops.
"""

from __future__ import annotations
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class GuardrailResult:
    """Standardized guardrail evaluation verdict."""
    is_safe: bool
    blocked: bool
    risk_score: float  # [0.0 - 1.0]
    category: Optional[str] = None
    reason: Optional[str] = None
    detected_entities: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "blocked": self.blocked,
            "risk_score": round(self.risk_score, 4),
            "category": self.category,
            "reason": self.reason,
            "detected_entities": self.detected_entities,
            "latency_ms": round(self.latency_ms, 3),
        }


class BaseGuardrail(ABC):
    """Abstract base guardrail."""

    name: str = "base"

    @abstractmethod
    def check(self, text: str) -> GuardrailResult:
        """Evaluates input text and returns a GuardrailResult."""
        pass


class PromptInjectionGuardrail(BaseGuardrail):
    """
    Sub-0.5ms semantic and regex detection for jailbreaks, prompt injections, 
    and system prompt extraction attempts.
    """

    name = "prompt_injection"

    # Compiled high-risk injection patterns
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(prior|previous|above)\s+(instructions?|rules?|directives?)", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(prior|previous|above)\s+(instructions?|rules?)", re.IGNORECASE),
        re.compile(r"forget\s+(all\s+)?(prior|previous)\s+rules?", re.IGNORECASE),
        re.compile(r"\byou\s+are\s+now\s+(in\s+)?(dan|developer\s+mode|unrestricted|jailbroken)\b", re.IGNORECASE),
        re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
        re.compile(r"print\s+(your\s+)?(secret\s+)?system\s+(prompt|instructions?)", re.IGNORECASE),
        re.compile(r"repeat\s+the\s+(text|words|instructions)\s+above\s+verbatim", re.IGNORECASE),
        re.compile(r"<\s*\|\s*im_start\s*\|\s*>\s*system", re.IGNORECASE),
        re.compile(r"```\s*system", re.IGNORECASE),
        re.compile(r"\[SYSTEM(?:\s+PROMPT)?\]", re.IGNORECASE),
    ]

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def check(self, text: str) -> GuardrailResult:
        start_time = time.perf_counter()
        matches = []

        for pat in self.INJECTION_PATTERNS:
            found = pat.search(text)
            if found:
                matches.append(found.group(0))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if matches:
            return GuardrailResult(
                is_safe=False,
                blocked=True,
                risk_score=0.98,
                category=self.name,
                reason=f"Detected prompt injection pattern: '{matches[0]}'",
                detected_entities=matches,
                latency_ms=latency_ms,
            )

        return GuardrailResult(
            is_safe=True,
            blocked=False,
            risk_score=0.02,
            category=self.name,
            reason=None,
            latency_ms=latency_ms,
        )


class PIIGuardrail(BaseGuardrail):
    """
    Sub-0.5ms scanner for Personally Identifiable Information (PII) 
    and API credentials (Credit Cards, SSNs, API Keys, Passwords).
    """

    name = "pii_leakage"

    SSN_PATTERN = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
    API_KEY_PATTERNS = [
        (re.compile(r"\bsk-[a-zA-Z0-9]{32,}\b"), "OpenAI API Key"),
        (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS Access Key"),
        (re.compile(r"\bghp_[a-zA-Z0-9]{36}\b"), "GitHub Personal Access Token"),
    ]
    # Visa / MasterCard / Amex formatted or raw 13-19 digits
    CC_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b|\b(?:\d[-\s]?){12,18}\d\b")

    def check(self, text: str) -> GuardrailResult:
        start_time = time.perf_counter()
        detected = []

        # Check API keys
        for pat, key_type in self.API_KEY_PATTERNS:
            if pat.search(text):
                detected.append(f"Secret Credential ({key_type})")

        # Check SSN
        if self.SSN_PATTERN.search(text):
            detected.append("Social Security Number (SSN)")

        # Check Credit Cards with Luhn check
        for match in self.CC_PATTERN.finditer(text):
            raw_digits = re.sub(r"\D", "", match.group(0))
            if self._luhn_verify(raw_digits):
                detected.append("Credit Card Number (Luhn Validated)")
                break

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if detected:
            return GuardrailResult(
                is_safe=False,
                blocked=True,
                risk_score=0.99,
                category=self.name,
                reason=f"PII / Credential leak detected: {', '.join(detected)}",
                detected_entities=detected,
                latency_ms=latency_ms,
            )

        return GuardrailResult(
            is_safe=True,
            blocked=False,
            risk_score=0.01,
            category=self.name,
            reason=None,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _luhn_verify(card_number: str) -> bool:
        """Validates credit card checksum via standard Luhn algorithm."""
        if len(card_number) < 13 or len(card_number) > 19:
            return False
        digits = [int(c) for c in card_number]
        checksum = 0
        reverse_digits = digits[::-1]
        for i, d in enumerate(reverse_digits):
            if i % 2 == 1:
                doubled = d * 2
                checksum += (doubled - 9) if doubled > 9 else doubled
            else:
                checksum += d
        return checksum % 10 == 0


class GuardrailSuite:
    """
    Composite Guardrail orchestrator executing multiple checks in sub-1ms.
    """

    def __init__(self, guardrails: Optional[List[BaseGuardrail]] = None, fail_fast: bool = True):
        self.guardrails = guardrails or [
            PromptInjectionGuardrail(),
            PIIGuardrail(),
        ]
        self.fail_fast = fail_fast

    def check(self, text: str) -> GuardrailResult:
        start_time = time.perf_counter()
        violations = []
        highest_risk = 0.0

        for g in self.guardrails:
            res = g.check(text)
            if res.risk_score > highest_risk:
                highest_risk = res.risk_score

            if not res.is_safe:
                violations.append(res)
                if self.fail_fast:
                    break

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        if violations:
            first_v = violations[0]
            all_entities = []
            for v in violations:
                all_entities.extend(v.detected_entities)

            return GuardrailResult(
                is_safe=False,
                blocked=True,
                risk_score=highest_risk,
                category=first_v.category,
                reason=first_v.reason,
                detected_entities=all_entities,
                latency_ms=total_latency_ms,
            )

        return GuardrailResult(
            is_safe=True,
            blocked=False,
            risk_score=highest_risk,
            category="all_clear",
            reason=None,
            latency_ms=total_latency_ms,
        )
