"""
TypeSafe AI & OpenRouter Jev backend implementation.
"""

from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.error
from typing import Dict, Any

from reflex.backends.base import BaseBackend
from reflex.primitives import PrimitiveType, Noul, Choice, Score, DecisionResult


class TypeSafeBackend(BaseBackend):
    """
    Client for TypeSafe AI Jev System 1 models.
    Supports native API and OpenRouter (~typesafe/jev-latest).
    """

    name: str = "typesafe"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = "typesafe/jev-latest"
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.endpoint = endpoint or os.environ.get(
            "TYPESAFE_ENDPOINT", "https://api.typesafe.ai/v1/systemone"
        )
        self.model = model

    def evaluate(self, state: str, questions: Dict[str, PrimitiveType]) -> DecisionResult:
        if not self.api_key:
            raise ValueError(
                "TypeSafe API key missing. Set TYPESAFE_API_KEY or pass api_key to Reflex."
            )

        start_time = time.perf_counter()

        # Format questions schema
        schema_questions: Dict[str, Any] = {}
        for key, q in questions.items():
            if isinstance(q, Noul):
                schema_questions[key] = {
                    "type": "noul",
                    "instructions": q.instructions
                }
            elif isinstance(q, Choice):
                payload = {
                    "type": "choice",
                    "instructions": q.instructions,
                    "options": q.options
                }
                if q.criteria:
                    payload["criteria"] = q.criteria
                schema_questions[key] = payload
            elif isinstance(q, Score):
                schema_questions[key] = {
                    "type": "score",
                    "instructions": q.instructions,
                    "range": [q.min_val, q.max_val]
                }

        payload_bytes = json.dumps({
            "model": self.model,
            "state": state,
            "questions": schema_questions
        }).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=payload_bytes,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Reflex-AI/0.1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"TypeSafe API request failed: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Parse decisions
        resolved_decisions: Dict[str, PrimitiveType] = {}
        raw_decisions = resp_data.get("decisions", resp_data.get("results", {}))

        for key, q in questions.items():
            raw = raw_decisions.get(key, {})
            if isinstance(q, Noul):
                prob = raw.get("probability", raw.get("noul", 0.5))
                resolved_decisions[key] = q.resolve(prob)
            elif isinstance(q, Choice):
                selected = raw.get("selected", raw.get("choice", q.options[0] if q.options else ""))
                dist = raw.get("distribution", {})
                resolved_decisions[key] = q.resolve(selected, dist)
            elif isinstance(q, Score):
                val = raw.get("score", q.min_val)
                conf = raw.get("confidence")
                resolved_decisions[key] = q.resolve(val, conf)

        # Telemetry
        tokens_in = resp_data.get("usage", {}).get("prompt_tokens", len(state) // 4)
        tokens_out = 0  # Jev output tokens are free
        cost = (tokens_in * 0.042) / 1_000_000.0

        return DecisionResult(
            decisions=resolved_decisions,
            latency_ms=round(elapsed_ms, 2),
            backend=self.name,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost
        )
