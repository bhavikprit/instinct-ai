"""
Fallback backend using traditional LLMs (OpenAI / Anthropic) with structured output emulation.
"""

from __future__ import annotations
import json
import os
import time
import urllib.request
from typing import Dict, Any

from sys1.backends.base import BaseBackend
from sys1.primitives import PrimitiveType, Noul, Choice, Score, DecisionResult


class FallbackLLMBackend(BaseBackend):
    """
    Fallback emulator for System 1 using OpenAI/Anthropic APIs.
    Translates Reflex primitives into JSON schema.
    Useful for local development or when proprietary decision models are unavailable.
    """

    name: str = "fallback-llm"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def evaluate(self, state: str, questions: Dict[str, PrimitiveType]) -> DecisionResult:
        if not self.api_key:
            # Fallback to deterministic heuristic if no key
            from sys1.backends.local import LocalEngine
            return LocalEngine().evaluate(state, questions)

        start_time = time.perf_counter()

        system_prompt = (
            "You are a machine-native decision engine. Evaluate the user state and return a JSON object "
            "where keys match the requested question IDs. For noul return a float probability [0.0 - 1.0]. "
            "For choice return the chosen string option. For score return a float number."
        )

        schema_props = {}
        for k, q in questions.items():
            if isinstance(q, Noul):
                schema_props[k] = {"type": "number", "description": f"Probability of: {q.instructions}"}
            elif isinstance(q, Choice):
                schema_props[k] = {"type": "string", "enum": q.options, "description": q.instructions}
            elif isinstance(q, Score):
                schema_props[k] = {"type": "number", "description": q.instructions}

        user_content = f"State:\n{state}\n\nQuestions:\n" + "\n".join(
            f"- {k}: {q.instructions}" for k, q in questions.items()
        )

        req_body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "response_format": {"type": "json_object"}
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=req_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        raw_json = json.loads(data["choices"][0]["message"]["content"])
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        resolved: Dict[str, PrimitiveType] = {}
        for k, q in questions.items():
            val = raw_json.get(k)
            if isinstance(q, Noul):
                resolved[k] = q.resolve(float(val) if val is not None else 0.5)
            elif isinstance(q, Choice):
                resolved[k] = q.resolve(str(val) if val else q.options[0])
            elif isinstance(q, Score):
                resolved[k] = q.resolve(float(val) if val is not None else q.min_val)

        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 0.15 + usage.get("completion_tokens", 0) * 0.60) / 1_000_000.0

        return DecisionResult(
            decisions=resolved,
            latency_ms=round(elapsed_ms, 2),
            backend=self.name,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=cost
        )
