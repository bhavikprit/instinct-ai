"""
Unified Reflex Client: The Universal System 1 Runtime Interface.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Union

from reflex.primitives import PrimitiveType, Noul, Choice, Score, DecisionResult
from reflex.backends.base import BaseBackend
from reflex.backends.typesafe import TypeSafeBackend
from reflex.backends.local import LocalEngine
from reflex.backends.fallback import FallbackLLMBackend
from reflex.backends.onnx_engine import ONNXEngine
from reflex.embeddings import PureSemanticEngine
from reflex.cache import InstinctCache
from reflex.telemetry import OpenTelemetryTracer
from reflex.learning import SelfTuningInstinctHead
from reflex.feedback import FeedbackCollector


class Reflex:
    """
    Reflex: The Universal System-1 Runtime & Dual-Brain Gateway.
    
    Usage:
        rx = Reflex(cache=True, learning=True)
        is_scam_prob = rx.noul("Is this a phishing email?", email_text)
        target_tool = rx.choice("Next agent action", ["search", "calculator", "finish"], context)
        
        # Online Active Learning: teach Reflex from System 2 escalations
        rx.teach("Reset my 2FA authentication token", "is_security_incident", ground_truth=True)
    """

    def __init__(
        self,
        backend: Union[str, BaseBackend] = "auto",
        policy: str = "dual-brain",
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        cache: Union[bool, InstinctCache] = False,
        tracer: Optional[OpenTelemetryTracer] = None,
        learning: bool = False,
        instinct_head: Optional[SelfTuningInstinctHead] = None,
        feedback: Optional[FeedbackCollector] = None,
        **backend_kwargs,
    ):
        self.policy = policy
        self.api_key = api_key
        self.model_path = model_path
        self.tracer = tracer

        # Active Learning & Feedback setup
        if instinct_head is not None or learning:
            self.instinct_head = instinct_head or SelfTuningInstinctHead()
        else:
            self.instinct_head = None
        self.feedback = feedback or FeedbackCollector()

        # Cache setup
        if isinstance(cache, InstinctCache):
            self.cache = cache
        elif cache is True:
            self.cache = InstinctCache()
        else:
            self.cache = None

        if isinstance(backend, BaseBackend):
            self.backend = backend
        elif backend in ("native", "c"):
            from reflex.backends.c_engine import NativeCEngine
            self.backend = NativeCEngine(**backend_kwargs)
        elif backend == "local":
            self.backend = LocalEngine()
        elif backend in ("semantic", "embeddings"):
            self.backend = PureSemanticEngine(**backend_kwargs)
        elif backend in ("onnx", "neural"):
            self.backend = ONNXEngine(model_path=model_path, **backend_kwargs)
        elif backend in ("typesafe", "jev"):
            self.backend = TypeSafeBackend(api_key=api_key)
        elif backend == "fallback":
            self.backend = FallbackLLMBackend(api_key=api_key)
        elif backend == "auto":
            # Auto-detection policy
            if os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or api_key:
                self.backend = TypeSafeBackend(api_key=api_key)
            elif model_path and os.path.exists(model_path):
                self.backend = ONNXEngine(model_path=model_path, **backend_kwargs)
            else:
                self.backend = LocalEngine()
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    def evaluate(self, state: str, questions: Dict[str, PrimitiveType]) -> DecisionResult:
        """Evaluates typed questions against state in a single pass."""
        if self.cache is not None:
            cached_res = self.cache.get(state, questions)
            if cached_res is not None:
                if self.tracer is not None:
                    with self.tracer.start_span("reflex.evaluate", attributes={"reflex.cached": True, "reflex.backend": cached_res.backend}):
                        pass
                return cached_res

        if self.tracer is not None:
            with self.tracer.start_span("reflex.evaluate", attributes={"reflex.backend": self.backend.name, "reflex.cached": False}) as span:
                result = self.backend.evaluate(state, questions)
                span.set_attribute("reflex.latency_ms", result.latency_ms)
                span.set_attribute("reflex.cost_usd", result.cost_usd)
        else:
            result = self.backend.evaluate(state, questions)

        # Apply online tuned instinct head predictions if available
        if self.instinct_head is not None:
            for k, q in questions.items():
                if isinstance(q, Noul) and k in self.instinct_head.weights:
                    prob = self.instinct_head.predict_noul(state, question_key=k)
                    result.decisions[k] = q.resolve(prob)
                elif isinstance(q, Choice):
                    prefix = f"{k}::{q.options[0]}" if q.options else ""
                    if any(head_k.startswith(f"{k}::") for head_k in self.instinct_head.weights):
                        selected, dist = self.instinct_head.predict_choice(state, q.options, prefix=k)
                        result.decisions[k] = q.resolve(selected, dist)

        if self.cache is not None:
            self.cache.set(state, questions, result)

        return result

    def teach(
        self,
        state: str,
        question_key: str,
        ground_truth: Union[bool, str, float],
        options: Optional[List[str]] = None,
        lr: float = 0.05,
        source_model: str = "claude-3.5-sonnet",
    ) -> float:
        """
        Online Active Learning: updates local instinct weights from System 2 ground truth.
        Takes <0.05ms and eliminates subsequent escalations.
        """
        if self.instinct_head is None:
            self.instinct_head = SelfTuningInstinctHead()

        self.feedback.record(
            state=state,
            question_key=question_key,
            ground_truth=ground_truth,
            source_model=source_model,
        )

        if isinstance(ground_truth, bool):
            target_val = 1.0 if ground_truth else 0.0
            return self.instinct_head.update_binary(state, target_val, question_key=question_key, lr=lr)
        elif isinstance(ground_truth, str):
            opts = options or [ground_truth]
            return self.instinct_head.update_choice(state, target_option=ground_truth, options=opts, prefix=question_key, lr=lr)
        else:
            return self.instinct_head.update_binary(state, float(ground_truth), question_key=question_key, lr=lr)

    def noul(self, instructions: str, state: str, threshold: float = 0.85) -> float:
        """Direct shortcut returning calibrated probability float [0.0 - 1.0]."""
        res = self.evaluate(state, {"_q": Noul(instructions=instructions, threshold=threshold)})
        return res["_q"].probability or 0.5

    def choice(self, instructions: str, options: List[str], state: str) -> str:
        """Direct shortcut returning the winning option string."""
        res = self.evaluate(state, {"_q": Choice(instructions=instructions, options=options)})
        return res["_q"].selected or (options[0] if options else "")

    def score(self, instructions: str, state: str, min_val: float = 1.0, max_val: float = 10.0) -> float:
        """Direct shortcut returning evaluated numerical score."""
        res = self.evaluate(state, {"_q": Score(instructions=instructions, min_val=min_val, max_val=max_val)})
        return res["_q"].score or min_val
