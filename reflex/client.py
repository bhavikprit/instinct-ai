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


class Reflex:
    """
    Reflex: The Universal System-1 Runtime & Dual-Brain Gateway.
    
    Usage:
        rx = Reflex()
        
        # 1. Multi-primitive evaluation
        result = rx.evaluate(
            state="Incoming email: I need a refund for my order #1234",
            questions={
                "is_refund": Noul("Does the customer demand a refund?"),
                "queue": Choice("Route to queue", options=["billing", "support", "sales"])
            }
        )
        
        # 2. Direct inline shortcuts
        is_scam_prob = rx.noul("Is this a phishing email?", email_text)
        target_tool = rx.choice("Next agent action", ["search", "calculator", "finish"], context)
    """

    def __init__(
        self,
        backend: Union[str, BaseBackend] = "auto",
        policy: str = "dual-brain",
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        **backend_kwargs,
    ):
        self.policy = policy
        self.api_key = api_key
        self.model_path = model_path

        if isinstance(backend, BaseBackend):
            self.backend = backend
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
        return self.backend.evaluate(state, questions)

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
