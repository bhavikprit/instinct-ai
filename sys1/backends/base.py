"""
BaseBackend interface for Reflex System 1 providers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
from sys1.primitives import PrimitiveType, DecisionResult


class BaseBackend(ABC):
    """Abstract base class for all Reflex System 1 execution backends."""

    name: str = "base"

    @abstractmethod
    def evaluate(self, state: str, questions: Dict[str, PrimitiveType]) -> DecisionResult:
        """
        Evaluates a set of typed primitives against the provided unstructured state.
        Must return a populated DecisionResult with latency and telemetry.
        """
        pass
