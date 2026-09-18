"""
Reflex Primitives: The Universal Machine-Native Decision Types.
Standardizing Noul, Choice, and Score across all System 1 backends.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Noul:
    """
    Noul: Probabilistic Boolean primitive.
    Portmanteau of 'no' and 'boolean'.
    Represents calibrated epistemic probability in [0.0, 1.0].
    """
    instructions: str
    probability: Optional[float] = None
    threshold: float = 0.85
    uncertainty_low: float = 0.35
    uncertainty_high: float = 0.65

    def resolve(self, prob: float) -> Noul:
        """Sets evaluated probability."""
        clamped = max(0.0, min(1.0, float(prob)))
        return Noul(
            instructions=self.instructions,
            probability=clamped,
            threshold=self.threshold,
            uncertainty_low=self.uncertainty_low,
            uncertainty_high=self.uncertainty_high
        )

    @property
    def is_true(self) -> bool:
        """Returns True if probability meets or exceeds confident threshold."""
        return self.probability is not None and self.probability >= self.threshold

    @property
    def is_false(self) -> bool:
        """Returns True if probability is definitively low."""
        return self.probability is not None and self.probability <= (1.0 - self.threshold)

    @property
    def is_uncertain(self) -> bool:
        """
        Returns True if the model admits epistemic uncertainty.
        This is the trigger to escalate to System 2 (reasoning model).
        """
        if self.probability is None:
            return True
        return self.uncertainty_low <= self.probability <= self.uncertainty_high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "noul",
            "instructions": self.instructions,
            "probability": self.probability,
            "is_true": self.is_true,
            "is_uncertain": self.is_uncertain
        }


@dataclass
class Choice:
    """
    Choice: Dynamic Rubric & Multi-Class Selection primitive.
    Returns the selected option and Dirichlet-like probability distribution.
    """
    instructions: str
    options: List[str] = field(default_factory=list)
    criteria: Optional[Dict[str, str]] = None
    selected: Optional[str] = None
    distribution: Dict[str, float] = field(default_factory=dict)

    def resolve(self, selected: str, distribution: Optional[Dict[str, float]] = None) -> Choice:
        dist = distribution or {selected: 1.0}
        return Choice(
            instructions=self.instructions,
            options=self.options,
            criteria=self.criteria,
            selected=selected,
            distribution=dist
        )

    def get_prob(self, option: str) -> float:
        return self.distribution.get(option, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "options": self.options,
            "selected": self.selected,
            "distribution": self.distribution
        }


@dataclass
class Score:
    """
    Score: Continuous or Rubric Evaluation primitive.
    """
    instructions: str
    min_val: float = 1.0
    max_val: float = 10.0
    score: Optional[float] = None
    confidence: Optional[float] = None

    def resolve(self, score: float, confidence: Optional[float] = None) -> Score:
        clamped = max(self.min_val, min(self.max_val, float(score)))
        return Score(
            instructions=self.instructions,
            min_val=self.min_val,
            max_val=self.max_val,
            score=clamped,
            confidence=confidence
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "score": self.score,
            "range": [self.min_val, self.max_val],
            "confidence": self.confidence
        }


PrimitiveType = Union[Noul, Choice, Score]


@dataclass
class DecisionResult:
    """
    Container for evaluated decision primitives with telemetry.
    """
    decisions: Dict[str, PrimitiveType]
    latency_ms: float
    backend: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __getitem__(self, key: str) -> PrimitiveType:
        return self.decisions[key]

    def get_noul(self, key: str) -> Noul:
        val = self.decisions.get(key)
        if isinstance(val, Noul):
            return val
        raise KeyError(f"Key '{key}' is not a Noul primitive")

    def get_choice(self, key: str) -> Choice:
        val = self.decisions.get(key)
        if isinstance(val, Choice):
            return val
        raise KeyError(f"Key '{key}' is not a Choice primitive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
            "meta": {
                "latency_ms": self.latency_ms,
                "backend": self.backend,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": self.cost_usd
            }
        }
