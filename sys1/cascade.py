"""
Cost-Aware Dual-Brain Cascades & Risk-Budgeted Routing.
Phase 40: Multi-tier model cascade optimization (FrugalML / Cascade; Chen et al., NeurIPS 2020; Wang et al., 2022).
Calibrates sequential routing thresholds to minimize inference cost and latency
while mathematically guaranteeing system-level risk bounds under an enterprise SLA.

Features:
- Multi-tier cascading across arbitrary K model tiers (System-1 Instinct -> Edge SLM -> Frontier LLM).
- Constrained optimization solving for optimal sequential thresholds minimizing expected cost subject to risk budget r*.
- Finite-sample statistical risk upper bounds via Wilson score confidence intervals.
- Pareto cost-risk frontier generation, cost reduction %, and terminal ASCII curve visualization.
- Contextual exploration (epsilon-greedy) and dynamic fallback on tier timeout/error.
- Zero-dependency binary persistence (.reflex-cascade, magic RFCS, 56-byte structured header, CRC32 trailer).
- Thread-safe runtime integration with Reflex client and primitives.
"""

from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
import random
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import zlib

from sys1.reject import binomial_risk_upper_bound


# Binary persistence constants
CASCADE_MAGIC: bytes = b"RFCS"  # Reflex Cascade
CASCADE_VERSION: int = 1
# Header: uint32 version, float64 target_risk, float64 target_quality, float64 calibrated_cost,
# float64 calibrated_risk, uint32 num_tiers, uint32 num_samples, 8s reserved = 56 bytes
CASCADE_HEADER_FORMAT: str = "<IddddII8s"
CASCADE_HEADER_SIZE: int = struct.calcsize(CASCADE_HEADER_FORMAT)
CASCADE_MIN_FILE_SIZE: int = 4 + CASCADE_HEADER_SIZE + 4 + 4  # magic(4) + header(56) + json_len(4) + crc32(4) = 68 bytes


@dataclass
class CascadeTier:
    """
    Specification of a single model tier in the cascade hierarchy.
    Tiers must be ordered by capability and cost (Tier 0 is cheapest/fastest).
    """
    name: str                                    # Descriptive name (e.g. "system1_instinct", "fast_slm", "frontier_llm")
    cost_per_query: float = 0.0                  # Incurred dollar cost per query (e.g. 0.0, 0.0002, 0.030)
    expected_latency_ms: float = 0.1             # Typical latency in milliseconds
    tier_index: int = 0                          # Tier rank (0, 1, ..., K-1)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cost_per_query": float(self.cost_per_query),
            "expected_latency_ms": float(self.expected_latency_ms),
            "tier_index": int(self.tier_index),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CascadeTier":
        return cls(
            name=str(d["name"]),
            cost_per_query=float(d.get("cost_per_query", 0.0)),
            expected_latency_ms=float(d.get("expected_latency_ms", 0.1)),
            tier_index=int(d.get("tier_index", 0)),
            description=str(d.get("description", "")),
        )


@dataclass
class CascadeConfig:
    """
    Configuration parameters for Cost-Aware Cascade Router.
    """
    target_risk: Optional[float] = 0.02          # Maximum acceptable blended error rate (e.g. 0.02 for <=2%)
    target_quality: Optional[float] = None       # Alternative: minimum acceptable blended accuracy (e.g. 0.98)
    confidence_bound: float = 0.05               # Significance level delta (e.g. 0.05 for 95% statistical confidence)
    min_calibration_samples: int = 30            # Minimum calibration samples before activating cascade routing
    exploration_rate: float = 0.02               # Epsilon for exploratory queries routed to higher tiers
    fallback_on_error: bool = True               # Escalate to next tier if lower tier fails/times out
    window_size: int = 1000                      # Maximum rolling calibration samples in memory

    def __post_init__(self) -> None:
        if self.target_risk is not None and not (0.0 < self.target_risk < 1.0):
            raise ValueError(f"target_risk must be in (0, 1), got {self.target_risk}")
        if self.target_quality is not None and not (0.0 < self.target_quality <= 1.0):
            raise ValueError(f"target_quality must be in (0, 1], got {self.target_quality}")
        if not (0.0 < self.confidence_bound < 1.0):
            raise ValueError(f"confidence_bound must be in (0, 1), got {self.confidence_bound}")
        if self.min_calibration_samples < 5:
            raise ValueError(f"min_calibration_samples must be >= 5, got {self.min_calibration_samples}")
        if not (0.0 <= self.exploration_rate < 1.0):
            raise ValueError(f"exploration_rate must be in [0, 1), got {self.exploration_rate}")
        if self.window_size < self.min_calibration_samples:
            raise ValueError("window_size must be >= min_calibration_samples")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_risk": self.target_risk,
            "target_quality": self.target_quality,
            "confidence_bound": self.confidence_bound,
            "min_calibration_samples": self.min_calibration_samples,
            "exploration_rate": self.exploration_rate,
            "fallback_on_error": self.fallback_on_error,
            "window_size": self.window_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CascadeConfig":
        return cls(
            target_risk=float(d["target_risk"]) if d.get("target_risk") is not None else None,
            target_quality=float(d["target_quality"]) if d.get("target_quality") is not None else None,
            confidence_bound=float(d.get("confidence_bound", 0.05)),
            min_calibration_samples=int(d.get("min_calibration_samples", 30)),
            exploration_rate=float(d.get("exploration_rate", 0.02)),
            fallback_on_error=bool(d.get("fallback_on_error", True)),
            window_size=int(d.get("window_size", 1000)),
        )


@dataclass
class CascadeDecision:
    """
    Outcome of a multi-tier cascade routing evaluation.
    """
    selected_tier: str                           # Name of the tier that served the final decision
    tier_index: int                              # 0-indexed rank of accepting tier
    score: float                                 # Confidence score that triggered acceptance
    threshold: float                             # Calibrated threshold applied at accepting tier
    cumulative_cost: float                       # Total dollar cost incurred across all evaluated tiers
    cumulative_latency_ms: float                 # Estimated total latency incurred in ms
    cost_savings_pct: float                      # Percentage savings vs always querying terminal tier
    guaranteed_risk: float                       # Blended risk upper bound under 1 - delta confidence
    trace: List[Dict[str, Any]]                  # Step-by-step trace of evaluations across tiers
    is_exploratory: bool = False                 # True if routing was triggered by epsilon exploration
    reason: str = ""

    def __contains__(self, key: Any) -> bool:
        return key in self.to_dict()

    def __getitem__(self, item: Any) -> Any:
        d = self.to_dict()
        if item in d:
            return d[item]
        return getattr(self, str(item))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_tier": self.selected_tier,
            "tier_index": self.tier_index,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "cumulative_cost": round(self.cumulative_cost, 6),
            "cumulative_latency_ms": round(self.cumulative_latency_ms, 2),
            "cost_savings_pct": round(self.cost_savings_pct, 2),
            "guaranteed_risk": round(self.guaranteed_risk, 4),
            "trace": self.trace,
            "is_exploratory": self.is_exploratory,
            "reason": self.reason,
        }


@dataclass
class CascadeFrontierPoint:
    """
    A single point along the Cost vs Risk Pareto trade-off frontier.
    """
    thresholds: List[float]                      # Threshold vector (theta_0, ..., theta_{K-2})
    expected_cost: float                         # Average cost per query in USD
    empirical_risk: float                        # Empirical blended error rate
    upper_bound_risk: float                      # Statistical 95% risk bound
    coverage_per_tier: Dict[str, float]          # Fraction of traffic accepted at each tier
    cost_reduction_pct: float                    # % cost saved vs terminal tier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": [round(t, 4) for t in self.thresholds],
            "expected_cost": round(self.expected_cost, 6),
            "empirical_risk": round(self.empirical_risk, 4),
            "upper_bound_risk": round(self.upper_bound_risk, 4),
            "coverage_per_tier": {k: round(v, 4) for k, v in self.coverage_per_tier.items()},
            "cost_reduction_pct": round(self.cost_reduction_pct, 2),
        }


class CascadeRouter:
    """
    Cost-Aware Model Cascade Optimizer & Router.
    FrugalML / Cascade (Chen et al., NeurIPS 2020; Wang et al., 2022).

    Given a sequence of candidate tiers with ascending capabilities and costs,
    learns sequential rejection thresholds theta_0, ..., theta_{K-2} that minimize
    total inference cost while guaranteeing system-level risk remains <= r*.
    """

    def __init__(
        self,
        tiers: Optional[List[CascadeTier]] = None,
        config: Optional[CascadeConfig] = None,
    ):
        self.tiers = tiers if tiers is not None else [
            CascadeTier(name="system1_instinct", cost_per_query=0.0, expected_latency_ms=0.05, tier_index=0, description="Reflex Sub-Millisecond Instinct"),
            CascadeTier(name="fast_slm", cost_per_query=0.0004, expected_latency_ms=45.0, tier_index=1, description="Fast Edge / SLM API"),
            CascadeTier(name="frontier_llm", cost_per_query=0.0300, expected_latency_ms=1200.0, tier_index=2, description="Frontier Heavy Reasoning Model"),
        ]
        self._validate_tiers(self.tiers)

        self.config = config or CascadeConfig()
        self._lock = threading.RLock()

        # Calibration buffer: list of (scores_dict, errors_dict)
        # scores_dict: {tier_idx: confidence_score}
        # errors_dict: {tier_idx: is_error (0 or 1)}
        self._samples: deque = deque(maxlen=self.config.window_size)
        self._total_samples: int = 0

        # Calibrated threshold vector theta_0, ..., theta_{K-2}
        # Default: 0.85 for all non-terminal tiers
        self._thresholds: List[float] = [0.85] * (len(self.tiers) - 1)
        self._calibrated_cost: float = self.tiers[-1].cost_per_query
        self._calibrated_empirical_risk: float = 0.0
        self._calibrated_upper_risk: float = 1.0
        self._calibrated_tier_shares: Dict[str, float] = {t.name: (1.0 if i == len(self.tiers) - 1 else 0.0) for i, t in enumerate(self.tiers)}
        self._calibrated: bool = False

        # Live performance tracking
        self._total_queries_routed: int = 0
        self._total_cost_incurred: float = 0.0
        self._total_baseline_cost: float = 0.0
        self._tier_query_counts: Dict[str, int] = {t.name: 0 for t in self.tiers}

    def _validate_tiers(self, tiers: List[CascadeTier]) -> None:
        if len(tiers) < 2:
            raise ValueError(f"CascadeRouter requires at least 2 tiers, got {len(tiers)}")
        # Check tier indices are 0, 1, ..., K-1
        for i, t in enumerate(tiers):
            if t.tier_index != i:
                t.tier_index = i
        # Check costs are non-decreasing
        for i in range(len(tiers) - 1):
            if tiers[i].cost_per_query > tiers[i + 1].cost_per_query:
                raise ValueError(
                    f"Cascade tiers must have non-decreasing costs: tier {tiers[i].name} (${tiers[i].cost_per_query}) > tier {tiers[i+1].name} (${tiers[i+1].cost_per_query})"
                )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._calibrated and (len(self._samples) >= self.config.min_calibration_samples)

    @property
    def num_tiers(self) -> int:
        return len(self.tiers)

    @property
    def thresholds(self) -> List[float]:
        with self._lock:
            return list(self._thresholds)

    @property
    def num_calibration_samples(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    @property
    def calibrated_cost(self) -> float:
        with self._lock:
            return self._calibrated_cost

    @property
    def calibrated_empirical_risk(self) -> float:
        with self._lock:
            return self._calibrated_empirical_risk

    @property
    def calibrated_upper_risk(self) -> float:
        with self._lock:
            return self._calibrated_upper_risk

    @property
    def calibrated_tier_shares(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._calibrated_tier_shares)

    @property
    def total_cost_saved_usd(self) -> float:
        with self._lock:
            return max(0.0, self._total_baseline_cost - self._total_cost_incurred)

    @property
    def overall_savings_pct(self) -> float:
        with self._lock:
            if self._total_baseline_cost <= 0.0:
                return 0.0
            return (self.total_cost_saved_usd / self._total_baseline_cost) * 100.0

    # -------------------------------------------------------------------------
    # Sample Ingestion & Calibration
    # -------------------------------------------------------------------------

    def add_sample(
        self,
        tier_scores: Dict[int, float],
        tier_errors: Dict[int, Union[bool, int]],
    ) -> None:
        """
        Adds a single calibration query record containing confidence scores and ground truth
        error indicators across candidate tiers.
        tier_scores: {tier_idx: confidence_score in [0, 1]}
        tier_errors: {tier_idx: 1 if error, 0 if correct}
        """
        with self._lock:
            clean_scores = {int(k): float(v) for k, v in tier_scores.items()}
            clean_errors = {int(k): (1 if (v is True or v == 1) else 0) for k, v in tier_errors.items()}
            self._samples.append((clean_scores, clean_errors))
            self._total_samples += 1

    add_calibration_sample = add_sample

    def fit(
        self,
        samples: Sequence[Tuple[Dict[int, float], Dict[int, Union[bool, int]]]],
    ) -> "CascadeRouter":
        """
        Fits or resets the calibration buffer with a batch of tier samples,
        then automatically executes threshold calibration.
        """
        with self._lock:
            self._samples.clear()
            for sc, er in samples:
                self.add_sample(sc, er)
            self.calibrate()
        return self

    def _simulate_cascade(
        self,
        thresholds: Sequence[float],
        samples: Sequence[Tuple[Dict[int, float], Dict[int, int]]],
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Simulates cascade policy with given thresholds over a sample set.
        Returns: (average_cost, empirical_risk, tier_shares)
        """
        num_tiers = len(self.tiers)
        num_non_terminal = num_tiers - 1
        n = len(samples)
        if n == 0:
            return 0.0, 0.0, {t.name: 0.0 for t in self.tiers}

        total_cost = 0.0
        total_errors = 0
        tier_counts = [0] * num_tiers

        for scores, errors in samples:
            accepted_tier = num_tiers - 1  # default to terminal
            cost_accum = 0.0

            for k in range(num_non_terminal):
                cost_accum += self.tiers[k].cost_per_query
                th = thresholds[k]
                sc = scores.get(k, 0.0)
                if sc >= th:
                    accepted_tier = k
                    break

            if accepted_tier == num_tiers - 1:
                cost_accum += self.tiers[-1].cost_per_query

            total_cost += cost_accum
            total_errors += errors.get(accepted_tier, 0)
            tier_counts[accepted_tier] += 1

        avg_cost = total_cost / float(n)
        emp_risk = float(total_errors) / float(n)
        shares = {self.tiers[i].name: float(tier_counts[i]) / float(n) for i in range(num_tiers)}
        return avg_cost, emp_risk, shares

    def calibrate(self) -> List[float]:
        """
        Solves the constrained optimization problem to find optimal sequential thresholds
        (theta_0, ..., theta_{K-2}) that minimize expected cost subject to the risk constraint.
        Uses discrete grid search over empirical candidate percentiles.
        """
        with self._lock:
            if len(self._samples) < self.config.min_calibration_samples:
                # Retain defaults if not enough samples
                return list(self._thresholds)

            samples = list(self._samples)
            n = len(samples)
            num_tiers = len(self.tiers)
            num_non_terminal = num_tiers - 1
            delta = self.config.confidence_bound
            target_r = self.config.target_risk if self.config.target_risk is not None else 0.02
            if self.config.target_quality is not None:
                target_r = min(target_r, 1.0 - self.config.target_quality)

            # Extract candidate score thresholds for each non-terminal tier
            grid_candidates: List[List[float]] = []
            for k in range(num_non_terminal):
                tier_k_scores = sorted(list(set(sc.get(k, 0.5) for sc, _ in samples)))
                if not tier_k_scores:
                    tier_k_scores = [0.85]
                # Downsample candidate thresholds for efficiency
                candidate_steps = 15
                if len(tier_k_scores) > candidate_steps:
                    step = len(tier_k_scores) / float(candidate_steps)
                    cands = [tier_k_scores[int(i * step)] for i in range(candidate_steps)]
                    if tier_k_scores[-1] not in cands:
                        cands.append(tier_k_scores[-1])
                else:
                    cands = tier_k_scores
                grid_candidates.append(cands)

            # Generate cartesian product or coordinate-wise search
            # For 2-tier (1 threshold): simple 1D sweep
            # For 3-tier (2 thresholds): 2D grid sweep (15 x 15 = 225 combos)
            # For K >= 4: coordinate descent
            combos: List[List[float]] = []
            if num_non_terminal == 1:
                combos = [[c] for c in grid_candidates[0]]
            elif num_non_terminal == 2:
                for c0 in grid_candidates[0]:
                    for c1 in grid_candidates[1]:
                        combos.append([c0, c1])
            else:
                # Multi-tier coordinate descent initialization
                base = [0.90] * num_non_terminal
                combos.append(list(base))
                for k in range(num_non_terminal):
                    for c in grid_candidates[k]:
                        variant = list(base)
                        variant[k] = c
                        combos.append(variant)

            best_thresholds = combos[-1]
            best_cost = float("inf")
            best_emp_risk = 1.0
            best_upper_risk = 1.0
            best_shares = {t.name: 0.0 for t in self.tiers}
            found_feasible = False

            # Evaluate each threshold combination
            for candidate_th in combos:
                avg_c, emp_r, shares = self._simulate_cascade(candidate_th, samples)
                # Compute statistical upper bound on error
                total_errors = int(round(emp_r * n))
                up_r = binomial_risk_upper_bound(total_errors, n, delta=delta)

                if up_r <= target_r:
                    # Feasible under enterprise SLA!
                    found_feasible = True
                    if avg_c < best_cost:
                        best_cost = avg_c
                        best_thresholds = candidate_th
                        best_emp_risk = emp_r
                        best_upper_risk = up_r
                        best_shares = shares

            if not found_feasible:
                # If target risk bound cannot be met strictly, select the most conservative thresholds
                # (highest thresholds -> maximum queries sent to frontier model)
                candidate_th = [max(grid_candidates[k]) for k in range(num_non_terminal)]
                avg_c, emp_r, shares = self._simulate_cascade(candidate_th, samples)
                total_errors = int(round(emp_r * n))
                best_cost = avg_c
                best_thresholds = candidate_th
                best_emp_risk = emp_r
                best_upper_risk = binomial_risk_upper_bound(total_errors, n, delta=delta)
                best_shares = shares

            self._thresholds = list(best_thresholds)
            self._calibrated_cost = best_cost
            self._calibrated_empirical_risk = best_emp_risk
            self._calibrated_upper_risk = best_upper_risk
            self._calibrated_tier_shares = best_shares
            self._calibrated = True

            return list(self._thresholds)

    # -------------------------------------------------------------------------
    # Real-Time Query Routing
    # -------------------------------------------------------------------------

    def route(
        self,
        query: Any,
        score_provider: Callable[[int, Any], float],
        fallback_errors: Optional[Sequence[int]] = None,
    ) -> CascadeDecision:
        """
        Executes sequential cost-aware routing for a single query.
        score_provider: callable(tier_index: int, query: Any) -> confidence_score in [0.0, 1.0].
        fallback_errors: optional list of tier indices experiencing operational errors/timeouts.
        """
        with self._lock:
            num_tiers = len(self.tiers)
            num_non_terminal = num_tiers - 1
            terminal_tier = self.tiers[-1]
            failed_tiers = set(fallback_errors) if fallback_errors else set()

            # Contextual epsilon exploration
            is_exploring = (random.random() < self.config.exploration_rate) if self.config.exploration_rate > 0.0 else False

            trace: List[Dict[str, Any]] = []
            selected_tier_idx = num_tiers - 1
            accepted_score = 0.0
            accepted_th = 0.0
            accum_cost = 0.0
            accum_lat = 0.0

            if is_exploring:
                # Route query directly to highest available tier for exploration / drift tracking
                selected_tier_idx = num_tiers - 1
                accum_cost = terminal_tier.cost_per_query
                accum_lat = terminal_tier.expected_latency_ms
                accepted_score = 1.0
                accepted_th = 0.0
                trace.append({
                    "tier": terminal_tier.name,
                    "tier_index": terminal_tier.tier_index,
                    "score": 1.0,
                    "threshold": 0.0,
                    "action": "EXPLORE",
                })
            else:
                routed = False
                for k in range(num_non_terminal):
                    tier_k = self.tiers[k]
                    accum_cost += tier_k.cost_per_query
                    accum_lat += tier_k.expected_latency_ms

                    if k in failed_tiers:
                        trace.append({
                            "tier": tier_k.name,
                            "tier_index": tier_k.tier_index,
                            "score": 0.0,
                            "threshold": self._thresholds[k] if k < len(self._thresholds) else 0.85,
                            "action": "FALLBACK_ERROR",
                        })
                        continue

                    try:
                        sc = float(score_provider(k, query))
                    except Exception as e:
                        if self.config.fallback_on_error:
                            trace.append({
                                "tier": tier_k.name,
                                "tier_index": tier_k.tier_index,
                                "error": str(e),
                                "action": "FALLBACK_EXCEPTION",
                            })
                            continue
                        raise

                    th = self._thresholds[k] if k < len(self._thresholds) else 0.85
                    if sc >= th:
                        # Confidence meets threshold: accept at tier k!
                        selected_tier_idx = k
                        accepted_score = sc
                        accepted_th = th
                        routed = True
                        trace.append({
                            "tier": tier_k.name,
                            "tier_index": tier_k.tier_index,
                            "score": round(sc, 4),
                            "threshold": round(th, 4),
                            "action": "ACCEPT",
                        })
                        break
                    else:
                        trace.append({
                            "tier": tier_k.name,
                            "tier_index": tier_k.tier_index,
                            "score": round(sc, 4),
                            "threshold": round(th, 4),
                            "action": "ESCALATE",
                        })

                if not routed:
                    # Escalated through all non-terminal tiers: served by terminal tier
                    selected_tier_idx = num_tiers - 1
                    accum_cost += terminal_tier.cost_per_query
                    accum_lat += terminal_tier.expected_latency_ms
                    accepted_score = 1.0
                    accepted_th = 0.0
                    trace.append({
                        "tier": terminal_tier.name,
                        "tier_index": terminal_tier.tier_index,
                        "score": 1.0,
                        "threshold": 0.0,
                        "action": "TERMINAL_FALLBACK",
                    })

            selected_tier = self.tiers[selected_tier_idx]
            frontier_cost = terminal_tier.cost_per_query
            savings_pct = max(0.0, (1.0 - (accum_cost / frontier_cost)) * 100.0) if frontier_cost > 0.0 else 0.0

            # Update live metrics
            self._total_queries_routed += 1
            self._total_cost_incurred += accum_cost
            self._total_baseline_cost += frontier_cost
            self._tier_query_counts[selected_tier.name] = self._tier_query_counts.get(selected_tier.name, 0) + 1

            reason = (
                f"Accepted at {selected_tier.name} (score {accepted_score:.4f} >= threshold {accepted_th:.4f})"
                if selected_tier_idx < num_non_terminal
                else f"Escalated to terminal tier {terminal_tier.name}"
            )
            if is_exploring:
                reason = f"Exploration query routed to {terminal_tier.name}"

            return CascadeDecision(
                selected_tier=selected_tier.name,
                tier_index=selected_tier_idx,
                score=accepted_score,
                threshold=accepted_th,
                cumulative_cost=accum_cost,
                cumulative_latency_ms=accum_lat,
                cost_savings_pct=savings_pct,
                guaranteed_risk=self._calibrated_upper_risk,
                trace=trace,
                is_exploratory=is_exploring,
                reason=reason,
            )

    # -------------------------------------------------------------------------
    # Pareto Frontier & Visualization
    # -------------------------------------------------------------------------

    def cost_risk_frontier(self, num_points: int = 15) -> List[CascadeFrontierPoint]:
        """
        Computes points along the Pareto Cost-Risk trade-off frontier.
        """
        with self._lock:
            samples = list(self._samples)
            if len(samples) < 5:
                return []

            frontier: List[CascadeFrontierPoint] = []
            terminal_cost = self.tiers[-1].cost_per_query
            delta = self.config.confidence_bound
            n = len(samples)

            # Sweep target risks from 0.005 to 0.30
            step = 0.295 / float(max(1, num_points - 1))
            for i in range(num_points):
                r_target = 0.005 + i * step
                cfg_temp = CascadeConfig(target_risk=r_target, confidence_bound=delta, min_calibration_samples=5)
                router_temp = CascadeRouter(tiers=self.tiers, config=cfg_temp)
                router_temp._samples = self._samples
                th = router_temp.calibrate()

                avg_c, emp_r, shares = router_temp._simulate_cascade(th, samples)
                total_err = int(round(emp_r * n))
                up_r = binomial_risk_upper_bound(total_err, n, delta=delta)
                savings = max(0.0, (1.0 - (avg_c / terminal_cost)) * 100.0) if terminal_cost > 0.0 else 0.0

                frontier.append(CascadeFrontierPoint(
                    thresholds=th,
                    expected_cost=avg_c,
                    empirical_risk=emp_r,
                    upper_bound_risk=up_r,
                    coverage_per_tier=shares,
                    cost_reduction_pct=savings,
                ))

            # Deduplicate and sort by expected cost ascending
            unique_frontier: List[CascadeFrontierPoint] = []
            seen_costs = set()
            for pt in sorted(frontier, key=lambda p: p.expected_cost):
                c_key = round(pt.expected_cost, 6)
                if c_key not in seen_costs:
                    seen_costs.add(c_key)
                    unique_frontier.append(pt)

            return unique_frontier

    def ascii_cost_risk_frontier(self, num_points: int = 10) -> str:
        """
        Renders an ASCII visualization of the Cost vs Risk Pareto frontier.
        """
        pts = self.cost_risk_frontier(num_points=num_points)
        if not pts:
            return "No calibration data available for Cascade Pareto frontier."

        lines = [
            f"{'Avg Cost ($)':<14} {'Cost Saved %':<14} {'Emp Risk':<12} {'Upper (95%)':<14} {'Cost vs Frontier Visual'}",
            "-" * 85,
        ]
        terminal_cost = self.tiers[-1].cost_per_query
        for p in pts:
            ratio = (p.expected_cost / terminal_cost) if terminal_cost > 0.0 else 0.0
            bar_len = min(25, int(ratio * 25))
            vis = f"|{'█' * bar_len}{' ' * (25 - bar_len)}| ({ratio * 100:5.1f}% of max)"
            lines.append(
                f"${p.expected_cost:<13.6f} {p.cost_reduction_pct:<13.1f}% {p.empirical_risk * 100:<10.2f}% {p.upper_bound_risk * 100:<12.2f}% {vis}"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Zero-Dependency Binary Persistence (.reflex-cascade)
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serializes CascadeRouter configuration, tiers, thresholds, and samples
        to a .reflex-cascade binary file with 32-bit CRC32 checksum verification.
        Uses atomic file replacement.
        """
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with self._lock:
            payload = {
                "config": self.config.to_dict(),
                "tiers": [t.to_dict() for t in self.tiers],
                "thresholds": self._thresholds,
                "calibrated_cost": self._calibrated_cost,
                "calibrated_empirical_risk": self._calibrated_empirical_risk,
                "calibrated_upper_risk": self._calibrated_upper_risk,
                "calibrated_tier_shares": self._calibrated_tier_shares,
                "samples": list(self._samples),
                "total_queries_routed": self._total_queries_routed,
                "total_cost_incurred": self._total_cost_incurred,
                "total_baseline_cost": self._total_baseline_cost,
                "tier_query_counts": self._tier_query_counts,
                "saved_at": time.time(),
            }
            json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            json_len = len(json_bytes)

            t_risk = float(self.config.target_risk) if self.config.target_risk is not None else -1.0
            t_qual = float(self.config.target_quality) if self.config.target_quality is not None else -1.0

            header = struct.pack(
                CASCADE_HEADER_FORMAT,
                CASCADE_VERSION,
                t_risk,
                t_qual,
                float(self._calibrated_cost),
                float(self._calibrated_empirical_risk),
                int(len(self.tiers)),
                int(self._total_samples),
                b"\x00" * 8,
            )

            body = bytearray()
            body.extend(CASCADE_MAGIC)
            body.extend(header)
            body.extend(struct.pack("<I", json_len))
            body.extend(json_bytes)

            checksum = zlib.crc32(body) & 0xFFFFFFFF
            body.extend(struct.pack("<I", checksum))

        tmp_path = f"{path}.tmp.{os.getpid()}_{int(time.time() * 1000)}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @classmethod
    def load(cls, path: str) -> "CascadeRouter":
        """
        Loads and validates a CascadeRouter from a .reflex-cascade file.
        Verifies magic bytes, header structure, and 32-bit CRC32 checksum.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cascade model file not found: {path}")

        file_size = os.path.getsize(path)
        if file_size < CASCADE_MIN_FILE_SIZE:
            raise ValueError(
                f"Invalid .reflex-cascade file: size {file_size} bytes below minimum {CASCADE_MIN_FILE_SIZE} bytes"
            )

        with open(path, "rb") as f:
            data = f.read()

        magic = data[:4]
        if magic != CASCADE_MAGIC:
            raise ValueError(f"Invalid magic identifier: expected {CASCADE_MAGIC!r}, got {magic!r}")

        # Verify CRC32 checksum
        body = data[:-4]
        expected_crc = struct.unpack("<I", data[-4:])[0]
        actual_crc = zlib.crc32(body) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"CRC32 checksum mismatch: calculated {actual_crc:#x} != recorded {expected_crc:#x}")

        header_bytes = data[4:4 + CASCADE_HEADER_SIZE]
        version, t_risk, t_qual, cal_cost, cal_risk, num_tiers, num_samples, _ = struct.unpack(
            CASCADE_HEADER_FORMAT, header_bytes
        )

        json_len = struct.unpack("<I", data[4 + CASCADE_HEADER_SIZE:4 + CASCADE_HEADER_SIZE + 4])[0]
        json_offset = 4 + CASCADE_HEADER_SIZE + 4
        json_bytes = data[json_offset:json_offset + json_len]
        payload = json.loads(json_bytes.decode("utf-8"))

        tiers = [CascadeTier.from_dict(t) for t in payload.get("tiers", [])]
        config = CascadeConfig.from_dict(payload.get("config", {}))
        router = cls(tiers=tiers, config=config)

        router._thresholds = [float(x) for x in payload.get("thresholds", [])]
        router._calibrated_cost = float(payload.get("calibrated_cost", cal_cost))
        router._calibrated_empirical_risk = float(payload.get("calibrated_empirical_risk", cal_risk))
        router._calibrated_upper_risk = float(payload.get("calibrated_upper_risk", 1.0))
        router._calibrated_tier_shares = payload.get("calibrated_tier_shares", {})
        router._total_samples = int(num_samples)
        router._calibrated = True

        for s in payload.get("samples", []):
            sc_dict = {int(k): float(v) for k, v in s[0].items()}
            er_dict = {int(k): int(v) for k, v in s[1].items()}
            router._samples.append((sc_dict, er_dict))

        router._total_queries_routed = int(payload.get("total_queries_routed", 0))
        router._total_cost_incurred = float(payload.get("total_cost_incurred", 0.0))
        router._total_baseline_cost = float(payload.get("total_baseline_cost", 0.0))
        router._tier_query_counts = payload.get("tier_query_counts", {})

        return router
