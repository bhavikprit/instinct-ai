"""
Reflex Shadow: Autonomous Canary Deployment & Decision Shadowing (Phase 21).
Delivers zero-latency background shadowing, real-time Cohen's Kappa agreement tracking,
dynamic canary traffic splitting, and autonomous safety rollback.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
from collections import defaultdict, deque
import concurrent.futures
import copy
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from reflex.primitives import PrimitiveType, Noul, Choice, Score, DecisionResult


class ShadowStage(str, Enum):
    """Progressive rollout stages for candidate decision models."""
    OBSERVATION = "OBSERVATION"    # 0% live canary traffic, 100% background shadow
    CANARY_10 = "CANARY_10"        # 10% live traffic to challenger
    CANARY_25 = "CANARY_25"        # 25% live traffic to challenger
    CANARY_50 = "CANARY_50"        # 50% live traffic to challenger
    PROMOTED = "PROMOTED"          # 100% live traffic to challenger (promoted to champion)
    ROLLED_BACK = "ROLLED_BACK"    # 0% traffic to challenger (halted due to safety trigger)

    @classmethod
    def from_percentage(cls, pct: float) -> "ShadowStage":
        if pct <= 0.0:
            return cls.OBSERVATION
        if pct <= 15.0:
            return cls.CANARY_10
        if pct <= 35.0:
            return cls.CANARY_25
        if pct < 100.0:
            return cls.CANARY_50
        return cls.PROMOTED


@dataclass
class ShadowConfig:
    """Configuration options for Decision Shadowing and Canary Rollouts."""
    shadow_traffic_pct: float = 100.0       # Percentage of live requests to shadow asynchronously (0.0 - 100.0)
    canary_traffic_pct: float = 0.0         # Percentage of live requests routed to challenger (0.0 - 100.0)
    stage: ShadowStage = ShadowStage.OBSERVATION
    concordance_threshold: float = 0.90     # Minimum agreement rate required for promotion (e.g. 90%)
    min_kappa: float = 0.70                 # Minimum Cohen's Kappa required for promotion
    rollback_threshold: float = 0.80        # Agreement threshold triggering immediate rollback (e.g. 80%)
    min_samples_for_promotion: int = 20     # Minimum samples required before checking promotion
    min_samples_for_rollback: int = 10      # Minimum samples required before checking rollback
    auto_promote: bool = True               # Automatically step up canary stages upon satisfying criteria
    auto_rollback: bool = True              # Automatically drop canary to 0% upon safety violation
    max_workers: int = 4                    # Background worker threads for async shadow execution
    max_history: int = 2000                 # Maximum rolling sample history kept in memory

    def __post_init__(self):
        if self.canary_traffic_pct > 0.0 and self.stage == ShadowStage.OBSERVATION:
            self.stage = ShadowStage.from_percentage(self.canary_traffic_pct)
        elif self.stage != ShadowStage.OBSERVATION and self.canary_traffic_pct == 0.0:
            if self.stage == ShadowStage.CANARY_10:
                self.canary_traffic_pct = 10.0
            elif self.stage == ShadowStage.CANARY_25:
                self.canary_traffic_pct = 25.0
            elif self.stage == ShadowStage.CANARY_50:
                self.canary_traffic_pct = 50.0
            elif self.stage == ShadowStage.PROMOTED:
                self.canary_traffic_pct = 100.0



@dataclass
class ShadowEvaluationRecord:
    """Detailed record of a single shadowed decision pair."""
    timestamp: float
    state: str
    question_keys: List[str]
    champion_decisions: Dict[str, Any]
    challenger_decisions: Dict[str, Any]
    champion_confidences: Dict[str, float]
    challenger_confidences: Dict[str, float]
    concordant: bool
    champion_latency_ms: float
    challenger_latency_ms: float
    served_by: str  # "champion" or "challenger"
    details: Dict[str, bool] = field(default_factory=dict)  # per-question match


class DivergenceTracker:
    """
    Thread-safe real-time statistical analytics engine for decision concordance.
    Computes Concordance Rate, Cohen's Kappa, Confusion Matrices, and Confidence Drift.
    Zero external dependencies (pure Python standard library).
    """

    def __init__(self, max_history: int = 2000):
        self.max_history = max_history
        self._lock = threading.Lock()
        self.total_samples: int = 0
        self.concordant_samples: int = 0
        self.history: deque[ShadowEvaluationRecord] = deque(maxlen=max_history)

        # Disagreement confusion matrices per question_key: champ_val -> chal_val -> count
        self.confusion_matrices: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.confidence_deltas: deque[float] = deque(maxlen=max_history)
        self.champion_latencies: deque[float] = deque(maxlen=max_history)
        self.challenger_latencies: deque[float] = deque(maxlen=max_history)

    def record(self, record: ShadowEvaluationRecord):
        """Thread-safely records a shadowed evaluation pair."""
        with self._lock:
            self.total_samples += 1
            if record.concordant:
                self.concordant_samples += 1

            self.history.append(record)
            self.champion_latencies.append(record.champion_latency_ms)
            self.challenger_latencies.append(record.challenger_latency_ms)

            # Record confusion matrices and confidence deltas
            for qk in record.question_keys:
                c_val = str(record.champion_decisions.get(qk, ""))
                ch_val = str(record.challenger_decisions.get(qk, ""))
                self.confusion_matrices[qk][c_val][ch_val] += 1

                c_conf = record.champion_confidences.get(qk, 0.0)
                ch_conf = record.challenger_confidences.get(qk, 0.0)
                self.confidence_deltas.append(ch_conf - c_conf)

    def concordance_rate(self) -> float:
        """Returns empirical agreement rate between champion and challenger [0.0 - 1.0]."""
        with self._lock:
            if self.total_samples == 0:
                return 1.0
            return round(self.concordant_samples / self.total_samples, 4)

    def cohen_kappa(self, question_key: Optional[str] = None) -> float:
        """
        Computes Cohen's Kappa coefficient (inter-rater reliability accounting for chance).
        Formula: kappa = (p_o - p_e) / (1 - p_e)
        Returns float in [-1.0, 1.0], where 1.0 = perfect agreement, 0.0 = chance agreement.
        """
        with self._lock:
            if not self.confusion_matrices:
                return 1.0

            if question_key is not None:
                matrix = self.confusion_matrices.get(question_key)
                if not matrix:
                    return 1.0
                return self._compute_kappa_from_matrix(matrix)

            # Aggregate across all question keys
            aggregated: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for q_matrix in self.confusion_matrices.values():
                for c_val, chal_dict in q_matrix.items():
                    for ch_val, cnt in chal_dict.items():
                        aggregated[c_val][ch_val] += cnt
            return self._compute_kappa_from_matrix(aggregated)

    def _compute_kappa_from_matrix(self, matrix: Dict[str, Dict[str, int]]) -> float:
        """Calculates Cohen's Kappa from a discrete 2D confusion matrix."""
        # Find all distinct categories
        categories = set(matrix.keys())
        for row in matrix.values():
            categories.update(row.keys())

        if not categories:
            return 1.0

        total_n = 0
        row_totals: Dict[str, int] = defaultdict(int)
        col_totals: Dict[str, int] = defaultdict(int)
        agree_n = 0

        for cat_i in categories:
            for cat_j in categories:
                count = matrix.get(cat_i, {}).get(cat_j, 0)
                total_n += count
                row_totals[cat_i] += count
                col_totals[cat_j] += count
                if cat_i == cat_j:
                    agree_n += count

        if total_n == 0:
            return 1.0

        p_o = agree_n / float(total_n)
        p_e = sum((row_totals[cat] * col_totals[cat]) for cat in categories) / float(total_n * total_n)

        denom = 1.0 - p_e
        if abs(denom) < 1e-9:
            # Complete agreement or degenerate marginal distributions
            return 1.0 if p_o >= 0.999 else 0.0

        kappa = (p_o - p_e) / denom
        return round(max(-1.0, min(1.0, kappa)), 4)

    def mean_confidence_delta(self) -> float:
        """Returns average difference in confidence: Challenger - Champion."""
        with self._lock:
            if not self.confidence_deltas:
                return 0.0
            return round(sum(self.confidence_deltas) / len(self.confidence_deltas), 4)

    def latency_profile(self) -> Dict[str, Dict[str, float]]:
        """Returns P50, P95, and P99 latency percentiles in ms for champion and challenger."""
        with self._lock:
            return {
                "champion": self._calc_percentiles(list(self.champion_latencies)),
                "challenger": self._calc_percentiles(list(self.challenger_latencies)),
            }

    @staticmethod
    def _calc_percentiles(data: List[float]) -> Dict[str, float]:
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        s = sorted(data)
        n = len(s)
        p50 = s[int(n * 0.50)]
        p95 = s[min(n - 1, int(n * 0.95))]
        p99 = s[min(n - 1, int(n * 0.99))]
        mean_val = sum(s) / n
        return {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "mean": round(mean_val, 3),
        }

    def get_confusion_matrix(self, question_key: Optional[str] = None) -> Dict[str, Any]:
        """Returns JSON-serializable confusion matrix representation."""
        with self._lock:
            if question_key is not None:
                raw = self.confusion_matrices.get(question_key, {})
                return {c: dict(chal) for c, chal in raw.items()}
            out = {}
            for qk, mat in self.confusion_matrices.items():
                out[qk] = {c: dict(chal) for c, chal in mat.items()}
            return out

    def reset(self):
        """Resets all metrics and sample history."""
        with self._lock:
            self.total_samples = 0
            self.concordant_samples = 0
            self.history.clear()
            self.confusion_matrices.clear()
            self.confidence_deltas.clear()
            self.champion_latencies.clear()
            self.challenger_latencies.clear()


class DecisionShadowRouter:
    """
    High-performance Autonomous Canary Deployment & Decision Shadowing Router.
    Routes live user traffic to Champion while asynchronously shadowing Challenger candidates
    in background worker threads with zero added latency penalty.
    """

    def __init__(
        self,
        champion: Any,
        challenger: Any,
        config: Optional[ShadowConfig] = None,
    ):
        self.champion = champion
        self.challenger = challenger
        self.config = config or ShadowConfig()
        self.tracker = DivergenceTracker(max_history=self.config.max_history)

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="reflex-shadow-worker",
        )
        self._pending_futures: List[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._incident_log: List[Dict[str, Any]] = []

    def evaluate(
        self,
        state: str,
        questions: Dict[str, PrimitiveType],
        client_key: Optional[str] = None,
    ) -> DecisionResult:
        """
        Evaluates decision request through canary router.
        1. Synchronously evaluates primary candidate (<1ms).
        2. Asynchronously forks request to shadow candidate without blocking primary response.
        3. Returns primary DecisionResult immediately.
        """
        # 1. Determine primary route (Champion vs Canary Challenger)
        use_challenger_as_primary = self._should_route_canary(state, client_key)
        primary = self.challenger if use_challenger_as_primary else self.champion
        shadow = self.champion if use_challenger_as_primary else self.challenger
        served_by = "challenger" if use_challenger_as_primary else "champion"

        # 2. Synchronous evaluation of primary model
        t0 = time.perf_counter()
        primary_res = self._call_model(primary, state, questions)
        primary_latency = (time.perf_counter() - t0) * 1000.0

        # 3. Asynchronous background shadowing
        if self._should_shadow():
            # Clone questions for isolated shadow execution
            q_copy = {k: copy.deepcopy(v) for k, v in questions.items()}
            future = self._executor.submit(
                self._run_shadow_task,
                shadow=shadow,
                state=state,
                questions=q_copy,
                primary_res=primary_res,
                primary_latency=primary_latency,
                served_by=served_by,
            )
            with self._lock:
                self._pending_futures.append(future)
                # Prune completed futures
                if len(self._pending_futures) > 200:
                    self._pending_futures = [f for f in self._pending_futures if not f.done()]

        return primary_res

    def _should_route_canary(self, state: str, client_key: Optional[str]) -> bool:
        """Determines if the request should be served by the challenger canary."""
        if self.config.stage == ShadowStage.ROLLED_BACK:
            return False
        if self.config.stage == ShadowStage.PROMOTED:
            return True
        if self.config.canary_traffic_pct <= 0.0:
            return False
        if self.config.canary_traffic_pct >= 100.0:
            return True

        # Deterministic hashing if client_key or state provided
        hash_seed = client_key if client_key is not None else state
        slot = (abs(hash(hash_seed)) % 10000) / 100.0  # [0.00, 99.99]
        return slot < self.config.canary_traffic_pct

    def _should_shadow(self) -> bool:
        """Determines if the unserved candidate should be shadowed asynchronously."""
        if self.config.stage == ShadowStage.ROLLED_BACK:
            return False
        if self.config.shadow_traffic_pct >= 100.0:
            return True
        if self.config.shadow_traffic_pct <= 0.0:
            return False
        return (random.random() * 100.0) < self.config.shadow_traffic_pct

    def _call_model(self, model: Any, state: str, questions: Dict[str, PrimitiveType]) -> DecisionResult:
        """Invokes a Reflex instance, backend, or callable."""
        if hasattr(model, "_evaluate_direct"):
            return model._evaluate_direct(state, questions)
        elif hasattr(model, "evaluate"):
            try:
                return model.evaluate(state, questions, shadow=False)
            except TypeError:
                return model.evaluate(state, questions)
        elif callable(model):
            return model(state, questions)
        raise TypeError(f"Target model {type(model)} does not implement evaluate(state, questions)")


    def _run_shadow_task(
        self,
        shadow: Any,
        state: str,
        questions: Dict[str, PrimitiveType],
        primary_res: DecisionResult,
        primary_latency: float,
        served_by: str,
    ):
        """Background thread executing shadow candidate and recording divergence metrics."""
        try:
            t0 = time.perf_counter()
            shadow_res = self._call_model(shadow, state, questions)
            shadow_latency = (time.perf_counter() - t0) * 1000.0

            if served_by == "champion":
                champ_res = primary_res
                chal_res = shadow_res
                champ_lat = primary_latency
                chal_lat = shadow_latency
            else:
                champ_res = shadow_res
                chal_res = primary_res
                champ_lat = shadow_latency
                chal_lat = primary_latency

            # Extract decisions and confidences
            champ_decisions: Dict[str, Any] = {}
            chal_decisions: Dict[str, Any] = {}
            champ_confs: Dict[str, float] = {}
            chal_confs: Dict[str, float] = {}
            details: Dict[str, bool] = {}
            all_match = True

            for qk, q_spec in questions.items():
                c_p = champ_res.decisions.get(qk)
                ch_p = chal_res.decisions.get(qk)
                c_val, c_conf = self._extract_value_and_confidence(c_p)
                ch_val, ch_conf = self._extract_value_and_confidence(ch_p)

                champ_decisions[qk] = c_val
                chal_decisions[qk] = ch_val
                champ_confs[qk] = c_conf
                chal_confs[qk] = ch_conf

                is_match = self._values_agree(c_val, ch_val, c_p)
                details[qk] = is_match
                if not is_match:
                    all_match = False

            record = ShadowEvaluationRecord(
                timestamp=time.time(),
                state=state,
                question_keys=list(questions.keys()),
                champion_decisions=champ_decisions,
                challenger_decisions=chal_decisions,
                champion_confidences=champ_confs,
                challenger_confidences=chal_confs,
                concordant=all_match,
                champion_latency_ms=champ_lat,
                challenger_latency_ms=chal_lat,
                served_by=served_by,
                details=details,
            )

            self.tracker.record(record)

            # Evaluate autonomous promotion / rollback gates
            self._check_autonomous_gates()

        except Exception as e:
            # Never raise exceptions from shadow workers that could destabilize runtime
            with self._lock:
                self._incident_log.append({
                    "timestamp": time.time(),
                    "type": "shadow_execution_error",
                    "error": str(e),
                })

    @staticmethod
    def _extract_value_and_confidence(primitive: Optional[PrimitiveType]) -> Tuple[Any, float]:
        if primitive is None:
            return None, 0.0
        if isinstance(primitive, Noul):
            return primitive.is_true, (primitive.confidence if hasattr(primitive, "confidence") else 0.5)
        if isinstance(primitive, Choice):
            return primitive.selected, (primitive.confidence if hasattr(primitive, "confidence") else 1.0)
        if isinstance(primitive, Score):
            return primitive.score, (primitive.confidence or 1.0)
        return str(primitive), 0.0

    @staticmethod
    def _values_agree(val1: Any, val2: Any, prim_type: Optional[PrimitiveType]) -> bool:
        if val1 is None or val2 is None:
            return val1 == val2
        if isinstance(prim_type, Score):
            # Float scores agree within 10% tolerance
            try:
                diff = abs(float(val1) - float(val2))
                span = prim_type.max_val - prim_type.min_val
                return diff <= max(0.5, 0.10 * span)
            except Exception:
                return val1 == val2
        return val1 == val2

    def _check_autonomous_gates(self):
        """Autonomous progression loop: checks auto-rollback and auto-promotion criteria."""
        samples = self.tracker.total_samples
        concordance = self.tracker.concordance_rate()
        kappa = self.tracker.cohen_kappa()

        # 1. Autonomous Auto-Rollback Gate
        if self.config.auto_rollback and self.config.stage not in (ShadowStage.ROLLED_BACK, ShadowStage.OBSERVATION):
            if samples >= self.config.min_samples_for_rollback:
                if concordance < self.config.rollback_threshold:
                    self.rollback(
                        reason=f"Concordance rate {concordance:.3f} dropped below safety threshold {self.config.rollback_threshold:.3f}"
                    )
                    return

        # 2. Autonomous Auto-Promotion Gate
        if self.config.auto_promote and self.config.stage not in (ShadowStage.PROMOTED, ShadowStage.ROLLED_BACK):
            if samples >= self.config.min_samples_for_promotion:
                if concordance >= self.config.concordance_threshold and kappa >= self.config.min_kappa:
                    self._advance_stage()

    def _advance_stage(self):
        """Progressively advances canary deployment to the next traffic tier."""
        current = self.config.stage
        if current == ShadowStage.OBSERVATION:
            self.set_stage(ShadowStage.CANARY_10)
        elif current == ShadowStage.CANARY_10:
            self.set_stage(ShadowStage.CANARY_25)
        elif current == ShadowStage.CANARY_25:
            self.set_stage(ShadowStage.CANARY_50)
        elif current == ShadowStage.CANARY_50:
            self.promote()

    def set_stage(self, stage: Union[str, ShadowStage]):
        """Sets the rollout stage and updates canary traffic percentage accordingly."""
        if isinstance(stage, str):
            stage = ShadowStage(stage)
        with self._lock:
            self.config.stage = stage
            if stage == ShadowStage.OBSERVATION:
                self.config.canary_traffic_pct = 0.0
            elif stage == ShadowStage.CANARY_10:
                self.config.canary_traffic_pct = 10.0
            elif stage == ShadowStage.CANARY_25:
                self.config.canary_traffic_pct = 25.0
            elif stage == ShadowStage.CANARY_50:
                self.config.canary_traffic_pct = 50.0
            elif stage == ShadowStage.PROMOTED:
                self.config.canary_traffic_pct = 100.0
            elif stage == ShadowStage.ROLLED_BACK:
                self.config.canary_traffic_pct = 0.0

    def set_canary_pct(self, pct: float):
        """Sets explicit canary traffic percentage [0.0 - 100.0]."""
        with self._lock:
            clamped = max(0.0, min(100.0, float(pct)))
            self.config.canary_traffic_pct = clamped
            self.config.stage = ShadowStage.from_percentage(clamped)

    def promote(self):
        """Promotes challenger candidate to 100% live traffic."""
        with self._lock:
            self.config.stage = ShadowStage.PROMOTED
            self.config.canary_traffic_pct = 100.0
            self._incident_log.append({
                "timestamp": time.time(),
                "type": "promotion",
                "message": "Challenger promoted to 100% live production traffic.",
            })

    def rollback(self, reason: str = "Manual or automated safety rollback"):
        """Instantly terminates live canary traffic and halts candidate."""
        with self._lock:
            self.config.stage = ShadowStage.ROLLED_BACK
            self.config.canary_traffic_pct = 0.0
            self._incident_log.append({
                "timestamp": time.time(),
                "type": "rollback",
                "reason": reason,
            })

    def flush(self, timeout: float = 5.0):
        """Waits for all pending background shadow tasks to complete."""
        with self._lock:
            pending = list(self._pending_futures)
        if pending:
            concurrent.futures.wait(pending, timeout=timeout)
        with self._lock:
            self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def stats(self) -> Dict[str, Any]:
        """Returns comprehensive JSON-serializable canary status and agreement telemetry."""
        with self._lock:
            concordance = self.tracker.concordance_rate()
            kappa = self.tracker.cohen_kappa()
            mean_conf_delta = self.tracker.mean_confidence_delta()
            latencies = self.tracker.latency_profile()

            return {
                "stage": self.config.stage.value,
                "canary_traffic_pct": self.config.canary_traffic_pct,
                "shadow_traffic_pct": self.config.shadow_traffic_pct,
                "total_shadowed_samples": self.tracker.total_samples,
                "concordant_samples": self.tracker.concordant_samples,
                "concordance_rate": concordance,
                "cohen_kappa": kappa,
                "mean_confidence_delta": mean_conf_delta,
                "concordance_threshold": self.config.concordance_threshold,
                "min_kappa": self.config.min_kappa,
                "rollback_threshold": self.config.rollback_threshold,
                "auto_promote": self.config.auto_promote,
                "auto_rollback": self.config.auto_rollback,
                "latencies_ms": latencies,
                "confusion_matrix": self.tracker.get_confusion_matrix(),
                "recent_incidents": list(self._incident_log[-10:]),
            }

    def shutdown(self, wait: bool = True):
        """Gracefully shuts down background executor pool."""
        self._executor.shutdown(wait=wait)
