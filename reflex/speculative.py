"""
Reflex Speculative: Decision-Level Speculative Execution & Parallel Pre-Fetch (Phase 22).
Enables sub-millisecond System-1 intention prediction and parallel idempotent pre-fetching
while heavy reasoning models generate tokens, reducing agent tool latency to 0ms.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
import asyncio
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum
import inspect
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from reflex.client import Reflex

from reflex.primitives import Choice, DecisionResult




class SpeculativeStatus(str, Enum):
    """Execution status for a speculative decision session."""
    IDLE = "IDLE"
    IN_FLIGHT = "IN_FLIGHT"
    HIT = "HIT"
    MISS = "MISS"
    SKIPPED = "SKIPPED"
    ABORTED = "ABORTED"


@dataclass
class SpeculativeAction:
    """
    Representation of a tool or operation eligible for speculative pre-fetching.
    """
    name: str
    handler: Callable[..., Any]
    description: str = ""
    idempotent: bool = True  # Safety flag: non-idempotent actions will NEVER be pre-fetched
    extractor: Optional[Callable[[str], Dict[str, Any]]] = None
    timeout: float = 5.0


class SpeculativeMetrics:
    """
    Thread-safe telemetry tracking speculative hit rates, time saved, and aborts.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.speculations_launched: int = 0
        self.speculative_hits: int = 0
        self.speculative_misses: int = 0
        self.speculative_skips: int = 0
        self.speculative_aborts: int = 0
        self.total_latency_saved_ms: float = 0.0
        self.action_hits: Dict[str, int] = defaultdict(int)
        self.action_misses: Dict[str, int] = defaultdict(int)

    def record_launch(self, action_name: str):
        with self._lock:
            self.total_requests += 1
            self.speculations_launched += 1

    def record_skip(self):
        with self._lock:
            self.total_requests += 1
            self.speculative_skips += 1

    def record_hit(self, action_name: str, latency_saved_ms: float):
        with self._lock:
            self.speculative_hits += 1
            self.total_latency_saved_ms += max(0.0, latency_saved_ms)
            self.action_hits[action_name] += 1

    def record_miss(self, predicted_action: str, confirmed_action: str):
        with self._lock:
            self.speculative_misses += 1
            self.action_misses[predicted_action] += 1

    def record_abort(self, action_name: str):
        with self._lock:
            self.speculative_aborts += 1

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            hit_rate = (self.speculative_hits / max(1, self.speculative_hits + self.speculative_misses)) if (self.speculative_hits + self.speculative_misses) > 0 else 0.0
            avg_saved = (self.total_latency_saved_ms / max(1, self.speculative_hits)) if self.speculative_hits > 0 else 0.0

            return {
                "total_requests": self.total_requests,
                "speculations_launched": self.speculations_launched,
                "speculative_hits": self.speculative_hits,
                "speculative_misses": self.speculative_misses,
                "speculative_skips": self.speculative_skips,
                "speculative_aborts": self.speculative_aborts,
                "hit_rate": round(hit_rate, 4),
                "total_latency_saved_ms": round(self.total_latency_saved_ms, 2),
                "average_latency_saved_ms": round(avg_saved, 2),
                "action_hits": dict(self.action_hits),
                "action_misses": dict(self.action_misses),
            }


class SpeculativeSession:
    """
    Active speculative context for an in-flight decision step.
    Supports both sync and async resolution with parallel execution.
    """

    def __init__(
        self,
        engine: "SpeculativeEngine",
        state: str,
        predicted_action: Optional[str],
        confidence: float,
        predicted_args: Dict[str, Any],
        speculated: bool,
        future: Optional[concurrent.futures.Future] = None,
        prefetch_start_time: float = 0.0,
    ):
        self.engine = engine
        self.state = state
        self.predicted_action = predicted_action
        self.confidence = confidence
        self.predicted_args = predicted_args
        self.speculated = speculated
        self.future = future
        self.prefetch_start_time = prefetch_start_time
        self.status = SpeculativeStatus.IN_FLIGHT if speculated else SpeculativeStatus.SKIPPED
        self.resolved_result: Optional[Any] = None
        self.abort_event = threading.Event()
        self._lock = threading.Lock()

    def resolve(self, confirmed_action: str, **kwargs) -> Tuple[Any, bool]:
        """
        Synchronously resolves the speculative session against confirmed LLM action.
        Returns: (result, is_hit: bool)
        """
        with self._lock:
            # 1. Speculative HIT: LLM confirmed predicted tool
            if self.speculated and confirmed_action == self.predicted_action and self.future is not None:
                try:
                    action_obj = self.engine.actions.get(confirmed_action)
                    timeout = action_obj.timeout if action_obj else 5.0
                    result = self.future.result(timeout=timeout)
                    time_saved_ms = (time.perf_counter() - self.prefetch_start_time) * 1000.0
                    self.engine.metrics.record_hit(confirmed_action, time_saved_ms)
                    self.status = SpeculativeStatus.HIT
                    self.resolved_result = result
                    return result, True
                except Exception as e:
                    # If speculative prefetch failed, fall through to re-run
                    self.status = SpeculativeStatus.MISS

            # 2. Speculative MISS or Skipped: Cancel in-flight future if still running
            if self.future is not None and not self.future.done():
                self.future.cancel()
                self.abort_event.set()
                if self.predicted_action:
                    self.engine.metrics.record_abort(self.predicted_action)

            if self.speculated and self.predicted_action:
                self.engine.metrics.record_miss(self.predicted_action, confirmed_action)
            self.status = SpeculativeStatus.MISS

            # Execute confirmed action on-demand
            action_obj = self.engine.actions.get(confirmed_action)
            if action_obj is None:
                raise KeyError(f"Action '{confirmed_action}' is not registered with SpeculativeEngine")

            # Merge predicted arguments if user provided none
            exec_args = dict(self.predicted_args) if (confirmed_action == self.predicted_action) else {}
            exec_args.update(kwargs)

            # Handle sync vs async handler
            if inspect.iscoroutinefunction(action_obj.handler):
                # Run async handler synchronously
                loop = asyncio.new_event_loop()
                try:
                    res = loop.run_until_complete(action_obj.handler(**exec_args))
                finally:
                    loop.close()
            else:
                res = action_obj.handler(**exec_args)

            self.resolved_result = res
            return res, False

    async def resolve_async(self, confirmed_action: str, **kwargs) -> Tuple[Any, bool]:
        """
        Asynchronously resolves the speculative session inside an asyncio event loop.
        Returns: (result, is_hit: bool)
        """
        # Speculative HIT
        if self.speculated and confirmed_action == self.predicted_action and self.future is not None:
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, self.future.result, 5.0)
                time_saved_ms = (time.perf_counter() - self.prefetch_start_time) * 1000.0
                self.engine.metrics.record_hit(confirmed_action, time_saved_ms)
                self.status = SpeculativeStatus.HIT
                self.resolved_result = result
                return result, True
            except Exception:
                self.status = SpeculativeStatus.MISS

        # Speculative MISS
        if self.future is not None and not self.future.done():
            self.future.cancel()
            self.abort_event.set()
            if self.predicted_action:
                self.engine.metrics.record_abort(self.predicted_action)

        if self.speculated and self.predicted_action:
            self.engine.metrics.record_miss(self.predicted_action, confirmed_action)
        self.status = SpeculativeStatus.MISS

        action_obj = self.engine.actions.get(confirmed_action)
        if action_obj is None:
            raise KeyError(f"Action '{confirmed_action}' is not registered with SpeculativeEngine")

        exec_args = dict(self.predicted_args) if (confirmed_action == self.predicted_action) else {}
        exec_args.update(kwargs)

        if inspect.iscoroutinefunction(action_obj.handler):
            res = await action_obj.handler(**exec_args)
        else:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, lambda: action_obj.handler(**exec_args))

        self.resolved_result = res
        return res, False

    def __enter__(self) -> "SpeculativeSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Gracefully cleanup in-flight future if abandoned
        with self._lock:
            if self.status == SpeculativeStatus.IN_FLIGHT:
                self.status = SpeculativeStatus.ABORTED
                self.abort_event.set()
                if self.future is not None and not self.future.done():
                    self.future.cancel()
                if self.predicted_action:
                    self.engine.metrics.record_abort(self.predicted_action)

    async def __aenter__(self) -> "SpeculativeSession":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            if self.status == SpeculativeStatus.IN_FLIGHT:
                self.status = SpeculativeStatus.ABORTED
                self.abort_event.set()
                if self.future is not None and not self.future.done():
                    self.future.cancel()
                if self.predicted_action:
                    self.engine.metrics.record_abort(self.predicted_action)


class SpeculativeEngine:
    """
    Sub-millisecond Speculative Decision Routing & Parallel Pre-Fetch Engine.
    Predicts likely target actions in <0.1ms and executes idempotent handlers
    in background worker threads while large reasoning models generate tokens.
    """

    def __init__(
        self,
        reflex_client: Optional[Any] = None,
        confidence_threshold: float = 0.75,
        max_workers: int = 8,
    ):
        if reflex_client is None:
            from reflex.client import Reflex
            self.rx = Reflex(backend="local")
        else:
            self.rx = reflex_client
        self.confidence_threshold = confidence_threshold

        self.actions: Dict[str, SpeculativeAction] = {}
        self.metrics = SpeculativeMetrics()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="reflex-speculative-worker",
        )
        self._lock = threading.Lock()

    def register_action(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        idempotent: bool = True,
        extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
        timeout: float = 5.0,
    ) -> None:
        """Registers an action or tool definition eligible for speculative execution."""
        with self._lock:
            self.actions[name] = SpeculativeAction(
                name=name,
                handler=handler,
                description=description or name,
                idempotent=idempotent,
                extractor=extractor,
                timeout=timeout,
            )

    def register_actions(self, actions: List[SpeculativeAction]) -> None:
        """Bulk registers a collection of SpeculativeActions."""
        for act in actions:
            self.register_action(
                name=act.name,
                handler=act.handler,
                description=act.description,
                idempotent=act.idempotent,
                extractor=act.extractor,
                timeout=act.timeout,
            )

    def speculate(
        self,
        state: str,
        action_names: Optional[List[str]] = None,
        override_action: Optional[str] = None,
        override_confidence: Optional[float] = None,
    ) -> SpeculativeSession:
        """
        Evaluates user state in <0.1ms and initiates parallel pre-fetching if confident.
        Returns active SpeculativeSession.
        """
        candidates = action_names or list(self.actions.keys())
        if not candidates:
            self.metrics.record_skip()
            return SpeculativeSession(
                engine=self,
                state=state,
                predicted_action=None,
                confidence=0.0,
                predicted_args={},
                speculated=False,
            )

        # 1. System-1 Instant Prediction (<0.1ms)
        if override_action is not None and override_action in self.actions:
            predicted_action = override_action
            confidence = override_confidence if override_confidence is not None else 1.0
        else:
            criteria = {
                name: self.actions[name].description
                for name in candidates
                if name in self.actions and self.actions[name].description
            }
            decision = self.rx.evaluate(
                state=state,
                questions={"action": Choice("Select target action", options=candidates, criteria=criteria if criteria else None)},
            )
            choice_res = decision["action"]
            predicted_action = choice_res.selected
            if override_confidence is not None:
                confidence = override_confidence
            else:
                confidence = choice_res.confidence if hasattr(choice_res, "confidence") else 0.85

        action_obj = self.actions.get(predicted_action)
        
        # 2. Safety & Confidence Checks
        # - Must meet confidence threshold
        # - Action must be registered
        # - Action MUST be idempotent (mutations are blocked)
        if (
            action_obj is None
            or confidence < self.confidence_threshold
            or not action_obj.idempotent
        ):
            self.metrics.record_skip()
            return SpeculativeSession(
                engine=self,
                state=state,
                predicted_action=predicted_action,
                confidence=confidence,
                predicted_args={},
                speculated=False,
            )

        # 3. Extract predicted parameters
        predicted_args: Dict[str, Any] = {}
        if action_obj.extractor is not None:
            try:
                predicted_args = action_obj.extractor(state) or {}
            except Exception:
                predicted_args = {}

        # 4. Dispatch background speculative pre-fetch
        self.metrics.record_launch(predicted_action)
        t_start = time.perf_counter()

        def _execute_task():
            if inspect.iscoroutinefunction(action_obj.handler):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(action_obj.handler(**predicted_args))
                finally:
                    loop.close()
            return action_obj.handler(**predicted_args)

        future = self._executor.submit(_execute_task)

        return SpeculativeSession(
            engine=self,
            state=state,
            predicted_action=predicted_action,
            confidence=confidence,
            predicted_args=predicted_args,
            speculated=True,
            future=future,
            prefetch_start_time=t_start,
        )

    def stats(self) -> Dict[str, Any]:
        """Returns JSON-serializable performance and hit rate metrics."""
        return self.metrics.to_dict()

    def shutdown(self, wait: bool = False):
        """Shuts down background executor pool."""
        self._executor.shutdown(wait=wait)
