"""
Reflex Flow: Fast Agent State Machine & Decision Graph (Phase 16).
Machine-native, sub-millisecond System-1 decision DAG for autonomous AI agent loops.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from sys1.client import Reflex
from sys1.primitives import Noul, Choice, DecisionResult

# Standard flow boundary identifiers
START: str = "__start__"
END: str = "__end__"


class FlowError(Exception):
    """Base exception for Reflex Flow errors."""
    pass


class MaxStepsExceededError(FlowError):
    """Raised when flow execution exceeds maximum permitted steps."""
    pass


class InvalidTransitionError(FlowError):
    """Raised when an edge cannot find a valid destination node."""
    pass


@dataclass
class FlowStep:
    """Represents an atomic execution step in an agent workflow."""
    step_index: int
    node_name: str
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    next_node: str
    decision_primitive: Optional[Union[Noul, Choice, str]] = None
    decision_result: Optional[DecisionResult] = None
    latency_us: float = 0.0
    escalated: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to serializable dictionary."""
        return {
            "step_index": self.step_index,
            "node_name": self.node_name,
            "input_state": copy.deepcopy(self.input_state),
            "output_state": copy.deepcopy(self.output_state),
            "next_node": self.next_node,
            "decision_primitive": str(self.decision_primitive) if self.decision_primitive else None,
            "decision_result": self.decision_result.to_dict() if self.decision_result else None,
            "latency_us": round(self.latency_us, 2),
            "escalated": self.escalated,
            "timestamp": self.timestamp,
        }


@dataclass
class FlowResult:
    """Encapsulates the complete execution trace and output of a compiled flow."""
    final_state: Dict[str, Any]
    steps: List[FlowStep]
    completed: bool
    total_latency_ms: float
    reflex_transitions_count: int = 0
    estimated_savings_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "final_state": self.final_state,
            "completed": self.completed,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "step_count": len(self.steps),
            "reflex_transitions_count": self.reflex_transitions_count,
            "estimated_savings_usd": round(self.estimated_savings_usd, 5),
            "steps": [s.to_dict() for s in self.steps],
        }

    def get_step(self, idx: int) -> FlowStep:
        """Retrieve a specific step by 0-based index."""
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        raise IndexError(f"Step index {idx} out of range (0..{len(self.steps) - 1})")

    def get_state_at_step(self, idx: int) -> Dict[str, Any]:
        """Retrieve the state snapshot immediately after step idx executed."""
        step = self.get_step(idx)
        return copy.deepcopy(step.output_state)


class ConditionalReflexEdge:
    """
    Evaluates dynamic transitions using Reflex System-1 primitives (Noul, Choice)
    or custom state callables with sub-millisecond execution.
    """

    def __init__(
        self,
        source_node: str,
        condition: Union[Noul, Choice, Callable[[Dict[str, Any]], Any]],
        path_map: Dict[Any, str],
        extractor: Optional[Union[str, Callable[[Dict[str, Any]], str]]] = None,
        confidence_threshold: float = 0.70,
        escalation_fallback: Optional[Callable[[Dict[str, Any], Optional[DecisionResult]], str]] = None,
    ):
        self.source_node = source_node
        self.condition = condition
        self.path_map = path_map
        self.extractor = extractor
        self.confidence_threshold = confidence_threshold
        self.escalation_fallback = escalation_fallback

    def extract_context(self, state: Dict[str, Any]) -> str:
        """Extract prompt or text context from state for Reflex evaluation."""
        if callable(self.extractor):
            return str(self.extractor(state))
        elif isinstance(self.extractor, str):
            val = state.get(self.extractor, "")
            return str(val) if val is not None else ""
        
        # Heuristic search for common agent context keys
        for candidate_key in ["context", "message", "text", "input", "query", "state", "output"]:
            if candidate_key in state and isinstance(state[candidate_key], str):
                return state[candidate_key]
        
        # Fallback to stringified state representation
        return str(state)

    def evaluate(
        self,
        state: Dict[str, Any],
        reflex_client: Reflex,
    ) -> Tuple[str, Optional[Union[Noul, Choice, str]], Optional[DecisionResult], float, bool]:
        """
        Evaluate edge transition.
        Returns (next_node, primitive, decision_result, latency_us, escalated).
        """
        t0 = time.perf_counter()
        
        # 1. Plain callable condition
        if callable(self.condition) and not isinstance(self.condition, (Noul, Choice)):
            val = self.condition(state)
            latency_us = (time.perf_counter() - t0) * 1_000_000.0
            next_node = self._resolve_target(val)
            return next_node, str(self.condition), None, latency_us, False

        context = self.extract_context(state)
        escalated = False

        # 2. Noul Boolean Primitive
        if isinstance(self.condition, Noul):
            result = reflex_client.evaluate(context, {"gate": self.condition})
            latency_us = (time.perf_counter() - t0) * 1_000_000.0
            noul_res = result["gate"]
            
            # Check confidence against threshold
            confidence = noul_res.confidence
            if confidence < self.confidence_threshold:
                escalated = True
                if self.escalation_fallback is not None:
                    next_node = self.escalation_fallback(state, result)
                    return next_node, self.condition, result, latency_us, True
                if "escalate" in self.path_map:
                    return self.path_map["escalate"], self.condition, result, latency_us, True

            val = noul_res.is_true
            next_node = self._resolve_target(val)
            return next_node, self.condition, result, latency_us, escalated

        # 3. Choice Multi-Class Primitive
        elif isinstance(self.condition, Choice):
            result = reflex_client.evaluate(context, {"routing": self.condition})
            latency_us = (time.perf_counter() - t0) * 1_000_000.0
            choice_res = result["routing"]
            
            confidence = choice_res.confidence
            if confidence < self.confidence_threshold:
                escalated = True
                if self.escalation_fallback is not None:
                    next_node = self.escalation_fallback(state, result)
                    return next_node, self.condition, result, latency_us, True
                if "escalate" in self.path_map:
                    return self.path_map["escalate"], self.condition, result, latency_us, True

            selected_key = choice_res.selected
            next_node = self._resolve_target(selected_key)
            return next_node, self.condition, result, latency_us, escalated

        raise InvalidTransitionError(f"Unsupported condition type: {type(self.condition)}")

    def _resolve_target(self, key: Any) -> str:
        """Map evaluated key to target node name."""
        # Direct lookup
        if key in self.path_map:
            return self.path_map[key]
        
        # Boolean normalization: True/"true"/1, False/"false"/0
        if isinstance(key, bool):
            str_key = "true" if key else "false"
            if str_key in self.path_map:
                return self.path_map[str_key]
        elif isinstance(key, str):
            lower_key = key.lower()
            if lower_key in self.path_map:
                return self.path_map[lower_key]
            if lower_key == "true" and True in self.path_map:
                return self.path_map[True]
            if lower_key == "false" and False in self.path_map:
                return self.path_map[False]

        # Default fallback key in path_map
        if "__default__" in self.path_map:
            return self.path_map["__default__"]
        if "default" in self.path_map:
            return self.path_map["default"]

        raise InvalidTransitionError(
            f"Evaluated condition '{key}' has no route in path_map keys: {list(self.path_map.keys())}"
        )


class StateGraph:
    """
    Builder for fast agent state machine decision graphs.
    """

    def __init__(self):
        self.nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, ConditionalReflexEdge] = {}
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, action: Callable[[Dict[str, Any]], Dict[str, Any]]) -> StateGraph:
        """Register a node action function."""
        if name in (START, END):
            raise ValueError(f"Cannot register reserved node name '{name}'")
        self.nodes[name] = action
        return self

    def set_entry_point(self, node_name: str) -> StateGraph:
        """Define the initial node to execute when the flow starts."""
        self.entry_point = node_name
        self.edges[START] = node_name
        return self

    def add_edge(self, start_node: str, end_node: str) -> StateGraph:
        """Add an unconditional transition edge from start_node to end_node."""
        if start_node == START:
            self.entry_point = end_node
        self.edges[start_node] = end_node
        return self

    def add_conditional_edge(
        self,
        source_node: str,
        condition: Union[Noul, Choice, Callable[[Dict[str, Any]], Any]],
        path_map: Dict[Any, str],
        extractor: Optional[Union[str, Callable[[Dict[str, Any]], str]]] = None,
        confidence_threshold: float = 0.70,
        escalation_fallback: Optional[Callable[[Dict[str, Any], Optional[DecisionResult]], str]] = None,
    ) -> StateGraph:
        """
        Add a conditional transition evaluated via System-1 Reflex primitives or callable.
        """
        edge = ConditionalReflexEdge(
            source_node=source_node,
            condition=condition,
            path_map=path_map,
            extractor=extractor,
            confidence_threshold=confidence_threshold,
            escalation_fallback=escalation_fallback,
        )
        self.conditional_edges[source_node] = edge
        return self

    def compile(self, reflex_client: Optional[Reflex] = None) -> CompiledFlow:
        """Compile graph into an executable CompiledFlow instance."""
        if not self.entry_point and START not in self.edges:
            raise FlowError("Flow has no entry point defined. Call set_entry_point() or add_edge(START, ...)")
        
        entry = self.entry_point or self.edges[START]
        if entry not in self.nodes:
            raise FlowError(f"Entry point node '{entry}' does not exist in registered nodes.")

        # Ensure Reflex client is available
        client = reflex_client or Reflex()
        return CompiledFlow(
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=entry,
            reflex_client=client,
        )


class CompiledFlow:
    """
    Compiled, thread-safe, high-speed execution engine for an agent state machine.
    """

    # Estimated cost savings per avoided LLM reasoning transition (e.g. Claude 3.5 / GPT-4o input/output)
    ESTIMATED_LLM_TRANSITION_COST_USD: float = 0.005

    def __init__(
        self,
        nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        edges: Dict[str, str],
        conditional_edges: Dict[str, ConditionalReflexEdge],
        entry_point: str,
        reflex_client: Reflex,
    ):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.reflex_client = reflex_client

    def step(
        self,
        current_node: str,
        current_state: Dict[str, Any],
        step_index: int = 0,
    ) -> Tuple[str, Dict[str, Any], FlowStep]:
        """
        Execute a single node action and compute the transition to the next node.
        Returns (next_node, output_state, FlowStep).
        """
        if current_node not in self.nodes:
            raise InvalidTransitionError(f"Target node '{current_node}' is not registered in flow.")

        action = self.nodes[current_node]
        input_state = copy.deepcopy(current_state)
        
        # Execute node logic
        updated = action(current_state)
        output_state = updated if isinstance(updated, dict) else current_state

        # Determine transition to next node
        decision_primitive = None
        decision_result = None
        latency_us = 0.0
        escalated = False

        if current_node in self.conditional_edges:
            cond_edge = self.conditional_edges[current_node]
            next_node, decision_primitive, decision_result, latency_us, escalated = cond_edge.evaluate(
                output_state, self.reflex_client
            )
        elif current_node in self.edges:
            next_node = self.edges[current_node]
        else:
            # If no edge is declared, terminate at END
            next_node = END

        flow_step = FlowStep(
            step_index=step_index,
            node_name=current_node,
            input_state=input_state,
            output_state=output_state,
            next_node=next_node,
            decision_primitive=decision_primitive,
            decision_result=decision_result,
            latency_us=latency_us,
            escalated=escalated,
        )

        return next_node, output_state, flow_step

    def stream(
        self,
        initial_state: Dict[str, Any],
        max_steps: int = 50,
    ) -> Iterator[FlowStep]:
        """
        Stream each executed FlowStep in real-time until reaching END or max_steps.
        """
        current_node = self.entry_point
        state = copy.deepcopy(initial_state)
        step_idx = 0

        while current_node != END and step_idx < max_steps:
            next_node, state, step_record = self.step(current_node, state, step_index=step_idx)
            yield step_record
            current_node = next_node
            step_idx += 1

        if current_node != END and step_idx >= max_steps:
            raise MaxStepsExceededError(
                f"Flow exceeded maximum allowed steps ({max_steps}) at node '{current_node}'."
            )

    def run(
        self,
        initial_state: Dict[str, Any],
        max_steps: int = 50,
    ) -> FlowResult:
        """
        Execute the complete agent graph to termination.
        Returns FlowResult with final state and full execution trace.
        """
        t0 = time.perf_counter()
        steps: List[FlowStep] = []
        state = copy.deepcopy(initial_state)
        current_node = self.entry_point
        step_idx = 0
        reflex_count = 0

        while current_node != END and step_idx < max_steps:
            next_node, state, step_record = self.step(current_node, state, step_index=step_idx)
            steps.append(step_record)
            if step_record.decision_primitive is not None:
                reflex_count += 1
            current_node = next_node
            step_idx += 1

        completed = (current_node == END)
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        estimated_savings = reflex_count * self.ESTIMATED_LLM_TRANSITION_COST_USD

        if not completed and step_idx >= max_steps:
            raise MaxStepsExceededError(
                f"Flow exceeded maximum allowed steps ({max_steps}) at node '{current_node}'."
            )

        return FlowResult(
            final_state=state,
            steps=steps,
            completed=completed,
            total_latency_ms=total_latency_ms,
            reflex_transitions_count=reflex_count,
            estimated_savings_usd=estimated_savings,
        )

    def to_mermaid(self) -> str:
        """
        Generate Mermaid flowchart syntax representing the decision graph structure.
        """
        lines = ["flowchart TD", f"    {START}([\"START\"]) --> {self.entry_point}"]

        # Direct edges
        for src, dst in self.edges.items():
            if src == START:
                continue
            dst_label = f"([\"{END}\"])" if dst == END else f"[\"{dst}\"]"
            lines.append(f"    {src}[\"{src}\"] --> {dst}{dst_label}")

        # Conditional edges
        for src, edge in self.conditional_edges.items():
            cond_id = f"cond_{src}"
            if isinstance(edge.condition, Noul):
                desc = f"Noul: {edge.condition.instructions[:30]}..." if len(edge.condition.instructions) > 30 else f"Noul: {edge.condition.instructions}"
            elif isinstance(edge.condition, Choice):
                desc = f"Choice: {edge.condition.instructions[:30]}..." if len(edge.condition.instructions) > 30 else f"Choice: {edge.condition.instructions}"
            else:
                desc = "Callable Condition"

            lines.append(f"    {src}[\"{src}\"] --> {cond_id}{{\"{desc}\"}}")
            for key, target in edge.path_map.items():
                target_label = f"([\"{END}\"])" if target == END else f"[\"{target}\"]"
                lines.append(f"    {cond_id} -- \"{key}\" --> {target}{target_label}")

        return "\n".join(lines)


# Aliases
Flow = StateGraph
