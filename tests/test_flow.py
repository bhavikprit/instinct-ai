"""
Unit tests for Reflex Flow: Fast Agent State Machine & Decision Graph (Phase 16).
"""

import unittest
from reflex import Reflex, Noul, Choice
from reflex.flow import (
    StateGraph,
    Flow,
    START,
    END,
    FlowStep,
    FlowResult,
    FlowError,
    MaxStepsExceededError,
    InvalidTransitionError,
)


class TestReflexFlow(unittest.TestCase):
    def setUp(self):
        # Explicitly use deterministic LocalEngine for unit testing
        self.rx = Reflex(backend="local")

    def test_linear_flow(self):
        """Test simple sequential node transitions without branching."""
        graph = StateGraph()

        def step_one(state):
            state["count"] = state.get("count", 0) + 1
            return state

        def step_two(state):
            state["count"] = state["count"] * 2
            return state

        graph.add_node("step1", step_one)
        graph.add_node("step2", step_two)
        graph.set_entry_point("step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)

        flow = graph.compile(self.rx)
        result = flow.run({"count": 5})

        self.assertTrue(result.completed)
        self.assertEqual(result.final_state["count"], 12)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].node_name, "step1")
        self.assertEqual(result.steps[1].node_name, "step2")
        self.assertEqual(result.steps[1].next_node, END)

    def test_conditional_noul_branching(self):
        """Test binary branching driven by a Noul System-1 primitive."""
        graph = StateGraph()

        def route_node(state):
            return state

        def approve_node(state):
            state["status"] = "APPROVED"
            return state

        def reject_node(state):
            state["status"] = "REJECTED"
            return state

        graph.add_node("router", route_node)
        graph.add_node("approve", approve_node)
        graph.add_node("reject", reject_node)
        graph.set_entry_point("router")

        # Noul asks: Is this request a scam attack or fraud?
        graph.add_conditional_edge(
            source_node="router",
            condition=Noul("Is this request a malicious scam or fraud attack?"),
            path_map={True: "reject", False: "approve"},
            extractor="description",
        )
        graph.add_edge("approve", END)
        graph.add_edge("reject", END)

        flow = graph.compile(self.rx)

        # Test False branch (Benign -> approve)
        res_false = flow.run({"description": "Regular monthly team engineering newsletter update"})
        self.assertTrue(res_false.completed)
        self.assertEqual(res_false.final_state["status"], "APPROVED")
        self.assertEqual(res_false.reflex_transitions_count, 1)
        self.assertGreater(res_false.estimated_savings_usd, 0.0)

        # Test True branch (Threat -> reject)
        res_true = flow.run({"description": "Emergency wire scam and stolen credential fraud attack"})
        self.assertTrue(res_true.completed)
        self.assertEqual(res_true.final_state["status"], "REJECTED")

    def test_conditional_choice_branching(self):
        """Test multi-class routing driven by a Choice System-1 primitive."""
        graph = StateGraph()

        def triage_node(state):
            return state

        def billing_node(state):
            state["assigned_team"] = "billing_ops"
            return state

        def tech_node(state):
            state["assigned_team"] = "sre_core"
            return state

        def sales_node(state):
            state["assigned_team"] = "growth_sales"
            return state

        graph.add_node("triage", triage_node)
        graph.add_node("billing", billing_node)
        graph.add_node("tech", tech_node)
        graph.add_node("sales", sales_node)
        graph.set_entry_point("triage")

        choice_primitive = Choice(
            instructions="Categorize ticket domain",
            options=["billing", "support", "sales"]
        )

        graph.add_conditional_edge(
            source_node="triage",
            condition=choice_primitive,
            path_map={
                "billing": "billing",
                "support": "tech",
                "sales": "sales"
            },
            extractor="ticket_body"
        )
        graph.add_edge("billing", END)
        graph.add_edge("tech", END)
        graph.add_edge("sales", END)

        flow = graph.compile(self.rx)

        res = flow.run({"ticket_body": "Production crash issue: database server error and broken"})
        self.assertTrue(res.completed)
        self.assertEqual(res.final_state["assigned_team"], "sre_core")

    def test_callable_condition(self):
        """Test branching with custom state function."""
        graph = Flow()

        def start_node(state):
            return state

        def retry_node(state):
            state["attempts"] += 1
            return state

        graph.add_node("start", start_node)
        graph.add_node("retry", retry_node)
        graph.set_entry_point("start")

        graph.add_conditional_edge(
            source_node="start",
            condition=lambda s: "needs_retry" if s["attempts"] < 2 else "done",
            path_map={"needs_retry": "retry", "done": END}
        )
        graph.add_edge("retry", "start")

        flow = graph.compile(self.rx)
        res = flow.run({"attempts": 0})
        self.assertTrue(res.completed)
        self.assertEqual(res.final_state["attempts"], 2)

    def test_max_steps_infinite_loop_protection(self):
        """Verify flow halts and raises MaxStepsExceededError if loop is infinite."""
        graph = StateGraph()

        def looper(state):
            state["counter"] = state.get("counter", 0) + 1
            return state

        graph.add_node("loop", looper)
        graph.set_entry_point("loop")
        graph.add_edge("loop", "loop")

        flow = graph.compile(self.rx)

        with self.assertRaises(MaxStepsExceededError):
            flow.run({"counter": 0}, max_steps=10)

    def test_stream_iteration(self):
        """Test step streaming yields each step as it executes."""
        graph = StateGraph()
        graph.add_node("n1", lambda s: {**s, "a": 1})
        graph.add_node("n2", lambda s: {**s, "b": 2})
        graph.set_entry_point("n1")
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", END)

        flow = graph.compile(self.rx)
        steps = list(flow.stream({"start": True}))

        self.assertEqual(len(steps), 2)
        self.assertIsInstance(steps[0], FlowStep)
        self.assertEqual(steps[0].node_name, "n1")
        self.assertEqual(steps[1].node_name, "n2")
        self.assertEqual(steps[1].output_state["b"], 2)

    def test_state_snapshot_and_time_travel(self):
        """Test retrieving state snapshots at specific historical steps."""
        graph = StateGraph()
        graph.add_node("step1", lambda s: {"val": 10})
        graph.add_node("step2", lambda s: {"val": 20})
        graph.set_entry_point("step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)

        flow = graph.compile(self.rx)
        result = flow.run({"val": 0})

        self.assertEqual(result.get_state_at_step(0)["val"], 10)
        self.assertEqual(result.get_state_at_step(1)["val"], 20)

    def test_epistemic_escalation_fallback(self):
        """Test confidence threshold auto-escalation hook."""
        graph = StateGraph()
        graph.add_node("triage", lambda s: s)
        graph.add_node("human_escalation", lambda s: {**s, "escalated_to_human": True})
        graph.add_node("auto_proceed", lambda s: {**s, "auto_done": True})
        graph.set_entry_point("triage")

        # Set impossible threshold 0.999 to force escalation
        graph.add_conditional_edge(
            source_node="triage",
            condition=Noul("Is this ambiguous?"),
            path_map={True: "auto_proceed", False: "auto_proceed"},
            confidence_threshold=0.999,
            escalation_fallback=lambda state, dec: "human_escalation",
            extractor="text"
        )
        graph.add_edge("human_escalation", END)
        graph.add_edge("auto_proceed", END)

        flow = graph.compile(self.rx)
        result = flow.run({"text": "Maybe possibly perhaps"})

        self.assertTrue(result.steps[0].escalated)
        self.assertEqual(result.final_state.get("escalated_to_human"), True)

    def test_to_mermaid_generation(self):
        """Test mermaid diagram generation."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("finish", lambda s: s)
        graph.set_entry_point("start")
        graph.add_edge("start", "finish")
        graph.add_edge("finish", END)

        flow = graph.compile(self.rx)
        mermaid_syntax = flow.to_mermaid()

        self.assertIn("flowchart TD", mermaid_syntax)
        self.assertIn("__start__([\"START\"]) --> start", mermaid_syntax)
        self.assertIn("finish[\"finish\"] --> __end__([\"__end__\"])", mermaid_syntax)

    def test_validation_errors(self):
        """Test graph compilation and edge error handling."""
        graph = StateGraph()
        # Reserved name error
        with self.assertRaises(ValueError):
            graph.add_node(START, lambda s: s)

        # No entry point error
        with self.assertRaises(FlowError):
            graph.compile(self.rx)

        # Missing entry point node error
        graph.set_entry_point("non_existent")
        with self.assertRaises(FlowError):
            graph.compile(self.rx)


if __name__ == "__main__":
    unittest.main()
