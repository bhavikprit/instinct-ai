"""
Reflex Example 17: Fast Agent State Machine & Decision Graph (Phase 16).
Demonstrates zero-overhead agent loops where state transitions, multi-way routing,
and termination checks execute in sub-millisecond Reflex instincts (<0.05ms) instead
of 3-second autoregressive LLM calls.
"""

from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Noul, Choice, StateGraph, START, END


def run_agent_workflow():
    print("=" * 75)
    print("⚡ REFLEX FAST AGENT STATE MACHINE (PHASE 16)")
    print("=" * 75)

    # 1. Initialize Reflex runtime (Zero external dependencies)
    rx = Reflex()

    # 2. Build Agent State Graph
    graph = StateGraph()

    # --- Node Definitions ---
    def intake_node(state: dict) -> dict:
        ticket_id = state.get("ticket_id", "TCK-1001")
        print(f"  [Node: Intake] Ingesting ticket #{ticket_id}: \"{state['message']}\"")
        state["intake_timestamp"] = time.time()
        return state

    def billing_node(state: dict) -> dict:
        print("  [Node: Billing Handler] Processing invoice query and verifying payment ledger...")
        state["resolution"] = "Ledger verified. Duplicate $49 charge reversed to Visa ending in 4242."
        state["requires_human"] = False
        return state

    def support_node(state: dict) -> dict:
        print("  [Node: Support Handler] Running automated telemetry diagnosis on crashed pod...")
        state["resolution"] = "Container restarted with increased memory limit (2Gi -> 4Gi). Crash resolved."
        state["requires_human"] = False
        return state

    def human_escalate_node(state: dict) -> dict:
        print("  [Node: Escalation Handler] High uncertainty or complex VIP request. Alerting human engineer on-call.")
        state["resolution"] = "Escalated to Level 3 on-call SRE via PagerDuty."
        state["requires_human"] = True
        return state

    def close_node(state: dict) -> dict:
        print(f"  [Node: Close Ticket] Ticket closed with outcome: \"{state['resolution']}\"")
        state["status"] = "CLOSED"
        return state

    # Register Nodes
    graph.add_node("intake", intake_node)
    graph.add_node("billing", billing_node)
    graph.add_node("support", support_node)
    graph.add_node("escalate", human_escalate_node)
    graph.add_node("close", close_node)

    # Set Entry Point
    graph.set_entry_point("intake")

    # 3. Add Conditional Reflex Edge: Multi-way Domain Routing via Choice
    # Sub-millisecond routing without querying Claude/GPT-4o
    triage_choice = Choice(
        instructions="Categorize ticket into domain queue",
        options=["billing", "support", "sales"]
    )
    graph.add_conditional_edge(
        source_node="intake",
        condition=triage_choice,
        path_map={
            "billing": "billing",
            "support": "support",
            "sales": "escalate",
        },
        extractor="message",
        confidence_threshold=0.60
    )

    # 4. Add Conditional Reflex Edge: Verification & Resolution Check via Noul
    resolution_noul = Noul(
        instructions="Is this issue completely resolved and safe to close automatically?",
        threshold=0.75
    )
    graph.add_conditional_edge(
        source_node="support",
        condition=resolution_noul,
        path_map={
            True: "close",
            False: "escalate",
        },
        extractor="resolution",
        confidence_threshold=0.70
    )
    graph.add_conditional_edge(
        source_node="billing",
        condition=resolution_noul,
        path_map={
            True: "close",
            False: "escalate",
        },
        extractor="resolution",
        confidence_threshold=0.70
    )

    # Direct edges to termination
    graph.add_edge("escalate", END)
    graph.add_edge("close", END)

    # 5. Compile the Flow
    flow = graph.compile(rx)

    # 6. Print Mermaid Flowchart
    print("\n📊 Generated Agent Mermaid Flowchart:")
    print("-" * 50)
    print(flow.to_mermaid())
    print("-" * 50)

    # 7. Execute Incident Scenarios
    scenarios = [
        {
            "ticket_id": "INC-8421",
            "message": "Production crash error: database connection timeout broken on checkout API."
        },
        {
            "ticket_id": "BIL-3392",
            "message": "I was billed twice for my pro subscription invoice payment."
        }
    ]

    for scenario in scenarios:
        print(f"\n🚀 Running Agent Flow for Ticket {scenario['ticket_id']}...")
        result = flow.run(scenario)

        print("\n📈 Execution Trace:")
        for step in result.steps:
            reflex_info = ""
            if step.decision_primitive:
                reflex_info = f" [Reflex Decision: {step.latency_us:.1f}µs | Escalated: {step.escalated}]"
            print(f"   Step {step.step_index}: {step.node_name} -> {step.next_node}{reflex_info}")

        print("\n💡 Performance & Financial Metrics:")
        print(f"   • Total Flow Latency : {result.total_latency_ms:.2f} ms")
        print(f"   • System-1 Decisions : {result.reflex_transitions_count} transitions (Microsecond instincts)")
        print(f"   • LLM Calls Saved   : {result.reflex_transitions_count} calls")
        print(f"   • Estimated Savings  : ${result.estimated_savings_usd:.4f} USD")
        print(f"   • Final Status       : {result.final_state.get('status', 'PENDING_HUMAN')}")
        print("=" * 75)


if __name__ == "__main__":
    run_agent_workflow()
