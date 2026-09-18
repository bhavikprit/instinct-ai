"""
Reflex Example 2: The Dual-Brain Agent Pattern
(System 1 Spinal Reflex + System 2 Heavy Reasoning Fallback)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Noul, Choice

rx = Reflex(backend="local")

def process_agent_request(incoming_message: str):
    print(f"\nIncoming Event: '{incoming_message}'")
    
    # Step 1: System 1 Spinal Reflex (<15ms, $0.00 cost)
    reflex_result = rx.evaluate(
        state=incoming_message,
        questions={
            "is_emergency": Noul("Does this demand immediate emergency shutdown or security isolation?"),
            "action": Choice("Action", options=["quarantine_host", "execute_tool", "escalate_to_cortex"])
        }
    )
    
    noul = reflex_result.get_noul("is_emergency")
    print(f"  [System 1 Reflex] Evaluated in {reflex_result.latency_ms}ms | Cost: ${reflex_result.cost_usd}")
    print(f"  Emergency Probability: {noul.probability:.3f}")
    
    # Step 2: Epistemic Gate
    if noul.is_true:
        print("  ⚡ [Reflex Action Taken] Instant emergency quarantine! (Zero LLM tokens burned)")
        return {"status": "quarantined", "latency": reflex_result.latency_ms}
    elif noul.is_uncertain:
        print("  🧠 [Escalating to System 2 Cortex] High uncertainty detected. Waking up Claude 3.5 Sonnet...")
        # Simulate System 2 reasoning
        return {"status": "escalated_to_reasoning_llm"}
    else:
        print("  ✓ [Standard Action Taken] Safe normal execution.")
        return {"status": "normal"}

if __name__ == "__main__":
    # 1. Emergency incident -> handled instantly by reflex in <15ms!
    process_agent_request("CRITICAL ALERT: Unauthorized SSH root login attempt with remote key injection!")
    
    # 2. Routine request -> safe execution
    process_agent_request("Show me the weekly sales chart for team alpha.")
