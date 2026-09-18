"""
Reflex Example 1: Basic Machine-Native Decisions
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Noul, Choice, Score

# Initialize Reflex (uses local sub-15ms engine if no API key set)
rx = Reflex()

state = "Customer says: My annual invoice charged me $399 twice today. I want an immediate refund!"

print("=== Evaluating Multi-Primitive Decision in Single Pass ===")
result = rx.evaluate(
    state=state,
    questions={
        "is_refund": Noul("Is the customer demanding a refund or payment reversal?"),
        "route": Choice(
            instructions="Triage destination",
            options=["billing", "technical_support", "sales", "spam"]
        ),
        "frustration": Score("Customer frustration rating 1-10", min_val=1.0, max_val=10.0)
    }
)

print(f"Latency: {result.latency_ms} ms")
print(f"Backend: {result.backend}")
print(f"Cost: ${result.cost_usd:.6f}")
print("-" * 50)

noul = result.get_noul("is_refund")
print(f"Refund Requested? -> Prob: {noul.probability:.3f} (Confident: {noul.is_true})")

choice = result.get_choice("route")
print(f"Target Queue -> {choice.selected} (Prob: {choice.get_prob(choice.selected):.1%})")
print(f"Full Distribution: {choice.distribution}")

score = result["frustration"]
print(f"Frustration Score -> {score.score}/10")
