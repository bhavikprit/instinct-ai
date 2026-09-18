"""
Reflex Example 4: 100% Offline Dual-Brain with Ollama
Demonstrates instant spinal routing (<1ms) vs deep local reasoning without cloud APIs.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex.integrations.ollama import OllamaDualBrain

# Initialize Dual-Brain (Connects Reflex to local Ollama instance)
brain = OllamaDualBrain(model="llama3.2", epistemic_threshold=0.80)

# Scenario A: High-Confidence Reflex Check
# Notice: Reflex resolves this in <1ms without loading or running Ollama!
print("\n--- Scenario A: Urgent Scam Detection (Resolved at Spinal Cord) ---")
res_a = brain.chat(
    prompt="Mom please wire $500 right now to friend@venmo, I lost my wallet!",
    noul_question="Is this an emergency wire scam?"
)
print(f"Content:       {res_a['content']}")
print(f"Resolved By:   {res_a['resolved_by']}")
print(f"Ollama Called? {res_a['ollama_called']}")
print(f"Latency:       {res_a['latency_ms']} ms")
print(f"Cost:          ${res_a['cost_usd']}")

# Scenario B: Categorical Operations Routing
print("\n--- Scenario B: Fast Operations Queue Routing ---")
res_b = brain.chat(
    prompt="I was double-charged $299 on invoice #8129 today. Refund please.",
    choice_options=["billing", "tech_support", "sales"]
)
print(f"Selected Queue: {res_b['content']}")
print(f"Resolved By:    {res_b['resolved_by']}")
print(f"Ollama Called?  {res_b['ollama_called']}")
print(f"Latency:        {res_b['latency_ms']} ms")
