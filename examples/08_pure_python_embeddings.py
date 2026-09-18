"""
Reflex Example 8: Pure-Python Semantic Vector Decisions
Demonstrates sub-0.1ms semantic classification with 100% standard library Python (zero C++ dependencies).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Noul, Choice, Score
from reflex.embeddings import SemanticVectorEncoder, cosine_similarity

print("=== 1. Direct Semantic Vector Embeddings ===")
encoder = SemanticVectorEncoder()

vec1 = encoder.encode("Severe production PostgreSQL database outage and connection timeout")
vec2 = encoder.encode("Critical MySQL database connection pool exhausted")
vec3 = encoder.encode("How to bake chocolate chip cookies")

sim_related = cosine_similarity(vec1, vec2)
sim_unrelated = cosine_similarity(vec1, vec3)

print(f"Vector Dimension:              {len(vec1)}")
print(f"Similarity (Database vs MySQL): {sim_related:.4f} (High Semantic Overlap)")
print(f"Similarity (Database vs Cookie):{sim_unrelated:.4f} (Low Semantic Overlap)")

print("\n=== 2. Reflex with Pure-Semantic Backend ===")
rx = Reflex(backend="semantic")

result = rx.evaluate(
    state="The customer cannot log in because two-factor authentication SMS is never arriving.",
    questions={
        "is_auth_issue": Noul("Is this an authentication or sign-in failure?"),
        "triage_queue": Choice("Operations Queue", options=["identity_access", "billing", "marketing"]),
        "urgency": Score("Customer frustration score 1-10", min_val=1.0, max_val=10.0)
    }
)

print(f"Backend Used:     {result.backend}")
print(f"Latency:          {result.latency_ms} ms")
print(f"Auth Issue Prob:  {result['is_auth_issue'].probability:.3f} (is_true={result['is_auth_issue'].is_true})")
print(f"Triage Queue:     {result['triage_queue'].selected}")
print(f"Dist:             {result['triage_queue'].distribution}")
print(f"Urgency Score:    {result['urgency'].score} / 10.0")
