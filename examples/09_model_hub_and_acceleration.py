"""
Example 09: Model Hub, Checkpoint Catalog & Hardware Acceleration.

Demonstrates:
1. Inspecting the open-weights model catalog programmatically.
2. Configuring hardware acceleration profiles (Apple CoreML/Metal, NVIDIA CUDA, CPU).
3. Zero-marginal-cost local decision execution.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Noul, Choice, Score, list_models, MODEL_CATALOG


def main():
    print("=" * 70)
    print("⚡ Reflex Model Hub & Hardware Acceleration")
    print("=" * 70)

    # 1. Programmatically inspect available open-weights checkpoints
    print("\n1. Querying Open-Weights Model Catalog:")
    catalog_models = list_models()
    for m in catalog_models:
        status = "Cached locally" if m["cached"] else "Remote (HuggingFace Hub)"
        print(f" • {m['name']:<24} [{m['size_mb']:.1f} MB] -> {status}")
        print(f"   {m['description']}")

    # 2. Hardware Acceleration Profiles
    print("\n2. Initializing Reflex with Hardware Acceleration Profile:")
    print("Available profiles: device='auto' | 'coreml' | 'cuda' | 'cpu'")
    
    # In mock_mode for demonstration without requiring physical model download
    rx_neural = Reflex(
        backend="onnx",
        device="auto",        # Automatically selects CoreML (macOS) or CUDA (Linux/Windows)
        mock_mode=True,       # Demonstrates pipeline
    )

    print(f"Initialized backend: {rx_neural.backend.name}")
    print(f"Execution providers: {rx_neural.backend.providers}")

    # 3. Fast Spinal Reflex Decision Pass
    incoming_ticket = (
        "CRITICAL ALERT: Production Postgres replica lag exceeded 60 seconds! "
        "User auth service is rejecting customer sessions."
    )
    print(f"\n3. Evaluating Production State:\n\"{incoming_ticket}\"")

    decision = rx_neural.evaluate(
        state=incoming_ticket,
        questions={
            "is_p0": Noul("Is this a P0 critical production outage?"),
            "escalation_team": Choice(
                instructions="Route to responder on-call team",
                options=["sre_infra", "security_ops", "frontend_support", "billing"]
            ),
            "urgency": Score("Incident urgency rating 1-10", min_val=1.0, max_val=10.0),
        }
    )

    print("\nDecision Results:")
    print(f" • P0 Outage Probability : {decision['is_p0'].probability:.4f} (Is True: {decision['is_p0'].is_true})")
    print(f" • Escalation Route       : {decision['escalation_team'].selected}")
    print(f" • Urgency Score          : {decision['urgency'].score:.1f}/10.0")
    print(f" • Execution Latency      : {decision.latency_ms} ms")
    print(f" • Marginal Cost          : ${decision.cost_usd:.4f}")

    print("\n✅ Model Hub & Acceleration demonstration complete!")


if __name__ == "__main__":
    main()
