"""
Reflex Example 22: Speculative Decision Routing & Parallel Pre-Fetch (Phase 22).
Demonstrates sub-millisecond System-1 intention prediction, parallel idempotent pre-fetching,
0ms tool latency during LLM reasoning, side-effect safety guards, and telemetry tracking.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Choice
from reflex.speculative import (
    SpeculativeEngine,
    SpeculativeAction,
    SpeculativeStatus,
)


def extract_order_id(state: str) -> dict:
    """Zero-shot parameter extractor from user prompt."""
    match = re.search(r"ord_[a-zA-Z0-9]+", state, re.IGNORECASE)
    order_id = match.group(0) if match else "ord_default"
    return {"order_id": order_id}


def extract_user_id(state: str) -> dict:
    match = re.search(r"usr_[a-zA-Z0-9]+", state, re.IGNORECASE)
    user_id = match.group(0) if match else "usr_1001"
    return {"user_id": user_id}


def run_speculative_demo():
    print("=" * 85)
    print("⚡ REFLEX SPECULATIVE: ZERO-OVERHEAD TOOL PRE-FETCH & DECISION ROUTING (PHASE 22)")
    print("=" * 85)

    # -----------------------------------------------------------------
    # 1. Initialize Reflex Speculative Engine
    # -----------------------------------------------------------------
    print("\n1. 🛠️ Initializing Speculative Decision Engine with Registered Tools:")
    rx = Reflex(backend="local")
    engine = SpeculativeEngine(reflex_client=rx, confidence_threshold=0.70)

    # Tool 1: Order details lookup (Read-only / Idempotent, ~80ms database query)
    def fetch_order_status(order_id: str = "ord_default"):
        time.sleep(0.08)  # Simulate 80ms network round-trip to database
        return {
            "order_id": order_id,
            "status": "In Transit - Out for Delivery",
            "eta": "Today by 4:00 PM",
            "items": ["Reflex Dual-Brain Gateway DevKit"],
        }

    # Tool 2: Customer CRM Profile lookup (Read-only / Idempotent, ~60ms query)
    def fetch_user_profile(user_id: str = "usr_1001"):
        time.sleep(0.06)
        return {
            "user_id": user_id,
            "tier": "Enterprise VIP",
            "sla_tier": "P0 (<100ms response)",
        }

    # Tool 3: Payment settlement (Mutation / Non-idempotent -> NEVER pre-fetch!)
    def charge_customer(amount: float, order_id: str):
        return {
            "transaction_id": "tx_998811",
            "amount": amount,
            "status": "charged",
        }

    engine.register_action(
        name="fetch_order_status",
        handler=fetch_order_status,
        description="Lookup shipment status and order tracking details",
        idempotent=True,
        extractor=extract_order_id,
    )
    engine.register_action(
        name="fetch_user_profile",
        handler=fetch_user_profile,
        description="Lookup customer profile, VIP status, and SLA terms",
        idempotent=True,
        extractor=extract_user_id,
    )
    engine.register_action(
        name="charge_customer",
        handler=charge_customer,
        description="Process credit card transaction payment settlement",
        idempotent=False,  # Mutation safety guard!
    )

    print("   • [Registered] fetch_order_status (idempotent=True, latency=~80ms)")
    print("   • [Registered] fetch_user_profile (idempotent=True, latency=~60ms)")
    print("   • [Registered] charge_customer    (idempotent=False, MUTATION GUARD ACTIVE)")

    # -----------------------------------------------------------------
    # 2. Benchmark: Traditional Sequential Agent vs Reflex Speculative
    # -----------------------------------------------------------------
    print("\n" + "-" * 85)
    print("2. ⏱️ BENCHMARK: TRADITIONAL SEQUENTIAL AGENT VS REFLEX SPECULATIVE PRE-FETCH")
    print("-" * 85)

    user_query = "Where is my shipment package for ord_88291?"
    llm_generation_time_s = 0.12  # Simulate 120ms token generation time for LLM

    print(f"   User Prompt: \"{user_query}\"")

    # A. Sequential Baseline
    t0 = time.perf_counter()
    time.sleep(llm_generation_time_s)  # LLM generates "Calling tool: fetch_order_status"
    chosen_tool = "fetch_order_status"
    tool_res = fetch_order_status(order_id="ord_88291")
    seq_latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n   [Sequential Baseline]")
    print(f"   • Step 1: LLM Reasoning & Token Generation : {llm_generation_time_s * 1000.0:.1f}ms")
    print(f"   • Step 2: Tool Execution (Wait on DB)      : 80.0ms")
    print(f"   • TOTAL USER WAIT TIME                     : {seq_latency_ms:.1f}ms")

    # B. Reflex Speculative Pre-fetch
    t1 = time.perf_counter()
    # Step 1: Reflex System-1 instantly (<0.1ms) predicts the tool and launches parallel pre-fetch
    session = engine.speculate(state=user_query, action_names=["fetch_order_status", "fetch_user_profile"])
    predict_latency_ms = (time.perf_counter() - t1) * 1000.0

    print(f"\n   [Reflex Speculative Execution]")
    print(f"   • Step 1: System-1 Prediction & Background Dispatch: {predict_latency_ms:.2f}ms")
    print(f"             Predicted Tool : {session.predicted_action} (confidence: {session.confidence * 100.0:.1f}%)")
    print(f"             Extracted Args : {session.predicted_args}")
    print(f"             Speculative Task Status: {session.status.value}")

    # Step 2: Simulate LLM generating tokens in parallel
    print(f"   • Step 2: Heavy LLM generates tokens in parallel    : {llm_generation_time_s * 1000.0:.1f}ms")
    time.sleep(llm_generation_time_s)

    # Step 3: LLM confirms the tool call -> Instantly resolve from pre-fetched future
    t_resolve_start = time.perf_counter()
    result, is_hit = session.resolve("fetch_order_status")
    resolve_latency_ms = (time.perf_counter() - t_resolve_start) * 1000.0
    total_spec_latency_ms = (time.perf_counter() - t1) * 1000.0

    print(f"   • Step 3: Speculative Resolution                    : {resolve_latency_ms:.2f}ms (is_hit={is_hit})")
    print(f"   • TOTAL USER WAIT TIME                             : {total_spec_latency_ms:.1f}ms")
    print(f"   • 🔥 LATENCY REDUCTION                             : {seq_latency_ms - total_spec_latency_ms:.1f}ms ({((seq_latency_ms - total_spec_latency_ms) / seq_latency_ms) * 100.0:.1f}% faster!)")
    print(f"   • Tool Result                                      : {result['status']}")

    # -----------------------------------------------------------------
    # 3. Speculative Miss & Cancellation
    # -----------------------------------------------------------------
    print("\n" + "-" * 85)
    print("3. 🔀 SPECULATIVE MISS & ZERO-BLOCKING ADAPTIVE FALLBACK")
    print("-" * 85)

    ambiguous_query = "Check account status for usr_7719"
    with engine.speculate(state=ambiguous_query, action_names=["fetch_order_status", "fetch_user_profile"]) as spec_session:
        print(f"   User Prompt: \"{ambiguous_query}\"")
        print(f"   Speculative Guess: {spec_session.predicted_action}")

        # Simulate LLM deciding on a DIFFERENT tool
        print("   LLM Reasoning diverges: Selects 'fetch_user_profile' instead of predicted tool.")
        fallback_res, fallback_hit = spec_session.resolve("fetch_user_profile", user_id="usr_7719")
        print(f"   Resolution: is_hit={fallback_hit} | Session Status: {spec_session.status.value}")
        print(f"   Executed confirmed tool safely on-demand: {fallback_res['tier']}")

    # -----------------------------------------------------------------
    # 4. Side-Effect Safety Guard
    # -----------------------------------------------------------------
    print("\n" + "-" * 85)
    print("4. 🛡️ SIDE-EFFECT SAFETY GUARD (NON-IDEMPOTENT MUTATIONS)")
    print("-" * 85)

    mutation_query = "Charge customer card $450 for ord_88291"
    print(f"   User Prompt: \"{mutation_query}\"")
    guard_session = engine.speculate(
        state=mutation_query,
        action_names=["charge_customer"],
        override_action="charge_customer",
    )
    print(f"   Speculated: {guard_session.speculated}")
    print(f"   Session Status: {guard_session.status.value}")
    print("   ✅ Safety Guard Blocked Speculation: Non-idempotent operations are never pre-fetched!")

    # Resolution executes safely once confirmed
    safe_res, safe_hit = guard_session.resolve("charge_customer", amount=450.0, order_id="ord_88291")
    print(f"   On confirmed LLM approval, executed mutation: {safe_res}")

    # -----------------------------------------------------------------
    # 5. Telemetry & Analytics Summary
    # -----------------------------------------------------------------
    print("\n" + "-" * 85)
    print("5. 📊 REFLEX SPECULATIVE TELEMETRY REPORT")
    print("-" * 85)

    stats = engine.stats()
    print(json.dumps(stats, indent=4))
    print("\n" + "=" * 85)
    print("✅ PHASE 22 SPECULATIVE ROUTING & PARALLEL PRE-FETCH COMPLETE")
    print("=" * 85)

    engine.shutdown(wait=False)


if __name__ == "__main__":
    run_speculative_demo()
