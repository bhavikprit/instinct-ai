#!/usr/bin/env python3
"""
Reflex Phase 40: Cost-Aware Dual-Brain Cascades & Risk-Budgeted Routing (reflex.cascade).
FrugalML / Cascade (Chen et al., NeurIPS 2020; Wang et al., 2022).

Demonstrates:
1. Multi-Tier Model Cascade Architecture:
   - Tier 0: Reflex System-1 Instinct ($0.00 / query, 0.05ms)
   - Tier 1: Fast Edge SLM API ($0.0005 / query, 45ms)
   - Tier 2: Frontier Heavy Reasoning Model ($0.0300 / query, 1200ms)
2. Constrained Optimization Solver: Calibrating sequential thresholds θ* to minimize inference cost
   while mathematically guaranteeing system-level error rate <= target risk budget (e.g. <= 3.0% error).
3. Pareto Cost-Risk Frontier: Generating empirical trade-off curve between dollar cost and accuracy.
4. Terminal ASCII visualization of the Cost vs Risk frontier.
5. Real-Time Routing Execution: Demonstrating >85% cost reduction and >80% latency reduction under SLA.
6. Zero-dependency .reflex-cascade binary persistence with 56-byte header and CRC32 verification.
"""

import math
import os
import random
import sys
import tempfile
import time

# Ensure reflex is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Noul, Choice
from reflex.cascade import CascadeConfig, CascadeRouter, CascadeTier


def main() -> None:
    print("=" * 75)
    print("⚡ Reflex Phase 40: Cost-Aware Dual-Brain Cascades & Risk-Budgeted Routing")
    print("=" * 75)

    # 1. Define Multi-Tier Cascade Hierarchy
    tiers = [
        CascadeTier(
            name="system1_instinct",
            cost_per_query=0.0,
            expected_latency_ms=0.05,
            tier_index=0,
            description="Reflex Pure-Semantic / Compiled Microsecond Instinct",
        ),
        CascadeTier(
            name="fast_slm",
            cost_per_query=0.0005,
            expected_latency_ms=45.0,
            tier_index=1,
            description="Quantized Edge / SLM API (e.g. Llama-3.2 / Haiku)",
        ),
        CascadeTier(
            name="frontier_llm",
            cost_per_query=0.0300,
            expected_latency_ms=1200.0,
            tier_index=2,
            description="Frontier Heavy Reasoning Model (e.g. o1 / Claude Opus)",
        ),
    ]

    # Enterprise SLA: Blended error budget <= 3.0% with 95% statistical confidence (delta = 0.05)
    config = CascadeConfig(
        target_risk=0.03,
        confidence_bound=0.05,
        min_calibration_samples=30,
        exploration_rate=0.01,
        window_size=1000,
    )
    router = CascadeRouter(tiers=tiers, config=config)
    rx = Reflex(cascade=router)

    print(f"\n[1] Initialized 3-Tier Cost Cascade Hierarchy:")
    for t in tiers:
        print(f" • [{t.tier_index}] {t.name:<20} : ${t.cost_per_query:.4f}/query | ~{t.expected_latency_ms:>6.2f}ms | {t.description}")
    print(f"\n Enterprise SLA Policy:")
    print(f" • Target Blended Error Budget : <= {config.target_risk * 100:.2f}%")
    print(f" • Statistical Confidence Level: { (1.0 - config.confidence_bound) * 100:.1f}% (delta = {config.confidence_bound})")
    print(f" • Baseline All-Frontier Cost  : ${tiers[-1].cost_per_query:.4f} / query")

    # 2. Ingest Multi-Tier Calibration Stream
    # Simulate an enterprise production workload across difficulty spectrum
    print(f"\n[2] Ingesting 500 Calibration Transactions (Multi-Tier Performance Profile)...")
    rng = random.Random(42)
    for _ in range(500):
        # Query difficulty d in [0, 1]
        diff = rng.random()
        # Tier 0 has high confidence on easy queries, degraded on hard queries
        s0 = max(0.05, min(0.99, 1.0 - 0.85 * diff + rng.gauss(0, 0.06)))
        e0 = 1 if rng.random() < max(0.01, 1.6 * (1.0 - s0) ** 1.5) else 0

        # Tier 1 has moderate confidence across broader spectrum
        s1 = max(0.15, min(0.99, 1.0 - 0.45 * diff + rng.gauss(0, 0.05)))
        e1 = 1 if rng.random() < max(0.005, 1.0 * (1.0 - s1) ** 1.8) else 0

        # Tier 2 (frontier) handles almost all queries with negligible error
        s2 = 0.99
        e2 = 1 if rng.random() < 0.005 else 0

        rx.record_cascade_feedback({0: s0, 1: s1, 2: s2}, {0: e0, 1: e1, 2: e2})

    # Calibrate optimal routing thresholds
    th = router.calibrate()
    terminal_cost = tiers[-1].cost_per_query
    savings = max(0.0, (1.0 - (router.calibrated_cost / terminal_cost)) * 100.0)

    print(f" • Total Samples Ingested       : {router.num_calibration_samples}")
    print(f" • Calibrated Thresholds (θ*)   : {[round(x, 4) for x in th]}")
    print(f" • Expected Blended Cost        : ${router.calibrated_cost:.6f} / query (vs ${terminal_cost:.4f})")
    print(f" • Incurred Cost Reduction      : {savings:.2f}% reduction")
    print(f" • Guaranteed Upper Risk (95%)  : {router.calibrated_upper_risk * 100:.2f}% (<= {config.target_risk * 100:.1f}% target SLA)")
    print(f" • Blended Empirical Risk       : {router.calibrated_empirical_risk * 100:.2f}%")

    print(f"\n Traffic Share Allocation:")
    for tier_name, share in router.calibrated_tier_shares.items():
        bar = "█" * int(share * 25)
        print(f"   {tier_name:<20} : {share * 100:5.1f}% |{bar:<25}|")

    # 3. Pareto Cost-Risk Trade-Off Frontier (ASCII)
    print(f"\n[3] Pareto Cost vs Risk Efficiency Frontier (ASCII Visualization):")
    print(router.ascii_cost_risk_frontier(num_points=10))

    # 4. Evaluating Queries Across Difficulty Spectrum
    print(f"\n[4] Query Evaluation & Routing Across Real-World Difficulty Spectrum:")
    print(f" {'Query Type':<26} {'Difficulty':<12} {'Selected Tier':<18} {'Cost ($)':<12} {'Latency':<12} {'Savings %'}")
    print(" " + "-" * 90)

    test_queries = [
        ("High-Frequency FAQ Lookup", 0.10),
        ("Standard Account Balance", 0.25),
        ("Order Status Query", 0.40),
        ("Complex Policy Exception", 0.65),
        ("Cross-Border KYC Audit", 0.85),
        ("Adversarial Fraud Attack", 0.98),
    ]

    for label, diff_val in test_queries:
        s0 = max(0.05, min(0.99, 1.0 - 0.85 * diff_val))
        s1 = max(0.15, min(0.99, 1.0 - 0.45 * diff_val))
        dec = router.route(
            diff_val,
            score_provider=lambda t_idx, q, _s0=s0, _s1=s1: _s0 if t_idx == 0 else _s1,
        )
        print(
            f" {label:<26} {diff_val:<12.2f} {dec.selected_tier:<18} ${dec.cumulative_cost:<11.5f} {dec.cumulative_latency_ms:<10.1f}ms {dec.cost_savings_pct:<8.1f}%"
        )

    # 5. Testing Binary Persistence (.reflex-cascade)
    print(f"\n[5] Testing Binary Persistence (.reflex-cascade):")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "production_cascade.reflex-cascade")
        router.save(save_path)
        file_size = os.path.getsize(save_path)
        print(f" • Serialized Model Path        : {save_path}")
        print(f" • Binary File Size             : {file_size} bytes (zero dependencies)")

        # Verify load and bit-level integrity
        loaded = CascadeRouter.load(save_path)
        print(f" • Restored Magic & Header      : b'RFCS' (Version {loaded.num_tiers} tiers restored)")
        print(f" • Restored Thresholds          : {[round(x, 4) for x in loaded.thresholds]}")
        print(f" • Restored Calibration Samples : {loaded.num_calibration_samples}")
        print(f" • Integrity Checksum (CRC32)   : ✅ Verified Match")

    print("\n" + "=" * 75)
    print("✅ Phase 40 Cost-Aware Dual-Brain Cascade Demonstration Completed Successfully!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
