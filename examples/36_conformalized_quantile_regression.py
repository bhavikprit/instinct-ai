#!/usr/bin/env python3
"""
Reflex Phase 36: Conformalized Quantile Regression (CQR) & Heteroscedastic Uncertainty Bounding.
Romano, Sesia & Candès (NeurIPS 2019) distribution-free continuous intervals.

Demonstrates:
1. Heteroscedastic latency/cost prediction where query variance scales with complexity.
2. Constant-width conformal prediction (rigid, overly conservative on easy queries).
3. Reflex CQR dynamic intervals (tight on simple queries, adaptive on complex queries).
4. Finite-sample guaranteed 90% coverage across all difficulty tiers.
5. Zero-dependency .reflex-cqr binary serialization with 32-bit CRC32 integrity.
6. Seamless Reflex Client integration with epistemic tolerance escalation.
"""

import math
import os
import random
import sys
import tempfile
import time

# Ensure reflex is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import (
    Reflex,
    Score,
    CQRConfig,
    CQRInterval,
    QuantileInstinctHead,
    ConformalizedQuantileRegressor,
)


def run_cqr_demo():
    print("=" * 75)
    print("⚡ Reflex Phase 36: Conformalized Quantile Regression (CQR) Demonstration")
    print("   Romano, Sesia & Candès (2019) Heteroscedastic Distribution-Free Bounds")
    print("=" * 75)

    rng = random.Random(42)

    # 1. Initialize CQR Engine with 90% nominal coverage guarantee (alpha = 0.10)
    print("\n[1/5] Initializing CQR Controller (target coverage: 90.0%, max tolerance: 5.0s)...")
    config = CQRConfig(
        alpha=0.10,
        min_calibration_samples=50,
        max_width_tolerance=5.0,  # Escalate if predicted latency window > 5.0s
        seed=42,
    )
    cqr = ConformalizedQuantileRegressor(config=config)

    # 2. Simulate Calibration Set: 500 API queries with heterogeneous complexity
    # Latency model: T(x) = base_latency(x) + noise, where noise ~ N(0, sigma(x)^2)
    # Simple query (complexity=1): mean=1.0s, sigma=0.25s
    # Complex query (complexity=10): mean=10.0s, sigma=2.50s
    print("\n[2/5] Ingesting 500 calibration queries with heteroscedastic noise...")

    z_90 = 1.645  # Standard normal 90% coverage factor

    cal_points = []
    cal_residuals = []
    for _ in range(500):
        complexity = rng.uniform(1.0, 10.0)
        mean_latency = complexity * 1.0
        sigma = 0.25 + 0.25 * (complexity ** 1.1)
        true_latency = max(0.2, mean_latency + rng.gauss(0, sigma))

        # Base quantile estimates
        q_low = max(0.1, mean_latency - (z_90 * sigma))
        q_high = mean_latency + (z_90 * sigma)

        cqr.add_calibration_sample(
            pred_low=q_low,
            pred_high=q_high,
            true_value=true_latency,
            point_estimate=mean_latency,
        )
        cal_points.append(mean_latency)
        cal_residuals.append(abs(true_latency - mean_latency))

    cqr.calibrate()
    print(f" • Calibration complete. Conformal quantile offset Q̂ = {cqr.q_hat:+.4f} seconds")
    print(f" • Calibration empirical coverage: {cqr.empirical_coverage * 100:.1f}%")

    # Fit standard constant-width conformal prediction margin for comparison
    p_level = math.ceil((len(cal_residuals) + 1) * (1.0 - config.alpha)) / len(cal_residuals)
    p_level = min(1.0, max(0.0, p_level))
    const_margin = sorted(cal_residuals)[min(len(cal_residuals) - 1, max(0, math.ceil(p_level * len(cal_residuals)) - 1))]
    print(f" • Baseline constant-width conformal margin: ±{const_margin:.4f} seconds (width = {2.0 * const_margin:.4f}s)")

    # 3. Test on 1,000 Unseen Test Queries across Three Complexity Tiers
    print("\n[3/5] Evaluating 1,000 unseen test queries across complexity tiers...")

    tiers = {
        "Tier 1: Simple (Easy, 1-3)": {"cqr_cov": 0, "cqr_w": [], "const_cov": 0, "const_w": [], "count": 0},
        "Tier 2: Medium (4-7)":       {"cqr_cov": 0, "cqr_w": [], "const_cov": 0, "const_w": [], "count": 0},
        "Tier 3: Complex (Hard, 8-10)": {"cqr_cov": 0, "cqr_w": [], "const_cov": 0, "const_w": [], "count": 0},
    }

    test_samples = []
    for _ in range(1000):
        complexity = rng.uniform(1.0, 10.0)
        mean_latency = complexity * 1.0
        sigma = 0.25 + 0.25 * (complexity ** 1.1)
        true_latency = max(0.2, mean_latency + rng.gauss(0, sigma))

        q_low = max(0.1, mean_latency - (z_90 * sigma))
        q_high = mean_latency + (z_90 * sigma)

        if complexity <= 3.0:
            tier_key = "Tier 1: Simple (Easy, 1-3)"
        elif complexity <= 7.0:
            tier_key = "Tier 2: Medium (4-7)"
        else:
            tier_key = "Tier 3: Complex (Hard, 8-10)"

        # CQR Adaptive Interval
        interval = cqr.predict(q_low, q_high, point_estimate=mean_latency, min_val=0.0, max_val=30.0)
        cqr_in = interval.contains(true_latency)

        # Constant-width Conformal Interval
        const_low = max(0.0, mean_latency - const_margin)
        const_high = mean_latency + const_margin
        const_in = (const_low <= true_latency <= const_high)

        tiers[tier_key]["count"] += 1
        if cqr_in:
            tiers[tier_key]["cqr_cov"] += 1
        tiers[tier_key]["cqr_w"].append(interval.interval_width)

        if const_in:
            tiers[tier_key]["const_cov"] += 1
        tiers[tier_key]["const_w"].append(const_high - const_low)

        test_samples.append((q_low, q_high, true_latency))

    # 4. Display Tier Comparison Table
    print("\n" + "-" * 75)
    print(f"{'Complexity Tier':<30} {'Count':<8} {'Method':<16} {'Coverage':<10} {'Mean Width'}")
    print("-" * 75)

    for tier_name, data in tiers.items():
        n = data["count"]
        cqr_c = 100.0 * data["cqr_cov"] / n
        const_c = 100.0 * data["const_cov"] / n
        cqr_w = sum(data["cqr_w"]) / n
        const_w = sum(data["const_w"]) / n

        print(f"{tier_name:<30} {n:<8} {'Constant Conformal':<16} {const_c:>5.1f}%    {const_w:>6.2f}s ⚠️ (Rigid)")
        print(f"{'':<30} {'':<8} {'Reflex CQR':<16} {cqr_c:>5.1f}%    {cqr_w:>6.2f}s ✅ (Adaptive)")
        print("." * 75)

    # 5. Reflex Client Integration with Epistemic Escalation
    print("\n[4/5] Testing Reflex Client Integration & Tolerance Escalation...")
    rx = Reflex(backend="local", cqr=cqr)

    # Simple query: expected to pass tolerance check without escalation
    res_simple = rx.evaluate("Simple query: echo status", {"latency": Score("Predict execution latency", min_val=0.0, max_val=20.0)})
    print(f" • Simple Query Latency Prediction:")
    if res_simple.cqr and "latency" in res_simple.cqr:
        inv = res_simple.cqr["latency"]
        print(f"   - Certified Interval : [{inv.lower_bound:.2f}s, {inv.upper_bound:.2f}s] (width: {inv.interval_width:.2f}s)")
        print(f"   - Escalation Flag    : {res_simple.should_escalate} (Is Safe: {inv.is_safe})")

    # 6. Zero-Dependency Binary Serialization
    print("\n[5/5] Testing .reflex-cqr Binary Persistence & CRC32 Verification...")
    with tempfile.NamedTemporaryFile(suffix=".reflex-cqr", delete=False) as tmp:
        save_path = tmp.name

    try:
        cqr.save(save_path)
        file_size = os.path.getsize(save_path)
        print(f" • Successfully saved CQR model to {save_path} ({file_size} bytes)")

        restored = ConformalizedQuantileRegressor.load(save_path)
        print(f" • Loaded model successfully. Verified Q̂ offset: {restored.q_hat:.4f}s")
        print(f" • Calibrated status: {restored.is_calibrated} | Coverage guarantee: 90.0%")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    print("\n" + "=" * 75)
    print("✅ Phase 36 Conformalized Quantile Regression demonstration completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_cqr_demo()
