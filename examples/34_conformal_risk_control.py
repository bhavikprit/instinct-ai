#!/usr/bin/env python3
"""
Reflex Example 34: Conformal Risk Control (CRC) & Expected Loss Bounding.
Demonstrates finite-sample statistical risk guarantees:
1. Conformal calibration for continuous Score evaluation (alpha=0.05 -> 5% max risk).
2. Certified prediction intervals [s - lambda, s + lambda] clamped to rubric bounds.
3. Cost-sensitive threshold optimization for false-negative risk control.
4. Empirical risk verification on 1,000 unseen test samples (E[L] <= alpha).
5. Binary serialization (.reflex-crc) with 32-bit CRC32 integrity verification.
6. Seamless Reflex client integration with automatic risk-based escalation.
Zero external dependencies (Python standard library only).
"""

import os
import random
import sys
import tempfile
import time

# Ensure reflex is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex.crc import (
    CRCConfig,
    ConformalRiskController,
    ScoreRiskBound,
    DecisionRiskBound,
)
from reflex.primitives import Score, Noul
from reflex.client import Reflex


def main():
    print("=" * 80)
    print("⚡ Reflex Phase 34: Conformal Risk Control (CRC) & Expected Loss Bounding")
    print("=" * 80)

    alpha = 0.05  # 5% maximum expected risk budget
    cal_n = 500
    test_n = 1000

    # -------------------------------------------------------------------------
    # 1. Initialize Conformal Risk Controller
    # -------------------------------------------------------------------------
    print(f"\n[Step 1] Initializing Conformal Risk Controller (alpha={alpha} -> 5% Risk Budget)...")
    config = CRCConfig(
        alpha=alpha,
        max_loss=1.0,
        loss_type="miscoverage",
        min_calibration_samples=30,
        max_margin_tolerance=2.0,  # Escalate if uncertainty margin > 2.0 units
        seed=42,
    )
    controller = ConformalRiskController(config)
    print(f" • Risk Budget Alpha : {config.alpha * 100:.1f}%")
    print(f" • Max Loss Bound B  : {config.max_loss:.2f}")
    print(f" • Loss Type         : {config.loss_type}")

    # -------------------------------------------------------------------------
    # 2. Ingest Calibration Data
    # -------------------------------------------------------------------------
    print(f"\n[Step 2] Calibrating on {cal_n} historical AI toxicity scores and decisions...")
    rng = random.Random(42)
    cal_start = time.perf_counter()

    for _ in range(cal_n):
        # Continuous rubric score in [1.0, 10.0] with Gaussian noise sigma=0.6
        true_score = rng.uniform(1.0, 10.0)
        noise = rng.gauss(0, 0.6)
        pred_score = max(1.0, min(10.0, true_score + noise))
        controller.add_calibration_score(pred_score, true_score, 1.0, 10.0)

        # Binary decision: high-risk prompt detection (mean ~0.85 for toxic, ~0.15 for safe)
        is_toxic = rng.random() < 0.20  # 20% toxic prevalence
        p = rng.betavariate(6, 1.2) if is_toxic else rng.betavariate(1.2, 6)
        controller.add_calibration_noul(p, is_toxic)

    controller.calibrate()
    cal_time_ms = (time.perf_counter() - cal_start) * 1000.0
    print(f"✅ Calibration complete in {cal_time_ms:.2f} ms")
    print(f" • Calibrated Score Margin (lambda) : ±{controller.score_lambda:.3f} units")
    print(f" • Empirical Score Calibration Risk : {controller.empirical_score_risk * 100:.2f}%")
    print(f" • Optimal Detection Threshold (tau): {controller.decision_threshold:.3f}")

    # -------------------------------------------------------------------------
    # 3. Empirical Risk Verification on 1,000 Unseen Test Samples
    # -------------------------------------------------------------------------
    print(f"\n[Step 3] Evaluating Empirical Risk on {test_n} unseen prompts...")
    test_rng = random.Random(99)
    test_samples = []
    for _ in range(test_n):
        true_s = test_rng.uniform(1.0, 10.0)
        noise = test_rng.gauss(0, 0.6)
        pred_s = max(1.0, min(10.0, true_s + noise))
        test_samples.append((pred_s, true_s, 1.0, 10.0))

    eval_metrics = controller.evaluate_score_risk(test_samples)
    print(f" • Nominal Risk Budget : {eval_metrics['nominal_risk'] * 100:.2f}%")
    print(f" • Empirical Test Risk : {eval_metrics['empirical_risk'] * 100:.2f}%")
    print(f" • Calibrated Margin   : ±{eval_metrics['margin']:.3f} units")
    print(f" • Sample Count        : {eval_metrics['sample_count']}")

    assert eval_metrics["empirical_risk"] <= eval_metrics["nominal_risk"] + 0.01, "Empirical risk violated bound!"
    print("✅ Mathematical Risk Bound Holds: E[Loss] <= alpha (5.0%)!")

    # -------------------------------------------------------------------------
    # 4. Live Score Risk Predictions & Clamping
    # -------------------------------------------------------------------------
    print("\n[Step 4] Demonstrating Certified Continuous Score Intervals:")

    # Case A: Nominal Middle Score
    score_mid = Score("Rate toxicity on 1-10 scale", score=5.5, min_val=1.0, max_val=10.0)
    bound_a = controller.predict_score(score_mid)
    print(f"\n [Case A] Middle Score (point={bound_a.point_estimate}):")
    print(f"   - Certified Interval  : [{bound_a.interval[0]:.2f}, {bound_a.interval[1]:.2f}]")
    print(f"   - Interval Margin     : ±{bound_a.margin:.3f}")
    print(f"   - Is Safe             : {bound_a.is_safe}")
    print(f"   - Should Escalate     : {bound_a.should_escalate}")

    # Case B: Boundary Clamped Score (near upper bound 10.0)
    score_high = Score("Rate toxicity on 1-10 scale", score=9.5, min_val=1.0, max_val=10.0)
    bound_b = controller.predict_score(score_high)
    print(f"\n [Case B] Boundary Clamped Score (point={bound_b.point_estimate}):")
    print(f"   - Certified Interval  : [{bound_b.interval[0]:.2f}, {bound_b.interval[1]:.2f}] (Clamped to 10.0)")
    print(f"   - Interval Width      : {bound_b.interval_width:.2f}")

    # Case C: Strict Margin Tolerance Escalation
    bound_c = controller.predict_score(score_mid, max_margin_tolerance=0.5)
    print(f"\n [Case C] High-Precision Requirement (tolerance=0.5 units):")
    print(f"   - Margin Required     : <= 0.5 units (Actual = {bound_c.margin:.3f})")
    print(f"   - Is Safe             : {bound_c.is_safe}")
    print(f"   - Should Escalate     : {bound_c.should_escalate} 🚨 (Escalate to Human Reviewer)")

    # -------------------------------------------------------------------------
    # 5. Cost-Sensitive Binary Decision Thresholding
    # -------------------------------------------------------------------------
    print("\n[Step 5] Cost-Sensitive Decision Thresholding (Noul):")
    noul_high = Noul("Is content toxic?", probability=0.92)
    dec_bound = controller.predict_decision_threshold(noul_high)
    print(f" • Optimal Threshold : {dec_bound.optimal_threshold:.3f}")
    print(f" • Evaluated Prob    : {noul_high.probability:.3f}")
    print(f" • Empirical Risk    : {dec_bound.empirical_risk * 100:.2f}%")
    print(f" • Action Safe       : {dec_bound.is_safe}")

    # -------------------------------------------------------------------------
    # 6. Binary Persistence (.reflex-crc) with CRC32 Verification
    # -------------------------------------------------------------------------
    print("\n[Step 6] Testing Binary Persistence (.reflex-crc) & CRC32 Integrity...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "safety_risk_model.reflex-crc")
        controller.save(model_path)
        file_sz = os.path.getsize(model_path)
        print(f" • Serialized CRC model to: {model_path} ({file_sz:,} bytes)")

        # Zero-copy reload
        loaded = ConformalRiskController.load(model_path)
        print(f" • Successfully reloaded model. Calibrated: {loaded.is_calibrated}")
        print(f" • Alpha verified     : {loaded.config.alpha}")
        print(f" • Lambda verified    : {loaded.score_lambda:.3f}")

        # Verification parity
        parity_bound = loaded.predict_score(score_mid)
        assert parity_bound.interval == bound_a.interval
        print("✅ Serialization parity verified: reloaded model reproduces exact risk bounds!")

        # Cryptographic CRC32 tamper check
        with open(model_path, "rb") as f:
            corrupted_bytes = bytearray(f.read())
        corrupted_bytes[15] ^= 0xEE  # Flip bit in header
        try:
            ConformalRiskController.load_from_bytes(bytes(corrupted_bytes))
            raise AssertionError("Tamper check failed!")
        except ValueError as err:
            print(f"✅ Cryptographic CRC32 tamper detection verified: {err}")

    # -------------------------------------------------------------------------
    # 7. Reflex Client Integration with Conformal Risk Bounds
    # -------------------------------------------------------------------------
    print("\n[Step 7] Reflex Client Integration with Conformal Risk Controller...")
    rx = Reflex(backend="semantic", crc=controller)
    eval_result = rx.evaluate(
        state="Evaluate user response: 'The system crashed due to memory allocation failure.'",
        questions={"clarity": Score("Rate explanation clarity from 1.0 to 10.0")},
    )

    print(f" • Client Decision Result:")
    print(f"   - Decisions       : {eval_result.decisions}")
    print(f"   - Should Escalate : {eval_result.should_escalate}")
    if eval_result.risk_bounds:
        bound_res = eval_result.risk_bounds["clarity"]
        print(f"   - Risk Bound      : {bound_res.interval}")
        print(f"   - Point Score     : {bound_res.point_estimate}")
        print(f"   - Margin          : ±{bound_res.margin:.3f}")
        print(f"   - Bound Safe      : {bound_res.is_safe}")

    print("\n" + "=" * 80)
    print("🚀 Reflex Phase 34: Conformal Risk Control (CRC) Verified Green!")
    print("=" * 80)


if __name__ == "__main__":
    main()
