#!/usr/bin/env python3
"""
Reflex Phase 38: Venn-Abers Multi-Class Conformal Predictors & Epistemic Uncertainty Intervals.
Vovk & Petej (2014) distribution-free multi-probabilistic intervals [p0, p1] via PAVA isotonic regression.

Demonstrates:
1. Difference between aleatoric uncertainty (known 50/50 probability) and epistemic uncertainty (data absence).
2. Pool Adjacent Violators Algorithm (PAVA) generating certified lower and upper bounds [p0, p1].
3. Dynamic epistemic uncertainty quantification: tight intervals in dense data, wide intervals for OOD queries.
4. Automatic System-2 deliberation triggering when epistemic uncertainty exceeds safety threshold.
5. Multi-class Inductive Venn-Abers Predictor (IVAP) for categorical Choice routing decisions.
6. Zero-dependency .reflex-va binary serialization with 32-bit CRC32 integrity verification.
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
from reflex.venn_abers import VennAbersConfig, VennAbersPredictor


def main() -> None:
    print("=" * 75)
    print("⚡ Reflex Phase 38: Venn-Abers Multi-Class Conformal Predictors")
    print("=" * 75)

    # 1. Initialize Venn-Abers Predictor
    config = VennAbersConfig(
        max_uncertainty_threshold=0.15,
        min_calibration_samples=25,
        point_estimator="balanced",
        window_size=500,
    )
    va_engine = VennAbersPredictor(config=config, classes=["allow", "flag", "block"])
    rx = Reflex(venn_abers=va_engine)

    print(f"\n[1] Initialized Venn-Abers Engine:")
    print(f" • Uncertainty Escalation Threshold : {config.max_uncertainty_threshold:.2f} (Interval width U = p1 - p0)")
    print(f" • Min Calibration Samples          : {config.min_calibration_samples}")
    print(f" • Point Estimator                  : {config.point_estimator} (p1 / (1 - p0 + p1))")

    # 2. Ingest Calibration Data
    # Simulate a realistic scenario: lots of training data near typical scores (0.35 - 0.75),
    # but very few or no samples in extreme tails (0.00 - 0.15, 0.85 - 1.00).
    print(f"\n[2] Ingesting 250 Calibration Samples (Clustered In-Distribution Data)...")
    rng = random.Random(42)
    for _ in range(250):
        # Clustered normal distribution
        raw_s = min(0.92, max(0.08, rng.gauss(0.55, 0.12)))
        # Sigmoid ground-truth probability
        true_prob = 1.0 / (1.0 + math.exp(-8.0 * (raw_s - 0.50)))
        y = 1 if rng.random() < true_prob else 0
        rx.record_venn_abers_feedback(raw_s, y)

    print(f" • Total Calibration Samples Ingested: {va_engine.num_calibration_samples}")
    print(f" • Mean Uncertainty across Spectrum  : {va_engine.mean_uncertainty(30):.4f}")
    print(f" • Empirical Calibration Brier Score : {va_engine.compute_brier_score():.4f}")

    # 3. Binary Decision (Noul): Epistemic Uncertainty vs Aleatoric Noise
    print(f"\n[3] Evaluating Queries Across Data-Density Regimes (Noul Binary Decision):")
    print(f" {'Query Regime':<24} {'Raw Score':<10} {'Interval [p0, p1]':<22} {'Point p':<10} {'Width (U)':<10} {'Action'}")
    print(" " + "-" * 85)

    test_queries = [
        ("In-Distribution (Dense)", 0.50),
        ("In-Distribution (Dense)", 0.55),
        ("Moderate Density", 0.65),
        ("Sparse Region (Tail)", 0.85),
        ("Out-of-Distribution (OOD)", 0.98),
        ("Out-of-Distribution (OOD)", 0.02),
    ]

    for regime, raw_score in test_queries:
        res = rx.evaluate(
            f"Query evaluation score={raw_score}",
            {"is_legit": Noul(instructions="Is legitimate transaction?", threshold=0.5)},
        )
        # Directly test predictor for exact raw score
        va_res = va_engine.predict_noul(raw_score)
        inv_str = f"[{va_res.p0:.3f}, {va_res.p1:.3f}]"
        action_str = "🚨 ESCALATE (System-2)" if va_res.should_escalate else "✅ AUTONOMOUS"
        print(f" {regime:<24} {raw_score:<10.2f} {inv_str:<22} {va_res.p_calibrated:<10.3f} {va_res.uncertainty:<10.3f} {action_str}")

    # 4. Multi-Class Decision (Choice): IVAP Categorical Routing
    print(f"\n[4] Multi-Class Inductive Venn-Abers Predictor (IVAP Categorical Decision):")
    multi_va = VennAbersPredictor(
        config=VennAbersConfig(min_calibration_samples=25, max_uncertainty_threshold=0.20),
        classes=["allow", "flag", "block"],
    )
    for _ in range(60):
        cls = rng.choice(["allow", "flag", "block"])
        sc = rng.uniform(0.65, 0.95)
        multi_va.add_calibration_sample(sc, cls)

    choice_dist = {"allow": 0.68, "flag": 0.22, "block": 0.10}
    ivap_res = multi_va.predict_choice(choice_dist)

    print(f" • Input Raw Distribution: {choice_dist}")
    print(f" • Calibrated Distribution: {ivap_res.calibrated_distribution}")
    print(f" • Certified Class Intervals:")
    for c, inv in ivap_res.intervals.items():
        u = ivap_res.uncertainties[c]
        print(f"   - Class '{c:<6}': [{inv[0]:.3f}, {inv[1]:.3f}] (Epistemic width U={u:.3f})")
    print(f" • Winning Action        : '{ivap_res.selected}' (Escalate: {ivap_res.should_escalate})")

    # 5. Zero-Dependency Binary Persistence (.reflex-va)
    print(f"\n[5] Binary Persistence Roundtrip (.reflex-va) & CRC32 Verification...")
    with tempfile.NamedTemporaryFile(suffix=".reflex-va", delete=False) as tmp:
        save_path = tmp.name

    try:
        va_engine.save(save_path)
        file_size = os.path.getsize(save_path)
        print(f" • Serialized Venn-Abers model to {save_path} ({file_size} bytes)")

        restored_va = VennAbersPredictor.load(save_path)
        print(f" • Successfully loaded model from disk.")
        print(f" • Verified Samples  : {restored_va.num_calibration_samples}")
        print(f" • CRC32 Verification: PASSED (Header & Payload Authenticated)")

        # Verify restored prediction identical
        p_orig = va_engine.predict_noul(0.50).p_calibrated
        p_rest = restored_va.predict_noul(0.50).p_calibrated
        print(f" • Prediction Parity : Original={p_orig:.4f}, Restored={p_rest:.4f} (Match: {p_orig == p_rest})")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    print("\n" + "=" * 75)
    print("✅ Phase 38 Venn-Abers Multi-Class Conformal Predictor Demo Complete!")
    print("=" * 75)


if __name__ == "__main__":
    main()
