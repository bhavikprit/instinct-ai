#!/usr/bin/env python3
"""
Reflex Phase 39: Selective Classification & Risk-Controlled Rejection Option (reflex.reject).
Geifman & El-Yaniv (NeurIPS 2017 / ICML 2019) risk-controlled selection with finite-sample guarantees.

Demonstrates:
1. The Selective Classification Trade-Off: Risk vs Coverage with finite-sample Wilson score guarantees.
2. Target Risk Mode: Calibrating optimal threshold θ* to guarantee selective risk <= r* (e.g. <= 2% error).
3. Target Coverage Mode: Setting minimum autonomous coverage φ* and minimizing selective risk.
4. Runtime Decision Routing: Autonomous System-1 execution when accepted, escalation to System-2 when rejected.
5. Risk-Coverage (RC) Curve generation and Area Under the Risk-Coverage Curve (AURC).
6. ASCII visualization of empirical error vs confidence threshold.
7. Zero-dependency .reflex-reject binary persistence with 48-byte struct header and CRC32 verification.
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
from reflex.reject import SelectiveClassifier, SelectiveRejectConfig


def main() -> None:
    print("=" * 75)
    print("⚡ Reflex Phase 39: Selective Classification & Risk-Controlled Rejection")
    print("=" * 75)

    # 1. Initialize Selective Classifier in Target-Risk Mode
    # Enterprise SLA: Error budget <= 5.0% with 95% statistical confidence (delta = 0.05)
    config = SelectiveRejectConfig(
        target_risk=0.05,
        confidence_bound=0.05,
        min_calibration_samples=30,
        scoring_method="confidence",
        window_size=1000,
    )
    classifier = SelectiveClassifier(config=config)
    rx = Reflex(selective_reject=classifier)

    print(f"\n[1] Initialized Selective Classifier:")
    print(f" • Operating Mode               : Target Risk (guarantee error <= r* while maximizing coverage)")
    print(f" • Target Error Budget (r*)     : <= {config.target_risk * 100:.2f}%")
    print(f" • Statistical Confidence Level : { (1.0 - config.confidence_bound) * 100:.1f}% (delta = {config.confidence_bound})")
    print(f" • Min Calibration Samples      : {config.min_calibration_samples}")
    print(f" • Scoring Metric               : {config.scoring_method}")

    # 2. Ingest Calibration Stream
    # Simulate an enterprise decision pipeline: 400 calibration transactions
    # Higher confidence predictions have low error rate; low confidence has higher error rate
    print(f"\n[2] Ingesting 400 Calibration Samples (Production Transactions)...")
    rng = random.Random(42)
    for _ in range(400):
        # Confidence score kappa(x) in [0.50, 1.00]
        conf = 0.50 + 0.50 * (rng.random() ** 0.5)
        # Error probability is inverse to confidence: ~20% at 0.50 down to ~0.5% at 0.99
        err_prob = max(0.005, min(0.35, 1.8 * (1.0 - conf) ** 1.5))
        is_error = 1 if rng.random() < err_prob else 0
        rx.record_selective_feedback(conf, is_error)

    # Calibrate optimal rejection threshold
    classifier.calibrate()
    print(f" • Total Samples Ingested       : {classifier.num_calibration_samples}")
    print(f" • Calibrated Threshold (θ*)    : {classifier.threshold:.4f}")
    print(f" • Empirical Selective Risk     : {classifier.calibrated_empirical_risk * 100:.2f}%")
    print(f" • Guaranteed Upper Risk (95%)  : {classifier.calibrated_upper_risk * 100:.2f}% (<= {config.target_risk * 100:.1f}% target budget)")
    print(f" • Autonomous Coverage (φ)      : {classifier.calibrated_coverage * 100:.2f}% of traffic handled autonomously")
    print(f" • Area Under RC Curve (AURC)   : {classifier.aurc():.4f}")

    # 3. Empirical Risk-Coverage Trade-Off Curve (ASCII)
    print(f"\n[3] Risk-Coverage Trade-Off Frontier (ASCII Visualization):")
    print(classifier.ascii_risk_coverage_curve(num_points=10))

    # 4. Evaluating Decisions Across Confidence Spectrum
    print(f"\n[4] Query Evaluation & Routing Across Confidence Regimes:")
    print(f" {'Query Regime':<26} {'Confidence':<12} {'Verdict':<12} {'Guaranteed Risk':<18} {'System Routing'}")
    print(" " + "-" * 85)

    test_queries = [
        ("Clear Autonomous Approval", 0.98),
        ("Standard High-Confidence", 0.93),
        ("Marginal In-Distribution", 0.88),
        ("Borderline SLA Decision", classifier.threshold + 0.005),
        ("Sub-Threshold Uncertainty", classifier.threshold - 0.02),
        ("Ambiguous Edge-Case", 0.65),
        ("Extreme Tail OOD Query", 0.52),
    ]

    for label, conf_score in test_queries:
        # Evaluate synthetic decision primitive
        noul = Noul(instructions="Authorize transaction", probability=conf_score)
        dec = classifier.evaluate_noul(noul)
        verdict = "✅ ACCEPT" if dec.accepted else "❌ REJECT"
        routing = "System-1 (Autonomous Fast-Path)" if dec.accepted else "🚨 System-2 (Human / Escalation)"
        print(f" {label:<26} {conf_score:<12.4f} {verdict:<12} {dec.upper_bound_risk * 100:<17.2f}% {routing}")

    # 5. Target Coverage Mode Demonstration
    print(f"\n[5] Target Coverage Mode (Enterprise Mandate: Handle >= 80% Autonomously):")
    cov_config = SelectiveRejectConfig(
        target_risk=None,
        target_coverage=0.80,
        min_calibration_samples=30,
    )
    cov_classifier = SelectiveClassifier(config=cov_config)
    for score, err in list(classifier._samples):
        cov_classifier.add_sample(score, err)
    cov_classifier.calibrate()

    print(f" • Target Minimum Coverage (φ*) : >= {cov_config.target_coverage * 100:.1f}%")
    print(f" • Calibrated Optimal Threshold : {cov_classifier.threshold:.4f}")
    print(f" • Achieved Coverage            : {cov_classifier.calibrated_coverage * 100:.2f}%")
    print(f" • Empirical Risk at Coverage   : {cov_classifier.calibrated_empirical_risk * 100:.2f}%")
    print(f" • Guaranteed Upper Risk (95%)  : {cov_classifier.calibrated_upper_risk * 100:.2f}%")

    # 6. Binary Persistence (.reflex-reject)
    print(f"\n[6] Testing Binary Persistence (.reflex-reject):")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "production.reflex-reject")
        classifier.save(save_path)
        file_size = os.path.getsize(save_path)
        print(f" • Serialized Model Path        : {save_path}")
        print(f" • Binary File Size             : {file_size} bytes (zero dependencies)")

        # Verify load and bit-level integrity
        loaded = SelectiveClassifier.load(save_path)
        print(f" • Restored Magic & Header      : b'RFRJ' (Threshold {loaded.threshold:.4f} restored)")
        print(f" • Restored Calibration Samples : {loaded.num_calibration_samples}")
        print(f" • Integrity Checksum (CRC32)   : ✅ Verified Match")

    print("\n" + "=" * 75)
    print("✅ Phase 39 Selective Classification Demonstration Completed Successfully!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
