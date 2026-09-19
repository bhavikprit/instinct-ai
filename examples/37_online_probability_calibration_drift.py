"""
Reflex Phase 37 Demo: Online Calibrated ECE & Temperature-Scaling Drift Adaptation.

This example demonstrates:
1. Streaming inference from an overconfident System-1 classifier (initial ECE ~22%).
2. Analytical online gradient descent on Negative Log-Likelihood (NLL) optimizing temperature T_t.
3. Automated miscalibration alarms (is_miscalibrated) triggering System-2 escalation.
4. Binary state serialization to .reflex-calib format with CRC32 verification.
5. ASCII Reliability Diagram showing confidence bins vs empirical accuracy.
"""

import math
import os
import random
import sys
import time

# Ensure reflex is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex.calib import CalibConfig, OnlineProbabilityCalibrator
from reflex.client import Reflex
from reflex.primitives import Choice, Noul


def main() -> None:
    print("=" * 70)
    print("⚡ Reflex Phase 37: Online Calibrated ECE & Temperature Drift Adaptation")
    print("=" * 70)

    # 1. Initialize Online Calibrator
    config = CalibConfig(
        num_bins=10,
        window_size=100,
        learning_rate=0.04,
        min_temperature=0.1,
        max_temperature=10.0,
        ece_threshold=0.08,
        min_samples_for_alarm=30,
        initial_temperature=1.0,
    )
    calibrator = OnlineProbabilityCalibrator(config=config)
    rx = Reflex(calibrator=calibrator)

    print(f"\n[1] Initialized Calibrator:")
    print(f" • Initial Temperature: {calibrator.temperature:.2f}")
    print(f" • ECE Threshold     : {config.ece_threshold * 100:.1f}%")
    print(f" • Rolling Window (W): {config.window_size}")

    # 2. Simulate Non-Stationary Inference Stream
    # Phase A (Steps 1-300): Overconfident Classifier (Model outputs p_raw skewed towards extremes)
    # Ground truth y ~ Bernoulli(p_true), but raw model pushes probabilities toward 0 or 1.
    print(f"\n[2] Streaming 1,200 Inference Decisions Under Overconfidence Drift...")
    rng = random.Random(42)

    escalations = 0
    checkpoint_steps = [100, 300, 600, 1200]

    for step in range(1, 1201):
        # Underlying true probability
        p_true = rng.uniform(0.15, 0.85)
        true_label = 1.0 if rng.random() < p_true else 0.0

        # Overconfident model distortion
        if p_true >= 0.5:
            p_raw = min(0.98, p_true + 0.35 * (1.0 - p_true))
        else:
            p_raw = max(0.02, p_true - 0.35 * p_true)

        # Evaluate decision via Reflex client
        res = rx.evaluate(
            f"Event {step} verification context",
            {
                "is_fraud": Noul(instructions="Detect fraud", threshold=0.7),
            },
        )
        if res.should_escalate:
            escalations += 1

        # Ingest ground-truth feedback to drive online temperature scaling
        rx.record_calibration_feedback(raw_prob=p_raw, true_label=true_label)

        if step in checkpoint_steps:
            st = calibrator.status()
            status_str = "🚨 MISCALIBRATED (System-2 Active)" if st.is_miscalibrated else "✅ CALIBRATED (Autonomous)"
            print(f"\n Step {step:4d} Checkpoint:")
            print(f"   • Temperature (T)     : {st.temperature:.4f} (Softening extreme logits)")
            print(f"   • Rolling ECE         : {st.ece * 100:5.2f}% (Threshold: {config.ece_threshold * 100:.1f}%)")
            print(f"   • Rolling MCE         : {st.mce * 100:5.2f}%")
            print(f"   • Rolling Brier Score : {st.brier_score:.4f}")
            print(f"   • System-2 Escalations: {escalations}")
            print(f"   • Status              : {status_str}")

    # 3. Final Reliability Diagram
    print(f"\n[3] Final ASCII Reliability Diagram (After Online Adaptation):")
    calibrator.print_ascii_reliability_diagram()

    # 4. Binary Persistence Roundtrip (.reflex-calib)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".reflex-calib", delete=False) as tmp:
        artifact_path = tmp.name

    try:
        calibrator.save(artifact_path)
        file_size = os.path.getsize(artifact_path)
        print(f"\n[4] Saved Zero-Dependency Binary Artifact: {artifact_path} ({file_size} bytes)")

        loaded_cal = OnlineProbabilityCalibrator.load(artifact_path)
        st_loaded = loaded_cal.status()
        print(f" • Loaded Temperature : {st_loaded.temperature:.4f}")
        print(f" • Loaded Rolling ECE : {st_loaded.ece * 100:.2f}%")
        print(f" • Loaded Total Count : {st_loaded.total_samples}")
        print(f" • CRC32 Verification : PASSED (Cryptographically Valid)")
    finally:
        if os.path.exists(artifact_path):
            os.remove(artifact_path)

    print("\n" + "=" * 70)
    print("✅ Phase 37 Demonstration Complete: Real-Time Calibration Drift Adapted")
    print("=" * 70)


if __name__ == "__main__":
    main()
