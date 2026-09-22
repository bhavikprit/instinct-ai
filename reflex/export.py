"""
Reflex Export & Quantization Tooling.
Dynamic INT8 ONNX quantization and temperature scaling calibration for System 1 models.
"""

from __future__ import annotations
import math
import os
from typing import Dict, List, Optional, Tuple

from reflex.rlcd.loss import brier_score, expected_calibration_error


def calibrate_temperature(
    logits: List[float],
    labels: List[int],
    init_temp: float = 1.0,
    lr: float = 0.05,
    max_iters: int = 100,
) -> float:
    """
    Optimizes temperature scaling factor T to minimize negative log likelihood (NLL).
    Zero external dependencies: uses native gradient descent with line search.
    
    Args:
        logits: Raw uncalibrated scalar model logits.
        labels: Binary ground truth labels (0 or 1).
        init_temp: Starting temperature (default: 1.0).
        lr: Learning rate.
        max_iters: Iteration steps.
        
    Returns:
        Calibrated temperature scalar T > 0.
    """
    if not logits or not labels or len(logits) != len(labels):
        raise ValueError("Logits and labels must be non-empty and of matching length.")

    # Work in log-temperature space: T = exp(s) ensures T > 0 strictly
    s = math.log(max(0.01, init_temp))
    n = len(logits)

    for _ in range(max_iters):
        t = math.exp(s)
        grad_s = 0.0

        for z, y in zip(logits, labels):
            scaled_z = z / t
            # Sigmoid probability
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, scaled_z))))
            # Gradient of NLL w.r.t scaled_z is (p - y)
            # d(scaled_z)/ds = d(z * exp(-s))/ds = -z * exp(-s) = -scaled_z
            grad_scaled = p - y
            grad_s += grad_scaled * (-scaled_z)

        grad_s = grad_s / n

        # Gradient update with clipping
        s -= lr * max(-5.0, min(5.0, grad_s))

    best_temp = round(math.exp(s), 4)
    return max(0.05, min(10.0, best_temp))


def evaluate_calibration(
    logits: List[float],
    labels: List[int],
    temperature: float = 1.0,
    num_bins: int = 10,
) -> Dict[str, float]:
    """Evaluates Brier Score and ECE for given logits under temperature T."""
    t = max(0.01, temperature)
    probs = [
        1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z / t))))
        for z in logits
    ]
    bs = brier_score(probs, labels)
    ece = expected_calibration_error(probs, labels, num_bins=num_bins)

    return {
        "temperature": t,
        "brier_score": round(bs, 4),
        "ece": round(ece, 4),
    }


def quantize_onnx_model(
    input_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Applies dynamic INT8 quantization to an existing ONNX model, reducing size by ~4x.
    Requires 'onnxruntime' package.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found at '{input_path}'")

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_int8{ext}"

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"Applying INT8 dynamic quantization to {input_path}...")
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"✅ Successfully exported quantized model: {output_path}")
        return output_path
    except ImportError as e:
        raise ImportError(
            "quantize_onnx_model requires 'onnxruntime'. "
            "Install via: pip install 'instinct-ai[local]'"
        ) from e
