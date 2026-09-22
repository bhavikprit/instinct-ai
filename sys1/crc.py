"""
Reflex Conformal Risk Control (CRC) & Expected Loss Bounding.
Phase 34: Generalizes distribution-free statistical guarantees from set coverage (0-1 loss)
to continuous rubric evaluations, score intervals, and arbitrary bounded loss functions:
    E[L(f_lambda(X), Y)] <= alpha
Based on the Conformal Risk Control framework (Angelopoulos, Bates, Candes, Jordan, Wainwright 2024;
Bates, Candes, Lei, Romano 2021).
Provides:
1. Finite-sample statistical risk guarantees without distributional assumptions.
2. Certified prediction intervals for continuous Score primitives [s - lambda, s + lambda].
3. Cost-sensitive threshold optimization (e.g. false negative risk control).
4. Binary persistence format (.reflex-crc) with 32-bit CRC32 checksum verification.
5. Integration with Reflex client runtime and decision primitives.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations

import json
import math
import os
import struct
import threading
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from sys1.primitives import Noul, Choice, Score

# Binary serialization constants
CRC_MAGIC: bytes = b"RFCR"  # Reflex Conformal Risk
CRC_FORMAT_VERSION: int = 1


@dataclass
class CRCConfig:
    """Configuration for Conformal Risk Control."""
    alpha: float = 0.05  # Upper bound on expected loss: E[L] <= alpha
    max_loss: float = 1.0  # Loss upper bound B: L in [0, B]
    loss_type: str = "miscoverage"  # "miscoverage", "excess_error", "false_negative"
    min_calibration_samples: int = 20
    max_margin_tolerance: Optional[float] = None  # Escalate if interval margin exceeds this
    seed: Optional[int] = None

    def __post_init__(self):
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {self.alpha}")
        if self.max_loss <= 0.0:
            raise ValueError(f"max_loss must be positive, got {self.max_loss}")
        if self.min_calibration_samples < 5:
            raise ValueError(f"min_calibration_samples must be >= 5, got {self.min_calibration_samples}")


@dataclass
class ScoreRiskBound:
    """Certified risk bound and prediction interval for a continuous Score primitive."""
    point_estimate: float
    interval: Tuple[float, float]
    margin: float
    alpha: float
    max_loss: float
    instructions: str = ""
    is_safe: bool = True
    should_escalate: bool = False
    empirical_risk: float = 0.0

    @property
    def interval_width(self) -> float:
        return self.interval[1] - self.interval[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_estimate": round(self.point_estimate, 4),
            "interval": [round(self.interval[0], 4), round(self.interval[1], 4)],
            "interval_width": round(self.interval_width, 4),
            "margin": round(self.margin, 4),
            "alpha": self.alpha,
            "max_loss": self.max_loss,
            "is_safe": self.is_safe,
            "should_escalate": self.should_escalate,
            "empirical_risk": round(self.empirical_risk, 6),
            "instructions": self.instructions,
        }


@dataclass
class DecisionRiskBound:
    """Certified risk bound and operating threshold for detection / binary decision."""
    optimal_threshold: float
    empirical_risk: float
    alpha: float
    max_loss: float
    loss_type: str
    is_safe: bool = True
    should_escalate: bool = False
    instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_threshold": round(self.optimal_threshold, 4),
            "empirical_risk": round(self.empirical_risk, 6),
            "alpha": self.alpha,
            "max_loss": self.max_loss,
            "loss_type": self.loss_type,
            "is_safe": self.is_safe,
            "should_escalate": self.should_escalate,
            "instructions": self.instructions,
        }


class ConformalRiskController:
    """
    Distribution-Free Conformal Risk Controller.
    Implements the finite-sample risk control criterion:
        (n / (n + 1)) * R_hat(lambda) + (B / (n + 1)) <= alpha
    guaranteeing E[L(lambda_hat)] <= alpha for bounded monotonic losses.
    """

    def __init__(self, config: Optional[CRCConfig] = None):
        self.config = config or CRCConfig()
        self._lock = threading.RLock()

        # Calibration data stores
        # Continuous score calibration: list of (predicted_score, true_score, min_val, max_val)
        self.score_cal_samples: List[Tuple[float, float, float, float]] = []

        # Decision/Noul calibration: list of (probability, true_label_bool)
        self.noul_cal_samples: List[Tuple[float, bool]] = []

        # Calibrated parameters
        self.score_lambda: Optional[float] = None
        self.decision_threshold: Optional[float] = None
        self.empirical_score_risk: float = 0.0
        self.empirical_decision_risk: float = 0.0
        self._is_calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._is_calibrated

    # -------------------------------------------------------------------------
    # Calibration Ingestion
    # -------------------------------------------------------------------------

    def add_calibration_score(
        self,
        predicted_score: float,
        true_score: float,
        min_val: float = 1.0,
        max_val: float = 10.0,
    ) -> None:
        """Adds a calibration sample for continuous Score evaluations."""
        with self._lock:
            self.score_cal_samples.append(
                (float(predicted_score), float(true_score), float(min_val), float(max_val))
            )
            self._is_calibrated = False

    def add_calibration_noul(self, prob: float, true_label: bool) -> None:
        """Adds a calibration sample for binary decisions / false-negative risk control."""
        with self._lock:
            p = max(0.0, min(1.0, float(prob)))
            self.noul_cal_samples.append((p, bool(true_label)))
            self._is_calibrated = False

    # -------------------------------------------------------------------------
    # Risk Calibration & Threshold Optimization
    # -------------------------------------------------------------------------

    def calibrate(self) -> None:
        """
        Fits Conformal Risk Control parameters.
        Solves for:
            lambda_hat = inf { lambda : (n / (n + 1)) * R_hat(lambda) + (B / (n + 1)) <= alpha }
        """
        with self._lock:
            alpha = self.config.alpha
            B = self.config.max_loss

            if not self.score_cal_samples and not self.noul_cal_samples:
                raise ValueError("Cannot calibrate: No calibration samples ingested.")

            # 1. Calibrate Continuous Score Interval (Margin lambda)
            if self.score_cal_samples:
                n = len(self.score_cal_samples)
                if n < self.config.min_calibration_samples:
                    raise ValueError(
                        f"Insufficient calibration samples ({n}) for Score CRC. "
                        f"Requires at least {self.config.min_calibration_samples}."
                    )

                # Check if sample size n is theoretically sufficient: (n + 1) * alpha > B
                if (n + 1) * alpha <= B:
                    raise ValueError(
                        f"Sample size n={n} is too small to guarantee risk budget alpha={alpha} "
                        f"with max_loss B={B}. Need at least n > {math.ceil(B / alpha - 1)} samples."
                    )

                # Determine maximum possible range across samples
                max_range = max(s[3] - s[2] for s in self.score_cal_samples)
                # Compute absolute errors |true - predicted|
                abs_errors = [abs(s[1] - s[0]) for s in self.score_cal_samples]

                # Search grid for lambda (candidate margin)
                # Loss function:
                # If "miscoverage": loss = 1 if |true - pred| > lambda else 0
                # If "excess_error": loss = max(0, |true - pred| - lambda) / max_range
                best_lambda: Optional[float] = None
                best_risk: float = 1.0

                # Candidate lambdas from sorted errors
                sorted_errors = sorted(abs_errors)
                # Candidate grid includes 0, all unique errors, and maximum range
                candidate_grid = [0.0] + sorted_errors + [max_range]

                for lam in candidate_grid:
                    if self.config.loss_type == "excess_error":
                        losses = [
                            min(B, max(0.0, err - lam) / max_range * B) for err in abs_errors
                        ]
                    else:  # "miscoverage" (default 0-1 coverage loss)
                        losses = [B if err > lam else 0.0 for err in abs_errors]

                    r_hat = sum(losses) / float(n)
                    crc_bound = (float(n) / float(n + 1)) * r_hat + (B / float(n + 1))

                    if crc_bound <= alpha:
                        best_lambda = lam
                        best_risk = r_hat
                        break

                if best_lambda is None:
                    # Fallback to maximum range if extreme risk
                    best_lambda = max_range
                    best_risk = 0.0

                self.score_lambda = best_lambda
                self.empirical_score_risk = best_risk

            # 2. Calibrate Binary Decision Risk Threshold (tau)
            if self.noul_cal_samples:
                n = len(self.noul_cal_samples)
                if n < self.config.min_calibration_samples:
                    raise ValueError(
                        f"Insufficient calibration samples ({n}) for Noul CRC. "
                        f"Requires at least {self.config.min_calibration_samples}."
                    )

                if (n + 1) * alpha <= B:
                    raise ValueError(
                        f"Sample size n={n} is too small to guarantee risk budget alpha={alpha} "
                        f"with max_loss B={B}. Need at least n > {math.ceil(B / alpha - 1)} samples."
                    )

                # Search candidate thresholds tau in [0.0, 1.0]
                # For False-Negative Risk Control:
                # Prediction is True if prob >= tau.
                # A false negative occurs when true_label=True, but prob < tau.
                # As tau decreases towards 0.0, false negatives decrease monotonically.
                tau_candidates = sorted({s[0] for s in self.noul_cal_samples} | {0.0, 1.0}, reverse=True)
                best_tau: Optional[float] = None
                best_noul_risk: float = 1.0

                for tau in tau_candidates:
                    # Loss = B if false negative, else 0
                    losses = [
                        B if (label and prob < tau) else 0.0
                        for prob, label in self.noul_cal_samples
                    ]
                    r_hat = sum(losses) / float(n)
                    crc_bound = (float(n) / float(n + 1)) * r_hat + (B / float(n + 1))

                    if crc_bound <= alpha:
                        best_tau = tau
                        best_noul_risk = r_hat
                        break

                if best_tau is None:
                    best_tau = 0.0
                    best_noul_risk = 0.0

                self.decision_threshold = best_tau
                self.empirical_decision_risk = best_noul_risk

            self._is_calibrated = True

    # -------------------------------------------------------------------------
    # Inference & Risk Bounds
    # -------------------------------------------------------------------------

    def predict_score(
        self,
        score: Score,
        max_margin_tolerance: Optional[float] = None,
    ) -> ScoreRiskBound:
        """
        Produces a certified risk prediction interval [s - lambda, s + lambda]
        for a continuous Score primitive.
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalRiskController is not calibrated. Call calibrate() first.")
        if self.score_lambda is None:
            raise RuntimeError("No Score calibration data was provided during calibration.")

        val = score.score if score.score is not None else (score.min_val + score.max_val) / 2.0
        lam = self.score_lambda

        # Clamp interval to Score min/max bounds
        low = max(score.min_val, val - lam)
        high = min(score.max_val, val + lam)

        tolerance = (
            max_margin_tolerance
            if max_margin_tolerance is not None
            else self.config.max_margin_tolerance
        )

        is_safe = True
        should_escalate = False
        if tolerance is not None and lam > tolerance:
            is_safe = False
            should_escalate = True

        return ScoreRiskBound(
            point_estimate=val,
            interval=(low, high),
            margin=lam,
            alpha=self.config.alpha,
            max_loss=self.config.max_loss,
            instructions=score.instructions,
            is_safe=is_safe,
            should_escalate=should_escalate,
            empirical_risk=self.empirical_score_risk,
        )

    def predict_decision_threshold(self, noul: Noul) -> DecisionRiskBound:
        """
        Evaluates a Noul boolean decision under the risk-controlled decision threshold.
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalRiskController is not calibrated. Call calibrate() first.")
        if self.decision_threshold is None:
            raise RuntimeError("No Noul calibration data was provided during calibration.")

        tau = self.decision_threshold
        p = noul.probability if noul.probability is not None else 0.5

        # If probability is ambiguous around threshold, flag escalation
        margin = abs(p - tau)
        is_safe = margin >= 0.05
        should_escalate = not is_safe

        return DecisionRiskBound(
            optimal_threshold=tau,
            empirical_risk=self.empirical_decision_risk,
            alpha=self.config.alpha,
            max_loss=self.config.max_loss,
            loss_type=self.config.loss_type,
            is_safe=is_safe,
            should_escalate=should_escalate,
            instructions=noul.instructions,
        )

    # -------------------------------------------------------------------------
    # Evaluation & Diagnostics
    # -------------------------------------------------------------------------

    def evaluate_score_risk(
        self, test_samples: Sequence[Tuple[float, float, float, float]]
    ) -> Dict[str, float]:
        """
        Evaluates empirical loss on an unseen test set to verify E[L] <= alpha.
        test_samples: list of (predicted_score, true_score, min_val, max_val).
        """
        if not test_samples or self.score_lambda is None:
            return {"empirical_risk": 0.0, "nominal_risk": self.config.alpha}

        B = self.config.max_loss
        lam = self.score_lambda
        max_range = max(s[3] - s[2] for s in test_samples)
        losses = []

        for pred, true_val, min_v, max_v in test_samples:
            err = abs(true_val - pred)
            if self.config.loss_type == "excess_error":
                l = min(B, max(0.0, err - lam) / max_range * B)
            else:
                l = B if err > lam else 0.0
            losses.append(l)

        n = len(test_samples)
        mean_loss = sum(losses) / float(n)
        return {
            "empirical_risk": round(mean_loss, 4),
            "nominal_risk": round(self.config.alpha, 4),
            "margin": round(lam, 4),
            "passed_guarantee": float(mean_loss <= self.config.alpha + 0.01),
            "sample_count": n,
        }

    # -------------------------------------------------------------------------
    # Binary Serialization (.reflex-crc) with 32-bit CRC32 Verification
    # -------------------------------------------------------------------------

    def save_to_bytes(self) -> bytes:
        """Serializes calibration model to binary bytes with CRC32 verification."""
        with self._lock:
            buf = bytearray()
            buf.extend(CRC_MAGIC)

            # 40-byte structured header:
            # <IddII12s:
            # version (uint32, 4B)
            # alpha (float64, 8B)
            # max_loss (float64, 8B)
            # min_samples (uint32, 4B)
            # is_calibrated (uint32, 4B)
            # reserved (12s, 12B)
            header = struct.pack(
                "<IddII12s",
                CRC_FORMAT_VERSION,
                self.config.alpha,
                self.config.max_loss,
                self.config.min_calibration_samples,
                1 if self._is_calibrated else 0,
                b"\x00" * 12,
            )
            buf.extend(header)

            payload_dict = {
                "loss_type": self.config.loss_type,
                "score_lambda": self.score_lambda,
                "decision_threshold": self.decision_threshold,
                "empirical_score_risk": self.empirical_score_risk,
                "empirical_decision_risk": self.empirical_decision_risk,
                "score_cal_samples": self.score_cal_samples,
                "noul_cal_samples": self.noul_cal_samples,
                "max_margin_tolerance": self.config.max_margin_tolerance,
            }
            json_bytes = json.dumps(payload_dict).encode("utf-8")
            buf.extend(struct.pack("<I", len(json_bytes)))
            buf.extend(json_bytes)

            # 32-bit CRC32 Trailer
            checksum = zlib.crc32(buf)
            buf.extend(struct.pack("<I", checksum))
            return bytes(buf)

    @classmethod
    def load_from_bytes(cls, data: bytes) -> ConformalRiskController:
        """Deserializes ConformalRiskController from binary bytes with CRC32 verification."""
        if len(data) < 52:
            raise ValueError(f"Data buffer too small for CRC header ({len(data)} bytes)")

        # Verify CRC32
        payload_data = data[:-4]
        expected_crc = struct.unpack("<I", data[-4:])[0]
        actual_crc = zlib.crc32(payload_data)
        if actual_crc != expected_crc:
            raise ValueError(
                f"CRC32 mismatch in CRC model: expected {hex(expected_crc)}, got {hex(actual_crc)}"
            )

        magic = payload_data[:4]
        if magic != CRC_MAGIC:
            raise ValueError(f"Invalid CRC magic bytes: {magic}, expected {CRC_MAGIC}")

        offset = 4
        (
            version,
            alpha,
            max_loss,
            min_samples,
            is_calibrated_flag,
            _,
        ) = struct.unpack("<IddII12s", payload_data[offset : offset + 40])
        offset += 40

        if version != CRC_FORMAT_VERSION:
            raise ValueError(f"Unsupported CRC version {version}, expected {CRC_FORMAT_VERSION}")

        json_len = struct.unpack("<I", payload_data[offset : offset + 4])[0]
        offset += 4
        json_bytes = payload_data[offset : offset + json_len]
        payload = json.loads(json_bytes.decode("utf-8"))

        cfg = CRCConfig(
            alpha=alpha,
            max_loss=max_loss,
            loss_type=payload.get("loss_type", "miscoverage"),
            min_calibration_samples=min_samples,
            max_margin_tolerance=payload.get("max_margin_tolerance"),
        )
        controller = cls(cfg)
        controller.score_lambda = payload.get("score_lambda")
        controller.decision_threshold = payload.get("decision_threshold")
        controller.empirical_score_risk = payload.get("empirical_score_risk", 0.0)
        controller.empirical_decision_risk = payload.get("empirical_decision_risk", 0.0)
        controller.score_cal_samples = [tuple(s) for s in payload.get("score_cal_samples", [])]
        controller.noul_cal_samples = [tuple(s) for s in payload.get("noul_cal_samples", [])]
        controller._is_calibrated = (is_calibrated_flag == 1)

        return controller

    def save(self, filepath: str) -> None:
        """Saves CRC model to file atomically."""
        data = self.save_to_bytes()
        tmp_file = f"{filepath}.tmp.{os.getpid()}"
        with open(tmp_file, "wb") as f:
            f.write(data)
        os.replace(tmp_file, filepath)

    @classmethod
    def load(cls, filepath: str) -> ConformalRiskController:
        """Loads CRC model from file."""
        with open(filepath, "rb") as f:
            data = f.read()
        return cls.load_from_bytes(data)
