"""
Conformalized Quantile Regression (CQR) for Arbitrary Continuous Target Intervals.
Phase 36: Romano, Sesia & Candès (NeurIPS 2019) distribution-free heteroscedastic uncertainty bounding.

Features:
- Finite-sample coverage guarantee: P(Y in C(X)) >= 1 - alpha for arbitrary distributions.
- Heteroscedastic adaptive intervals: widths adapt dynamically to local epistemic difficulty.
- Dual-head quantile regression with asymmetric pinball loss (quantile loss).
- Built-in QuantileInstinctHead for zero-dependency online state embedding quantile prediction.
- Epistemic tolerance escalation: triggers System-2 escalation when interval width exceeds tolerance.
- Zero-dependency binary persistence format (.reflex-cqr, magic RFCQ, 48-byte header, CRC32).
- Seamless integration with Reflex Client and Score primitives.
"""

from dataclasses import dataclass
import json
import math
import os
import random
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zlib

from sys1.embeddings import SemanticVectorEncoder


# Binary persistence constants
CQR_MAGIC: bytes = b"RFCQ"  # Reflex Conformalized Quantiles
CQR_VERSION: int = 1
# Header: uint32 version, float64 alpha, float64 q_hat, float64 empirical_coverage,
# float64 mean_width, uint32 total_cal_samples, uint32 flags, 4s reserved = 48 bytes
CQR_HEADER_FORMAT: str = "<IddddII4s"
CQR_HEADER_SIZE: int = struct.calcsize(CQR_HEADER_FORMAT)
CQR_MIN_FILE_SIZE: int = 4 + CQR_HEADER_SIZE + 4 + 4  # magic(4) + header(48) + json_len(4) + crc32(4) = 60 bytes


@dataclass
class CQRConfig:
    """
    Configuration parameters for Conformalized Quantile Regression.
    """
    alpha: float = 0.10                      # Significance level (e.g. 0.10 for 90% coverage guarantee)
    min_calibration_samples: int = 20        # Minimum samples required before calibrate() succeeds
    max_width_tolerance: Optional[float] = None  # Escalate if interval width exceeds this threshold
    learning_rate: float = 0.05              # Learning rate for online pinball loss updates
    l2_reg: float = 0.001                    # L2 weight regularization parameter
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.min_calibration_samples < 5:
            raise ValueError(f"min_calibration_samples must be >= 5, got {self.min_calibration_samples}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.l2_reg < 0.0:
            raise ValueError(f"l2_reg must be non-negative, got {self.l2_reg}")
        if self.max_width_tolerance is not None and self.max_width_tolerance <= 0.0:
            raise ValueError(f"max_width_tolerance must be positive, got {self.max_width_tolerance}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "min_calibration_samples": self.min_calibration_samples,
            "max_width_tolerance": self.max_width_tolerance,
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CQRConfig":
        return cls(
            alpha=float(d.get("alpha", 0.10)),
            min_calibration_samples=int(d.get("min_calibration_samples", 20)),
            max_width_tolerance=float(d["max_width_tolerance"]) if d.get("max_width_tolerance") is not None else None,
            learning_rate=float(d.get("learning_rate", 0.05)),
            l2_reg=float(d.get("l2_reg", 0.001)),
            seed=d.get("seed"),
        )


@dataclass
class CQRInterval:
    """
    Finite-sample mathematically certified prediction interval for continuous variables.
    P(Y in [lower_bound, upper_bound]) >= 1 - alpha.
    """
    lower_bound: float
    upper_bound: float
    point_estimate: float
    interval_width: float
    q_hat: float
    alpha: float
    coverage_guarantee: float
    is_safe: bool = True
    should_escalate: bool = False
    instructions: str = ""

    def contains(self, value: float) -> bool:
        """Returns True if the true value is bounded within this interval."""
        return self.lower_bound <= value <= self.upper_bound

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
            "point_estimate": round(self.point_estimate, 4),
            "interval_width": round(self.interval_width, 4),
            "q_hat": round(self.q_hat, 6),
            "alpha": self.alpha,
            "coverage_guarantee": self.coverage_guarantee,
            "is_safe": self.is_safe,
            "should_escalate": self.should_escalate,
            "instructions": self.instructions,
        }


class QuantileInstinctHead:
    """
    Online-trainable dual linear quantile regression head over 384-d semantic embeddings.
    Optimizes pinball loss: L_tau(y, y_hat) = max(tau*(y - y_hat), (tau - 1)*(y - y_hat)).
    Predicts lower quantile q_{alpha/2}(x) and upper quantile q_{1 - alpha/2}(x).
    """

    def __init__(
        self,
        dim: int = 384,
        tau_low: float = 0.05,
        tau_high: float = 0.95,
        encoder: Optional[SemanticVectorEncoder] = None,
        seed: Optional[int] = None,
    ):
        self.encoder = encoder or SemanticVectorEncoder()
        self.dim = getattr(self.encoder, "DIM", dim)
        self.tau_low = tau_low
        self.tau_high = tau_high

        rng = random.Random(seed if seed is not None else 42)
        std = 1.0 / math.sqrt(self.dim)
        self.weights_low: List[float] = [rng.gauss(0.0, std) for _ in range(self.dim)]
        self.bias_low: float = 0.0

        self.weights_high: List[float] = [rng.gauss(0.0, std) for _ in range(self.dim)]
        self.bias_high: float = 0.0
        self.training_steps: int = 0

    def predict_quantiles(self, state_or_vec: Union[str, List[float]]) -> Tuple[float, float]:
        """
        Computes (q_low, q_high) for an input prompt or vector.
        Enforces monotonicity: q_low <= q_high.
        """
        vec = state_or_vec if isinstance(state_or_vec, list) else self.encoder.encode(state_or_vec)
        z_low = sum(w * x for w, x in zip(self.weights_low, vec)) + self.bias_low
        z_high = sum(w * x for w, x in zip(self.weights_high, vec)) + self.bias_high

        if z_low > z_high:
            # Monotonicity adjustment
            mid = 0.5 * (z_low + z_high)
            z_low, z_high = mid - 0.01, mid + 0.01

        return z_low, z_high

    def update(
        self,
        state_or_vec: Union[str, List[float]],
        target_y: float,
        lr: float = 0.05,
        l2_reg: float = 0.001,
    ) -> Tuple[float, float]:
        """
        Executes an analytical subgradient update on pinball loss for both quantile heads.
        Returns the pinball losses (loss_low, loss_high).
        """
        vec = state_or_vec if isinstance(state_or_vec, list) else self.encoder.encode(state_or_vec)
        y = float(target_y)

        # 1. Lower quantile head update
        z_low = sum(w * x for w, x in zip(self.weights_low, vec)) + self.bias_low
        diff_low = y - z_low
        loss_low = max(self.tau_low * diff_low, (self.tau_low - 1.0) * diff_low)

        # Subgradient w.r.t z_low: g = -tau if y > z else (1 - tau)
        if diff_low > 0:
            g_low = -self.tau_low
        elif diff_low < 0:
            g_low = 1.0 - self.tau_low
        else:
            g_low = 0.0

        for i in range(self.dim):
            self.weights_low[i] -= lr * (g_low * vec[i] + l2_reg * self.weights_low[i])
        self.bias_low -= lr * g_low

        # 2. Upper quantile head update
        z_high = sum(w * x for w, x in zip(self.weights_high, vec)) + self.bias_high
        diff_high = y - z_high
        loss_high = max(self.tau_high * diff_high, (self.tau_high - 1.0) * diff_high)

        if diff_high > 0:
            g_high = -self.tau_high
        elif diff_high < 0:
            g_high = 1.0 - self.tau_high
        else:
            g_high = 0.0

        for i in range(self.dim):
            self.weights_high[i] -= lr * (g_high * vec[i] + l2_reg * self.weights_high[i])
        self.bias_high -= lr * g_high

        self.training_steps += 1
        return loss_low, loss_high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "tau_low": self.tau_low,
            "tau_high": self.tau_high,
            "weights_low": [round(w, 6) for w in self.weights_low],
            "bias_low": round(self.bias_low, 6),
            "weights_high": [round(w, 6) for w in self.weights_high],
            "bias_high": round(self.bias_high, 6),
            "training_steps": self.training_steps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], encoder: Optional[SemanticVectorEncoder] = None) -> "QuantileInstinctHead":
        head = cls(
            dim=d["dim"],
            tau_low=d["tau_low"],
            tau_high=d["tau_high"],
            encoder=encoder,
        )
        head.weights_low = [float(w) for w in d["weights_low"]]
        head.bias_low = float(d["bias_low"])
        head.weights_high = [float(w) for w in d["weights_high"]]
        head.bias_high = float(d["bias_high"])
        head.training_steps = int(d.get("training_steps", 0))
        return head


class ConformalizedQuantileRegressor:
    """
    Distribution-Free Conformalized Quantile Regression (CQR) Controller.
    Romano, Sesia & Candès (NeurIPS 2019).

    Produces certified continuous intervals:
      C(x) = [q_{alpha/2}(x) - Q_hat, q_{1 - alpha/2}(x) + Q_hat]
    guaranteeing P(Y in C(X)) >= 1 - alpha.
    """

    def __init__(
        self,
        config: Optional[CQRConfig] = None,
        head: Optional[QuantileInstinctHead] = None,
    ):
        self.config = config or CQRConfig()
        self._lock = threading.RLock()

        # Quantile instinct head (for end-to-end prompt/state prediction)
        tau_low = self.config.alpha / 2.0
        tau_high = 1.0 - (self.config.alpha / 2.0)
        self.head: QuantileInstinctHead = head or QuantileInstinctHead(
            tau_low=tau_low, tau_high=tau_high, seed=self.config.seed
        )

        # Calibration dataset: list of (pred_low, pred_high, true_value, point_estimate)
        self.calibration_samples: List[Tuple[float, float, float, float]] = []

        # Calibrated conformal parameters
        self.q_hat: Optional[float] = None
        self.empirical_coverage: float = 0.0
        self.mean_interval_width: float = 0.0
        self._is_calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._is_calibrated

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    def add_calibration_sample(
        self,
        pred_low: float,
        pred_high: float,
        true_value: float,
        point_estimate: Optional[float] = None,
    ) -> None:
        """
        Adds a single calibration sample for CQR adjustment.
        pred_low: estimated lower quantile q_{alpha/2}(x)
        pred_high: estimated upper quantile q_{1 - alpha/2}(x)
        true_value: ground truth continuous response y
        point_estimate: optional point prediction (defaults to midpoint)
        """
        with self._lock:
            low = float(pred_low)
            high = float(pred_high)
            y = float(true_value)
            point = float(point_estimate) if point_estimate is not None else 0.5 * (low + high)

            if low > high:
                low, high = high, low

            self.calibration_samples.append((low, high, y, point))
            self._is_calibrated = False

    def add_calibration_from_state(self, state: str, true_value: float) -> None:
        """
        Computes quantiles from state using internal QuantileInstinctHead and records sample.
        """
        low, high = self.head.predict_quantiles(state)
        self.add_calibration_sample(low, high, true_value)

    def calibrate(self) -> None:
        """
        Fits conformal quantile offset Q_hat on accumulated calibration samples:
          E_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))
          Q_hat = Quantile(E, ceil((n + 1)(1 - alpha)) / n)
        """
        with self._lock:
            n = len(self.calibration_samples)
            if n < self.config.min_calibration_samples:
                raise ValueError(
                    f"Insufficient calibration samples for CQR: {n} provided, "
                    f"minimum {self.config.min_calibration_samples} required."
                )

            # 1. Compute signed non-conformity scores: E_i = max(q_low - y, y - q_high)
            scores: List[float] = []
            for low, high, y, _ in self.calibration_samples:
                err_low = low - y
                err_high = y - high
                score = max(err_low, err_high)
                scores.append(score)

            # 2. Conformal quantile level with finite-sample inflation: ceil((n + 1)(1 - alpha)) / n
            p_level = math.ceil((n + 1) * (1.0 - self.config.alpha)) / float(n)
            p_level = min(1.0, max(0.0, p_level))

            sorted_scores = sorted(scores)
            idx = min(n - 1, max(0, math.ceil(p_level * n) - 1))
            self.q_hat = sorted_scores[idx]

            # 3. Calculate calibration statistics
            covered = sum(1 for s in scores if s <= self.q_hat)
            self.empirical_coverage = float(covered) / float(n)

            widths = [(high - low + 2.0 * self.q_hat) for low, high, _, _ in self.calibration_samples]
            self.mean_interval_width = sum(widths) / float(n)

            self._is_calibrated = True

    # -------------------------------------------------------------------------
    # Inference & Epistemic Prediction Intervals
    # -------------------------------------------------------------------------

    def predict(
        self,
        pred_low: float,
        pred_high: float,
        point_estimate: Optional[float] = None,
        min_val: float = 0.0,
        max_val: float = 10.0,
        instructions: str = "",
    ) -> CQRInterval:
        """
        Constructs a mathematically certified CQR interval:
          C(x) = [clamp(q_low - Q_hat), clamp(q_high + Q_hat)]
        guaranteeing P(Y in C(X)) >= 1 - alpha.
        """
        if not self.is_calibrated or self.q_hat is None:
            raise RuntimeError("ConformalizedQuantileRegressor is not calibrated. Call calibrate() first.")

        with self._lock:
            q_hat = self.q_hat

        low = float(pred_low)
        high = float(pred_high)
        if low > high:
            low, high = high, low

        # Conformal quantile adjustment
        c_low = low - q_hat
        c_high = high + q_hat

        # Domain boundary clamping
        c_low = max(min_val, min(max_val, c_low))
        c_high = max(min_val, min(max_val, c_high))

        if c_low > c_high:
            mid = 0.5 * (c_low + c_high)
            c_low = mid
            c_high = mid

        width = c_high - c_low
        point = float(point_estimate) if point_estimate is not None else 0.5 * (c_low + c_high)
        point = max(min_val, min(max_val, point))

        # Epistemic escalation check
        should_escalate = False
        is_safe = True
        if self.config.max_width_tolerance is not None and width > self.config.max_width_tolerance:
            should_escalate = True
            is_safe = False

        return CQRInterval(
            lower_bound=c_low,
            upper_bound=c_high,
            point_estimate=point,
            interval_width=width,
            q_hat=q_hat,
            alpha=self.config.alpha,
            coverage_guarantee=1.0 - self.config.alpha,
            is_safe=is_safe,
            should_escalate=should_escalate,
            instructions=instructions,
        )

    def predict_state(
        self,
        state: str,
        min_val: float = 0.0,
        max_val: float = 10.0,
        instructions: str = "",
    ) -> CQRInterval:
        """
        End-to-end inference from unstructured input text via QuantileInstinctHead.
        """
        pred_low, pred_high = self.head.predict_quantiles(state)
        return self.predict(
            pred_low=pred_low,
            pred_high=pred_high,
            min_val=min_val,
            max_val=max_val,
            instructions=instructions,
        )

    def evaluate_coverage(
        self, test_samples: Sequence[Tuple[float, float, float, Optional[float]]]
    ) -> Dict[str, float]:
        """
        Evaluates empirical coverage, mean interval width, and efficiency metrics on unseen test samples.
        test_samples: sequence of (pred_low, pred_high, true_value, [point_est])
        """
        if not test_samples:
            return {"coverage": 0.0, "mean_width": 0.0, "sample_count": 0}

        covered_count = 0
        total_width = 0.0

        for item in test_samples:
            low, high, true_val = item[0], item[1], item[2]
            point_est = item[3] if len(item) > 3 else None
            interval = self.predict(low, high, point_estimate=point_est, min_val=-1e9, max_val=1e9)
            if interval.contains(true_val):
                covered_count += 1
            total_width += interval.interval_width

        n = len(test_samples)
        emp_cov = float(covered_count) / float(n)
        mean_w = total_width / float(n)

        return {
            "empirical_coverage": emp_cov,
            "nominal_coverage": 1.0 - self.config.alpha,
            "mean_width": mean_w,
            "sample_count": n,
            "q_hat": self.q_hat or 0.0,
        }

    # -------------------------------------------------------------------------
    # Zero-Dependency Binary Persistence (.reflex-cqr)
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serializes CQR state and weights to a zero-dependency .reflex-cqr binary file.
        Format:
          - 4 bytes magic: b"RFCQ"
          - 48 bytes structured header: version, alpha, q_hat, empirical_cov, mean_width, cal_samples, flags, reserved
          - 4 bytes json_len (uint32)
          - json_len bytes: UTF-8 JSON payload (config, quantile head weights, calibration summary)
          - 4 bytes CRC32 checksum (uint32)
        Uses atomic file write and rename.
        """
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with self._lock:
            # 1. Prepare JSON payload
            payload = {
                "config": self.config.to_dict(),
                "head": self.head.to_dict(),
                "calibration_stats": {
                    "sample_count": len(self.calibration_samples),
                    "empirical_coverage": self.empirical_coverage,
                    "mean_interval_width": self.mean_interval_width,
                    "q_hat": self.q_hat,
                },
                "saved_at": time.time(),
            }
            json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            json_len = len(json_bytes)

            # 2. Build 48-byte structured binary header
            header = struct.pack(
                CQR_HEADER_FORMAT,
                CQR_VERSION,
                float(self.config.alpha),
                float(self.q_hat if self.q_hat is not None else 0.0),
                float(self.empirical_coverage),
                float(self.mean_interval_width),
                int(len(self.calibration_samples)),
                int(1 if self._is_calibrated else 0),
                b"\x00\x00\x00\x00",
            )

            # 3. Assemble binary body
            body = bytearray()
            body.extend(CQR_MAGIC)
            body.extend(header)
            body.extend(struct.pack("<I", json_len))
            body.extend(json_bytes)

            # 4. CRC32 checksum trailer
            checksum = zlib.crc32(body) & 0xFFFFFFFF
            body.extend(struct.pack("<I", checksum))

        # 5. Atomic write
        tmp_path = f"{path}.tmp.{os.getpid()}_{int(time.time() * 1000)}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @classmethod
    def load(cls, path: str, encoder: Optional[SemanticVectorEncoder] = None) -> "ConformalizedQuantileRegressor":
        """
        Loads and validates a CQR controller from a .reflex-cqr binary file.
        Verifies magic bytes and 32-bit CRC32 checksum.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CQR file not found: {path}")

        file_size = os.path.getsize(path)
        if file_size < CQR_MIN_FILE_SIZE:
            raise ValueError(
                f"Invalid .reflex-cqr file: size {file_size} bytes below minimum {CQR_MIN_FILE_SIZE} bytes"
            )

        with open(path, "rb") as f:
            data = f.read()

        # 1. Magic check
        magic = data[:4]
        if magic != CQR_MAGIC:
            raise ValueError(f"Invalid magic header: expected {CQR_MAGIC!r}, got {magic!r}")

        # 2. CRC32 verification
        body_bytes = data[:-4]
        stored_crc = struct.unpack("<I", data[-4:])[0]
        computed_crc = zlib.crc32(body_bytes) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC32 checksum mismatch: expected {computed_crc:#010x}, got {stored_crc:#010x}. "
                f"File may be corrupted or tampered."
            )

        # 3. Unpack header
        version, alpha, q_hat, empirical_cov, mean_width, cal_samples, flags, _ = struct.unpack(
            CQR_HEADER_FORMAT, data[4:4 + CQR_HEADER_SIZE]
        )

        # 4. Unpack JSON payload
        offset = 4 + CQR_HEADER_SIZE
        json_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        json_bytes = data[offset:offset + json_len]
        payload = json.loads(json_bytes.decode("utf-8"))

        config = CQRConfig.from_dict(payload.get("config", {}))
        head = QuantileInstinctHead.from_dict(payload.get("head", {}), encoder=encoder)

        cqr = cls(config=config, head=head)
        cqr.q_hat = float(q_hat)
        cqr.empirical_coverage = float(empirical_cov)
        cqr.mean_interval_width = float(mean_width)
        cqr._is_calibrated = bool(flags & 1)

        return cqr
