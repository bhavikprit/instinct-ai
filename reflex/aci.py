"""
Adaptive Conformal Inference (ACI) & Online Distribution Shift Adaptation.
Phase 35: Gibbs & Candès (2021, 2022) online quantile adaptation for non-stationary streams.

Features:
- Closed-loop online controller: alpha_{t+1} = clamp(alpha_t + gamma * (alpha - err_t), alpha_min, alpha_max)
- Long-term coverage guarantee ~ 1 - alpha under arbitrary non-stationary shifts and concept drift.
- Rolling window empirical coverage tracking (W=100) and automated drift alarm (is_drifting).
- Asymmetric penalty multiplier (gamma_down_multiplier) for safety-critical systems.
- Zero-dependency binary persistence format (.reflex-aci, magic RFAC, 40-byte header, CRC32).
- Thread-safe atomic operations.
"""

from collections import deque
from dataclasses import dataclass
import json
import math
import os
import struct
import threading
import time
from typing import Any, Collection, Dict, List, Optional, Tuple, Union
import zlib


# Magic identifier for .reflex-aci binary persistence files
ACI_MAGIC = b"RFAC"
ACI_VERSION = 1
# Header format: uint32 version, float64 target_alpha, float64 current_alpha, float64 gamma,
# uint64 total_steps, uint32 total_errors, 4s reserved = 44 bytes
ACI_HEADER_FORMAT = "<IdddQI4s"
ACI_HEADER_SIZE = struct.calcsize(ACI_HEADER_FORMAT)
ACI_MIN_FILE_SIZE = 4 + ACI_HEADER_SIZE + 4 + 4  # magic(4) + header(44) + json_len(4) + crc32(4) = 56 bytes


@dataclass
class ACIConfig:
    """
    Configuration parameters for Adaptive Conformal Inference.
    """
    target_alpha: float = 0.10             # Target significance level (e.g. 0.10 for 90% coverage)
    gamma: float = 0.01                    # Online step size parameter
    alpha_min: float = 0.001               # Minimum clamped alpha
    alpha_max: float = 0.999               # Maximum clamped alpha
    window_size: int = 100                 # Rolling window size for empirical coverage
    drift_threshold: float = 0.08          # Threshold for drift alarm: empirical < (target_cov - drift_threshold)
    min_samples_for_drift: int = 30        # Minimum samples in rolling window before triggering drift alarms
    gamma_down_multiplier: float = 1.0     # Multiplier when err_t == 1 (asymmetric penalty for safety)

    def __post_init__(self) -> None:
        if not (0.0 < self.target_alpha < 1.0):
            raise ValueError(f"target_alpha must be in (0, 1), got {self.target_alpha}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.alpha_min >= self.alpha_max:
            raise ValueError(f"alpha_min ({self.alpha_min}) must be strictly less than alpha_max ({self.alpha_max})")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.gamma_down_multiplier <= 0.0:
            raise ValueError(f"gamma_down_multiplier must be positive, got {self.gamma_down_multiplier}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_alpha": self.target_alpha,
            "gamma": self.gamma,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "window_size": self.window_size,
            "drift_threshold": self.drift_threshold,
            "min_samples_for_drift": self.min_samples_for_drift,
            "gamma_down_multiplier": self.gamma_down_multiplier,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ACIConfig":
        return cls(
            target_alpha=float(d.get("target_alpha", 0.10)),
            gamma=float(d.get("gamma", 0.01)),
            alpha_min=float(d.get("alpha_min", 0.001)),
            alpha_max=float(d.get("alpha_max", 0.999)),
            window_size=int(d.get("window_size", 100)),
            drift_threshold=float(d.get("drift_threshold", 0.08)),
            min_samples_for_drift=int(d.get("min_samples_for_drift", 30)),
            gamma_down_multiplier=float(d.get("gamma_down_multiplier", 1.0)),
        )


@dataclass
class ACIStatus:
    """
    Snapshot of Adaptive Conformal Inference online controller telemetry.
    """
    target_alpha: float
    current_alpha: float
    nominal_coverage: float
    target_coverage: float
    empirical_coverage: float
    total_steps: int
    total_errors: int
    rolling_steps: int
    drift_score: float
    is_drifting: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_alpha": round(self.target_alpha, 6),
            "current_alpha": round(self.current_alpha, 6),
            "nominal_coverage": round(self.nominal_coverage, 6),
            "target_coverage": round(self.target_coverage, 6),
            "empirical_coverage": round(self.empirical_coverage, 6),
            "total_steps": self.total_steps,
            "total_errors": self.total_errors,
            "rolling_steps": self.rolling_steps,
            "drift_score": round(self.drift_score, 6),
            "is_drifting": self.is_drifting,
        }


class AdaptiveConformalTracker:
    """
    Online Adaptive Conformal Inference (ACI) Controller (Gibbs & Candès 2021).

    Dynamically adjusts the test-time significance level alpha_t based on online feedback:
      alpha_{t+1} = clamp(alpha_t + gamma * (target_alpha - err_t), alpha_min, alpha_max)
    where err_t = 0 if ground truth y_t in C_{alpha_t}(x_t) else 1.

    Guarantees long-term empirical coverage ~= 1 - target_alpha under non-stationary streams,
    covariate shift, and concept drift.
    """

    def __init__(
        self,
        config: Optional[ACIConfig] = None,
        initial_alpha: Optional[float] = None,
    ):
        self.config = config or ACIConfig()
        init_a = float(initial_alpha) if initial_alpha is not None else self.config.target_alpha
        self._current_alpha: float = max(self.config.alpha_min, min(self.config.alpha_max, init_a))
        self._lock = threading.RLock()
        self._history: deque = deque(maxlen=self.config.window_size)  # 1 for covered, 0 for miscovered
        self._total_steps: int = 0
        self._total_errors: int = 0

    @property
    def current_alpha(self) -> float:
        with self._lock:
            return self._current_alpha

    def get_alpha(self) -> float:
        """Returns the current adaptive significance level alpha_t."""
        return self.current_alpha

    @property
    def target_alpha(self) -> float:
        return self.config.target_alpha

    @property
    def total_steps(self) -> int:
        with self._lock:
            return self._total_steps

    @property
    def total_errors(self) -> int:
        with self._lock:
            return self._total_errors

    @property
    def empirical_coverage(self) -> float:
        """Rolling window empirical coverage over recent steps."""
        with self._lock:
            if not self._history:
                return 1.0 - self.config.target_alpha
            return sum(self._history) / float(len(self._history))

    @property
    def cumulative_coverage(self) -> float:
        """Cumulative empirical coverage across all steps since inception."""
        with self._lock:
            if self._total_steps == 0:
                return 1.0 - self.config.target_alpha
            return float(self._total_steps - self._total_errors) / float(self._total_steps)

    @property
    def drift_score(self) -> float:
        """Absolute divergence between target coverage and rolling window coverage."""
        target_cov = 1.0 - self.config.target_alpha
        return abs(target_cov - self.empirical_coverage)

    @property
    def is_drifting(self) -> bool:
        """
        True when rolling coverage has degraded beyond drift_threshold below target coverage.
        Triggers automated System-2 escalation alarms.
        """
        with self._lock:
            if len(self._history) < self.config.min_samples_for_drift:
                return False
            rolling_cov = sum(self._history) / float(len(self._history))
            target_cov = 1.0 - self.config.target_alpha
            return rolling_cov < (target_cov - self.config.drift_threshold)

    # -------------------------------------------------------------------------
    # Online Controller Updates
    # -------------------------------------------------------------------------

    def update(self, is_covered: bool) -> float:
        """
        Ingests online feedback for step t and updates the controller:
          err_t = 0 if is_covered else 1
          step = gamma * (target_alpha - err_t)
          if err_t == 1: step *= gamma_down_multiplier
          alpha_{t+1} = clamp(alpha_t + step, alpha_min, alpha_max)

        Returns the new alpha_{t+1}.
        """
        with self._lock:
            covered = bool(is_covered)
            err_t = 0.0 if covered else 1.0

            if covered:
                step = self.config.gamma * self.config.target_alpha
            else:
                step = self.config.gamma * (self.config.target_alpha - 1.0) * self.config.gamma_down_multiplier

            new_alpha = self._current_alpha + step
            self._current_alpha = max(self.config.alpha_min, min(self.config.alpha_max, new_alpha))

            self._total_steps += 1
            if not covered:
                self._total_errors += 1
            self._history.append(1 if covered else 0)

            return self._current_alpha

    def update_with_result(self, true_label: Any, prediction_set: Collection[Any]) -> float:
        """
        Convenience helper: checks if true_label in prediction_set and updates controller.
        Returns the updated alpha_{t+1}.
        """
        is_covered = true_label in prediction_set
        return self.update(is_covered=is_covered)

    def reset(self, initial_alpha: Optional[float] = None) -> None:
        """Resets tracking history and restores initial alpha."""
        with self._lock:
            init_a = float(initial_alpha) if initial_alpha is not None else self.config.target_alpha
            self._current_alpha = max(self.config.alpha_min, min(self.config.alpha_max, init_a))
            self._history.clear()
            self._total_steps = 0
            self._total_errors = 0

    def status(self) -> ACIStatus:
        """Generates a point-in-time telemetry snapshot of the ACI controller."""
        with self._lock:
            target_cov = 1.0 - self.config.target_alpha
            nom_cov = 1.0 - self._current_alpha
            rolling_cov = sum(self._history) / float(len(self._history)) if self._history else target_cov
            drift_sc = abs(target_cov - rolling_cov)
            drifting = (
                len(self._history) >= self.config.min_samples_for_drift
                and rolling_cov < (target_cov - self.config.drift_threshold)
            )

            return ACIStatus(
                target_alpha=self.config.target_alpha,
                current_alpha=self._current_alpha,
                nominal_coverage=nom_cov,
                target_coverage=target_cov,
                empirical_coverage=rolling_cov,
                total_steps=self._total_steps,
                total_errors=self._total_errors,
                rolling_steps=len(self._history),
                drift_score=drift_sc,
                is_drifting=drifting,
            )

    # -------------------------------------------------------------------------
    # Zero-Dependency Binary Serialization (.reflex-aci)
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serializes ACI state to a zero-dependency .reflex-aci binary file with CRC32 integrity.
        Uses atomic file renaming to prevent corruption during concurrent writes.
        """
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with self._lock:
            # 1. Prepare dynamic metadata payload (JSON)
            payload_dict = {
                "config": self.config.to_dict(),
                "history": list(self._history),
                "saved_at": time.time(),
            }
            json_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
            json_len = len(json_bytes)

            # 2. Build 40-byte structured binary header
            header = struct.pack(
                ACI_HEADER_FORMAT,
                ACI_VERSION,
                float(self.config.target_alpha),
                float(self._current_alpha),
                float(self.config.gamma),
                int(self._total_steps),
                int(self._total_errors),
                b"\x00\x00\x00\x00",  # 4 bytes reserved
            )

            # 3. Assemble binary body before checksum
            body = bytearray()
            body.extend(ACI_MAGIC)
            body.extend(header)
            body.extend(struct.pack("<I", json_len))
            body.extend(json_bytes)

            # 4. Compute 32-bit CRC32 checksum over body
            checksum = zlib.crc32(body) & 0xFFFFFFFF
            body.extend(struct.pack("<I", checksum))

        # 5. Atomic write to temporary file then replace
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
    def load(cls, path: str) -> "AdaptiveConformalTracker":
        """
        Loads and validates an ACI tracker from a .reflex-aci binary file.
        Verifies magic header and CRC32 checksum trailer.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"ACI model file not found: {path}")

        file_size = os.path.getsize(path)
        if file_size < ACI_MIN_FILE_SIZE:
            raise ValueError(f"Invalid .reflex-aci file: size {file_size} bytes is below minimum {ACI_MIN_FILE_SIZE} bytes")

        with open(path, "rb") as f:
            data = f.read()

        # 1. Verify magic header
        magic = data[:4]
        if magic != ACI_MAGIC:
            raise ValueError(f"Invalid magic identifier: expected {ACI_MAGIC!r}, got {magic!r}")

        # 2. Verify CRC32 checksum trailer
        body_bytes = data[:-4]
        stored_crc = struct.unpack("<I", data[-4:])[0]
        computed_crc = zlib.crc32(body_bytes) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC32 checksum mismatch: expected {computed_crc:#010x}, got {stored_crc:#010x}. File may be corrupted or tampered."
            )

        # 3. Unpack 40-byte structured header
        version, target_alpha, current_alpha, gamma, total_steps, total_errors, _ = struct.unpack(
            ACI_HEADER_FORMAT, data[4:4 + ACI_HEADER_SIZE]
        )

        # 4. Unpack JSON payload
        offset = 4 + ACI_HEADER_SIZE
        json_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        json_bytes = data[offset:offset + json_len]
        payload = json.loads(json_bytes.decode("utf-8"))

        config_dict = payload.get("config", {})
        config = ACIConfig.from_dict(config_dict)

        tracker = cls(config=config, initial_alpha=current_alpha)
        tracker._total_steps = int(total_steps)
        tracker._total_errors = int(total_errors)

        history_list = payload.get("history", [])
        tracker._history = deque(history_list, maxlen=config.window_size)

        return tracker
