"""
Online Calibrated ECE & Temperature-Scaling Drift Adaptation.
Phase 37: Real-time Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and closed-loop online temperature scaling under non-stationary streams (Guo et al. 2017).

Features:
- Streaming ECE, MCE, and Brier score tracking over rolling deque window (W=100).
- Analytical online gradient descent on Negative Log-Likelihood (NLL) optimizing temperature T_t.
- Reliability diagrams with binned confidence vs accuracy calibration curves.
- Automated miscalibration alarms (is_miscalibrated) triggering System-2 escalation.
- Zero-dependency binary persistence (.reflex-calib, magic RFCL, 48-byte header, CRC32).
- Thread-safe runtime integration with Reflex client and Noul / Choice primitives.
"""

from collections import deque
from dataclasses import dataclass
import json
import math
import os
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zlib


# Binary persistence constants
CALIB_MAGIC: bytes = b"RFCL"  # Reflex Calibration
CALIB_VERSION: int = 1
# Header: uint32 version, float64 temperature, float64 ece, float64 mce,
# float64 brier_score, uint32 total_samples, uint32 num_bins, 4s reserved = 48 bytes
CALIB_HEADER_FORMAT: str = "<IddddII4s"
CALIB_HEADER_SIZE: int = struct.calcsize(CALIB_HEADER_FORMAT)
CALIB_MIN_FILE_SIZE: int = 4 + CALIB_HEADER_SIZE + 4 + 4  # magic(4) + header(48) + json_len(4) + crc32(4) = 60 bytes


def _logit(p: float) -> float:
    """Clamped inverse sigmoid logit: log(p / (1 - p))."""
    clamped = max(1e-7, min(1.0 - 1e-7, float(p)))
    return math.log(clamped / (1.0 - clamped))


def _sigmoid(z: float) -> float:
    """Clamped logistic sigmoid."""
    clamped = max(-30.0, min(30.0, float(z)))
    return 1.0 / (1.0 + math.exp(-clamped))


def _softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """Temperature-scaled numerically stable softmax."""
    if not logits:
        return []
    t = max(0.01, float(temperature))
    max_l = max(logits)
    exps = [math.exp(max(-30.0, min(30.0, (z - max_l) / t))) for z in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


@dataclass
class CalibConfig:
    """
    Configuration parameters for Online Probability Calibration.
    """
    num_bins: int = 10                     # Number of confidence bins for ECE/MCE calculation
    window_size: int = 100                 # Rolling sample window size for streaming metrics
    learning_rate: float = 0.05            # Online learning rate for temperature NLL optimization
    min_temperature: float = 0.05          # Minimum clamped temperature (avoids division by zero)
    max_temperature: float = 10.0          # Maximum clamped temperature
    ece_threshold: float = 0.08            # ECE threshold above which is_miscalibrated alarm triggers
    min_samples_for_alarm: int = 30        # Minimum samples in window before firing drift alarms
    initial_temperature: float = 1.0       # Initial temperature (1.0 = identity scaling)

    def __post_init__(self) -> None:
        if self.num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {self.num_bins}")
        if self.window_size < 10:
            raise ValueError(f"window_size must be >= 10, got {self.window_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not (0.0 < self.min_temperature < self.max_temperature):
            raise ValueError(f"Invalid temperature bounds: min={self.min_temperature}, max={self.max_temperature}")
        if not (0.0 < self.ece_threshold < 1.0):
            raise ValueError(f"ece_threshold must be in (0, 1), got {self.ece_threshold}")
        if self.min_samples_for_alarm <= 0:
            raise ValueError(f"min_samples_for_alarm must be positive, got {self.min_samples_for_alarm}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_bins": self.num_bins,
            "window_size": self.window_size,
            "learning_rate": self.learning_rate,
            "min_temperature": self.min_temperature,
            "max_temperature": self.max_temperature,
            "ece_threshold": self.ece_threshold,
            "min_samples_for_alarm": self.min_samples_for_alarm,
            "initial_temperature": self.initial_temperature,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibConfig":
        return cls(
            num_bins=int(d.get("num_bins", 10)),
            window_size=int(d.get("window_size", 100)),
            learning_rate=float(d.get("learning_rate", 0.05)),
            min_temperature=float(d.get("min_temperature", 0.05)),
            max_temperature=float(d.get("max_temperature", 10.0)),
            ece_threshold=float(d.get("ece_threshold", 0.08)),
            min_samples_for_alarm=int(d.get("min_samples_for_alarm", 30)),
            initial_temperature=float(d.get("initial_temperature", 1.0)),
        )


@dataclass
class CalibrationStatus:
    """
    Snapshot of online calibration telemetry, reliability bins, and drift state.
    """
    temperature: float
    ece: float
    mce: float
    brier_score: float
    total_samples: int
    rolling_samples: int
    is_miscalibrated: bool
    bins: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": round(self.temperature, 4),
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier_score": round(self.brier_score, 4),
            "total_samples": self.total_samples,
            "rolling_samples": self.rolling_samples,
            "is_miscalibrated": self.is_miscalibrated,
            "bins": self.bins,
        }


class OnlineProbabilityCalibrator:
    """
    Closed-Loop Online Probability Calibrator & Drift Detector.
    Guo et al. (ICML 2017) / Platt scaling runtime for System-1 streams.

    Continuously tracks streaming ECE/MCE/Brier score, detects model overconfidence drift,
    and updates temperature parameter T in log-space via analytical NLL subgradient descent.
    """

    def __init__(
        self,
        config: Optional[CalibConfig] = None,
        initial_temperature: Optional[float] = None,
    ):
        self.config = config or CalibConfig()
        init_t = initial_temperature if initial_temperature is not None else self.config.initial_temperature
        self._temperature: float = max(self.config.min_temperature, min(self.config.max_temperature, float(init_t)))
        self._log_temperature: float = math.log(self._temperature)
        self._lock = threading.RLock()

        # Rolling history of (calibrated_prob, true_label_int)
        self._history: deque = deque(maxlen=self.config.window_size)
        self._total_samples: int = 0

    @property
    def temperature(self) -> float:
        with self._lock:
            return self._temperature

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    # -------------------------------------------------------------------------
    # Inference: Probability & Distribution Calibration
    # -------------------------------------------------------------------------

    def calibrate_probability(self, prob: float) -> float:
        """
        Applies temperature scaling to a scalar probability:
          z = logit(prob)
          prob_calib = sigmoid(z / T)
        """
        with self._lock:
            t = self._temperature
        z = _logit(prob)
        return _sigmoid(z / t)

    def calibrate_distribution(self, distribution: Dict[str, float]) -> Dict[str, float]:
        """
        Applies temperature scaling to a multi-class categorical probability distribution.
        """
        if not distribution:
            return {}

        with self._lock:
            t = self._temperature

        keys = list(distribution.keys())
        # Invert probabilities to pseudo-logits: log(p)
        raw_logits = [math.log(max(1e-7, float(distribution[k]))) for k in keys]
        scaled_probs = _softmax(raw_logits, temperature=t)

        return {k: round(p, 4) for k, p in zip(keys, scaled_probs)}

    # -------------------------------------------------------------------------
    # Online Feedback & NLL Gradient Descent Adaptation
    # -------------------------------------------------------------------------

    def update(self, raw_prob: float, true_label: Union[bool, int, float]) -> float:
        """
        Ingests feedback (p_raw, y) and executes an instantaneous online gradient descent
        step on NLL w.r.t log-temperature s = log(T):
          d(NLL)/ds = (p_calib - y) * (-z / T)
          s_{t+1} = clamp(s_t - lr * d(NLL)/ds)
          T_{t+1} = exp(s_{t+1})

        Returns the updated temperature.
        """
        with self._lock:
            y = 1.0 if (true_label is True or true_label == 1) else (0.0 if (true_label is False or true_label == 0) else float(true_label))
            y = max(0.0, min(1.0, y))

            t = self._temperature
            z = _logit(raw_prob)
            p_calib = _sigmoid(z / t)

            # Gradient of NLL w.r.t s = log(T):
            # scaled_z = z / T = z * exp(-s)
            # d(NLL)/d(scaled_z) = (p_calib - y)
            # d(scaled_z)/ds = -z * exp(-s) = -z / T
            # d(NLL)/ds = (p_calib - y) * (-z / T)
            grad_s = (p_calib - y) * (-z / t)
            # Clip gradient for numerical stability
            grad_s = max(-10.0, min(10.0, grad_s))

            self._log_temperature -= self.config.learning_rate * grad_s
            min_log = math.log(self.config.min_temperature)
            max_log = math.log(self.config.max_temperature)
            self._log_temperature = max(min_log, min(max_log, self._log_temperature))
            self._temperature = max(self.config.min_temperature, min(self.config.max_temperature, math.exp(self._log_temperature)))

            # Record sample in rolling history
            self._history.append((p_calib, y))
            self._total_samples += 1

            return self._temperature

    # -------------------------------------------------------------------------
    # Calibration Metrics & Reliability Diagram
    # -------------------------------------------------------------------------

    def compute_metrics(self) -> Tuple[float, float, float, List[Dict[str, Any]]]:
        """
        Computes (ECE, MCE, Brier Score, bin_records) over current rolling window history.
        """
        with self._lock:
            samples = list(self._history)

        if not samples:
            return 0.0, 0.0, 0.0, []

        n = len(samples)
        num_bins = self.config.num_bins
        bin_width = 1.0 / float(num_bins)

        # Initialize bins
        bins_conf = [0.0] * num_bins
        bins_acc = [0.0] * num_bins
        bins_count = [0] * num_bins
        brier_sum = 0.0

        for p, y in samples:
            brier_sum += (p - y) ** 2
            # Bin assignment [0, num_bins - 1]
            b_idx = min(num_bins - 1, max(0, int(p / bin_width)))
            bins_count[b_idx] += 1
            bins_conf[b_idx] += p
            bins_acc[b_idx] += y

        ece = 0.0
        mce = 0.0
        bin_records = []

        for i in range(num_bins):
            cnt = bins_count[i]
            low_edge = i * bin_width
            high_edge = (i + 1) * bin_width

            if cnt > 0:
                avg_conf = bins_conf[i] / float(cnt)
                avg_acc = bins_acc[i] / float(cnt)
                gap = abs(avg_acc - avg_conf)
                ece += (float(cnt) / float(n)) * gap
                if gap > mce:
                    mce = gap
            else:
                avg_conf = 0.5 * (low_edge + high_edge)
                avg_acc = 0.0
                gap = 0.0

            bin_records.append({
                "bin_index": i,
                "range": [round(low_edge, 2), round(high_edge, 2)],
                "count": cnt,
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
                "calibration_gap": round(gap, 4),
            })

        brier = brier_sum / float(n)
        return ece, mce, brier, bin_records

    @property
    def is_miscalibrated(self) -> bool:
        """
        True when rolling ECE exceeds ece_threshold after minimum sample warm-up.
        """
        with self._lock:
            if len(self._history) < self.config.min_samples_for_alarm:
                return False
        ece, _, _, _ = self.compute_metrics()
        return ece > self.config.ece_threshold

    def status(self) -> CalibrationStatus:
        """Returns a snapshot of live calibration metrics and reliability curve."""
        with self._lock:
            t = self._temperature
            total = self._total_samples
            rolling = len(self._history)
            warm = rolling >= self.config.min_samples_for_alarm

        ece, mce, brier, bins = self.compute_metrics()
        miscalibrated = warm and (ece > self.config.ece_threshold)

        return CalibrationStatus(
            temperature=t,
            ece=ece,
            mce=mce,
            brier_score=brier,
            total_samples=total,
            rolling_samples=rolling,
            is_miscalibrated=miscalibrated,
            bins=bins,
        )

    def reset(self, initial_temperature: Optional[float] = None) -> None:
        """Resets history and restores initial temperature."""
        with self._lock:
            init_t = initial_temperature if initial_temperature is not None else self.config.initial_temperature
            self._temperature = max(self.config.min_temperature, min(self.config.max_temperature, float(init_t)))
            self._log_temperature = math.log(self._temperature)
            self._history.clear()
            self._total_samples = 0

    def ascii_reliability_diagram(self) -> str:
        """
        Generates an ASCII visualization of the calibration curve:
        Plots confidence vs accuracy for each bin alongside gap bars.
        """
        _, _, _, bins = self.compute_metrics()
        lines = [
            f"{'Bin':<6} {'Conf Range':<14} {'Count':<8} {'Avg Conf':<10} {'Avg Acc':<10} {'Gap (ECE)':<10} {'Visual Gap'}",
            "-" * 75,
        ]
        for b in bins:
            b_idx = b["bin_index"]
            r_str = f"[{b['range'][0]:.2f}, {b['range'][1]:.2f})"
            cnt = b["count"]
            conf = b["avg_confidence"]
            acc = b["avg_accuracy"]
            gap = b["calibration_gap"]
            if cnt == 0:
                visual = "·"
            else:
                bar_len = min(20, int(gap * 40))
                if acc < conf:
                    visual = f"|{'=' * bar_len} (overconf)"
                elif acc > conf:
                    visual = f"|{'=' * bar_len} (underconf)"
                else:
                    visual = "| (perfect)"
            lines.append(f" {b_idx:<5} {r_str:<14} {cnt:<8} {conf:<10.3f} {acc:<10.3f} {gap:<10.3f} {visual}")
        return "\n".join(lines)

    def print_ascii_reliability_diagram(self) -> None:
        """Prints the ASCII reliability diagram to stdout."""
        print(self.ascii_reliability_diagram())

    # -------------------------------------------------------------------------
    # Zero-Dependency Binary Persistence (.reflex-calib)
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serializes online calibration state and reliability curves to .reflex-calib binary.
        Format:
          - 4 bytes magic: b"RFCL"
          - 48 bytes structured header: version, temperature, ece, mce, brier, total_samples, num_bins, reserved
          - 4 bytes json_len (uint32)
          - json_len bytes: UTF-8 JSON payload (config, recent history, reliability bins)
          - 4 bytes CRC32 checksum trailer
        Uses atomic file renaming.
        """
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        st = self.status()

        with self._lock:
            payload = {
                "config": self.config.to_dict(),
                "history": list(self._history),
                "bins": st.bins,
                "saved_at": time.time(),
            }
            json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            json_len = len(json_bytes)

            header = struct.pack(
                CALIB_HEADER_FORMAT,
                CALIB_VERSION,
                float(self._temperature),
                float(st.ece),
                float(st.mce),
                float(st.brier_score),
                int(self._total_samples),
                int(self.config.num_bins),
                b"\x00\x00\x00\x00",
            )

            body = bytearray()
            body.extend(CALIB_MAGIC)
            body.extend(header)
            body.extend(struct.pack("<I", json_len))
            body.extend(json_bytes)

            checksum = zlib.crc32(body) & 0xFFFFFFFF
            body.extend(struct.pack("<I", checksum))

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
    def load(cls, path: str) -> "OnlineProbabilityCalibrator":
        """
        Loads and validates an online calibrator from a .reflex-calib file.
        Verifies magic bytes and 32-bit CRC32 checksum.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")

        file_size = os.path.getsize(path)
        if file_size < CALIB_MIN_FILE_SIZE:
            raise ValueError(
                f"Invalid .reflex-calib file: size {file_size} bytes below minimum {CALIB_MIN_FILE_SIZE} bytes"
            )

        with open(path, "rb") as f:
            data = f.read()

        magic = data[:4]
        if magic != CALIB_MAGIC:
            raise ValueError(f"Invalid magic header: expected {CALIB_MAGIC!r}, got {magic!r}")

        body_bytes = data[:-4]
        stored_crc = struct.unpack("<I", data[-4:])[0]
        computed_crc = zlib.crc32(body_bytes) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC32 checksum mismatch: expected {computed_crc:#010x}, got {stored_crc:#010x}. "
                f"File may be corrupted or tampered."
            )

        version, temperature, ece, mce, brier, total_samples, num_bins, _ = struct.unpack(
            CALIB_HEADER_FORMAT, data[4:4 + CALIB_HEADER_SIZE]
        )

        offset = 4 + CALIB_HEADER_SIZE
        json_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        json_bytes = data[offset:offset + json_len]
        payload = json.loads(json_bytes.decode("utf-8"))

        config = CalibConfig.from_dict(payload.get("config", {}))
        calibrator = cls(config=config, initial_temperature=float(temperature))
        calibrator._total_samples = int(total_samples)

        history_list = payload.get("history", [])
        calibrator._history = deque(history_list, maxlen=config.window_size)

        return calibrator
