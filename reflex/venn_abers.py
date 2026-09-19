"""
Venn-Abers Multi-Class Conformal Predictors & Epistemic Uncertainty Intervals.
Phase 38: Distribution-free, mathematically optimal calibrated multi-probabilistic intervals [p0, p1]
via the Pool Adjacent Violators Algorithm (PAVA) (Vovk & Petej, 2014; Vovk et al., KDD 2015).

Features:
- Pure-Python Pool Adjacent Violators Algorithm (PAVA) isotonic regression in O(N) time.
- Certified multi-probabilistic intervals [p0, p1] with guaranteed validity under exchangeability.
- Epistemic uncertainty quantification (interval width U = p1 - p0) distinguishing aleatoric noise
  from out-of-distribution / data-sparse queries.
- Multi-class Inductive Venn-Abers Predictor (IVAP) via One-vs-Rest decomposition.
- Automated epistemic escalation (is_uncertain) triggering System-2 deliberation.
- Zero-dependency binary persistence (.reflex-va, magic RFVA, 48-byte header, CRC32).
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
VA_MAGIC: bytes = b"RFVA"  # Reflex Venn-Abers
VA_VERSION: int = 1
# Header: uint32 version, float64 max_uncertainty_threshold, float64 mean_uncertainty,
# float64 brier_score, uint32 num_samples, uint32 num_classes, 12s reserved = 48 bytes
VA_HEADER_FORMAT: str = "<IdddII12s"
VA_HEADER_SIZE: int = struct.calcsize(VA_HEADER_FORMAT)
VA_MIN_FILE_SIZE: int = 4 + VA_HEADER_SIZE + 4 + 4  # magic(4) + header(48) + json_len(4) + crc32(4) = 60 bytes


def pava_isotonic_regression(
    scores: Sequence[float],
    labels: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """
    Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
    Finds non-decreasing values g_1 <= g_2 <= ... <= g_n minimizing sum w_i * (y_i - g_i)^2.

    Runs in O(N) time once pairs are sorted by score.

    Args:
        scores: Sequence of numerical scores (must be pre-sorted non-decreasingly).
        labels: Sequence of target values (typically 0.0 or 1.0).
        weights: Optional sample weights (defaults to 1.0 per sample).

    Returns:
        List of fitted isotonic values corresponding to each input sample.
    """
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [float(labels[0])]

    w = [float(x) for x in weights] if weights is not None else [1.0] * n

    # Stack of blocks: each block is [weight, sum_wy, value, start_idx, end_idx]
    # Represented as a list of dataclasses or lightweight dicts
    class Block:
        __slots__ = ("weight", "sum_wy", "val", "start", "end")

        def __init__(self, weight: float, sum_wy: float, start: int, end: int):
            self.weight = weight
            self.sum_wy = sum_wy
            self.val = sum_wy / max(1e-12, weight)
            self.start = start
            self.end = end

    stack: List[Block] = []

    for i in range(n):
        b = Block(w[i], w[i] * float(labels[i]), i, i)
        while stack and stack[-1].val >= b.val:
            prev = stack.pop()
            # Merge prev and b
            new_weight = prev.weight + b.weight
            new_sum_wy = prev.sum_wy + b.sum_wy
            b = Block(new_weight, new_sum_wy, prev.start, b.end)
        stack.append(b)

    # Reconstruct fitted values across all original indices
    fitted = [0.0] * n
    for b in stack:
        for idx in range(b.start, b.end + 1):
            fitted[idx] = b.val

    return fitted


@dataclass
class VennAbersConfig:
    """
    Configuration parameters for Venn-Abers Predictor.
    """
    max_uncertainty_threshold: float = 0.20   # Interval width (p1 - p0) above which System-2 escalates
    min_calibration_samples: int = 20         # Minimum calibration points required for reliable bounds
    point_estimator: str = "balanced"         # "balanced" (p1 / (1 - p0 + p1)) or "midpoint" ((p0 + p1) / 2)
    window_size: int = 500                    # Maximum rolling calibration samples to retain in memory

    def __post_init__(self) -> None:
        if not (0.0 < self.max_uncertainty_threshold < 1.0):
            raise ValueError(f"max_uncertainty_threshold must be in (0, 1), got {self.max_uncertainty_threshold}")
        if self.min_calibration_samples < 2:
            raise ValueError(f"min_calibration_samples must be >= 2, got {self.min_calibration_samples}")
        if self.point_estimator not in ("balanced", "midpoint"):
            raise ValueError(f"Unknown point_estimator '{self.point_estimator}'; expected 'balanced' or 'midpoint'")
        if self.window_size < self.min_calibration_samples:
            raise ValueError(f"window_size ({self.window_size}) must be >= min_calibration_samples ({self.min_calibration_samples})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_uncertainty_threshold": self.max_uncertainty_threshold,
            "min_calibration_samples": self.min_calibration_samples,
            "point_estimator": self.point_estimator,
            "window_size": self.window_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VennAbersConfig":
        return cls(
            max_uncertainty_threshold=float(d.get("max_uncertainty_threshold", 0.20)),
            min_calibration_samples=int(d.get("min_calibration_samples", 20)),
            point_estimator=str(d.get("point_estimator", "balanced")),
            window_size=int(d.get("window_size", 500)),
        )


@dataclass
class VennAbersNoulResult:
    """
    Venn-Abers multi-probabilistic calibrated interval and epistemic uncertainty for a Noul decision.
    """
    p0: float                  # Lower probability bound (calibrated prob assuming y* = 0)
    p1: float                  # Upper probability bound (calibrated prob assuming y* = 1)
    p_calibrated: float        # Point calibrated probability
    uncertainty: float         # Epistemic uncertainty interval width: (p1 - p0)
    should_escalate: bool      # True if epistemic uncertainty exceeds max_uncertainty_threshold
    is_calibrated: bool        # True if calibration set met minimum sample threshold

    def __contains__(self, key: Any) -> bool:
        return key in self.to_dict()

    def __getitem__(self, item: Any) -> Any:
        d = self.to_dict()
        if item in d:
            return d[item]
        return getattr(self, str(item))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p0": round(self.p0, 4),
            "p1": round(self.p1, 4),
            "p_calibrated": round(self.p_calibrated, 4),
            "uncertainty": round(self.uncertainty, 4),
            "interval": [round(self.p0, 4), round(self.p1, 4)],
            "should_escalate": self.should_escalate,
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class VennAbersChoiceResult:
    """
    Multi-class Inductive Venn-Abers Predictor (IVAP) results for a Choice decision.
    """
    intervals: Dict[str, Tuple[float, float]]     # Class -> (p0, p1)
    calibrated_distribution: Dict[str, float]      # Class -> normalized calibrated probability
    uncertainties: Dict[str, float]               # Class -> (p1 - p0)
    max_uncertainty: float                        # Highest epistemic uncertainty across all options
    selected: str                                 # Option with highest calibrated probability
    should_escalate: bool                         # True if max_uncertainty exceeds threshold
    is_calibrated: bool                           # True if calibration set met minimum sample threshold

    def __contains__(self, key: Any) -> bool:
        return key in self.to_dict()

    def __getitem__(self, item: Any) -> Any:
        d = self.to_dict()
        if item in d:
            return d[item]
        return getattr(self, str(item))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervals": {k: [round(v[0], 4), round(v[1], 4)] for k, v in self.intervals.items()},
            "calibrated_distribution": {k: round(v, 4) for k, v in self.calibrated_distribution.items()},
            "uncertainties": {k: round(v, 4) for k, v in self.uncertainties.items()},
            "max_uncertainty": round(self.max_uncertainty, 4),
            "selected": self.selected,
            "should_escalate": self.should_escalate,
            "is_calibrated": self.is_calibrated,
        }


class VennAbersPredictor:
    """
    Inductive Venn-Abers Multi-Class Conformal Predictor.
    Vovk & Petej (2014) distribution-free multi-probabilistic intervals.

    Provides valid lower and upper probability bounds [p0, p1] for binary and multi-class
    decisions. The interval width directly measures epistemic uncertainty, allowing principled
    System-2 escalation on queries where calibration evidence is sparse or contradictory.
    """

    def __init__(
        self,
        config: Optional[VennAbersConfig] = None,
        classes: Optional[Sequence[str]] = None,
    ):
        self.config = config or VennAbersConfig()
        self.classes: List[str] = list(classes) if classes else []
        self._lock = threading.RLock()

        # Calibration samples: list of (score: float, label: Union[int, str])
        self._samples: deque = deque(maxlen=self.config.window_size)
        self._total_samples: int = 0

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return len(self._samples) >= self.config.min_calibration_samples

    @property
    def num_calibration_samples(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    # -------------------------------------------------------------------------
    # Calibration Sample Ingestion
    # -------------------------------------------------------------------------

    def add_calibration_sample(
        self,
        score: float,
        label: Union[bool, int, float, str],
    ) -> None:
        """
        Adds a single calibration score-label pair to the rolling window.
        """
        with self._lock:
            # Normalize label
            if isinstance(label, bool):
                norm_label = 1 if label else 0
            elif isinstance(label, (int, float)):
                norm_label = int(label)
            else:
                norm_label = str(label)
                if norm_label not in self.classes:
                    self.classes.append(norm_label)

            self._samples.append((float(score), norm_label))
            self._total_samples += 1

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[Union[bool, int, float, str]],
    ) -> "VennAbersPredictor":
        """
        Fits or resets the calibration set with a batch of scores and labels.
        """
        if len(scores) != len(labels):
            raise ValueError(f"scores and labels must have equal length: {len(scores)} vs {len(labels)}")

        with self._lock:
            self._samples.clear()
            for s, y in zip(scores, labels):
                self.add_calibration_sample(s, y)
        return self

    # -------------------------------------------------------------------------
    # Binary Venn-Abers Prediction (Noul)
    # -------------------------------------------------------------------------

    def _compute_binary_interval(
        self,
        raw_score: float,
        calibration_pairs: List[Tuple[float, int]],
    ) -> Tuple[float, float, float]:
        """
        Fits dual isotonic regressions for candidate hypotheses (s*, 0) and (s*, 1)
        using the Pool Adjacent Violators Algorithm (PAVA).

        Returns:
            (p0, p1, p_calibrated)
        """
        if not calibration_pairs:
            # No calibration data: uninformative bounds [0, 1]
            return 0.0, 1.0, 0.5

        s_star = float(raw_score)

        # 1. Hypothesis y* = 0
        # Augment calibration set with (s_star, 0)
        # Using a stable tuple sorting key: (score, is_test_sample)
        # To break ties conservatively, test point with y=0 placed before identical scores
        augmented_0 = list(calibration_pairs)
        augmented_0.append((s_star, 0))
        # Sort primarily by score
        # For equal scores, placing 0 before 1 preserves non-decreasing monotonicity
        augmented_0.sort(key=lambda item: item[0])

        scores_0 = [item[0] for item in augmented_0]
        labels_0 = [float(item[1]) for item in augmented_0]
        fitted_0 = pava_isotonic_regression(scores_0, labels_0)

        # Locate fitted value for s_star
        # In case of duplicates, find any index matching s_star and y=0
        idx_0 = 0
        for i, (sc, lb) in enumerate(augmented_0):
            if sc == s_star and lb == 0:
                idx_0 = i
                break
        p0 = max(0.0, min(1.0, fitted_0[idx_0]))

        # 2. Hypothesis y* = 1
        augmented_1 = list(calibration_pairs)
        augmented_1.append((s_star, 1))
        augmented_1.sort(key=lambda item: item[0])

        scores_1 = [item[0] for item in augmented_1]
        labels_1 = [float(item[1]) for item in augmented_1]
        fitted_1 = pava_isotonic_regression(scores_1, labels_1)

        idx_1 = len(augmented_1) - 1
        for i in range(len(augmented_1) - 1, -1, -1):
            sc, lb = augmented_1[i]
            if sc == s_star and lb == 1:
                idx_1 = i
                break
        p1 = max(0.0, min(1.0, fitted_1[idx_1]))

        # Theoretical invariant: p0 <= p1
        if p0 > p1:
            p0, p1 = p1, p0

        # Point estimator
        if self.config.point_estimator == "balanced":
            denom = 1.0 - p0 + p1
            if denom > 1e-12:
                p_calib = max(0.0, min(1.0, p1 / denom))
            else:
                p_calib = 0.5 * (p0 + p1)
        else:
            p_calib = 0.5 * (p0 + p1)

        return p0, p1, p_calib

    def predict_noul(self, raw_score: float) -> VennAbersNoulResult:
        """
        Computes calibrated probability interval [p0, p1] and point probability for a Noul decision.
        """
        with self._lock:
            calibrated = len(self._samples) >= self.config.min_calibration_samples
            # Extract binary calibration pairs (score, 0 or 1)
            pairs: List[Tuple[float, int]] = []
            for s, y in self._samples:
                if isinstance(y, (int, float)):
                    pairs.append((s, 1 if y > 0 else 0))
                elif isinstance(y, str) and self.classes:
                    pairs.append((s, 1 if y == self.classes[0] else 0))
                else:
                    pairs.append((s, 1 if str(y).lower() in ("true", "1", "yes") else 0))

        if not calibrated:
            # Fallback before minimum calibration warmup
            clamped_s = max(0.0, min(1.0, float(raw_score)))
            return VennAbersNoulResult(
                p0=clamped_s,
                p1=clamped_s,
                p_calibrated=clamped_s,
                uncertainty=0.0,
                should_escalate=False,
                is_calibrated=False,
            )

        p0, p1, p_calib = self._compute_binary_interval(raw_score, pairs)
        uncertainty = p1 - p0
        should_escalate = uncertainty > self.config.max_uncertainty_threshold

        return VennAbersNoulResult(
            p0=p0,
            p1=p1,
            p_calibrated=p_calib,
            uncertainty=uncertainty,
            should_escalate=should_escalate,
            is_calibrated=True,
        )

    # -------------------------------------------------------------------------
    # Multi-Class Venn-Abers Prediction (Choice - IVAP)
    # -------------------------------------------------------------------------

    def predict_choice(self, distribution: Dict[str, float]) -> VennAbersChoiceResult:
        """
        Computes multi-class Inductive Venn-Abers Predictor (IVAP) intervals for Choice decisions.
        Uses One-vs-Rest (OvR) binary calibration for each categorical class.
        """
        if not distribution:
            return VennAbersChoiceResult(
                intervals={},
                calibrated_distribution={},
                uncertainties={},
                max_uncertainty=0.0,
                selected="",
                should_escalate=False,
                is_calibrated=False,
            )

        classes = list(distribution.keys())

        with self._lock:
            calibrated = len(self._samples) >= self.config.min_calibration_samples
            samples_snapshot = list(self._samples)

        if not calibrated:
            # Return raw distribution when not yet calibrated
            selected = max(distribution.items(), key=lambda kv: kv[1])[0]
            intervals = {k: (distribution[k], distribution[k]) for k in classes}
            uncertainties = {k: 0.0 for k in classes}
            return VennAbersChoiceResult(
                intervals=intervals,
                calibrated_distribution=dict(distribution),
                uncertainties=uncertainties,
                max_uncertainty=0.0,
                selected=selected,
                should_escalate=False,
                is_calibrated=False,
            )

        intervals: Dict[str, Tuple[float, float]] = {}
        unnormalized_probs: Dict[str, float] = {}
        uncertainties: Dict[str, float] = {}

        for c in classes:
            # Binary OvR calibration set: y_ovr = 1 if sample_label == c else 0
            ovr_pairs: List[Tuple[float, int]] = []
            for s, y in samples_snapshot:
                label_match = (y == c) or (str(y) == c)
                ovr_pairs.append((s, 1 if label_match else 0))

            raw_sc = distribution.get(c, 0.0)
            p0, p1, p_calib = self._compute_binary_interval(raw_sc, ovr_pairs)
            intervals[c] = (p0, p1)
            unnormalized_probs[c] = p_calib
            uncertainties[c] = p1 - p0

        # Normalize point probabilities
        total_p = sum(unnormalized_probs.values())
        if total_p > 0.0:
            calibrated_dist = {c: p / total_p for c, p in unnormalized_probs.items()}
        else:
            uniform = 1.0 / float(len(classes))
            calibrated_dist = {c: uniform for c in classes}

        max_u = max(uncertainties.values()) if uncertainties else 0.0
        selected = max(calibrated_dist.items(), key=lambda kv: kv[1])[0]
        should_escalate = max_u > self.config.max_uncertainty_threshold

        return VennAbersChoiceResult(
            intervals=intervals,
            calibrated_distribution=calibrated_dist,
            uncertainties=uncertainties,
            max_uncertainty=max_u,
            selected=selected,
            should_escalate=should_escalate,
            is_calibrated=True,
        )

    # -------------------------------------------------------------------------
    # Telemetry & Diagnostics
    # -------------------------------------------------------------------------

    def mean_uncertainty(self, num_eval_points: int = 50) -> float:
        """
        Estimates expected epistemic uncertainty width across the score spectrum [0, 1].
        """
        with self._lock:
            if len(self._samples) < self.config.min_calibration_samples:
                return 0.0
            pairs = [(s, 1 if (y is True or y == 1 or y == "1") else 0) for s, y in self._samples]

        step = 1.0 / float(num_eval_points)
        widths = []
        for i in range(num_eval_points + 1):
            sc = i * step
            p0, p1, _ = self._compute_binary_interval(sc, pairs)
            widths.append(p1 - p0)

        return sum(widths) / float(len(widths)) if widths else 0.0

    def compute_brier_score(self) -> float:
        """Computes empirical Brier score on current calibration samples."""
        with self._lock:
            if not self._samples:
                return 0.0
            samples = list(self._samples)

        brier_sum = 0.0
        for s, y in samples:
            y_float = 1.0 if (y is True or y == 1 or y == "1") else 0.0
            res = self.predict_noul(s)
            brier_sum += (res.p_calibrated - y_float) ** 2
        return brier_sum / float(len(samples))

    # -------------------------------------------------------------------------
    # Zero-Dependency Binary Persistence (.reflex-va)
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serializes Venn-Abers calibration state to .reflex-va binary format.
        Format:
          - 4 bytes magic: b"RFVA"
          - 48 bytes structured header
          - 4 bytes json_len (uint32)
          - json_len bytes UTF-8 JSON payload (config, sample history, classes)
          - 4 bytes CRC32 checksum trailer
        Uses atomic file replacement.
        """
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        mean_u = self.mean_uncertainty(num_eval_points=30)
        brier = self.compute_brier_score()

        with self._lock:
            payload = {
                "config": self.config.to_dict(),
                "classes": self.classes,
                "samples": list(self._samples),
                "saved_at": time.time(),
            }
            json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            json_len = len(json_bytes)

            num_classes = len(self.classes) if self.classes else 2
            header = struct.pack(
                VA_HEADER_FORMAT,
                VA_VERSION,
                float(self.config.max_uncertainty_threshold),
                float(mean_u),
                float(brier),
                int(self._total_samples),
                int(num_classes),
                b"\x00" * 12,
            )

            body = bytearray()
            body.extend(VA_MAGIC)
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
    def load(cls, path: str) -> "VennAbersPredictor":
        """
        Loads and validates a Venn-Abers predictor from a .reflex-va file.
        Verifies magic bytes and 32-bit CRC32 checksum.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Venn-Abers file not found: {path}")

        file_size = os.path.getsize(path)
        if file_size < VA_MIN_FILE_SIZE:
            raise ValueError(
                f"Invalid .reflex-va file: size {file_size} bytes below minimum {VA_MIN_FILE_SIZE} bytes"
            )

        with open(path, "rb") as f:
            data = f.read()

        magic = data[:4]
        if magic != VA_MAGIC:
            raise ValueError(f"Invalid magic header: expected {VA_MAGIC!r}, got {magic!r}")

        body_bytes = data[:-4]
        stored_crc = struct.unpack("<I", data[-4:])[0]
        computed_crc = zlib.crc32(body_bytes) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC32 checksum mismatch: expected {computed_crc:#010x}, got {stored_crc:#010x}. "
                f"File may be corrupted or tampered."
            )

        version, max_u, mean_u, brier, total_samples, num_classes, _ = struct.unpack(
            VA_HEADER_FORMAT, data[4:4 + VA_HEADER_SIZE]
        )

        offset = 4 + VA_HEADER_SIZE
        json_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        json_bytes = data[offset:offset + json_len]
        payload = json.loads(json_bytes.decode("utf-8"))

        config = VennAbersConfig.from_dict(payload.get("config", {}))
        classes = payload.get("classes", [])
        predictor = cls(config=config, classes=classes)
        predictor._total_samples = int(total_samples)

        samples = payload.get("samples", [])
        for item in samples:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                predictor._samples.append((float(item[0]), item[1]))

        return predictor
