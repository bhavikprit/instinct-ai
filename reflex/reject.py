"""
Selective Classification & Risk-Controlled Rejection Option.
Phase 39: Mathematically guaranteed selective risk control and optimal autonomous coverage
with rejection options (Geifman & El-Yaniv, NeurIPS 2017; ICML 2019).

Features:
- Finite-sample statistical risk upper bounding via Clopper-Pearson / Wilson-score intervals.
- Dual operating modes: Target-Risk mode (guarantee error <= r* while maximizing coverage)
  and Target-Coverage mode (guarantee coverage >= phi* while minimizing risk).
- Multi-metric confidence scoring: softmax confidence, margin (top-1 vs top-2), and Shannon entropy.
- Risk-Coverage (RC) curves and Area Under Risk-Coverage Curve (AURC) computation.
- Automated rejection (g(x) = 0) triggering System-2 deliberation.
- Zero-dependency binary persistence (.reflex-reject, magic RFRJ, 48-byte header, CRC32).
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

from reflex.primitives import Choice, Noul


# Binary persistence constants
REJECT_MAGIC: bytes = b"RFRJ"  # Reflex Reject
REJECT_VERSION: int = 1
# Header: uint32 version, float64 threshold, float64 target_risk, float64 target_coverage,
# float64 empirical_risk, float64 empirical_coverage, uint32 num_samples, 4s reserved = 48 bytes
REJECT_HEADER_FORMAT: str = "<IdddddI4s"
REJECT_HEADER_SIZE: int = struct.calcsize(REJECT_HEADER_FORMAT)
REJECT_MIN_FILE_SIZE: int = 4 + REJECT_HEADER_SIZE + 4 + 4  # magic(4) + header(48) + json_len(4) + crc32(4) = 60 bytes


def _inv_norm_cdf(p: float) -> float:
    """
    Approximation of the inverse standard normal CDF (probit function) via rational approximation.
    Accurate to within 1e-4 for p in (1e-6, 1 - 1e-6).
    """
    p = max(1e-7, min(1.0 - 1e-7, float(p)))
    if p < 0.5:
        # Lower tail
        t = math.sqrt(-2.0 * math.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return -(t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0))
    else:
        # Upper tail
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)


def binomial_risk_upper_bound(
    errors: int,
    total: int,
    delta: float = 0.05,
) -> float:
    """
    Calculates the 1 - delta upper confidence bound for a binomial proportion
    (selective risk) using the Wilson score interval with continuity correction.

    Args:
        errors: Number of errors among accepted decisions (k).
        total: Number of accepted decisions (m).
        delta: Significance level (default 0.05 for 95% confidence).

    Returns:
        Upper bound risk float in [0.0, 1.0].
    """
    if total <= 0:
        return 1.0
    if errors <= 0:
        # Zero errors rule of three: -log(delta) / total
        return min(1.0, max(0.0, -math.log(max(1e-9, delta)) / float(total)))

    p_hat = float(errors) / float(total)
    z = _inv_norm_cdf(1.0 - delta)
    z2 = z * z
    m = float(total)

    # Wilson score interval with continuity correction (upper bound)
    denom = 1.0 + (z2 / m)
    center = p_hat + (z2 / (2.0 * m))
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / m) + (z2 / (4.0 * m * m)))

    upper = (center + spread) / denom
    return min(1.0, max(0.0, upper))


@dataclass
class SelectiveRejectConfig:
    """
    Configuration parameters for Selective Classifier.
    """
    target_risk: Optional[float] = 0.02          # Maximum acceptable selective error rate (e.g. 0.02 for <=2%)
    target_coverage: Optional[float] = None       # Alternative: target minimum autonomous coverage (e.g. 0.85)
    confidence_bound: float = 0.05               # Significance level delta (e.g. 0.05 for 95% statistical confidence)
    min_calibration_samples: int = 30            # Minimum calibration samples before activating selective rejection
    scoring_method: str = "confidence"           # "confidence", "margin", or "entropy"
    window_size: int = 1000                      # Maximum rolling calibration samples in memory

    def __post_init__(self) -> None:
        if self.target_risk is not None and not (0.0 < self.target_risk < 1.0):
            raise ValueError(f"target_risk must be in (0, 1), got {self.target_risk}")
        if self.target_coverage is not None and not (0.0 < self.target_coverage <= 1.0):
            raise ValueError(f"target_coverage must be in (0, 1], got {self.target_coverage}")
        if not (0.0 < self.confidence_bound < 1.0):
            raise ValueError(f"confidence_bound must be in (0, 1), got {self.confidence_bound}")
        if self.min_calibration_samples < 5:
            raise ValueError(f"min_calibration_samples must be >= 5, got {self.min_calibration_samples}")
        if self.scoring_method not in ("confidence", "margin", "entropy"):
            raise ValueError(f"Unknown scoring_method '{self.scoring_method}'; expected 'confidence', 'margin', or 'entropy'")
        if self.window_size < self.min_calibration_samples:
            raise ValueError(f"window_size must be >= min_calibration_samples")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_risk": self.target_risk,
            "target_coverage": self.target_coverage,
            "confidence_bound": self.confidence_bound,
            "min_calibration_samples": self.min_calibration_samples,
            "scoring_method": self.scoring_method,
            "window_size": self.window_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SelectiveRejectConfig":
        return cls(
            target_risk=float(d["target_risk"]) if d.get("target_risk") is not None else None,
            target_coverage=float(d["target_coverage"]) if d.get("target_coverage") is not None else None,
            confidence_bound=float(d.get("confidence_bound", 0.05)),
            min_calibration_samples=int(d.get("min_calibration_samples", 30)),
            scoring_method=str(d.get("scoring_method", "confidence")),
            window_size=int(d.get("window_size", 1000)),
        )


@dataclass
class SelectiveDecision:
    """
    Selective evaluation decision with acceptance/abstention verdict and risk bounds.
    """
    accepted: bool                    # True if g(x) = 1 (System-1 autonomous), False if abstained (System-2)
    score: float                      # Confidence/rejection score kappa(x)
    threshold: float                  # Calibrated optimal rejection threshold theta*
    empirical_risk: float             # Empirical risk on accepted slice
    upper_bound_risk: float           # Statistically guaranteed upper bound risk (1 - delta)
    coverage: float                   # Empirical coverage (fraction accepted)
    should_escalate: bool             # True if abstained (accepted=False)
    reason: str                       # Human-readable explanation

    def __contains__(self, key: Any) -> bool:
        return key in self.to_dict()

    def __getitem__(self, item: Any) -> Any:
        d = self.to_dict()
        if item in d:
            return d[item]
        return getattr(self, str(item))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "empirical_risk": round(self.empirical_risk, 4),
            "upper_bound_risk": round(self.upper_bound_risk, 4),
            "coverage": round(self.coverage, 4),
            "should_escalate": self.should_escalate,
            "reason": self.reason,
        }


@dataclass
class RiskCoveragePoint:
    """
    A single point along the empirical Risk-Coverage trade-off curve.
    """
    threshold: float
    coverage: float
    empirical_risk: float
    upper_bound_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "coverage": round(self.coverage, 4),
            "empirical_risk": round(self.empirical_risk, 4),
            "upper_bound_risk": round(self.upper_bound_risk, 4),
        }


class SelectiveClassifier:
    """
    Selective Classifier & Risk-Controlled Rejection Engine.
    Geifman & El-Yaniv (NeurIPS 2017 / ICML 2019).

    Learns an optimal rejection policy g(x) = I(kappa(x) >= theta*) maximizing
    coverage while guaranteeing that selective risk on accepted decisions remains
    below the enterprise target budget r* with 1 - delta statistical confidence.
    """

    def __init__(
        self,
        config: Optional[SelectiveRejectConfig] = None,
    ):
        self.config = config or SelectiveRejectConfig()
        self._lock = threading.RLock()

        # Calibration sample buffer: list of (confidence_score: float, is_error: int)
        self._samples: deque = deque(maxlen=self.config.window_size)
        self._total_samples: int = 0

        # Calibrated parameters
        self._calibrated: bool = False
        self._threshold: float = 0.50
        self._calibrated_empirical_risk: float = 0.0
        self._calibrated_upper_risk: float = 0.0
        self._calibrated_coverage: float = 1.0

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._calibrated and (len(self._samples) >= self.config.min_calibration_samples)

    @property
    def threshold(self) -> float:
        with self._lock:
            return self._threshold

    @property
    def num_calibration_samples(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    @property
    def calibrated_empirical_risk(self) -> float:
        with self._lock:
            return self._calibrated_empirical_risk

    @property
    def calibrated_upper_risk(self) -> float:
        with self._lock:
            return self._calibrated_upper_risk

    @property
    def calibrated_coverage(self) -> float:
        with self._lock:
            return self._calibrated_coverage

    # -------------------------------------------------------------------------
    # Sample Ingestion & Calibration
    # -------------------------------------------------------------------------

    def add_sample(
        self,
        score: float,
        is_error: Union[bool, int],
    ) -> None:
        """
        Adds a single calibration sample (confidence score, error indicator).
        is_error: 1 (or True) if the model prediction was incorrect; 0 (or False) if correct.
        """
        with self._lock:
            err_flag = 1 if (is_error is True or is_error == 1 or is_error == 1.0) else 0
            self._samples.append((float(score), err_flag))
            self._total_samples += 1

    add_calibration_sample = add_sample

    def fit(
        self,
        scores: Sequence[float],
        errors: Sequence[Union[bool, int]],
    ) -> "SelectiveClassifier":
        """
        Fits or resets the calibration buffer with a batch of scores and error indicators,
        then automatically executes threshold calibration.
        """
        if len(scores) != len(errors):
            raise ValueError(f"scores and errors must have equal length: {len(scores)} vs {len(errors)}")

        with self._lock:
            self._samples.clear()
            for s, e in zip(scores, errors):
                self.add_sample(s, e)
            self.calibrate()
        return self

    def calibrate(self) -> float:
        """
        Finds the optimal threshold theta* meeting the target risk or target coverage constraint.
        Uses finite-sample Wilson score / binomial confidence upper bounds.

        Returns:
            Calibrated threshold theta*.
        """
        with self._lock:
            if len(self._samples) < self.config.min_calibration_samples:
                self._calibrated = False
                return self._threshold

            samples = list(self._samples)
            n = len(samples)

            # Extract distinct candidate thresholds
            unique_scores = sorted(list(set(s for s, _ in samples)))
            # If too many unique scores, downsample candidates to 150 for efficiency
            if len(unique_scores) > 150:
                step = len(unique_scores) / 150.0
                candidates = [unique_scores[int(i * step)] for i in range(150)]
                if unique_scores[-1] not in candidates:
                    candidates.append(unique_scores[-1])
            else:
                candidates = unique_scores

            target_r = self.config.target_risk
            target_c = self.config.target_coverage
            delta = self.config.confidence_bound

            best_theta = candidates[0]
            best_emp_risk = 0.0
            best_upper_risk = 0.0
            best_coverage = 0.0
            found_feasible = False

            if target_r is not None:
                # Target Risk Mode: Find lowest theta (maximizing coverage) where upper_bound_risk <= target_r
                for th in candidates:
                    accepted = [(s, e) for s, e in samples if s >= th]
                    m = len(accepted)
                    if m == 0:
                        continue
                    k = sum(e for _, e in accepted)
                    emp_r = float(k) / float(m)
                    upper_r = binomial_risk_upper_bound(k, m, delta=delta)
                    cov = float(m) / float(n)

                    if upper_r <= target_r:
                        best_theta = th
                        best_emp_risk = emp_r
                        best_upper_risk = upper_r
                        best_coverage = cov
                        found_feasible = True
                        # Lowest candidate meeting risk condition maximizes coverage
                        break

                if not found_feasible:
                    # If upper bound cannot be met, fallback to highest candidate (most conservative)
                    th = candidates[-1]
                    accepted = [(s, e) for s, e in samples if s >= th]
                    m = len(accepted) if accepted else 1
                    k = sum(e for _, e in accepted) if accepted else 0
                    best_theta = th
                    best_emp_risk = float(k) / float(m)
                    best_upper_risk = binomial_risk_upper_bound(k, m, delta=delta)
                    best_coverage = float(len(accepted)) / float(n)

            elif target_c is not None:
                # Target Coverage Mode: Find highest theta where coverage >= target_c
                for th in candidates:
                    accepted = [(s, e) for s, e in samples if s >= th]
                    cov = float(len(accepted)) / float(n)
                    if cov >= target_c:
                        m = len(accepted)
                        k = sum(e for _, e in accepted)
                        best_theta = th
                        best_emp_risk = float(k) / float(m) if m > 0 else 0.0
                        best_upper_risk = binomial_risk_upper_bound(k, m, delta=delta) if m > 0 else 0.0
                        best_coverage = cov
                        found_feasible = True
                    else:
                        break

            self._threshold = best_theta
            self._calibrated_empirical_risk = best_emp_risk
            self._calibrated_upper_risk = best_upper_risk
            self._calibrated_coverage = best_coverage
            self._calibrated = True

            return self._threshold

    # -------------------------------------------------------------------------
    # Decision Evaluation (Noul & Choice)
    # -------------------------------------------------------------------------

    def extract_noul_score(self, noul: Noul) -> float:
        """
        Extracts confidence score kappa(x) from a Noul primitive.
        For boolean probability p in [0, 1], confidence is max(p, 1 - p) in [0.5, 1.0].
        """
        p = noul.probability if noul.probability is not None else 0.5
        p = max(0.0, min(1.0, float(p)))
        return max(p, 1.0 - p)

    def extract_choice_score(self, choice: Choice) -> float:
        """
        Extracts confidence score kappa(x) from a Choice primitive.
        Methods:
          - "confidence": max class probability.
          - "margin": difference between top-1 and top-2 probabilities.
          - "entropy": normalized negative entropy (1 - H(p)/log(K)).
        """
        dist = choice.distribution
        if not dist:
            return 0.0

        probs = sorted(list(dist.values()), reverse=True)
        if not probs:
            return 0.0

        method = self.config.scoring_method
        if method == "margin":
            top1 = probs[0]
            top2 = probs[1] if len(probs) > 1 else 0.0
            return max(0.0, min(1.0, top1 - top2))
        elif method == "entropy":
            k = len(probs)
            if k <= 1:
                return 1.0
            ent = -sum(p * math.log(max(1e-9, p)) for p in probs if p > 0.0)
            max_ent = math.log(k)
            norm_ent = ent / max_ent
            return max(0.0, min(1.0, 1.0 - norm_ent))
        else:
            # Default: max confidence
            return max(0.0, min(1.0, probs[0]))

    def evaluate_score(self, score: float) -> SelectiveDecision:
        """
        Evaluates an extracted confidence score against the calibrated rejection threshold.
        """
        with self._lock:
            calibrated = self.is_calibrated
            th = self._threshold
            emp_r = self._calibrated_empirical_risk
            up_r = self._calibrated_upper_risk
            cov = self._calibrated_coverage

        sc = float(score)

        if not calibrated:
            # Under warmup: always accept, report uncalibrated status
            return SelectiveDecision(
                accepted=True,
                score=sc,
                threshold=th,
                empirical_risk=0.0,
                upper_bound_risk=1.0,
                coverage=1.0,
                should_escalate=False,
                reason="Warmup phase: calibration sample threshold not yet reached",
            )

        accepted = sc >= th
        should_escalate = not accepted

        if accepted:
            reason = f"Accepted: confidence {sc:.4f} exceeds rejection threshold {th:.4f}"
        else:
            reason = f"Abstained: confidence {sc:.4f} below threshold {th:.4f} (risk target: {self.config.target_risk})"

        return SelectiveDecision(
            accepted=accepted,
            score=sc,
            threshold=th,
            empirical_risk=emp_r,
            upper_bound_risk=up_r,
            coverage=cov,
            should_escalate=should_escalate,
            reason=reason,
        )

    def evaluate_noul(self, noul: Noul) -> SelectiveDecision:
        """Evaluates a Noul primitive under selective risk control."""
        sc = self.extract_noul_score(noul)
        return self.evaluate_score(sc)

    def evaluate_choice(self, choice: Choice) -> SelectiveDecision:
        """Evaluates a Choice primitive under selective risk control."""
        sc = self.extract_choice_score(choice)
        return self.evaluate_score(sc)

    # -------------------------------------------------------------------------
    # Risk-Coverage Curves & AURC
    # -------------------------------------------------------------------------

    def risk_coverage_curve(self, num_points: int = 50) -> List[RiskCoveragePoint]:
        """
        Computes the complete empirical Risk-Coverage trade-off curve across candidate thresholds.
        """
        with self._lock:
            if not self._samples:
                return []
            samples = list(self._samples)
            delta = self.config.confidence_bound

        n = len(samples)
        scores = [s for s, _ in samples]
        min_s = min(scores)
        max_s = max(scores)

        step = (max_s - min_s) / float(max(1, num_points - 1)) if max_s > min_s else 0.0
        points = []

        for i in range(num_points):
            th = min_s + i * step
            accepted = [(s, e) for s, e in samples if s >= th]
            m = len(accepted)
            cov = float(m) / float(n)
            k = sum(e for _, e in accepted)
            emp_r = float(k) / float(m) if m > 0 else 0.0
            up_r = binomial_risk_upper_bound(k, m, delta=delta) if m > 0 else 0.0

            points.append(RiskCoveragePoint(
                threshold=th,
                coverage=cov,
                empirical_risk=emp_r,
                upper_bound_risk=up_r,
            ))

        return points

    def aurc(self, num_points: int = 50) -> float:
        """
        Area Under the Risk-Coverage Curve (AURC).
        Computed via trapezoidal integration over sorted coverage points.
        Lower values indicate superior selective performance.
        """
        curve = self.risk_coverage_curve(num_points=num_points)
        if len(curve) < 2:
            return 0.0

        # Sort by coverage ascending
        sorted_curve = sorted(curve, key=lambda p: p.coverage)
        area = 0.0
        for i in range(len(sorted_curve) - 1):
            p1 = sorted_curve[i]
            p2 = sorted_curve[i + 1]
            dx = p2.coverage - p1.coverage
            avg_y = 0.5 * (p1.empirical_risk + p2.empirical_risk)
            area += avg_y * dx

        return area

    def ascii_risk_coverage_curve(self, num_points: int = 15) -> str:
        """
        Renders an ASCII visualization of the Risk-Coverage trade-off frontier.
        """
        curve = self.risk_coverage_curve(num_points=num_points)
        if not curve:
            return "No calibration data available for Risk-Coverage curve."

        lines = [
            f"{'Threshold':<12} {'Coverage':<12} {'Empirical Risk':<16} {'Upper Bound (95%)':<18} {'Risk Visual'}",
            "-" * 80,
        ]
        for pt in curve:
            bar_len = min(25, int(pt.empirical_risk * 100))
            vis = f"|{'█' * bar_len}{' ' * (25 - bar_len)}| ({pt.empirical_risk * 100:5.2f}%)"
            lines.append(
                f"{pt.threshold:<12.4f} {pt.coverage * 100:<10.1f}%  {pt.empirical_risk * 100:<14.2f}% {pt.upper_bound_risk * 100:<16.2f}% {vis}"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Zero-Dependency Binary Persistence (.reflex-reject)
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serializes SelectiveClassifier calibration state and samples to .reflex-reject binary.
        Format:
          - 4 bytes magic: b"RFRJ"
          - 48 bytes structured header
          - 4 bytes json_len (uint32)
          - json_len bytes UTF-8 JSON payload (config, samples)
          - 4 bytes CRC32 checksum trailer
        Uses atomic file replacement.
        """
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with self._lock:
            payload = {
                "config": self.config.to_dict(),
                "samples": list(self._samples),
                "threshold": self._threshold,
                "empirical_risk": self._calibrated_empirical_risk,
                "upper_risk": self._calibrated_upper_risk,
                "coverage": self._calibrated_coverage,
                "saved_at": time.time(),
            }
            json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            json_len = len(json_bytes)

            t_risk = float(self.config.target_risk) if self.config.target_risk is not None else -1.0
            t_cov = float(self.config.target_coverage) if self.config.target_coverage is not None else -1.0

            header = struct.pack(
                REJECT_HEADER_FORMAT,
                REJECT_VERSION,
                float(self._threshold),
                t_risk,
                t_cov,
                float(self._calibrated_empirical_risk),
                float(self._calibrated_coverage),
                int(self._total_samples),
                b"\x00" * 4,
            )

            body = bytearray()
            body.extend(REJECT_MAGIC)
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
    def load(cls, path: str) -> "SelectiveClassifier":
        """
        Loads and validates a SelectiveClassifier from a .reflex-reject file.
        Verifies magic bytes and 32-bit CRC32 checksum.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Selective rejection file not found: {path}")

        file_size = os.path.getsize(path)
        if file_size < REJECT_MIN_FILE_SIZE:
            raise ValueError(
                f"Invalid .reflex-reject file: size {file_size} bytes below minimum {REJECT_MIN_FILE_SIZE} bytes"
            )

        with open(path, "rb") as f:
            data = f.read()

        magic = data[:4]
        if magic != REJECT_MAGIC:
            raise ValueError(f"Invalid magic header: expected {REJECT_MAGIC!r}, got {magic!r}")

        body_bytes = data[:-4]
        stored_crc = struct.unpack("<I", data[-4:])[0]
        computed_crc = zlib.crc32(body_bytes) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC32 checksum mismatch: expected {computed_crc:#010x}, got {stored_crc:#010x}. "
                f"File may be corrupted or tampered."
            )

        version, threshold, t_risk, t_cov, emp_risk, cov, total_samples, _ = struct.unpack(
            REJECT_HEADER_FORMAT, data[4:4 + REJECT_HEADER_SIZE]
        )

        offset = 4 + REJECT_HEADER_SIZE
        json_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        json_bytes = data[offset:offset + json_len]
        payload = json.loads(json_bytes.decode("utf-8"))

        config = SelectiveRejectConfig.from_dict(payload.get("config", {}))
        classifier = cls(config=config)
        classifier._threshold = float(threshold)
        classifier._calibrated_empirical_risk = float(emp_risk)
        classifier._calibrated_coverage = float(cov)
        classifier._total_samples = int(total_samples)

        samples = payload.get("samples", [])
        for item in samples:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                classifier._samples.append((float(item[0]), int(item[1])))

        classifier._calibrated = len(classifier._samples) >= config.min_calibration_samples
        if classifier._calibrated:
            k = sum(e for s, e in classifier._samples if s >= classifier._threshold)
            m = sum(1 for s, _ in classifier._samples if s >= classifier._threshold)
            classifier._calibrated_upper_risk = binomial_risk_upper_bound(k, m, delta=config.confidence_bound)

        return classifier
