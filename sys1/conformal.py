"""
Reflex Distribution-Free Conformal Prediction & Calibration Bounds (Phase 33).
Provides finite-sample statistical safety guarantees:
    P(Y in C(X)) >= 1 - alpha
Transforms heuristic confidence thresholds into certifiable prediction sets:
- Singleton sets (|C(X)| = 1) -> Safe System 1 shortcut (<0.1ms, $0 cost).
- Ambiguous sets (|C(X)| > 1) -> Certified ambiguity -> Escalate to System 2.
- Empty sets (C(X) = 0) -> Out-of-Distribution / Anomaly -> Escalate with Security Alert.
Supports global split conformal prediction and Mondrian (class-conditional) conformal prediction.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import os
import struct
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import zlib

from sys1.primitives import Noul, Choice

CONFORMAL_MAGIC = b"RFCF"
CONFORMAL_FORMAT_VERSION = 1


@dataclass
class ConformalConfig:
    """Hyperparameters for Conformal Prediction calibration and evaluation."""
    alpha: float = 0.05  # Significance level (1 - alpha coverage guarantee, e.g. 0.05 -> 95%)
    mondrian: bool = True  # Class-conditional calibration (guarantees coverage per class)
    min_calibration_samples: int = 30  # Minimum samples required per calibration partition
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {self.alpha}")
        if self.min_calibration_samples < 5:
            raise ValueError(
                f"min_calibration_samples must be at least 5, got {self.min_calibration_samples}"
            )

    @property
    def coverage_guarantee(self) -> float:
        """Nominal coverage guarantee (1 - alpha)."""
        return 1.0 - self.alpha


@dataclass
class ConformalNoulResult:
    """Certifiable conformal prediction result for a boolean Noul primitive."""
    prediction_set: Set[bool]
    p_value_true: float
    p_value_false: float
    alpha: float
    coverage_guarantee: float
    instructions: str = ""
    raw_probability: Optional[float] = None

    @property
    def is_singleton(self) -> bool:
        """True if the prediction set contains exactly one decisive outcome."""
        return len(self.prediction_set) == 1

    @property
    def is_empty(self) -> bool:
        """True if the prediction set is empty (out-of-distribution anomaly)."""
        return len(self.prediction_set) == 0

    @property
    def is_ambiguous(self) -> bool:
        """True if the prediction set contains multiple conflicting hypotheses."""
        return len(self.prediction_set) > 1

    @property
    def should_escalate(self) -> bool:
        """Certified escalation rule: Escalate to System 2 if not singleton."""
        return not self.is_singleton

    @property
    def certified_value(self) -> Optional[bool]:
        """Returns the certifiably safe boolean value, or None if ambiguous/empty."""
        if self.is_singleton:
            return next(iter(self.prediction_set))
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_set": sorted(list(self.prediction_set)),
            "is_singleton": self.is_singleton,
            "is_ambiguous": self.is_ambiguous,
            "is_empty": self.is_empty,
            "should_escalate": self.should_escalate,
            "certified_value": self.certified_value,
            "p_value_true": round(self.p_value_true, 6),
            "p_value_false": round(self.p_value_false, 6),
            "alpha": self.alpha,
            "coverage_guarantee": self.coverage_guarantee,
            "raw_probability": round(self.raw_probability, 4) if self.raw_probability is not None else None,
        }


@dataclass
class ConformalChoiceResult:
    """Certifiable conformal prediction result for a multi-class Choice primitive."""
    prediction_set: List[str]
    p_values: Dict[str, float]
    alpha: float
    coverage_guarantee: float
    instructions: str = ""
    distribution: Dict[str, float] = field(default_factory=dict)

    @property
    def is_singleton(self) -> bool:
        """True if exactly one option is in the certified set."""
        return len(self.prediction_set) == 1

    @property
    def is_empty(self) -> bool:
        """True if no option met the conformal threshold (anomaly/OOD)."""
        return len(self.prediction_set) == 0

    @property
    def is_ambiguous(self) -> bool:
        """True if multiple options remain in the prediction set."""
        return len(self.prediction_set) > 1

    @property
    def should_escalate(self) -> bool:
        """Escalate to System 2 if not singleton."""
        return not self.is_singleton

    @property
    def certified_selection(self) -> Optional[str]:
        """Returns the unique winning option if singleton, else None."""
        if self.is_singleton:
            return self.prediction_set[0]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_set": self.prediction_set,
            "is_singleton": self.is_singleton,
            "is_ambiguous": self.is_ambiguous,
            "is_empty": self.is_empty,
            "should_escalate": self.should_escalate,
            "certified_selection": self.certified_selection,
            "p_values": {k: round(v, 6) for k, v in self.p_values.items()},
            "alpha": self.alpha,
            "coverage_guarantee": self.coverage_guarantee,
        }


class ConformalPredictor:
    """
    Distribution-Free Conformal Predictor for System-1 Decisions.
    Calibrates on historical (probability, true_label) observations and produces
    mathematically certifiable prediction sets and p-values.
    """

    def __init__(self, config: Optional[ConformalConfig] = None):
        self.config = config or ConformalConfig()
        self._lock = threading.RLock()

        # Calibration data stores:
        # Noul calibration:
        # Mondrian: "true" -> List[float], "false" -> List[float]
        # Global: "global" -> List[float]
        self.noul_cal_scores: Dict[str, List[float]] = {
            "true": [],
            "false": [],
            "global": [],
        }

        # Choice calibration:
        # option -> List[float]
        self.choice_cal_scores: Dict[str, List[float]] = {
            "global": [],
        }
        self.choice_cumsum_scores: Dict[str, List[float]] = {
            "global": [],
        }

        # Fitted conformal quantile thresholds
        self.noul_thresholds: Dict[str, float] = {}
        self.choice_thresholds: Dict[str, float] = {}
        self.choice_cumsum_thresholds: Dict[str, float] = {}
        self._is_calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._is_calibrated

    # -------------------------------------------------------------------------
    # Calibration Ingestion
    # -------------------------------------------------------------------------

    def add_calibration_noul(self, prob_true: float, true_label: bool) -> None:
        """
        Adds a single calibration observation for boolean Noul decisions.
        Non-conformity score: s = 1 - P(true_label).
        """
        p = max(0.0, min(1.0, float(prob_true)))
        label = bool(true_label)
        score = (1.0 - p) if label else p

        with self._lock:
            if label:
                self.noul_cal_scores["true"].append(score)
            else:
                self.noul_cal_scores["false"].append(score)
            self.noul_cal_scores["global"].append(score)
            self._is_calibrated = False

    def add_calibration_choice(self, distribution: Dict[str, float], true_label: str) -> None:
        """
        Adds a single calibration observation for multi-class Choice decisions.
        Computes both raw non-conformity score (1 - P(y)) and APS cumulative probability score.
        """
        p = max(0.0, min(1.0, float(distribution.get(true_label, 0.0))))
        score = 1.0 - p

        # Compute APS cumulative probability score: sum of probabilities for options ranked >= true_label
        sorted_opts = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        cumsum = 0.0
        for opt, prob in sorted_opts:
            cumsum += prob
            if opt == true_label:
                break
        else:
            cumsum = 1.0

        with self._lock:
            if true_label not in self.choice_cal_scores:
                self.choice_cal_scores[true_label] = []
            self.choice_cal_scores[true_label].append(score)
            self.choice_cal_scores["global"].append(score)

            if true_label not in self.choice_cumsum_scores:
                self.choice_cumsum_scores[true_label] = []
            self.choice_cumsum_scores[true_label].append(cumsum)
            self.choice_cumsum_scores["global"].append(cumsum)

            self._is_calibrated = False

    def calibrate(self) -> None:
        """
        Fits conformal quantile thresholds on the accumulated calibration scores.
        Computes q_hat = Quantile(S, ceil((n + 1)(1 - alpha)) / n).
        """
        with self._lock:
            alpha = self.config.alpha
            self.noul_thresholds.clear()
            self.choice_thresholds.clear()
            self.choice_cumsum_thresholds.clear()

            # 1. Calibrate Noul thresholds
            if self.config.mondrian:
                for partition in ("true", "false"):
                    scores = self.noul_cal_scores[partition]
                    if len(scores) < self.config.min_calibration_samples:
                        # Fallback to global if class-conditional partition is too small
                        scores = self.noul_cal_scores["global"]
                    if len(scores) == 0:
                        raise ValueError(f"No calibration data available for Noul partition '{partition}'")
                    self.noul_thresholds[partition] = self._compute_conformal_quantile(scores, alpha)
            else:
                scores = self.noul_cal_scores["global"]
                if len(scores) == 0:
                    raise ValueError("No calibration data available for Noul")
                q = self._compute_conformal_quantile(scores, alpha)
                self.noul_thresholds["true"] = q
                self.noul_thresholds["false"] = q
                self.noul_thresholds["global"] = q

            # 2. Calibrate Choice thresholds
            if self.config.mondrian:
                for opt, scores in self.choice_cal_scores.items():
                    if opt == "global":
                        continue
                    if len(scores) < self.config.min_calibration_samples:
                        scores = self.choice_cal_scores["global"]
                    self.choice_thresholds[opt] = self._compute_conformal_quantile(scores, alpha)
            # Global choice threshold
            if len(self.choice_cal_scores["global"]) > 0:
                self.choice_thresholds["global"] = self._compute_conformal_quantile(
                    self.choice_cal_scores["global"], alpha
                )

            # 3. Calibrate Choice APS cumsum threshold
            if len(self.choice_cumsum_scores["global"]) > 0:
                self.choice_cumsum_thresholds["global"] = self._compute_conformal_quantile(
                    self.choice_cumsum_scores["global"], alpha
                )

            self._is_calibrated = True

    @staticmethod
    def _compute_conformal_quantile(scores: List[float], alpha: float) -> float:
        """
        Computes standard split conformal empirical quantile.
        q = sorted_scores[ceil((n + 1)(1 - alpha)) - 1].
        """
        n = len(scores)
        if n == 0:
            return 1.0

        p_level = math.ceil((n + 1) * (1.0 - alpha)) / float(n)
        if p_level >= 1.0:
            return 1.0

        sorted_s = sorted(scores)
        idx = min(n - 1, max(0, math.ceil(p_level * n) - 1))
        return sorted_s[idx]

    # -------------------------------------------------------------------------
    # Inference & Prediction Sets
    # -------------------------------------------------------------------------

    def predict_noul(self, noul: Noul, alpha: Optional[float] = None) -> ConformalNoulResult:
        """
        Constructs a distribution-free conformal prediction set for a Noul decision.
        Guarantees P(true_label in C(X)) >= 1 - alpha.
        If alpha is provided, computes quantiles dynamically on-the-fly (used for ACI).
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor is not calibrated. Call calibrate() first.")

        p = noul.probability if noul.probability is not None else 0.5
        p = max(0.0, min(1.0, float(p)))

        # Candidate non-conformity scores
        score_true = 1.0 - p
        score_false = p

        effective_alpha = max(0.0001, min(0.9999, float(alpha))) if alpha is not None else self.config.alpha
        effective_coverage = 1.0 - effective_alpha

        with self._lock:
            scores_for_true = (
                self.noul_cal_scores["true"]
                if (self.config.mondrian and len(self.noul_cal_scores["true"]) >= self.config.min_calibration_samples)
                else self.noul_cal_scores["global"]
            )
            scores_for_false = (
                self.noul_cal_scores["false"]
                if (self.config.mondrian and len(self.noul_cal_scores["false"]) >= self.config.min_calibration_samples)
                else self.noul_cal_scores["global"]
            )

            if alpha is not None:
                q_true = self._compute_conformal_quantile(scores_for_true, effective_alpha)
                q_false = self._compute_conformal_quantile(scores_for_false, effective_alpha)
            else:
                q_true = self.noul_thresholds.get("true", 1.0)
                q_false = self.noul_thresholds.get("false", 1.0)

        # Compute p-values: (1 + sum(I(s_cal >= s_test))) / (n + 1)
        p_val_true = self._compute_p_value(scores_for_true, score_true)
        p_val_false = self._compute_p_value(scores_for_false, score_false)

        prediction_set: Set[bool] = set()
        if score_true <= q_true:
            prediction_set.add(True)
        if score_false <= q_false:
            prediction_set.add(False)

        return ConformalNoulResult(
            prediction_set=prediction_set,
            p_value_true=p_val_true,
            p_value_false=p_val_false,
            alpha=effective_alpha,
            coverage_guarantee=effective_coverage,
            instructions=noul.instructions,
            raw_probability=noul.probability,
        )

    def predict_choice(self, choice: Choice, alpha: Optional[float] = None) -> ConformalChoiceResult:
        """
        Constructs a distribution-free conformal prediction set for a Choice decision
        using Adaptive Prediction Sets (APS).
        Guarantees P(true_option in C(X)) >= 1 - alpha.
        If alpha is provided, computes quantiles dynamically on-the-fly (used for ACI).
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor is not calibrated. Call calibrate() first.")

        effective_alpha = max(0.0001, min(0.9999, float(alpha))) if alpha is not None else self.config.alpha
        effective_coverage = 1.0 - effective_alpha

        options = choice.options or list(choice.distribution.keys())
        if not options:
            return ConformalChoiceResult(
                prediction_set=[],
                p_values={},
                alpha=effective_alpha,
                coverage_guarantee=effective_coverage,
                instructions=choice.instructions,
                distribution=choice.distribution,
            )

        p_values: Dict[str, float] = {}

        with self._lock:
            # 1. Exact conformal p-values for all candidate hypotheses
            for opt in options:
                prob = choice.get_prob(opt)
                score = 1.0 - prob

                # Select calibration partition
                if self.config.mondrian and opt in self.choice_cal_scores:
                    cal_scores = self.choice_cal_scores.get(opt, self.choice_cal_scores["global"])
                else:
                    cal_scores = self.choice_cal_scores.get("global", [])

                p_val = self._compute_p_value(cal_scores, score)
                p_values[opt] = p_val

            # 2. Adaptive Prediction Sets (APS)
            # Sort candidate options descending by probability
            sorted_opts = sorted(options, key=lambda opt: choice.get_prob(opt), reverse=True)
            if alpha is not None:
                q_cumsum = (
                    self._compute_conformal_quantile(self.choice_cumsum_scores.get("global", []), effective_alpha)
                    if len(self.choice_cumsum_scores.get("global", [])) > 0
                    else (1.0 - effective_alpha)
                )
            else:
                q_cumsum = self.choice_cumsum_thresholds.get("global", 1.0 - self.config.alpha)

            prediction_set: List[str] = []
            cumsum = 0.0
            for opt in sorted_opts:
                prediction_set.append(opt)
                cumsum += choice.get_prob(opt)
                if cumsum >= q_cumsum:
                    break

        return ConformalChoiceResult(
            prediction_set=prediction_set,
            p_values=p_values,
            alpha=effective_alpha,
            coverage_guarantee=effective_coverage,
            instructions=choice.instructions,
            distribution=choice.distribution,
        )

    def evaluate_coverage_choice(
        self, test_samples: Sequence[Tuple[Dict[str, float], str]]
    ) -> Dict[str, float]:
        """
        Evaluates empirical coverage, mean prediction set size, and singleton ratio for Choice decisions.
        test_samples: list of (predicted_distribution, true_label_str).
        """
        if not test_samples:
            return {"coverage": 0.0, "mean_set_size": 0.0, "singleton_ratio": 0.0}

        covered = 0
        total_set_size = 0
        singletons = 0
        ambiguous = 0

        for dist, true_label in test_samples:
            choice = Choice("test", distribution=dist)
            res = self.predict_choice(choice)
            if true_label in res.prediction_set:
                covered += 1
            sz = len(res.prediction_set)
            total_set_size += sz
            if sz == 1:
                singletons += 1
            elif sz > 1:
                ambiguous += 1

        n = len(test_samples)
        return {
            "empirical_coverage": round(covered / float(n), 4),
            "nominal_coverage": round(self.config.coverage_guarantee, 4),
            "mean_set_size": round(total_set_size / float(n), 3),
            "singleton_ratio": round(singletons / float(n), 4),
            "ambiguous_ratio": round(ambiguous / float(n), 4),
            "sample_count": n,
        }

    @staticmethod
    def _compute_p_value(calibration_scores: List[float], test_score: float) -> float:
        """Computes exact conformal p-value."""
        n = len(calibration_scores)
        if n == 0:
            return 1.0
        exceeds = sum(1 for s in calibration_scores if s >= test_score)
        return (1.0 + float(exceeds)) / float(n + 1)

    # -------------------------------------------------------------------------
    # Evaluation & Coverage Diagnostics
    # -------------------------------------------------------------------------

    def evaluate_coverage_noul(self, test_samples: Sequence[Tuple[float, bool]]) -> Dict[str, float]:
        """
        Evaluates empirical coverage, mean prediction set size, and singleton ratio.
        test_samples: list of (predicted_prob_true, true_label_bool).
        """
        if not test_samples:
            return {"coverage": 0.0, "mean_set_size": 0.0, "singleton_ratio": 0.0}

        covered = 0
        total_set_size = 0
        singletons = 0
        ambiguous = 0
        empty = 0

        for prob, true_label in test_samples:
            noul = Noul("test", probability=prob)
            res = self.predict_noul(noul)
            if true_label in res.prediction_set:
                covered += 1
            sz = len(res.prediction_set)
            total_set_size += sz
            if sz == 1:
                singletons += 1
            elif sz > 1:
                ambiguous += 1
            else:
                empty += 1

        n = len(test_samples)
        return {
            "empirical_coverage": round(covered / float(n), 4),
            "nominal_coverage": round(self.config.coverage_guarantee, 4),
            "mean_set_size": round(total_set_size / float(n), 3),
            "singleton_ratio": round(singletons / float(n), 4),
            "ambiguous_ratio": round(ambiguous / float(n), 4),
            "empty_ratio": round(empty / float(n), 4),
            "sample_count": n,
        }

    # -------------------------------------------------------------------------
    # Binary Serialization (.reflex-conformal) with CRC32 Verification
    # -------------------------------------------------------------------------

    def save_to_bytes(self) -> bytes:
        """Serializes calibration distributions and thresholds with CRC32 trailer."""
        with self._lock:
            buf = bytearray()
            buf.extend(CONFORMAL_MAGIC)

            mondrian_flag = 1 if self.config.mondrian else 0

            # 40-byte header with 64-bit double precision for alpha & coverage
            header = struct.pack(
                "<IddIII8s",
                CONFORMAL_FORMAT_VERSION,
                self.config.alpha,
                self.config.coverage_guarantee,
                mondrian_flag,
                self.config.min_calibration_samples,
                1 if self._is_calibrated else 0,
                b"\x00" * 8,
            )
            buf.extend(header)

            # JSON payload encoding calibration distributions and thresholds
            payload_dict = {
                "noul_cal_scores": self.noul_cal_scores,
                "choice_cal_scores": self.choice_cal_scores,
                "choice_cumsum_scores": self.choice_cumsum_scores,
                "noul_thresholds": self.noul_thresholds,
                "choice_thresholds": self.choice_thresholds,
                "choice_cumsum_thresholds": self.choice_cumsum_thresholds,
            }
            json_bytes = json.dumps(payload_dict).encode("utf-8")
            buf.extend(struct.pack("<I", len(json_bytes)))
            buf.extend(json_bytes)

            # 32-bit CRC32 Trailer
            checksum = zlib.crc32(buf)
            buf.extend(struct.pack("<I", checksum))
            return bytes(buf)

    @classmethod
    def load_from_bytes(cls, data: bytes) -> ConformalPredictor:
        """Deserializes ConformalPredictor from binary bytes with CRC32 verification."""
        if len(data) < 52:
            raise ValueError(f"Data buffer too small for Conformal header ({len(data)} bytes)")

        # Verify CRC32
        payload_data = data[:-4]
        expected_crc = struct.unpack("<I", data[-4:])[0]
        actual_crc = zlib.crc32(payload_data)
        if actual_crc != expected_crc:
            raise ValueError(
                f"CRC32 mismatch in conformal model: expected {hex(expected_crc)}, got {hex(actual_crc)}"
            )

        magic = payload_data[:4]
        if magic != CONFORMAL_MAGIC:
            raise ValueError(f"Invalid conformal magic bytes: {magic}, expected {CONFORMAL_MAGIC}")

        offset = 4
        (
            version,
            alpha,
            _,
            mondrian_flag,
            min_samples,
            is_calibrated_flag,
            _,
        ) = struct.unpack("<IddIII8s", payload_data[offset : offset + 40])
        offset += 40

        if version != CONFORMAL_FORMAT_VERSION:
            raise ValueError(f"Unsupported conformal version {version}, expected {CONFORMAL_FORMAT_VERSION}")

        cfg = ConformalConfig(
            alpha=alpha,
            mondrian=(mondrian_flag == 1),
            min_calibration_samples=min_samples,
        )
        predictor = cls(cfg)

        json_len = struct.unpack("<I", payload_data[offset : offset + 4])[0]
        offset += 4
        json_bytes = payload_data[offset : offset + json_len]
        payload = json.loads(json_bytes.decode("utf-8"))

        predictor.noul_cal_scores = payload.get("noul_cal_scores", {"true": [], "false": [], "global": []})
        predictor.choice_cal_scores = payload.get("choice_cal_scores", {"global": []})
        predictor.choice_cumsum_scores = payload.get("choice_cumsum_scores", {"global": []})
        predictor.noul_thresholds = payload.get("noul_thresholds", {})
        predictor.choice_thresholds = payload.get("choice_thresholds", {})
        predictor.choice_cumsum_thresholds = payload.get("choice_cumsum_thresholds", {})
        predictor._is_calibrated = (is_calibrated_flag == 1)

        return predictor

    def save(self, filepath: str) -> None:
        """Saves conformal model to file."""
        data = self.save_to_bytes()
        tmp_file = f"{filepath}.tmp.{os.getpid()}"
        with open(tmp_file, "wb") as f:
            f.write(data)
        os.replace(tmp_file, filepath)

    @classmethod
    def load(cls, filepath: str) -> ConformalPredictor:
        """Loads conformal model from file."""
        with open(filepath, "rb") as f:
            data = f.read()
        return cls.load_from_bytes(data)
