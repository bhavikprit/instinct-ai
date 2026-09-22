"""
Reflex Mixture-of-Reflexes (MoR) & Hierarchical Instinct Ensembles (Phase 26).
Enables dynamic routing across specialized .reflex models with uncertainty-weighted
Dirichlet voting, epistemic Shannon entropy attenuation, and 3-tier cascading.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import os
import struct
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import zlib

from sys1.primitives import Choice, Noul, Score, DecisionResult
from sys1.embeddings import SemanticVectorEncoder
from sys1.compiler import CompiledInstinct

MAGIC_ENSEMBLE_HEADER = b"RFXE"


def _shannon_entropy(distribution: Dict[str, float]) -> float:
    """Computes normalized Shannon entropy H(P) in [0.0, 1.0] for a discrete distribution."""
    probs = [max(1e-12, float(p)) for p in distribution.values() if p > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs)
    max_h = math.log2(len(distribution))
    if max_h <= 0.0:
        return 0.0
    return min(1.0, max(0.0, h / max_h))


def _cosine_sim(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 * n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


@dataclass
class SpecialistModel:
    """Represents a domain specialist model within a Mixture-of-Reflexes ensemble."""
    name: str
    domain: str
    model: Union[CompiledInstinct, Callable[[str], DecisionResult], Any]
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    weight: float = 1.0
    centroid: Optional[List[float]] = None

    def predict(self, state: str) -> DecisionResult:
        if isinstance(self.model, CompiledInstinct):
            return self.model.predict(state)
        elif hasattr(self.model, "evaluate_raw"):
            return self.model.evaluate_raw(state)
        elif callable(self.model):
            res = self.model(state)
            if isinstance(res, DecisionResult):
                return res
            return DecisionResult(decisions={"choice": Choice(instructions="", options=[], selected=str(res))})
        elif hasattr(self.model, "predict"):
            return self.model.predict(state)
        raise TypeError(f"Specialist '{self.name}' model type {type(self.model)} is not callable")

    def to_dict(self) -> Dict[str, Any]:
        compiled_payload = None
        if isinstance(self.model, CompiledInstinct):
            compiled_payload = {
                "name": self.model.name,
                "decision_type": self.model.decision_type,
                "options": self.model.options,
                "weights": self.model.weights,
                "biases": self.model.biases,
                "temperature": self.model.temperature,
                "metrics": self.model.metrics.to_dict() if self.model.metrics else None,
            }
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "keywords": self.keywords,
            "weight": self.weight,
            "centroid": self.centroid,
            "compiled_payload": compiled_payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecialistModel":
        model = None
        if data.get("compiled_payload"):
            cp = data["compiled_payload"]
            from sys1.compiler import CalibrationMetrics
            metrics = None
            if cp.get("metrics"):
                m = cp["metrics"]
                metrics = CalibrationMetrics(
                    accuracy=m.get("accuracy", 0.0),
                    brier_score=m.get("brier_score", 0.0),
                    ece=m.get("ece", 0.0),
                    total_samples=m.get("total_samples", 0),
                    training_time_ms=m.get("training_time_ms", 0.0),
                )
            model = CompiledInstinct(
                name=cp["name"],
                decision_type=cp["decision_type"],
                options=cp["options"],
                weights=cp["weights"],
                biases=cp["biases"],
                temperature=cp.get("temperature", 1.0),
                metrics=metrics,
            )
        return cls(
            name=data["name"],
            domain=data["domain"],
            model=model,
            description=data.get("description", ""),
            keywords=data.get("keywords", []),
            weight=data.get("weight", 1.0),
            centroid=data.get("centroid"),
        )


class MoRGatingNetwork:
    """
    Sub-50us System-1 Gating Network that routes incoming requests
    to candidate domain specialists using dense semantic centroids.
    """

    def __init__(self, encoder: Optional[SemanticVectorEncoder] = None):
        self.encoder = encoder or SemanticVectorEncoder()
        self.specialist_centroids: Dict[str, List[float]] = {}
        self.specialist_weights: Dict[str, float] = {}

    def register(self, specialist: SpecialistModel) -> None:
        """Registers a specialist and computes or assigns its semantic centroid."""
        if specialist.centroid is not None:
            self.specialist_centroids[specialist.name] = specialist.centroid
        else:
            # Build representative text for domain specialization
            text_corpus = f"{specialist.domain} {specialist.description} {' '.join(specialist.keywords)}"
            centroid = self.encoder.encode(text_corpus.strip())
            specialist.centroid = centroid
            self.specialist_centroids[specialist.name] = centroid
        self.specialist_weights[specialist.name] = max(0.01, float(specialist.weight))

    def compute_gates(self, state: str, temperature: float = 1.0) -> Dict[str, float]:
        """
        Computes normalized routing gating coefficients g_k across all registered specialists.
        """
        if not self.specialist_centroids:
            return {}

        state_vec = self.encoder.encode(state)
        temp = max(0.05, temperature)

        # 1. Cosine similarity to domain centroids weighted by specialist prior
        raw_scores = {}
        for name, centroid in self.specialist_centroids.items():
            sim = _cosine_sim(state_vec, centroid)
            prior = self.specialist_weights.get(name, 1.0)
            # Logit incorporates semantic alignment and prior credibility
            raw_scores[name] = (sim * 4.0 * prior) / temp

        # 2. Numerically stable softmax
        max_s = max(raw_scores.values()) if raw_scores else 0.0
        exps = {name: math.exp(min(40.0, max(-40.0, s - max_s))) for name, s in raw_scores.items()}
        sum_exp = sum(exps.values())
        if sum_exp <= 0.0:
            uniform = 1.0 / len(raw_scores)
            return {name: uniform for name in raw_scores}

        return {name: round(exp_val / sum_exp, 4) for name, exp_val in exps.items()}

    def top_k(self, gates: Dict[str, float], k: int = 2) -> Dict[str, float]:
        """Applies Top-K sparse routing filtering and renormalizes remaining gates."""
        if not gates or k >= len(gates):
            return gates

        sorted_items = sorted(gates.items(), key=lambda x: x[1], reverse=True)[:k]
        total = sum(score for _, score in sorted_items)
        if total <= 0:
            return {name: 1.0 / len(sorted_items) for name, _ in sorted_items}
        return {name: round(score / total, 4) for name, score in sorted_items}


@dataclass
class EnsembleResult:
    """Comprehensive typed result from an Instinct Ensemble or Mixture-of-Reflexes evaluation."""
    selected: str
    confidence: float
    entropy: float
    tier: str  # "L1_FAST_PATH", "L2_ENSEMBLE_CONSENSUS", "L3_SYSTEM2_ESCALATION"
    routed_to_system2: bool
    specialist_predictions: Dict[str, Dict[str, Any]]
    gating_weights: Dict[str, float]
    voting_weights: Dict[str, float]
    blended_distribution: Dict[str, float]
    latency_ms: float
    decision_type: str = "choice"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected,
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "tier": self.tier,
            "routed_to_system2": self.routed_to_system2,
            "specialist_predictions": self.specialist_predictions,
            "gating_weights": self.gating_weights,
            "voting_weights": self.voting_weights,
            "blended_distribution": self.blended_distribution,
            "latency_ms": round(self.latency_ms, 3),
            "decision_type": self.decision_type,
        }


class InstinctEnsemble:
    """
    Mixture-of-Reflexes Coordinator combining multiple domain specialists
    with uncertainty-weighted Dirichlet voting and hierarchical cascading.
    """

    def __init__(
        self,
        name: str = "default_ensemble",
        specialists: Optional[List[SpecialistModel]] = None,
        top_k: int = 2,
        temperature: float = 1.0,
        entropy_attenuation: float = 1.5,
    ):
        self.name = name
        self.top_k = max(1, top_k)
        self.temperature = max(0.01, temperature)
        self.entropy_attenuation = entropy_attenuation  # tau parameter: w_k \propto g_k * exp(-tau * H_k)
        self.gating = MoRGatingNetwork()
        self.specialists: Dict[str, SpecialistModel] = {}
        self.stats = {
            "total_queries": 0,
            "l1_fast_path": 0,
            "l2_consensus": 0,
            "l3_escalation": 0,
        }

        if specialists:
            for s in specialists:
                self.add_specialist(s)

    def add_specialist(self, specialist: SpecialistModel) -> None:
        """Adds a domain specialist to the ensemble and registers with the gating network."""
        self.specialists[specialist.name] = specialist
        self.gating.register(specialist)

    def predict(
        self,
        state: str,
        top_k: Optional[int] = None,
    ) -> EnsembleResult:
        """
        Standard MoR evaluation:
        1. Gating network computes routing weights across specialists.
        2. Top-K specialists execute sub-50us hot-path predictions.
        3. Predictions are blended using Dirichlet uncertainty attenuation.
        """
        t0 = time.perf_counter()
        self.stats["total_queries"] += 1

        if not self.specialists:
            raise ValueError("InstinctEnsemble has no registered specialists")

        effective_k = top_k if top_k is not None else self.top_k
        raw_gates = self.gating.compute_gates(state, temperature=self.temperature)
        active_gates = self.gating.top_k(raw_gates, k=effective_k)

        specialist_preds = {}
        specialist_entropies = {}
        specialist_dists = {}

        # 1. Execute inference on active specialists
        for name, gate_val in active_gates.items():
            spec = self.specialists[name]
            pred = spec.predict(state)
            
            # Extract choice or primary decision
            if "choice" in pred.decisions:
                c = pred.decisions["choice"]
                sel = c.selected
                dist = c.distribution or {sel: 1.0}
            elif "noul" in pred.decisions:
                n = pred.decisions["noul"]
                p = n.probability
                sel = "true" if p >= 0.5 else "false"
                dist = {"true": p, "false": round(1.0 - p, 4)}
            elif "score" in pred.decisions:
                s = pred.decisions["score"]
                sel = f"{s.score:.1f}"
                dist = {sel: s.confidence}
            else:
                sel = "unknown"
                dist = {"unknown": 1.0}

            h = _shannon_entropy(dist)
            top_prob = dist.get(sel, 0.0)
            specialist_preds[name] = {
                "selected": sel,
                "confidence": top_prob,
                "entropy": h,
                "domain": spec.domain,
            }
            specialist_entropies[name] = h
            specialist_dists[name] = dist

        # 2. Dirichlet Uncertainty-Weighted Voting
        # Specialists with low Shannon entropy (high local domain confidence) get higher weight
        raw_weights = {}
        for name, gate_val in active_gates.items():
            h = specialist_entropies.get(name, 0.5)
            # w_k = g_k * exp(-tau * H_k) * specialist_weight
            spec_prior = self.specialists[name].weight
            unc_factor = math.exp(-self.entropy_attenuation * h)
            raw_weights[name] = gate_val * unc_factor * spec_prior

        total_w = sum(raw_weights.values())
        if total_w <= 0:
            voting_weights = {name: 1.0 / len(raw_weights) for name in raw_weights}
        else:
            voting_weights = {name: round(w / total_w, 4) for name, w in raw_weights.items()}

        # 3. Blend distributions across all options
        blended_dist: Dict[str, float] = {}
        for name, dist in specialist_dists.items():
            v_weight = voting_weights.get(name, 0.0)
            for opt, p in dist.items():
                blended_dist[opt] = blended_dist.get(opt, 0.0) + (p * v_weight)

        # Normalize blended distribution
        sum_blended = sum(blended_dist.values())
        if sum_blended > 0:
            blended_dist = {opt: round(p / sum_blended, 4) for opt, p in blended_dist.items()}

        best_opt = max(blended_dist.keys(), key=lambda k: blended_dist[k]) if blended_dist else "unknown"
        best_conf = blended_dist.get(best_opt, 0.0)
        ensemble_entropy = _shannon_entropy(blended_dist)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return EnsembleResult(
            selected=best_opt,
            confidence=best_conf,
            entropy=ensemble_entropy,
            tier="L2_ENSEMBLE_CONSENSUS",
            routed_to_system2=False,
            specialist_predictions=specialist_preds,
            gating_weights=active_gates,
            voting_weights=voting_weights,
            blended_distribution=blended_dist,
            latency_ms=latency_ms,
        )

    def cascade_predict(
        self,
        state: str,
        confidence_threshold: float = 0.85,
        entropy_threshold: float = 0.40,
        consensus_threshold: float = 0.65,
        max_consensus_entropy: float = 0.70,
    ) -> EnsembleResult:
        """
        3-Tier Hierarchical Cascade Routing:
        - Tier 1 (L1 Fast-Path): If the top specialist is decisive (high confidence, low entropy),
          return immediately in <30us without invoking other models.
        - Tier 2 (L2 Ensemble Consensus): If L1 is ambiguous, evaluate Top-K specialists and blend
          votes. If blended consensus is decisive, return in <80us.
        - Tier 3 (L3 Escalation): If epistemic uncertainty remains high, flag routed_to_system2 = True
          for cloud LLM or human escalation.
        """
        t0 = time.perf_counter()
        self.stats["total_queries"] += 1

        if not self.specialists:
            raise ValueError("InstinctEnsemble has no registered specialists")

        # 1. Compute Gating Network
        raw_gates = self.gating.compute_gates(state, temperature=self.temperature)
        sorted_specialists = sorted(raw_gates.items(), key=lambda x: x[1], reverse=True)
        top_name, top_gate = sorted_specialists[0]

        # 2. Tier 1: L1 Fast-Path Check on Leading Specialist
        lead_spec = self.specialists[top_name]
        lead_pred = lead_spec.predict(state)

        if "choice" in lead_pred.decisions:
            lead_choice = lead_pred.decisions["choice"]
            sel = lead_choice.selected
            dist = lead_choice.distribution or {sel: 1.0}
        elif "noul" in lead_pred.decisions:
            p = lead_pred.decisions["noul"].probability
            sel = "true" if p >= 0.5 else "false"
            dist = {"true": p, "false": round(1.0 - p, 4)}
        else:
            sel = "unknown"
            dist = {"unknown": 1.0}

        lead_conf = dist.get(sel, 0.0)
        lead_entropy = _shannon_entropy(dist)

        # Check L1 decisive condition:
        # Specialist must have high gating affinity, high local confidence, and low entropy
        if top_gate >= 0.45 and lead_conf >= confidence_threshold and lead_entropy <= entropy_threshold:
            self.stats["l1_fast_path"] += 1
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return EnsembleResult(
                selected=sel,
                confidence=lead_conf,
                entropy=lead_entropy,
                tier="L1_FAST_PATH",
                routed_to_system2=False,
                specialist_predictions={
                    top_name: {
                        "selected": sel,
                        "confidence": lead_conf,
                        "entropy": lead_entropy,
                        "domain": lead_spec.domain,
                    }
                },
                gating_weights={top_name: top_gate},
                voting_weights={top_name: 1.0},
                blended_distribution=dist,
                latency_ms=latency_ms,
            )

        # 3. Tier 2: L2 Ensemble Consensus across Top-K
        # Gather remaining top-k specialists
        effective_k = min(len(sorted_specialists), self.top_k)
        active_gates = self.gating.top_k(raw_gates, k=effective_k)

        specialist_preds = {
            top_name: {
                "selected": sel,
                "confidence": lead_conf,
                "entropy": lead_entropy,
                "domain": lead_spec.domain,
            }
        }
        specialist_entropies = {top_name: lead_entropy}
        specialist_dists = {top_name: dist}

        for name, gate_val in active_gates.items():
            if name == top_name:
                continue
            spec = self.specialists[name]
            pred = spec.predict(state)
            if "choice" in pred.decisions:
                c = pred.decisions["choice"]
                s_sel = c.selected
                s_dist = c.distribution or {s_sel: 1.0}
            elif "noul" in pred.decisions:
                p = pred.decisions["noul"].probability
                s_sel = "true" if p >= 0.5 else "false"
                s_dist = {"true": p, "false": round(1.0 - p, 4)}
            else:
                s_sel = "unknown"
                s_dist = {"unknown": 1.0}

            h = _shannon_entropy(s_dist)
            specialist_preds[name] = {
                "selected": s_sel,
                "confidence": s_dist.get(s_sel, 0.0),
                "entropy": h,
                "domain": spec.domain,
            }
            specialist_entropies[name] = h
            specialist_dists[name] = s_dist

        # Compute uncertainty weights
        raw_weights = {}
        for name, gate_val in active_gates.items():
            h = specialist_entropies.get(name, 0.5)
            spec_prior = self.specialists[name].weight
            unc_factor = math.exp(-self.entropy_attenuation * h)
            raw_weights[name] = gate_val * unc_factor * spec_prior

        tot_w = sum(raw_weights.values())
        voting_weights = {name: round(w / tot_w, 4) for name, w in raw_weights.items()} if tot_w > 0 else {}

        # Blend distributions
        blended_dist = {}
        for name, d in specialist_dists.items():
            v_w = voting_weights.get(name, 0.0)
            for opt, p in d.items():
                blended_dist[opt] = blended_dist.get(opt, 0.0) + (p * v_w)

        sum_b = sum(blended_dist.values())
        if sum_b > 0:
            blended_dist = {opt: round(p / sum_b, 4) for opt, p in blended_dist.items()}

        ens_selected = max(blended_dist.keys(), key=lambda k: blended_dist[k]) if blended_dist else "unknown"
        ens_confidence = blended_dist.get(ens_selected, 0.0)
        ens_entropy = _shannon_entropy(blended_dist)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Check if Tier 2 consensus is satisfied
        if ens_confidence >= consensus_threshold and ens_entropy <= max_consensus_entropy:
            self.stats["l2_consensus"] += 1
            return EnsembleResult(
                selected=ens_selected,
                confidence=ens_confidence,
                entropy=ens_entropy,
                tier="L2_ENSEMBLE_CONSENSUS",
                routed_to_system2=False,
                specialist_predictions=specialist_preds,
                gating_weights=active_gates,
                voting_weights=voting_weights,
                blended_distribution=blended_dist,
                latency_ms=latency_ms,
            )

        # 4. Tier 3: High Epistemic Uncertainty -> System-2 Escalation
        self.stats["l3_escalation"] += 1
        return EnsembleResult(
            selected=ens_selected,
            confidence=ens_confidence,
            entropy=ens_entropy,
            tier="L3_SYSTEM2_ESCALATION",
            routed_to_system2=True,
            specialist_predictions=specialist_preds,
            gating_weights=active_gates,
            voting_weights=voting_weights,
            blended_distribution=blended_dist,
            latency_ms=latency_ms,
        )

    def save(self, path: str) -> None:
        """
        Serializes the ensemble and all specialists into a portable .reflex-ensemble artifact.
        Format: [RFXE 4B][CRC32 4B][LENGTH 4B][JSON PAYLOAD UTF-8]
        """
        specialists_data = [s.to_dict() for s in self.specialists.values()]
        payload = {
            "name": self.name,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "entropy_attenuation": self.entropy_attenuation,
            "specialists": specialists_data,
            "created_at": time.time(),
            "version": "0.2.0",
        }
        json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        crc = zlib.crc32(json_bytes) & 0xFFFFFFFF
        length = len(json_bytes)

        header = struct.pack(">4sII", MAGIC_ENSEMBLE_HEADER, crc, length)
        with open(path, "wb") as f:
            f.write(header)
            f.write(json_bytes)

    @classmethod
    def load(cls, path: str) -> "InstinctEnsemble":
        """
        Loads and verifies a .reflex-ensemble artifact from disk.
        """
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise ValueError("Corrupt .reflex-ensemble file: header too short")
            magic, expected_crc, length = struct.unpack(">4sII", header)
            if magic != MAGIC_ENSEMBLE_HEADER:
                raise ValueError(f"Invalid magic header: expected {MAGIC_ENSEMBLE_HEADER}, got {magic}")

            payload_bytes = f.read(length)
            if len(payload_bytes) != length:
                raise ValueError("Incomplete .reflex-ensemble file: truncated payload")

            actual_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ValueError(f"CRC32 checksum mismatch: expected {expected_crc}, got {actual_crc}")

        data = json.loads(payload_bytes.decode("utf-8"))
        specialists = [SpecialistModel.from_dict(s) for s in data.get("specialists", [])]
        ensemble = cls(
            name=data.get("name", "loaded_ensemble"),
            specialists=specialists,
            top_k=data.get("top_k", 2),
            temperature=data.get("temperature", 1.0),
            entropy_attenuation=data.get("entropy_attenuation", 1.5),
        )
        return ensemble


class HierarchicalCascade:
    """Convenience class for managing multi-tier cascading routing."""

    def __init__(
        self,
        ensemble: InstinctEnsemble,
        confidence_threshold: float = 0.85,
        entropy_threshold: float = 0.40,
        consensus_threshold: float = 0.65,
        max_consensus_entropy: float = 0.70,
    ):
        self.ensemble = ensemble
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.consensus_threshold = consensus_threshold
        self.max_consensus_entropy = max_consensus_entropy

    def route(self, state: str) -> EnsembleResult:
        return self.ensemble.cascade_predict(
            state,
            confidence_threshold=self.confidence_threshold,
            entropy_threshold=self.entropy_threshold,
            consensus_threshold=self.consensus_threshold,
            max_consensus_entropy=self.max_consensus_entropy,
        )
