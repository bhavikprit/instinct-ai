"""
Reflex Distill: Continuous Autonomous Distillation & Self-Synthesizing Model Factory (Phase 29).
Captures live System-2 LLM escalations from Dual-Brain gateways, mines latent semantic clusters,
synthesizes contrastive calibration datasets, and autonomously compiles updated .reflex instinct models.
Couples with reflex.shadow for safe canary validation and auto-promotion.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sys1.embeddings import SemanticVectorEncoder, cosine_similarity
from sys1.compiler import InstinctCompiler, PromptSpec, CompiledInstinct, CalibrationMetrics
from sys1.guardrails import PIIGuardrail


@dataclass
class DistillationTrace:
    """A single harvested decision trace from System-2 or high-uncertainty gateway traffic."""
    trace_id: str
    prompt: str
    response: str
    model: str
    latency_ms: float
    timestamp: float
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistillationTrace":
        return cls(
            trace_id=data.get("trace_id", ""),
            prompt=data.get("prompt", ""),
            response=data.get("response", ""),
            model=data.get("model", ""),
            latency_ms=float(data.get("latency_ms", 0.0)),
            timestamp=float(data.get("timestamp", time.time())),
            label=data.get("label"),
            metadata=data.get("metadata", {}),
        )


class DistillationBuffer:
    """
    Thread-safe bounded ring buffer capturing System-2 query/response trajectories.
    Automatically redacts PII before persisting or buffering.
    """

    def __init__(
        self,
        max_size: int = 2000,
        storage_path: Optional[str] = None,
        redact_pii: bool = True,
    ):
        self.max_size = max(1, max_size)
        self.storage_path = storage_path
        self.redact_pii = redact_pii
        self._pii_guard = PIIGuardrail() if redact_pii else None

        self._lock = threading.Lock()
        self._traces: deque[DistillationTrace] = deque(maxlen=self.max_size)
        self._total_recorded: int = 0
        self._pii_redacted_count: int = 0

        if self.storage_path and os.path.exists(self.storage_path):
            self.load_jsonl(self.storage_path)

    def record(
        self,
        prompt: str,
        response: str,
        model: str = "upstream",
        latency_ms: float = 0.0,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DistillationTrace:
        """Records a new trace into the buffer, with automatic PII sanitization."""
        clean_prompt = prompt
        clean_response = response

        if self.redact_pii:
            clean_p = self._sanitize_text(prompt)
            if clean_p != prompt:
                clean_prompt = clean_p
                with self._lock:
                    self._pii_redacted_count += 1
            elif self._pii_guard and self._pii_guard.check(prompt).blocked:
                clean_prompt = self._sanitize_text(prompt)
                with self._lock:
                    self._pii_redacted_count += 1

            clean_r = self._sanitize_text(response)
            if clean_r != response:
                clean_response = clean_r
                with self._lock:
                    self._pii_redacted_count += 1
            elif self._pii_guard and self._pii_guard.check(response).blocked:
                clean_response = self._sanitize_text(response)
                with self._lock:
                    self._pii_redacted_count += 1

        trace = DistillationTrace(
            trace_id=hashlib.sha256(f"{time.time()}_{random.random()}_{clean_prompt[:32]}".encode("utf-8")).hexdigest()[:16],
            prompt=clean_prompt.strip(),
            response=clean_response.strip(),
            model=model,
            latency_ms=latency_ms,
            timestamp=time.time(),
            label=label,
            metadata=metadata or {},
        )

        with self._lock:
            self._traces.append(trace)
            self._total_recorded += 1

            # Append to file if persistence enabled
            if self.storage_path:
                try:
                    with open(self.storage_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trace.to_dict()) + "\n")
                except Exception:
                    pass

        return trace

    def _sanitize_text(self, text: str) -> str:
        """Lightweight regex redaction for common sensitive tokens."""
        # Redact credit card patterns (13 to 19 digits)
        text = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "[REDACTED_CARD]", text)
        # Redact emails
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", text)
        # Redact api keys / secret tokens
        text = re.sub(r"\bsk-[a-zA-Z0-9_\-]{16,}\b", "[REDACTED_SECRET]", text)
        text = re.sub(r"(?i)(api[_-]?key|secret|token|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", r"\1: [REDACTED_SECRET]", text)
        return text

    def get_traces(
        self,
        limit: Optional[int] = None,
        since: Optional[float] = None,
    ) -> List[DistillationTrace]:
        """Returns buffered traces filtered by timestamp and limit."""
        with self._lock:
            results = list(self._traces)

        if since is not None:
            results = [t for t in results if t.timestamp >= since]

        if limit is not None and limit > 0:
            results = results[-limit:]

        return results

    def clear(self):
        """Empties the in-memory buffer."""
        with self._lock:
            self._traces.clear()

    def size(self) -> int:
        """Returns the current number of in-memory traces."""
        with self._lock:
            return len(self._traces)

    def stats(self) -> Dict[str, Any]:
        """Returns buffer telemetry statistics."""
        with self._lock:
            models = list(set(t.model for t in self._traces))
            oldest = self._traces[0].timestamp if self._traces else 0.0
            newest = self._traces[-1].timestamp if self._traces else 0.0
            return {
                "current_size": len(self._traces),
                "max_size": self.max_size,
                "total_recorded": self._total_recorded,
                "pii_redacted_count": self._pii_redacted_count,
                "unique_models": models,
                "oldest_timestamp": oldest,
                "newest_timestamp": newest,
                "storage_path": self.storage_path,
            }

    def load_jsonl(self, path: str):
        """Loads traces from a JSONL file into the buffer."""
        if not os.path.exists(path):
            return
        loaded = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        loaded.append(DistillationTrace.from_dict(json.loads(line)))
                    except Exception:
                        continue
        with self._lock:
            for t in loaded[-self.max_size:]:
                self._traces.append(t)
            self._total_recorded += len(loaded)


@dataclass
class MinedCluster:
    """A discovered intent cluster within the 384-d semantic embedding space."""
    cluster_id: int
    label: str
    centroid: List[float]
    size: int
    coherence: float
    exemplars: List[str]
    all_prompts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "size": self.size,
            "coherence": round(self.coherence, 4),
            "exemplars": self.exemplars,
            "prompt_count": len(self.all_prompts),
        }


class ClusterMiner:
    """
    Pure-Python semantic clustering engine operating over 384-d micro-embeddings.
    Uses k-means with cosine dot product maximization to discover latent user intents.
    """

    def __init__(self, encoder: Optional[SemanticVectorEncoder] = None):
        self.encoder = encoder or SemanticVectorEncoder()

    def mine_clusters(
        self,
        prompts: List[str],
        k: Optional[int] = None,
        min_cluster_size: int = 1,
        max_iterations: int = 25,
        seed: Optional[int] = 42,
    ) -> List[MinedCluster]:
        """
        Clusters a collection of prompt utterances into k intent clusters.
        Uses deterministic seeding and reassigns orphan points to preserve sample volume.
        """
        unique_prompts = list(dict.fromkeys(p.strip() for p in prompts if p.strip()))
        n = len(unique_prompts)
        if n < 2:
            return []

        # Determine k
        if k is None:
            k = max(2, min(5, n // 3))
        k = max(2, min(k, n))

        rng = random.Random(seed if seed is not None else 42)

        # 1. Project prompts into 384-dimensional normalized vectors
        vectors = [self.encoder.encode(p) for p in unique_prompts]
        dim = len(vectors[0])

        # 2. Initialize centroids using k-means++ spread heuristic with seeded rng
        centroids: List[List[float]] = []
        centroids.append(vectors[rng.randint(0, n - 1)])

        for _ in range(1, k):
            dists = []
            for v in vectors:
                min_dist = min(2.0 - 2.0 * self._dot(v, c) for c in centroids)
                dists.append(max(0.0, min_dist))
            total_d = sum(dists)
            if total_d <= 0.0:
                centroids.append(vectors[rng.randint(0, n - 1)])
                continue
            r = rng.random() * total_d
            cum = 0.0
            chosen = vectors[0]
            for v, d in zip(vectors, dists):
                cum += d
                if cum >= r:
                    chosen = v
                    break
            centroids.append(chosen)

        # 3. Iterative k-means optimization
        assignments = [0] * n
        for _ in range(max_iterations):
            changed = False
            for i, v in enumerate(vectors):
                best_sim = -2.0
                best_c = 0
                for c_idx, c in enumerate(centroids):
                    sim = self._dot(v, c)
                    if sim > best_sim:
                        best_sim = sim
                        best_c = c_idx
                if assignments[i] != best_c:
                    assignments[i] = best_c
                    changed = True

            if not changed:
                break

            for c_idx in range(k):
                cluster_members = [vectors[i] for i in range(n) if assignments[i] == c_idx]
                if not cluster_members:
                    continue
                mean_v = [0.0] * dim
                for vec in cluster_members:
                    for d_idx in range(dim):
                        mean_v[d_idx] += vec[d_idx]
                norm = math.sqrt(sum(x * x for x in mean_v)) or 1.0
                centroids[c_idx] = [x / norm for x in mean_v]

        # 4. Extract cluster metadata and reassign any orphan points
        cluster_groups: Dict[int, List[int]] = defaultdict(list)
        for i, a in enumerate(assignments):
            cluster_groups[a].append(i)

        surviving = [c for c, m in cluster_groups.items() if len(m) >= min_cluster_size]
        if not surviving:
            surviving = list(cluster_groups.keys())

        # Reassign orphans to nearest surviving cluster
        for c_idx, members in list(cluster_groups.items()):
            if c_idx not in surviving:
                for i in members:
                    v = vectors[i]
                    best_c = surviving[0]
                    best_sim = -2.0
                    for sc in surviving:
                        sim = self._dot(v, centroids[sc])
                        if sim > best_sim:
                            best_sim = sim
                            best_c = sc
                    cluster_groups[best_c].append(i)
                del cluster_groups[c_idx]

        clusters: List[MinedCluster] = []
        for out_id, (c_idx, member_indices) in enumerate(cluster_groups.items()):
            member_prompts = [unique_prompts[i] for i in member_indices]
            member_vectors = [vectors[i] for i in member_indices]

            c_vec = centroids[c_idx]
            sims = [self._dot(v, c_vec) for v in member_vectors]
            coherence = sum(sims) / len(sims) if sims else 0.0

            ranked = sorted(zip(member_prompts, sims), key=lambda x: x[1], reverse=True)
            exemplars = [p for p, _ in ranked[:3]]
            label = self._derive_label(out_id, member_prompts)

            clusters.append(MinedCluster(
                cluster_id=out_id,
                label=label,
                centroid=c_vec,
                size=len(member_prompts),
                coherence=coherence,
                exemplars=exemplars,
                all_prompts=member_prompts,
            ))

        return clusters

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _derive_label(cluster_id: int, prompts: List[str]) -> str:
        """Extracts top distinguishing non-stop words to construct a human-readable intent label."""
        stopwords = {
            "what", "is", "the", "a", "an", "and", "or", "to", "for", "in", "on", "of",
            "at", "by", "with", "from", "how", "can", "i", "my", "me", "you", "your",
            "we", "our", "it", "this", "that", "these", "those", "please", "help",
            "need", "want", "would", "like", "do", "does", "did", "have", "has", "had",
            "be", "are", "was", "were", "been", "get", "got", "about", "query", "user",
        }
        word_freq: Dict[str, int] = defaultdict(int)
        for p in prompts:
            tokens = re.findall(r"[a-zA-Z]{3,}", p.lower())
            for t in tokens:
                if t not in stopwords:
                    word_freq[t] += 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        if sorted_words:
            top_words = [w for w, _ in sorted_words[:2]]
            return "_".join(top_words)
        return f"intent_{cluster_id}"


class IntentSynthesizer:
    """
    Synthesizes a complete PromptSpec and contrastive training dataset from mined clusters.
    """

    def __init__(self, compiler: Optional[InstinctCompiler] = None):
        self.compiler = compiler or InstinctCompiler()

    def build_spec(
        self,
        clusters: List[MinedCluster],
        name: str = "distilled_model",
    ) -> PromptSpec:
        """Converts mined clusters into a structured PromptSpec ready for compilation."""
        options = [c.label for c in clusters]
        guidelines = {}
        few_shots = []

        for c in clusters:
            exemplar_summary = "; ".join(c.exemplars[:3])
            guidelines[c.label] = f"Queries regarding {c.label.replace('_', ' ')}. Exemplars: {exemplar_summary}"
            for p in c.all_prompts:
                few_shots.append({
                    "text": p,
                    "label": c.label,
                })

        return PromptSpec(
            name=name,
            prompt="Classify customer query into the appropriate operational category.",
            decision_type="choice",
            options=options,
            guidelines=guidelines,
            few_shot_examples=few_shots,
        )


@dataclass
class DistillationResult:
    """Metadata and metrics of a newly compiled distilled instinct model."""
    model_name: str
    model_path: Optional[str]
    options: List[str]
    cluster_count: int
    sample_count: int
    accuracy: float
    brier_score: float
    ece: float
    training_time_ms: float
    timestamp: float
    crc32: int
    clusters: List[MinedCluster]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "options": self.options,
            "cluster_count": self.cluster_count,
            "sample_count": self.sample_count,
            "accuracy": round(self.accuracy, 4),
            "brier_score": round(self.brier_score, 4),
            "ece": round(self.ece, 4),
            "training_time_ms": round(self.training_time_ms, 2),
            "timestamp": self.timestamp,
            "crc32": self.crc32,
            "clusters": [c.to_dict() for c in self.clusters],
        }


class AutonomousDistiller:
    """
    End-to-end continuous model compiler distilling harvested gateway trajectories into .reflex models.
    """

    def __init__(
        self,
        miner: Optional[ClusterMiner] = None,
        synthesizer: Optional[IntentSynthesizer] = None,
        compiler: Optional[InstinctCompiler] = None,
    ):
        self.miner = miner or ClusterMiner()
        self.synthesizer = synthesizer or IntentSynthesizer()
        self.compiler = compiler or InstinctCompiler()

    def distill_from_buffer(
        self,
        buffer: DistillationBuffer,
        output_path: Optional[str] = None,
        min_samples: int = 10,
        k: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Optional[DistillationResult]:
        """
        Reads traces from buffer, mines intent clusters, and compiles a calibrated .reflex model.
        Returns DistillationResult, or None if insufficient samples or clusters.
        """
        traces = buffer.get_traces()
        if len(traces) < min_samples:
            return None

        prompts = [t.prompt for t in traces]
        clusters = self.miner.mine_clusters(prompts, k=k, min_cluster_size=2)
        if len(clusters) < 2:
            return None

        name = model_name or f"distilled_instinct_{int(time.time())}"
        spec = self.synthesizer.build_spec(clusters, name=name)

        # Compile model using InstinctCompiler
        compiled = self.compiler.compile(
            spec=spec,
            samples_per_class=25,
            epochs=35,
            lr=0.08,
        )

        actual_path = output_path
        crc32_val = 0
        if output_path:
            compiled.save(output_path)
            # Read CRC32 from header
            try:
                with open(output_path, "rb") as f:
                    f.seek(12)
                    header_bytes = f.read(12)
                    import struct
                    _, crc32_val = struct.unpack(">II", header_bytes[:8])
            except Exception:
                crc32_val = 0

        m = compiled.metrics or CalibrationMetrics(0.0, 0.0, 0.0, 0, 0.0)

        return DistillationResult(
            model_name=name,
            model_path=actual_path,
            options=spec.options,
            cluster_count=len(clusters),
            sample_count=len(prompts),
            accuracy=m.accuracy,
            brier_score=m.brier_score,
            ece=m.ece,
            training_time_ms=m.training_time_ms,
            timestamp=time.time(),
            crc32=crc32_val,
            clusters=clusters,
        )


class DistillationWorker:
    """
    Background daemon thread periodically running autonomous distillation cycles.
    Notifies a callback when a new candidate model is compiled.
    """

    def __init__(
        self,
        buffer: DistillationBuffer,
        distiller: Optional[AutonomousDistiller] = None,
        interval_seconds: float = 60.0,
        min_new_samples: int = 15,
        output_dir: Optional[str] = None,
        on_candidate_ready: Optional[Callable[[DistillationResult], None]] = None,
    ):
        self.buffer = buffer
        self.distiller = distiller or AutonomousDistiller()
        self.interval_seconds = interval_seconds
        self.min_new_samples = min_new_samples
        self.output_dir = output_dir or "/tmp/reflex_distill"
        self.on_candidate_ready = on_candidate_ready

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_processed_count: int = 0
        self._latest_result: Optional[DistillationResult] = None
        self._cycle_count: int = 0
        self._lock = threading.Lock()

    def start(self):
        """Starts the background worker loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        """Signals worker to stop and waits for exit."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def trigger_cycle(self) -> Optional[DistillationResult]:
        """Manually runs an immediate distillation cycle synchronously."""
        with self._lock:
            cur_count = self.buffer.size()
            new_samples = cur_count - self._last_processed_count
            if new_samples < self.min_new_samples and self._cycle_count > 0:
                return None

            model_filename = f"distilled_model_{int(time.time())}.reflex"
            model_path = os.path.join(self.output_dir, model_filename)

            res = self.distiller.distill_from_buffer(
                buffer=self.buffer,
                output_path=model_path,
                min_samples=self.min_new_samples,
            )

            if res:
                self._latest_result = res
                self._last_processed_count = cur_count
                self._cycle_count += 1
                if self.on_candidate_ready:
                    try:
                        self.on_candidate_ready(res)
                    except Exception:
                        pass

            return res

    def _run_loop(self):
        """Internal background loop."""
        while not self._stop_event.is_set():
            try:
                self.trigger_cycle()
            except Exception:
                pass
            self._stop_event.wait(self.interval_seconds)

    def status(self) -> Dict[str, Any]:
        """Returns worker status and latest distillation metrics."""
        with self._lock:
            return {
                "running": self._thread.is_alive() if self._thread else False,
                "interval_seconds": self.interval_seconds,
                "min_new_samples": self.min_new_samples,
                "total_cycles": self._cycle_count,
                "last_processed_count": self._last_processed_count,
                "latest_candidate": self._latest_result.to_dict() if self._latest_result else None,
            }
