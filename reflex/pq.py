"""
Reflex Product Quantization (PQ) & Asymmetric Distance Computation (ADC) (Phase 31).
Compresses 384-dimensional dense vectors from 1,536 bytes down to 48 bytes (32x compression)
or 24 bytes (64x compression) with sub-microsecond ADC table lookup search.
Pure Python standard library with optional C99 SIMD batch acceleration. Zero external dependencies.
"""

from __future__ import annotations
import ctypes
from dataclasses import dataclass, field
import json
import math
import os
import random
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zlib

from reflex.backends.c_engine import find_libreflex

# Binary format identifiers
PQ_CODEBOOK_MAGIC = b"RFPQ"
PQ_INDEX_MAGIC = b"RFPI"
PQ_FORMAT_VERSION = 1

METRIC_COSINE = 0
METRIC_L2 = 1


@dataclass
class PQConfig:
    """Hyperparameters for Product Quantization and Asymmetric Distance Computation."""
    dim: int = 384
    num_subvectors: int = 48  # M (384 / 48 = 8 dimensions per sub-vector -> 48 bytes/vector)
    num_centroids: int = 256  # K (256 centroids = 1 byte per sub-quantizer)
    metric: str = "cosine"  # "cosine" or "l2"
    seed: Optional[int] = 42
    M: Optional[int] = None  # Alias for num_subvectors
    K: Optional[int] = None  # Alias for num_centroids

    def __post_init__(self) -> None:
        if self.M is not None:
            self.num_subvectors = self.M
        if self.K is not None:
            self.num_centroids = self.K
        self.M = self.num_subvectors
        self.K = self.num_centroids
        if self.dim <= 0:
            raise ValueError(f"Vector dimension must be positive, got {self.dim}")
        if self.num_subvectors <= 0:
            raise ValueError(f"Number of sub-vectors must be positive, got {self.num_subvectors}")
        if self.dim % self.num_subvectors != 0:
            raise ValueError(
                f"Dimension ({self.dim}) must be divisible by num_subvectors ({self.num_subvectors})"
            )
        if self.num_centroids < 2 or self.num_centroids > 256:
            raise ValueError(
                f"num_centroids must be between 2 and 256 for 1-byte code indices, got {self.num_centroids}"
            )

    @property
    def d_sub(self) -> int:
        """Dimension of each sub-vector."""
        return self.dim // self.num_subvectors

    @property
    def bytes_per_vector(self) -> int:
        """Memory consumed per quantized vector in bytes."""
        return self.num_subvectors

    @property
    def compression_ratio(self) -> float:
        """Memory compression factor relative to FP32."""
        return (self.dim * 4.0) / float(self.bytes_per_vector)


class ProductQuantizer:
    """
    Sub-vector Product Quantizer decomposing D-dimensional space into M sub-spaces.
    Provides vector encoding, vector reconstruction (decoding), and Asymmetric Distance
    Computation (ADC) Lookup Table (LUT) generation.
    """

    def __init__(self, config: Optional[PQConfig] = None):
        self.config = config or PQConfig()
        # centroids shape: [M][K][d_sub]
        self.centroids: List[List[List[float]]] = []
        self._flat_centroids: Optional[List[float]] = None
        self._c_lib = None
        self._c_centroids = None
        self._init_c_bindings()

    def _init_c_bindings(self) -> None:
        """Loads native C99 PQ/ADC kernels from libreflex if available."""
        lib_path = find_libreflex()
        if lib_path and os.path.exists(lib_path):
            try:
                lib = ctypes.CDLL(lib_path)
                if hasattr(lib, "reflex_compute_adc_lut_f32"):
                    lib.reflex_compute_adc_lut_f32.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                    ]
                    lib.reflex_compute_adc_lut_f32.restype = None

                if hasattr(lib, "reflex_batch_adc_dist_u8"):
                    lib.reflex_batch_adc_dist_u8.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                    ]
                    lib.reflex_batch_adc_dist_u8.restype = None

                if hasattr(lib, "reflex_quantize_vector_pq"):
                    lib.reflex_quantize_vector_pq.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                    ]
                    lib.reflex_quantize_vector_pq.restype = None

                if hasattr(lib, "reflex_assign_centroids_subvector"):
                    lib.reflex_assign_centroids_subvector.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_int),
                    ]
                    lib.reflex_assign_centroids_subvector.restype = None

                self._c_lib = lib
            except Exception:
                self._c_lib = None

    @property
    def is_trained(self) -> bool:
        """Returns True if the quantizer codebook has been trained."""
        return len(self.centroids) == self.config.num_subvectors

    @property
    def is_native_accelerated(self) -> bool:
        """Returns True if native C99 ADC hardware kernels are loaded."""
        return self._c_lib is not None

    def _sync_c_centroids(self) -> None:
        """Flattens centroids into contiguous C-compatible float buffer."""
        if not self.is_trained:
            return
        flat = []
        for m_centroids in self.centroids:
            for c in m_centroids:
                flat.extend(c)
        self._flat_centroids = flat

        if self._c_lib:
            try:
                arr_type = ctypes.c_float * len(flat)
                self._c_centroids = arr_type(*flat)
            except Exception:
                self._c_centroids = None

    # -------------------------------------------------------------------------
    # Codebook Training
    # -------------------------------------------------------------------------

    def train(
        self,
        vectors: Sequence[Sequence[float]],
        max_iters: int = 15,
    ) -> None:
        """
        Trains sub-vector codebooks using deterministic Lloyd's k-means across M sub-spaces.
        """
        if len(vectors) == 0:
            raise ValueError("Training vectors cannot be empty")
        for v in vectors:
            if len(v) != self.config.dim:
                raise ValueError(
                    f"Training vector dimension mismatch: expected {self.config.dim}, got {len(v)}"
                )

        rng = random.Random(self.config.seed)
        M = self.config.num_subvectors
        K = self.config.num_centroids
        d_sub = self.config.d_sub

        self.centroids = []

        # Train each sub-space independently
        for m in range(M):
            start_d = m * d_sub
            end_d = start_d + d_sub
            sub_vecs = [v[start_d:end_d] for v in vectors]

            # Initialize K centroids deterministically
            if len(sub_vecs) >= K:
                chosen_indices = rng.sample(range(len(sub_vecs)), K)
                m_centroids = [[float(x) for x in sub_vecs[idx]] for idx in chosen_indices]
            else:
                # Pad with small random perturbations
                m_centroids = []
                for k in range(K):
                    base = sub_vecs[k % len(sub_vecs)]
                    m_centroids.append([float(x + rng.uniform(-0.01, 0.01)) for x in base])

            # Lloyd's K-Means iterations
            c_assign_func = getattr(self._c_lib, "reflex_assign_centroids_subvector", None) if self._c_lib else None
            metric_code = METRIC_COSINE if self.config.metric == "cosine" else METRIC_L2
            count = len(sub_vecs)

            if c_assign_func:
                flat_sub = []
                for sv in sub_vecs:
                    flat_sub.extend(sv)
                c_sub_vecs = (ctypes.c_float * len(flat_sub))(*flat_sub)
                c_assignments = (ctypes.c_int * count)()

                for _ in range(max_iters):
                    flat_c = []
                    for c in m_centroids:
                        flat_c.extend(c)
                    c_centroids_iter = (ctypes.c_float * len(flat_c))(*flat_c)

                    c_assign_func(
                        c_sub_vecs,
                        count,
                        c_centroids_iter,
                        K,
                        d_sub,
                        metric_code,
                        c_assignments,
                    )

                    clusters = [[] for _ in range(K)]
                    for idx in range(count):
                        k_assigned = int(c_assignments[idx])
                        clusters[k_assigned].append(sub_vecs[idx])

                    converged = True
                    for k in range(K):
                        pts = clusters[k]
                        if pts:
                            new_c = [sum(p[i] for p in pts) / float(len(pts)) for i in range(d_sub)]
                            if self.config.metric == "cosine":
                                norm = math.sqrt(sum(x * x for x in new_c))
                                if norm > 1e-12:
                                    new_c = [x / norm for x in new_c]
                            if any(abs(new_c[i] - m_centroids[k][i]) > 1e-4 for i in range(d_sub)):
                                converged = False
                            m_centroids[k] = new_c

                    if converged:
                        break
            else:
                for _ in range(max_iters):
                    clusters: List[List[List[float]]] = [[] for _ in range(K)]

                    # E-step: Assign each sub-vector to closest centroid
                    for sv in sub_vecs:
                        best_k = 0
                        best_dist = float("inf")
                        for k in range(K):
                            c = m_centroids[k]
                            if self.config.metric == "cosine":
                                dot = sum(sv[i] * c[i] for i in range(d_sub))
                                d = 1.0 - max(-1.0, min(1.0, dot))
                            else:
                                d = sum((sv[i] - c[i]) ** 2 for i in range(d_sub))
                            if d < best_dist:
                                best_dist = d
                                best_k = k
                        clusters[best_k].append(sv)

                    # M-step: Update centroids
                    converged = True
                    for k in range(K):
                        pts = clusters[k]
                        if pts:
                            new_c = [sum(p[i] for p in pts) / float(len(pts)) for i in range(d_sub)]
                            if self.config.metric == "cosine":
                                norm = math.sqrt(sum(x * x for x in new_c))
                                if norm > 1e-12:
                                    new_c = [x / norm for x in new_c]
                            if any(abs(new_c[i] - m_centroids[k][i]) > 1e-4 for i in range(d_sub)):
                                converged = False
                            m_centroids[k] = new_c

                    if converged:
                        break

            self.centroids.append(m_centroids)

        self._sync_c_centroids()

    # -------------------------------------------------------------------------
    # Encoding & Decoding
    # -------------------------------------------------------------------------

    def encode(self, vector: Sequence[float]) -> bytes:
        """
        Quantizes a D-dimensional float vector into M 1-byte code indices.
        Returns a bytes object of length M.
        """
        if not self.is_trained:
            raise RuntimeError("ProductQuantizer must be trained before encoding")
        if len(vector) != self.config.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.config.dim}, got {len(vector)}"
            )

        M = self.config.num_subvectors
        K = self.config.num_centroids
        d_sub = self.config.d_sub

        if self._c_lib and self._c_centroids is not None:
            try:
                c_vec = (ctypes.c_float * self.config.dim)(*vector)
                c_out = (ctypes.c_uint8 * M)()
                metric_code = METRIC_COSINE if self.config.metric == "cosine" else METRIC_L2

                self._c_lib.reflex_quantize_vector_pq(
                    c_vec,
                    self._c_centroids,
                    M,
                    d_sub,
                    K,
                    metric_code,
                    c_out,
                )
                return bytes(c_out)
            except Exception:
                pass

        # Pure Python fallback
        code_list = []
        for m in range(M):
            start_d = m * d_sub
            end_d = start_d + d_sub
            sv = vector[start_d:end_d]

            best_k = 0
            best_dist = float("inf")
            for k in range(K):
                c = self.centroids[m][k]
                if self.config.metric == "cosine":
                    dot = sum(sv[i] * c[i] for i in range(d_sub))
                    d = 1.0 - max(-1.0, min(1.0, dot))
                else:
                    d = sum((sv[i] - c[i]) ** 2 for i in range(d_sub))
                if d < best_dist:
                    best_dist = d
                    best_k = k
            code_list.append(best_k)

        return bytes(code_list)

    def decode(self, codes: Union[bytes, Sequence[int]]) -> List[float]:
        """
        Reconstructs the approximate D-dimensional float vector from M code indices.
        """
        if not self.is_trained:
            raise RuntimeError("ProductQuantizer must be trained before decoding")
        if len(codes) != self.config.num_subvectors:
            raise ValueError(
                f"Codes length mismatch: expected {self.config.num_subvectors}, got {len(codes)}"
            )

        reconstructed: List[float] = []
        for m, code in enumerate(codes):
            centroid = self.centroids[m][int(code)]
            reconstructed.extend(centroid)
        return reconstructed

    # -------------------------------------------------------------------------
    # Asymmetric Distance Computation (ADC)
    # -------------------------------------------------------------------------

    def compute_lut(self, query: Sequence[float]) -> List[List[float]]:
        """
        Computes the M x K distance lookup table (LUT) between query subvectors
        and codebook centroids. Precomputed once per query.
        """
        if not self.is_trained:
            raise RuntimeError("ProductQuantizer must be trained before computing LUT")
        if len(query) != self.config.dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.config.dim}, got {len(query)}"
            )

        M = self.config.num_subvectors
        K = self.config.num_centroids
        d_sub = self.config.d_sub

        if self._c_lib and self._c_centroids is not None:
            try:
                c_query = (ctypes.c_float * self.config.dim)(*query)
                c_lut = (ctypes.c_float * (M * K))()
                metric_code = METRIC_COSINE if self.config.metric == "cosine" else METRIC_L2

                self._c_lib.reflex_compute_adc_lut_f32(
                    c_query,
                    self._c_centroids,
                    M,
                    d_sub,
                    K,
                    metric_code,
                    c_lut,
                )

                lut: List[List[float]] = []
                for m in range(M):
                    base = m * K
                    lut.append([float(c_lut[base + k]) for k in range(K)])
                return lut
            except Exception:
                pass

        # Pure Python fallback
        lut = []
        for m in range(M):
            start_d = m * d_sub
            end_d = start_d + d_sub
            q_sub = query[start_d:end_d]

            sub_lut = []
            for k in range(K):
                c = self.centroids[m][k]
                if self.config.metric == "cosine":
                    dot = sum(q_sub[i] * c[i] for i in range(d_sub))
                    d = 1.0 - max(-1.0, min(1.0, dot))
                else:
                    d = sum((q_sub[i] - c[i]) ** 2 for i in range(d_sub))
                sub_lut.append(d)
            lut.append(sub_lut)
        return lut

    def asymmetric_distance(
        self,
        lut: List[List[float]],
        codes: Union[bytes, Sequence[int]],
    ) -> float:
        """
        Calculates asymmetric distance in sub-microsecond time using M table lookups.
        Multiplication-free: consists purely of additions.
        """
        dist = 0.0
        for m in range(self.config.num_subvectors):
            dist += lut[m][codes[m]]
        return dist

    # -------------------------------------------------------------------------
    # Binary Serialization (.reflex-pq)
    # -------------------------------------------------------------------------

    def save_to_bytes(self) -> bytes:
        """Serializes quantizer codebook into portable byte stream with CRC32 integrity check."""
        if not self.is_trained:
            raise RuntimeError("Cannot serialize untrained ProductQuantizer")

        buf = bytearray()
        buf.extend(PQ_CODEBOOK_MAGIC)

        metric_code = METRIC_COSINE if self.config.metric == "cosine" else METRIC_L2
        header = struct.pack(
            ">6I",
            PQ_FORMAT_VERSION,
            self.config.dim,
            self.config.num_subvectors,
            self.config.d_sub,
            self.config.num_centroids,
            metric_code,
        )
        buf.extend(header)

        # Pack all centroids
        flat = []
        for m in range(self.config.num_subvectors):
            for k in range(self.config.num_centroids):
                flat.extend(self.centroids[m][k])

        buf.extend(struct.pack(f">{len(flat)}f", *flat))

        # Trailer: 32-bit CRC32
        crc = zlib.crc32(buf) & 0xFFFFFFFF
        buf.extend(struct.pack(">I", crc))
        return bytes(buf)

    @classmethod
    def load_from_bytes(cls, data: bytes) -> ProductQuantizer:
        """Restores ProductQuantizer from byte stream with CRC32 integrity verification."""
        min_len = 4 + 24 + 4
        if len(data) < min_len:
            raise ValueError("Invalid PQ codebook byte stream: payload too short")

        # Verify CRC32
        payload = data[:-4]
        expected_crc = struct.unpack(">I", data[-4:])[0]
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(
                f"Corrupt PQ codebook: CRC32 mismatch (expected {expected_crc:#010x}, got {actual_crc:#010x})"
            )

        if payload[:4] != PQ_CODEBOOK_MAGIC:
            raise ValueError(f"Invalid PQ codebook magic: expected {PQ_CODEBOOK_MAGIC!r}, got {payload[:4]!r}")

        version, dim, M, d_sub, K, metric_code = struct.unpack(">6I", payload[4:28])
        if version != PQ_FORMAT_VERSION:
            raise ValueError(f"Unsupported PQ format version: {version}")

        metric = "cosine" if metric_code == METRIC_COSINE else "l2"
        config = PQConfig(dim=dim, num_subvectors=M, num_centroids=K, metric=metric)
        quantizer = cls(config)

        total_floats = M * K * d_sub
        offset = 28
        float_bytes = total_floats * 4
        if len(payload) < offset + float_bytes:
            raise ValueError("Corrupt PQ codebook: missing centroid float payloads")

        flat_floats = struct.unpack(f">{total_floats}f", payload[offset : offset + float_bytes])

        # Reconstruct [M][K][d_sub]
        idx = 0
        quantizer.centroids = []
        for m in range(M):
            m_c = []
            for k in range(K):
                c = list(flat_floats[idx : idx + d_sub])
                idx += d_sub
                m_c.append(c)
            quantizer.centroids.append(m_c)

        quantizer._sync_c_centroids()
        return quantizer

    def save(self, filepath: str) -> None:
        """Saves codebook to a binary .reflex-pq file."""
        data = self.save_to_bytes()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        temp_path = f"{filepath}.tmp.{os.getpid()}"
        with open(temp_path, "wb") as f:
            f.write(data)
        os.replace(temp_path, filepath)

    @classmethod
    def load(cls, filepath: str) -> ProductQuantizer:
        """Loads codebook from a binary .reflex-pq file."""
        with open(filepath, "rb") as f:
            data = f.read()
        return cls.load_from_bytes(data)


@dataclass
class PQSearchResult:
    """Result returned by Product Quantization vector search."""
    node_id: int
    similarity: float  # Estimated cosine similarity or normalized score
    distance: float  # Asymmetric distance (ADC)
    payload: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "similarity": round(self.similarity, 6),
            "distance": round(self.distance, 6),
            "payload": self.payload,
        }


class PQIndex:
    """
    High-Capacity Flat Product Quantization Vector Index.
    Stores millions of vectors in contiguous 48-byte codes, enabling multiplication-free
    ADC retrieval across millions of memories with 32x RAM reduction.
    """

    def __init__(self, quantizer: ProductQuantizer):
        self.quantizer = quantizer
        self.codes = bytearray()  # Flat buffer of length N * M
        self.payloads: List[Any] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self.payloads)

    @property
    def count(self) -> int:
        """Returns the number of vectors stored in the index."""
        return len(self)

    def insert(self, vector: Sequence[float], payload: Any = None) -> int:
        """
        Quantizes and indexes a high-dimensional vector. Returns the assigned node_id.
        """
        code = self.quantizer.encode(vector)
        with self._lock:
            node_id = len(self.payloads)
            self.codes.extend(code)
            self.payloads.append(payload)
            return node_id

    def search(self, query: Sequence[float], k: int = 5, top_k: Optional[int] = None) -> List[PQSearchResult]:
        """
        Performs Asymmetric Distance Computation (ADC) search over all indexed vectors.
        """
        if top_k is not None:
            k = top_k
        if k <= 0 or not self.payloads:
            return []

        lut = self.quantizer.compute_lut(query)
        N = len(self.payloads)
        M = self.quantizer.config.num_subvectors
        K = self.quantizer.config.num_centroids

        with self._lock:
            # Native C99 acceleration if available
            if self.quantizer._c_lib:
                try:
                    flat_lut = []
                    for m_lut in lut:
                        flat_lut.extend(m_lut)

                    c_lut = (ctypes.c_float * (M * K))(*flat_lut)
                    c_codes = (ctypes.c_uint8 * (N * M)).from_buffer(self.codes)
                    c_dists = (ctypes.c_float * N)()

                    self.quantizer._c_lib.reflex_batch_adc_dist_u8(
                        c_lut,
                        c_codes,
                        N,
                        M,
                        K,
                        c_dists,
                    )

                    scored = []
                    for i in range(N):
                        d = float(c_dists[i])
                        sim = max(-1.0, min(1.0, 1.0 - d)) if self.quantizer.config.metric == "cosine" else (1.0 / (1.0 + d))
                        scored.append((d, sim, i))

                    scored.sort(key=lambda x: x[0])
                    top = scored[:k]

                    return [
                        PQSearchResult(
                            node_id=nid,
                            similarity=sim,
                            distance=d,
                            payload=self.payloads[nid],
                        )
                        for d, sim, nid in top
                    ]
                except Exception:
                    pass

            # Pure Python table lookup
            scored = []
            for i in range(N):
                base = i * M
                d = 0.0
                for m in range(M):
                    code_val = self.codes[base + m]
                    d += lut[m][code_val]

                sim = max(-1.0, min(1.0, 1.0 - d)) if self.quantizer.config.metric == "cosine" else (1.0 / (1.0 + d))
                scored.append((d, sim, i))

            scored.sort(key=lambda x: x[0])
            top = scored[:k]

            return [
                PQSearchResult(
                    node_id=nid,
                    similarity=sim,
                    distance=d,
                    payload=self.payloads[nid],
                )
                for d, sim, nid in top
            ]

    def compute_recall(
        self,
        query: Sequence[float],
        ground_truth_ids: Sequence[int],
        k: int = 5,
    ) -> float:
        """Computes Recall@K of ADC search against exact ground truth."""
        if k <= 0:
            return 1.0
        results = self.search(query, k=k)
        retrieved = {r.node_id for r in results}
        gt_set = set(ground_truth_ids[:k])
        overlap = len(retrieved.intersection(gt_set))
        return float(overlap) / float(k)

    def stats(self) -> Dict[str, Any]:
        """Returns statistics on the quantized index."""
        with self._lock:
            raw_fp32_bytes = len(self.payloads) * self.quantizer.config.dim * 4
            pq_bytes = len(self.codes)
            return {
                "vector_count": len(self.payloads),
                "count": len(self.payloads),
                "dimension": self.quantizer.config.dim,
                "num_subvectors": self.quantizer.config.num_subvectors,
                "bytes_per_vector": self.quantizer.config.bytes_per_vector,
                "raw_fp32_bytes": raw_fp32_bytes,
                "compressed_code_bytes": pq_bytes,
                "memory_compressed_kb": round(pq_bytes / 1024.0, 2),
                "memory_raw_fp32_kb": round(raw_fp32_bytes / 1024.0, 2),
                "compression_ratio": self.quantizer.config.compression_ratio,
                "native_accelerated": self.quantizer.is_native_accelerated,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Alias for stats() matching HNSWIndex API."""
        return self.stats()

    # -------------------------------------------------------------------------
    # Binary Serialization (.reflex-pq-index)
    # -------------------------------------------------------------------------

    def save_to_bytes(self) -> bytes:
        """Serializes the PQ index into byte stream with CRC32 verification."""
        with self._lock:
            buf = bytearray()
            buf.extend(PQ_INDEX_MAGIC)

            # 1. Codebook bytes
            cb_bytes = self.quantizer.save_to_bytes()
            buf.extend(struct.pack(">I", len(cb_bytes)))
            buf.extend(cb_bytes)

            # 2. Codes bytes
            buf.extend(struct.pack(">I", len(self.codes)))
            buf.extend(self.codes)

            # 3. Payloads JSON
            payload_json = json.dumps(self.payloads).encode("utf-8")
            buf.extend(struct.pack(">I", len(payload_json)))
            buf.extend(payload_json)

            # 4. CRC32
            crc = zlib.crc32(buf) & 0xFFFFFFFF
            buf.extend(struct.pack(">I", crc))
            return bytes(buf)

    @classmethod
    def load_from_bytes(cls, data: bytes) -> PQIndex:
        """Restores a PQ index from a byte stream with CRC32 integrity check."""
        if len(data) < 16:
            raise ValueError("Invalid PQ index byte stream: payload too short")

        payload_bytes = data[:-4]
        expected_crc = struct.unpack(">I", data[-4:])[0]
        actual_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(
                f"Corrupt PQ index: CRC32 mismatch (expected {expected_crc:#010x}, got {actual_crc:#010x})"
            )

        if payload_bytes[:4] != PQ_INDEX_MAGIC:
            raise ValueError(f"Invalid PQ index magic: expected {PQ_INDEX_MAGIC!r}, got {payload_bytes[:4]!r}")

        offset = 4
        cb_len = struct.unpack(">I", payload_bytes[offset : offset + 4])[0]
        offset += 4
        cb_bytes = payload_bytes[offset : offset + cb_len]
        offset += cb_len

        quantizer = ProductQuantizer.load_from_bytes(cb_bytes)
        index = cls(quantizer)

        codes_len = struct.unpack(">I", payload_bytes[offset : offset + 4])[0]
        offset += 4
        index.codes = bytearray(payload_bytes[offset : offset + codes_len])
        offset += codes_len

        payload_len = struct.unpack(">I", payload_bytes[offset : offset + 4])[0]
        offset += 4
        if payload_len > 0:
            raw_str = payload_bytes[offset : offset + payload_len].decode("utf-8")
            index.payloads = json.loads(raw_str)

        return index

    def save(self, filepath: str) -> None:
        """Saves PQ index to disk."""
        data = self.save_to_bytes()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        temp_path = f"{filepath}.tmp.{os.getpid()}"
        with open(temp_path, "wb") as f:
            f.write(data)
        os.replace(temp_path, filepath)

    @classmethod
    def load(cls, filepath: str) -> PQIndex:
        """Loads PQ index from disk."""
        with open(filepath, "rb") as f:
            data = f.read()
        return cls.load_from_bytes(data)
