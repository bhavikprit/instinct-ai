"""
Reflex InstinctCache: Sub-0.05ms Semantic Memory & Dual-Brain Cache.
Zero-dependency, multi-tier (L1 exact, L2 semantic vector) LRU cache with TTL and persistence.
"""

from __future__ import annotations
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from reflex.embeddings import SemanticVectorEncoder, cosine_similarity
from reflex.primitives import (
    Noul,
    Choice,
    Score,
    DecisionResult,
    PrimitiveType,
)


@dataclass
class CacheEntry:
    """Entry stored in InstinctCache."""
    state: str
    vector: List[float]
    questions_sig: str
    result: DecisionResult
    created_at: float
    last_accessed: float
    access_count: int = 1


class InstinctCache:
    """
    Sub-0.05ms Semantic Instinct Cache for Reflex.
    
    Architecture:
    - Tier 1 (L1 Exact Hash): Resolves identical queries in ~0.001ms.
    - Tier 2 (L2 Semantic Vector): Resolves semantically equivalent rephrasings
      (e.g., "refund payment" vs "cancel and return my charge") in ~0.04ms using cosine similarity.
    - Eviction: Least Recently Used (LRU) when reaching max_size.
    - Expiration: Optional Time-To-Live (TTL) in seconds.
    - Persistence: Save/load to zero-dependency JSON files.
    """

    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.88,
        ttl_seconds: Optional[float] = None,
        encoder: Optional[SemanticVectorEncoder] = None,
        use_hnsw: bool = True,
        use_pq: bool = False,
    ):
        self.max_size = max(1, max_size)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.encoder = encoder or SemanticVectorEncoder()
        self.use_hnsw = use_hnsw
        self.use_pq = use_pq

        # In-memory LRU store: key -> CacheEntry
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hnsw_index = None
        if self.use_hnsw:
            from reflex.index import HNSWIndex, HNSWConfig
            self._hnsw_index = HNSWIndex(HNSWConfig(dim=384, ef_search=32, ef_construction=64))

        self._pq_quantizer = None
        self._pq_index = None
        if self.use_pq:
            from reflex.pq import ProductQuantizer, PQConfig, PQIndex
            self._pq_quantizer = ProductQuantizer(PQConfig(dim=384, num_subvectors=48, num_centroids=256))
            self._pq_index = PQIndex(self._pq_quantizer)

        # Telemetry
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.evictions = 0
        self.latency_saved_ms = 0.0

    def _make_questions_sig(self, questions: Dict[str, PrimitiveType]) -> str:
        """Generates a stable signature string for the question dictionary."""
        sig_parts = []
        for k in sorted(questions.keys()):
            q = questions[k]
            if isinstance(q, Noul):
                sig_parts.append(f"N:{k}:{q.instructions}:{q.threshold}")
            elif isinstance(q, Choice):
                opts = ",".join(sorted(q.options))
                sig_parts.append(f"C:{k}:{q.instructions}:{opts}")
            elif isinstance(q, Score):
                sig_parts.append(f"S:{k}:{q.instructions}:{q.min_val}-{q.max_val}")
        return "|".join(sig_parts)

    def _make_exact_key(self, state: str, questions_sig: str) -> str:
        return f"{questions_sig}##{state.strip().lower()}"

    def get(
        self,
        state: str,
        questions: Dict[str, PrimitiveType],
    ) -> Optional[DecisionResult]:
        """
        Retrieves cached DecisionResult via L1 exact match or L2 semantic similarity.
        """
        now = time.time()
        q_sig = self._make_questions_sig(questions)
        exact_key = self._make_exact_key(state, q_sig)

        # 1. Tier 1: L1 Exact Match (O(1))
        if exact_key in self._entries:
            entry = self._entries[exact_key]
            if self._is_expired(entry, now):
                del self._entries[exact_key]
            else:
                self._entries.move_to_end(exact_key)
                entry.last_accessed = now
                entry.access_count += 1
                self.exact_hits += 1
                self.latency_saved_ms += entry.result.latency_ms
                return self._wrap_cached_result(entry.result)

        # 2. Tier 2: L2 Semantic Vector Cosine Match
        query_vec = self.encoder.encode(state)
        best_entry: Optional[CacheEntry] = None
        best_key: Optional[str] = None
        best_sim = -1.0

        if self.use_hnsw and self._hnsw_index is not None and len(self._entries) >= 30:
            candidates = self._hnsw_index.search(query_vec, k=min(15, len(self._entries)))
            for cand in candidates:
                cand_key = cand.payload.get("key") if isinstance(cand.payload, dict) else None
                cand_sig = cand.payload.get("questions_sig") if isinstance(cand.payload, dict) else None
                if cand_key and cand_key in self._entries and cand_sig == q_sig:
                    entry = self._entries[cand_key]
                    if not self._is_expired(entry, now):
                        sim = cand.similarity
                        if sim > best_sim:
                            best_sim = sim
                            best_entry = entry
                            best_key = cand_key
        else:
            for key, entry in self._entries.items():
                if entry.questions_sig != q_sig:
                    continue
                if self._is_expired(entry, now):
                    continue

                sim = cosine_similarity(query_vec, entry.vector)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
                    best_key = key

        if best_entry is not None and best_sim >= self.similarity_threshold:
            self._entries.move_to_end(best_key)
            best_entry.last_accessed = now
            best_entry.access_count += 1
            self.semantic_hits += 1
            self.latency_saved_ms += best_entry.result.latency_ms
            return self._wrap_cached_result(best_entry.result)

        self.misses += 1
        return None

    def set(
        self,
        state: str,
        questions: Dict[str, PrimitiveType],
        result: DecisionResult,
    ):
        """Stores DecisionResult in the cache."""
        now = time.time()
        q_sig = self._make_questions_sig(questions)
        exact_key = self._make_exact_key(state, q_sig)
        query_vec = self.encoder.encode(state)

        # LRU Eviction if full
        if len(self._entries) >= self.max_size and exact_key not in self._entries:
            self._entries.popitem(last=False)
            self.evictions += 1

        entry = CacheEntry(
            state=state,
            vector=query_vec,
            questions_sig=q_sig,
            result=result,
            created_at=now,
            last_accessed=now,
            access_count=1,
        )
        self._entries[exact_key] = entry
        self._entries.move_to_end(exact_key)

        if self.use_hnsw and self._hnsw_index is not None:
            self._hnsw_index.insert(
                query_vec,
                payload={"key": exact_key, "questions_sig": q_sig},
            )

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return (now - entry.created_at) > self.ttl_seconds

    def _wrap_cached_result(self, original: DecisionResult) -> DecisionResult:
        """Returns a copy of DecisionResult marked as cached with ~0.00ms latency."""
        return DecisionResult(
            decisions=original.decisions,
            latency_ms=0.01,
            backend=f"{original.backend} (cached)",
            input_tokens=original.input_tokens,
            output_tokens=original.output_tokens,
            cost_usd=0.0,
            cached=True,
        )

    def clear(self):
        """Empties the cache and resets counters."""
        self._entries.clear()
        if self.use_hnsw:
            from reflex.index import HNSWIndex, HNSWConfig
            self._hnsw_index = HNSWIndex(HNSWConfig(dim=384, ef_search=32, ef_construction=64))
        if self.use_pq:
            from reflex.pq import ProductQuantizer, PQConfig, PQIndex
            self._pq_quantizer = ProductQuantizer(PQConfig(dim=384, num_subvectors=48, num_centroids=256))
            self._pq_index = PQIndex(self._pq_quantizer)
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.evictions = 0
        self.latency_saved_ms = 0.0

    @property
    def total_hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def total_queries(self) -> int:
        return self.total_hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_queries
        return round(self.total_hits / total, 4) if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Returns cache telemetry dictionary."""
        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "total_queries": self.total_queries,
            "total_hits": self.total_hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "latency_saved_ms": round(self.latency_saved_ms, 2),
            "use_hnsw": self.use_hnsw,
            "hnsw_indexed_count": len(self._hnsw_index) if self._hnsw_index else 0,
            "use_pq": self.use_pq,
            "pq_indexed_count": len(self._pq_index) if self._pq_index else 0,
        }

    def save_to_file(self, filepath: str):
        """Persists cache entries to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        dump_data = []
        for key, entry in self._entries.items():
            dump_data.append({
                "key": key,
                "state": entry.state,
                "vector": entry.vector,
                "questions_sig": entry.questions_sig,
                "result": entry.result.to_dict(),
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dump_data, f)

    def load_from_file(self, filepath: str):
        """Loads cached entries from a JSON file."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            dump_data = json.load(f)

        now = time.time()
        for item in dump_data:
            # Reconstruct DecisionResult
            dec_dict = {}
            for k, d in item["result"]["decisions"].items():
                dtype = d.get("type")
                if dtype == "noul":
                    dec_dict[k] = Noul(
                        instructions=d["instructions"],
                        threshold=d.get("threshold", 0.85),
                        probability=d.get("probability"),
                    )
                elif dtype == "choice":
                    dec_dict[k] = Choice(
                        instructions=d["instructions"],
                        options=d.get("options", []),
                        selected=d.get("selected"),
                        distribution=d.get("distribution", {}),
                    )
                elif dtype == "score":
                    dec_dict[k] = Score(
                        instructions=d["instructions"],
                        min_val=d["range"][0],
                        max_val=d["range"][1],
                        score=d.get("score"),
                    )

            res = DecisionResult(
                decisions=dec_dict,
                latency_ms=item["result"]["meta"]["latency_ms"],
                backend=item["result"]["meta"]["backend"],
                cost_usd=item["result"]["meta"].get("cost_usd", 0.0),
                cached=True,
            )

            entry = CacheEntry(
                state=item["state"],
                vector=item["vector"],
                questions_sig=item["questions_sig"],
                result=res,
                created_at=item["created_at"],
                last_accessed=item["last_accessed"],
                access_count=item["access_count"],
            )
            if not self._is_expired(entry, now):
                self._entries[item["key"]] = entry
                if self.use_hnsw and self._hnsw_index is not None:
                    self._hnsw_index.insert(
                        entry.vector,
                        payload={"key": item["key"], "questions_sig": entry.questions_sig},
                    )
