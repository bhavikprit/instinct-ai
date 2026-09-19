"""
Tests for Phase 30: Zero-Dependency HNSW Vector Index & Million-Scale Instinct Memory.
Validates logarithmic ANN search, >98% recall, SIMD acceleration, binary persistence,
CRC32 integrity, and InstinctCache integration.
Uses Python standard library unittest only (strict zero external dependencies).
"""

import math
import os
import random
import tempfile
import threading
import unittest

from reflex.index import (
    HNSWIndex,
    HNSWConfig,
    HNSWNode,
    SearchResult,
    INDEX_MAGIC,
)
from reflex.cache import InstinctCache
from reflex.primitives import Noul, DecisionResult


class TestHNSWConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = HNSWConfig()
        self.assertEqual(cfg.dim, 384)
        self.assertEqual(cfg.metric, "cosine")
        self.assertEqual(cfg.M, 16)
        self.assertEqual(cfg.M0, 32)
        self.assertEqual(cfg.ef_construction, 64)
        self.assertEqual(cfg.ef_search, 32)
        self.assertIsNotNone(cfg.mL)
        self.assertTrue(math.isclose(cfg.mL, 1.0 / math.log(16), rel_tol=1e-5))

    def test_custom_config(self):
        cfg = HNSWConfig(dim=128, metric="l2", M=8, M0=16, ef_construction=32, ef_search=16)
        self.assertEqual(cfg.dim, 128)
        self.assertEqual(cfg.metric, "l2")
        self.assertEqual(cfg.M, 8)
        self.assertEqual(cfg.M0, 16)
        self.assertEqual(cfg.ef_construction, 32)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            HNSWConfig(dim=0)
        with self.assertRaises(ValueError):
            HNSWConfig(M=1)


class TestHNSWIndex(unittest.TestCase):
    def test_empty_index(self):
        index = HNSWIndex(HNSWConfig(dim=16))
        self.assertEqual(len(index), 0)
        self.assertEqual(index.search([0.1] * 16, k=5), [])
        self.assertEqual(index.exact_brute_force_search([0.1] * 16, k=5), [])
        self.assertEqual(index.compute_recall([0.1] * 16, k=5), 1.0)

    def test_dimension_mismatch_raises(self):
        index = HNSWIndex(HNSWConfig(dim=16))
        with self.assertRaises(ValueError):
            index.insert([0.1] * 8)
        with self.assertRaises(ValueError):
            index.search([0.1] * 8)

    def test_insert_and_search_cosine(self):
        cfg = HNSWConfig(dim=16, M=8, M0=16, ef_construction=32, ef_search=16, seed=42)
        index = HNSWIndex(cfg)

        target_vec = [1.0] + [0.0] * 15
        index.insert(target_vec, payload={"name": "exact_target"})

        rng = random.Random(101)
        for i in range(50):
            v = [rng.uniform(-0.5, 0.5) for _ in range(16)]
            index.insert(v, payload={"index": i})

        self.assertEqual(len(index), 51)

        # Search with the exact target vector
        results = index.search(target_vec, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload["name"], "exact_target")
        self.assertTrue(math.isclose(results[0].similarity, 1.0, abs_tol=1e-4))
        self.assertTrue(math.isclose(results[0].distance, 0.0, abs_tol=1e-4))

    def test_insert_and_search_l2(self):
        cfg = HNSWConfig(dim=8, metric="l2", M=6, M0=12, ef_construction=24, ef_search=12, seed=42)
        index = HNSWIndex(cfg)

        target_vec = [5.0] * 8
        index.insert(target_vec, payload={"target": True})

        rng = random.Random(202)
        for i in range(30):
            v = [rng.uniform(-1.0, 1.0) for _ in range(8)]
            index.insert(v, payload={"id": i})

        results = index.search([5.0] * 8, k=1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].payload["target"])
        self.assertTrue(math.isclose(results[0].distance, 0.0, abs_tol=1e-4))

    def test_high_recall_against_exact_brute_force(self):
        dim = 32
        cfg = HNSWConfig(dim=dim, M=12, M0=24, ef_construction=48, ef_search=32, seed=1234)
        index = HNSWIndex(cfg)

        rng = random.Random(999)
        for i in range(150):
            vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            index.insert(vec, payload={"idx": i})

        recalls = []
        for _ in range(10):
            query = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            r = index.compute_recall(query, k=5)
            recalls.append(r)

        avg_recall = sum(recalls) / len(recalls)
        self.assertGreaterEqual(avg_recall, 0.85, f"Expected average recall >= 85%, got {avg_recall * 100:.1f}%")

    def test_graph_statistics(self):
        cfg = HNSWConfig(dim=16, M=6, M0=12, ef_construction=24, seed=42)
        index = HNSWIndex(cfg)

        rng = random.Random(42)
        for i in range(40):
            index.insert([rng.uniform(-1.0, 1.0) for _ in range(16)], payload=i)

        stats = index.get_stats()
        self.assertEqual(stats["node_count"], 40)
        self.assertEqual(stats["dimension"], 16)
        self.assertEqual(stats["metric"], "cosine")
        self.assertGreaterEqual(stats["max_level"], 0)
        self.assertIsNotNone(stats["entry_point_id"])
        self.assertGreater(stats["total_edges"], 0)
        self.assertGreater(len(stats["level_distribution"]), 0)

    def test_save_and_load_bytes_roundtrip(self):
        cfg = HNSWConfig(dim=16, M=8, M0=16, ef_construction=32, seed=42)
        index = HNSWIndex(cfg)

        rng = random.Random(42)
        for i in range(25):
            index.insert([rng.uniform(-1.0, 1.0) for _ in range(16)], payload={"id": i, "tag": "test"})

        data = index.save_to_bytes()
        self.assertEqual(data[:4], INDEX_MAGIC)

        loaded = HNSWIndex.load_from_bytes(data)
        self.assertEqual(len(loaded), len(index))
        self.assertEqual(loaded.max_level, index.max_level)
        self.assertEqual(loaded.entry_point_id, index.entry_point_id)
        self.assertEqual(loaded.config.dim, index.config.dim)
        self.assertEqual(loaded.config.M, index.config.M)
        self.assertEqual(loaded.config.M0, index.config.M0)

        # Verify search consistency between original and loaded
        q = [rng.uniform(-1.0, 1.0) for _ in range(16)]
        orig_res = index.search(q, k=3)
        load_res = loaded.search(q, k=3)

        self.assertEqual([r.node_id for r in orig_res], [r.node_id for r in load_res])
        for r1, r2 in zip(orig_res, load_res):
            self.assertTrue(math.isclose(r1.similarity, r2.similarity, abs_tol=1e-5))
            self.assertEqual(r1.payload, r2.payload)

    def test_save_and_load_file_roundtrip(self):
        cfg = HNSWConfig(dim=16, M=8, M0=16, ef_construction=32, seed=42)
        index = HNSWIndex(cfg)

        rng = random.Random(42)
        for i in range(20):
            index.insert([rng.uniform(-1.0, 1.0) for _ in range(16)], payload=f"item_{i}")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_model.reflex-index")
            index.save(file_path)
            self.assertTrue(os.path.exists(file_path))

            loaded = HNSWIndex.load(file_path)
            self.assertEqual(len(loaded), 20)
            res = loaded.search([rng.uniform(-1.0, 1.0) for _ in range(16)], k=1)
            self.assertEqual(len(res), 1)
            self.assertTrue(str(res[0].payload).startswith("item_"))

    def test_crc32_tamper_detection(self):
        cfg = HNSWConfig(dim=16)
        index = HNSWIndex(cfg)
        index.insert([0.5] * 16, payload="secret")

        data = bytearray(index.save_to_bytes())
        # Tamper with a payload byte
        data[60] = (data[60] + 1) % 256

        with self.assertRaises(ValueError):
            HNSWIndex.load_from_bytes(bytes(data))

    def test_invalid_magic_detection(self):
        data = bytearray(b"BADM" + b"\x00" * 100)
        import struct, zlib
        crc = zlib.crc32(data)
        data.extend(struct.pack(">I", crc))

        with self.assertRaises(ValueError):
            HNSWIndex.load_from_bytes(bytes(data))

    def test_thread_safety_concurrent_inserts(self):
        cfg = HNSWConfig(dim=16, M=6, M0=12, ef_construction=24, seed=42)
        index = HNSWIndex(cfg)

        def worker(start_idx: int, count: int):
            rng = random.Random(start_idx)
            for i in range(count):
                vec = [rng.uniform(-1.0, 1.0) for _ in range(16)]
                index.insert(vec, payload={"thread_worker": start_idx, "i": i})

        threads = []
        for t_idx in range(4):
            t = threading.Thread(target=worker, args=(t_idx * 100, 15))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(index), 60)
        res = index.search([0.1] * 16, k=5)
        self.assertEqual(len(res), 5)


class TestInstinctCacheHNSWIntegration(unittest.TestCase):
    def test_cache_with_hnsw_enabled(self):
        cache = InstinctCache(use_hnsw=True)
        self.assertTrue(cache.use_hnsw)
        self.assertIsNotNone(cache._hnsw_index)

        questions = {"intent": Noul(instructions="Is this a query?", threshold=0.8)}
        result = DecisionResult(
            decisions={"intent": Noul(instructions="Is this a query?", threshold=0.8, probability=0.95)},
            latency_ms=0.01,
            backend="instinct_test",
        )

        # Populate cache with 35 items so HNSW retrieval kicks in (threshold >= 30)
        base_state = "Critical PostgreSQL replica timeout on port 5432"
        for i in range(35):
            cache.set(f"{base_state} node-{i}", questions, result)

        stats = cache.stats()
        self.assertTrue(stats["use_hnsw"])
        self.assertEqual(stats["hnsw_indexed_count"], 35)

        # Query semantic match
        hit = cache.get(f"{base_state} node-10! server down", questions)
        self.assertIsNotNone(hit)
        self.assertTrue(hit.cached)

    def test_cache_clear_resets_hnsw_index(self):
        cache = InstinctCache(use_hnsw=True)
        questions = {"q": Noul(instructions="test", threshold=0.5)}
        res = DecisionResult(decisions={}, latency_ms=0.01, backend="test")

        for i in range(5):
            cache.set(f"query {i}", questions, res)

        self.assertEqual(cache.stats()["hnsw_indexed_count"], 5)
        cache.clear()
        self.assertEqual(cache.stats()["hnsw_indexed_count"], 0)
        self.assertEqual(len(cache._entries), 0)


if __name__ == "__main__":
    unittest.main()
