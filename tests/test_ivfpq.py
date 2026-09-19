"""
Unit tests for Reflex Phase 32: Inverted File Product Quantization (IVF-PQ) & Hybrid HNSW-PQ.
Validates coarse Voronoi space partitioning, residual product quantization, pruned list search,
binary persistence with CRC32 integrity, thread safety, and InstinctCache integration.
Uses Python standard library unittest only (strict zero external dependencies).
"""

import math
import os
import random
import tempfile
import threading
import unittest

from reflex.ivfpq import (
    IVFPQConfig,
    IVFPQIndex,
    IVFPQSearchResult,
    IVFPQ_MAGIC,
)
from reflex.cache import InstinctCache


class TestIVFPQConfig(unittest.TestCase):
    def test_config_defaults_and_validation(self):
        cfg = IVFPQConfig()
        self.assertEqual(cfg.dim, 384)
        self.assertEqual(cfg.nlist, 256)
        self.assertEqual(cfg.nprobe, 8)
        self.assertEqual(cfg.M, 48)
        self.assertEqual(cfg.K, 256)
        self.assertEqual(cfg.metric, "cosine")
        self.assertFalse(cfg.use_hnsw_coarse)

        # Validation errors
        with self.assertRaises(ValueError):
            IVFPQConfig(dim=0)
        with self.assertRaises(ValueError):
            IVFPQConfig(nlist=0)
        with self.assertRaises(ValueError):
            IVFPQConfig(nprobe=0)
        with self.assertRaises(ValueError):
            IVFPQConfig(nlist=10, nprobe=20)  # nprobe > nlist
        with self.assertRaises(ValueError):
            IVFPQConfig(dim=384, M=50)  # 384 not divisible by 50
        with self.assertRaises(ValueError):
            IVFPQConfig(K=512)  # Exceeds 256
        with self.assertRaises(ValueError):
            IVFPQConfig(metric="manhattan")

    def test_config_properties(self):
        cfg = IVFPQConfig(dim=128, nlist=16, nprobe=4, M=16, K=64)
        self.assertEqual(cfg.d_sub, 8)
        self.assertEqual(cfg.bytes_per_vector, 16)
        self.assertAlmostEqual(cfg.compression_ratio, 32.0)
        self.assertAlmostEqual(cfg.theoretical_pruning_ratio, 75.0)


class TestIVFPQIndex(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.nlist = 4
        self.nprobe = 2
        self.M = 4
        self.K = 16
        self.cfg = IVFPQConfig(
            dim=self.dim,
            nlist=self.nlist,
            nprobe=self.nprobe,
            M=self.M,
            K=self.K,
            metric="l2",
            seed=42,
        )
        self.index = IVFPQIndex(self.cfg)

        # Generate reproducible training dataset
        rng = random.Random(42)
        self.training_data = [
            [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
            for _ in range(60)
        ]

    def test_untrained_index_raises(self):
        with self.assertRaises(RuntimeError):
            self.index.insert([0.1] * self.dim)
        with self.assertRaises(RuntimeError):
            self.index.save_to_bytes()
        # Search on untrained index returns empty list
        self.assertEqual(self.index.search([0.1] * self.dim), [])

    def test_training_coarse_clustering_and_residuals(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        self.assertTrue(self.index.is_trained)
        self.assertEqual(len(self.index.coarse_centroids), self.nlist)
        self.assertTrue(self.index.residual_quantizer.is_trained)

    def test_empty_search(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        res = self.index.search([0.1] * self.dim, k=5)
        self.assertEqual(res, [])

    def test_insert_and_single_search(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        node_id = self.index.insert(self.training_data[0], payload={"tag": "first"})
        self.assertEqual(node_id, 0)
        self.assertEqual(len(self.index), 1)

        res = self.index.search(self.training_data[0], k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].node_id, 0)
        self.assertEqual(res[0].payload["tag"], "first")
        self.assertGreaterEqual(res[0].similarity, 0.0)
        self.assertLessEqual(res[0].similarity, 1.0)
        self.assertGreaterEqual(res[0].distance, 0.0)

    def test_search_top_k_ordering(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        for i, vec in enumerate(self.training_data):
            self.index.insert(vec, payload={"idx": i})

        query = self.training_data[5]
        results = self.index.search(query, k=5, nprobe=self.nlist)
        self.assertGreaterEqual(len(results), 1)

        # Distances must be sorted ascending
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i].distance, results[i + 1].distance)
            self.assertGreaterEqual(results[i].similarity, results[i + 1].similarity)

    def test_nprobe_tradeoff_and_pruning(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        for i, vec in enumerate(self.training_data):
            self.index.insert(vec, payload={"idx": i})

        query = self.training_data[10]
        res_probe1 = self.index.search(query, k=1, nprobe=1)
        res_probe_all = self.index.search(query, k=1, nprobe=self.nlist)

        self.assertTrue(len(res_probe1) > 0)
        self.assertTrue(len(res_probe_all) > 0)
        # Full scan distance should be <= single probe distance
        self.assertLessEqual(res_probe_all[0].distance, res_probe1[0].distance + 1e-5)

    def test_hnsw_coarse_routing(self):
        cfg = IVFPQConfig(
            dim=self.dim,
            nlist=self.nlist,
            nprobe=self.nprobe,
            M=self.M,
            K=self.K,
            metric="l2",
            use_hnsw_coarse=True,
            seed=42,
        )
        idx = IVFPQIndex(cfg)
        idx.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        self.assertIsNotNone(idx.coarse_hnsw)

        for i in range(15):
            idx.insert(self.training_data[i], payload={"id": i})

        res = idx.search(self.training_data[0], k=3)
        self.assertEqual(len(res), 3)

    def test_batch_insert_parity(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        ids = self.index.batch_insert(
            self.training_data[:10],
            payloads=[{"i": i} for i in range(10)],
        )
        self.assertEqual(ids, list(range(10)))
        self.assertEqual(len(self.index), 10)

    def test_stats_and_imbalance_factor(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        for v in self.training_data:
            self.index.insert(v)

        st = self.index.stats()
        self.assertEqual(st["total_vectors"], len(self.training_data))
        self.assertEqual(st["nlist"], self.nlist)
        self.assertEqual(st["nprobe"], self.nprobe)
        self.assertGreaterEqual(st["imbalance_factor"], 1.0)
        self.assertGreater(st["compression_ratio"], 1.0)
        self.assertEqual(self.index.get_stats()["total_vectors"], len(self.training_data))

    def test_save_and_load_roundtrip(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        for i, vec in enumerate(self.training_data[:20]):
            self.index.insert(vec, payload={"item": i})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index.reflex-ivfpq")
            self.index.save(path)
            self.assertTrue(os.path.exists(path))

            loaded = IVFPQIndex.load(path)
            self.assertEqual(len(loaded), 20)
            self.assertEqual(loaded.config.nlist, self.nlist)
            self.assertEqual(loaded.config.M, self.M)
            self.assertTrue(loaded.is_trained)

            # Query parity check
            query = self.training_data[3]
            orig_res = self.index.search(query, k=3)
            loaded_res = loaded.search(query, k=3)

            self.assertEqual(len(orig_res), len(loaded_res))
            for r1, r2 in zip(orig_res, loaded_res):
                self.assertEqual(r1.node_id, r2.node_id)
                self.assertAlmostEqual(r1.distance, r2.distance, places=4)

    def test_crc32_tamper_detection(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        self.index.insert(self.training_data[0])
        data = bytearray(self.index.save_to_bytes())

        # Corrupt one byte in the body
        data[25] ^= 0xFF
        with self.assertRaises(ValueError) as ctx:
            IVFPQIndex.load_from_bytes(bytes(data))
        self.assertIn("CRC32", str(ctx.exception))

    def test_thread_safety_concurrent_inserts(self):
        self.index.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)
        rng = random.Random(99)

        def worker(count):
            for _ in range(count):
                vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
                self.index.insert(vec, payload={"thread": threading.get_ident()})

        threads = []
        for _ in range(4):
            t = threading.Thread(target=worker, args=(15,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(self.index), 60)
        res = self.index.search([0.0] * self.dim, k=5)
        self.assertEqual(len(res), 5)

    def test_cache_with_ivfpq_integration(self):
        cache = InstinctCache(use_ivfpq=True, ivfpq_nlist=8, ivfpq_nprobe=2)
        self.assertTrue(cache.use_ivfpq)
        self.assertIsNotNone(cache._ivfpq_index)
        self.assertEqual(cache._ivfpq_index.config.nlist, 8)
        self.assertEqual(cache._ivfpq_index.config.nprobe, 2)

        st = cache.stats()
        self.assertTrue(st["use_ivfpq"])
        self.assertEqual(st["ivfpq_indexed_count"], 0)

        cache.clear()
        self.assertTrue(cache.use_ivfpq)
        self.assertIsNotNone(cache._ivfpq_index)

    def test_cosine_metric_clustering_and_search(self):
        cfg = IVFPQConfig(
            dim=self.dim,
            nlist=self.nlist,
            nprobe=self.nprobe,
            M=self.M,
            K=self.K,
            metric="cosine",
            seed=42,
        )
        idx = IVFPQIndex(cfg)
        idx.train(self.training_data, max_coarse_iters=5, max_sub_iters=5)

        for i in range(10):
            idx.insert(self.training_data[i], payload={"id": i})

        res = idx.search(self.training_data[0], k=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].payload["id"], 0)


if __name__ == "__main__":
    unittest.main()
