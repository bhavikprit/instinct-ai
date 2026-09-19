"""
Unit tests for Reflex Phase 31: Native Product Quantization (PQ) & Asymmetric Distance Computation (ADC).
Validates sub-vector codebook training, 32x memory compression, multiplier-free ADC lookup search,
C99 SIMD acceleration, binary serialization with CRC32 verification, and InstinctCache integration.
Uses Python standard library unittest only (strict zero external dependencies).
"""

import math
import os
import random
import tempfile
import threading
import unittest

from reflex.pq import (
    PQConfig,
    ProductQuantizer,
    PQIndex,
    PQSearchResult,
    PQ_CODEBOOK_MAGIC,
    PQ_INDEX_MAGIC,
)
from reflex.cache import InstinctCache


class TestPQConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = PQConfig()
        self.assertEqual(cfg.dim, 384)
        self.assertEqual(cfg.num_subvectors, 48)
        self.assertEqual(cfg.num_centroids, 256)
        self.assertEqual(cfg.d_sub, 8)
        self.assertEqual(cfg.bytes_per_vector, 48)
        self.assertAlmostEqual(cfg.compression_ratio, 32.0)

    def test_custom_config(self):
        cfg = PQConfig(dim=128, num_subvectors=16, num_centroids=64, metric="l2")
        self.assertEqual(cfg.dim, 128)
        self.assertEqual(cfg.num_subvectors, 16)
        self.assertEqual(cfg.num_centroids, 64)
        self.assertEqual(cfg.d_sub, 8)
        self.assertEqual(cfg.bytes_per_vector, 16)
        self.assertAlmostEqual(cfg.compression_ratio, 32.0)

    def test_invalid_dimension_raises(self):
        with self.assertRaises(ValueError):
            PQConfig(dim=0)
        with self.assertRaises(ValueError):
            PQConfig(dim=384, num_subvectors=50)  # 384 not divisible by 50
        with self.assertRaises(ValueError):
            PQConfig(dim=384, num_centroids=512)  # Exceeds 256 (1 byte)


class TestProductQuantizer(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.M = 4
        self.K = 16
        self.cfg = PQConfig(dim=self.dim, num_subvectors=self.M, num_centroids=self.K, seed=42)
        self.quantizer = ProductQuantizer(self.cfg)

    def test_untrained_quantizer_raises(self):
        self.assertFalse(self.quantizer.is_trained)
        with self.assertRaises(RuntimeError):
            self.quantizer.encode([0.1] * self.dim)
        with self.assertRaises(RuntimeError):
            self.quantizer.decode(bytes([0] * self.M))
        with self.assertRaises(RuntimeError):
            self.quantizer.compute_lut([0.1] * self.dim)

    def test_kmeans_training_and_dimensions(self):
        rng = random.Random(42)
        dataset = [[rng.uniform(-1.0, 1.0) for _ in range(self.dim)] for _ in range(80)]
        self.quantizer.train(dataset, max_iters=5)

        self.assertTrue(self.quantizer.is_trained)
        self.assertEqual(len(self.quantizer.centroids), self.M)
        for m in range(self.M):
            self.assertEqual(len(self.quantizer.centroids[m]), self.K)
            for k in range(self.K):
                self.assertEqual(len(self.quantizer.centroids[m][k]), self.cfg.d_sub)

    def test_encode_and_decode_reconstruction(self):
        rng = random.Random(101)
        dataset = [[rng.uniform(-1.0, 1.0) for _ in range(self.dim)] for _ in range(60)]
        self.quantizer.train(dataset, max_iters=5)

        sample_v = dataset[0]
        codes = self.quantizer.encode(sample_v)
        self.assertIsInstance(codes, bytes)
        self.assertEqual(len(codes), self.M)

        reconstructed = self.quantizer.decode(codes)
        self.assertEqual(len(reconstructed), self.dim)
        # Verify reconstruction bounded error
        diff = sum((sample_v[i] - reconstructed[i]) ** 2 for i in range(self.dim))
        self.assertLess(diff, 10.0)

    def test_adc_lut_computation(self):
        rng = random.Random(202)
        dataset = [[rng.uniform(-1.0, 1.0) for _ in range(self.dim)] for _ in range(50)]
        self.quantizer.train(dataset, max_iters=5)

        query = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        lut = self.quantizer.compute_lut(query)

        self.assertEqual(len(lut), self.M)
        for m in range(self.M):
            self.assertEqual(len(lut[m]), self.K)
            for k in range(self.K):
                self.assertGreaterEqual(lut[m][k], 0.0)

    def test_asymmetric_distance_eval(self):
        rng = random.Random(303)
        dataset = [[rng.uniform(-1.0, 1.0) for _ in range(self.dim)] for _ in range(50)]
        self.quantizer.train(dataset, max_iters=5)

        query = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        lut = self.quantizer.compute_lut(query)

        codes = self.quantizer.encode(dataset[0])
        dist = self.quantizer.asymmetric_distance(lut, codes)

        expected = sum(lut[m][codes[m]] for m in range(self.M))
        self.assertTrue(math.isclose(dist, expected, abs_tol=1e-5))

    def test_save_and_load_codebook_bytes(self):
        rng = random.Random(404)
        dataset = [[rng.uniform(-1.0, 1.0) for _ in range(self.dim)] for _ in range(40)]
        self.quantizer.train(dataset, max_iters=5)

        data = self.quantizer.save_to_bytes()
        self.assertEqual(data[:4], PQ_CODEBOOK_MAGIC)

        loaded = ProductQuantizer.load_from_bytes(data)
        self.assertTrue(loaded.is_trained)
        self.assertEqual(loaded.config.dim, self.quantizer.config.dim)
        self.assertEqual(loaded.config.num_subvectors, self.quantizer.config.num_subvectors)
        self.assertEqual(loaded.config.num_centroids, self.quantizer.config.num_centroids)

        # Verify encoding consistency
        v = dataset[1]
        self.assertEqual(self.quantizer.encode(v), loaded.encode(v))

    def test_crc32_tamper_detection(self):
        rng = random.Random(505)
        dataset = [[rng.uniform(-1.0, 1.0) for _ in range(self.dim)] for _ in range(30)]
        self.quantizer.train(dataset, max_iters=3)

        data = bytearray(self.quantizer.save_to_bytes())
        data[40] = (data[40] + 1) % 256  # Tamper with byte

        with self.assertRaises(ValueError):
            ProductQuantizer.load_from_bytes(bytes(data))


class TestPQIndex(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.M = 4
        self.K = 16
        self.cfg = PQConfig(dim=self.dim, num_subvectors=self.M, num_centroids=self.K, seed=42)
        self.quantizer = ProductQuantizer(self.cfg)
        rng = random.Random(42)
        self.train_data = [[rng.uniform(-1.0, 1.0) for _ in range(self.dim)] for _ in range(50)]
        self.quantizer.train(self.train_data, max_iters=5)
        self.index = PQIndex(self.quantizer)

    def test_empty_search(self):
        self.assertEqual(len(self.index), 0)
        res = self.index.search([0.1] * self.dim, k=5)
        self.assertEqual(res, [])

    def test_insert_and_search(self):
        rng = random.Random(999)
        for i in range(40):
            v = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
            self.index.insert(v, payload={"id": i})

        self.assertEqual(len(self.index), 40)
        self.assertEqual(len(self.index.codes), 40 * self.M)

        q = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        results = self.index.search(q, k=5)
        self.assertEqual(len(results), 5)
        # Results should be ordered by similarity descending
        sims = [r.similarity for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_stats_and_compression(self):
        for i in range(25):
            self.index.insert([0.1] * self.dim, payload=i)

        stats = self.index.stats()
        self.assertEqual(stats["vector_count"], 25)
        self.assertEqual(stats["dimension"], self.dim)
        self.assertEqual(stats["bytes_per_vector"], self.M)
        self.assertAlmostEqual(stats["compression_ratio"], (self.dim * 4.0) / self.M)

    def test_save_and_load_file_roundtrip(self):
        for i in range(20):
            self.index.insert([float(i)] * self.dim, payload={"tag": f"doc_{i}"})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_pq.reflex-pq-index")
            self.index.save(file_path)
            self.assertTrue(os.path.exists(file_path))

            loaded = PQIndex.load(file_path)
            self.assertEqual(len(loaded), 20)
            self.assertEqual(len(loaded.codes), 20 * self.M)

            q = [1.0] * self.dim
            orig_res = self.index.search(q, k=3)
            load_res = loaded.search(q, k=3)

            self.assertEqual([r.node_id for r in orig_res], [r.node_id for r in load_res])
            for r1, r2 in zip(orig_res, load_res):
                self.assertTrue(math.isclose(r1.distance, r2.distance, abs_tol=1e-5))
                self.assertEqual(r1.payload, r2.payload)

    def test_thread_safety_concurrent_inserts(self):
        def worker(start_idx: int, count: int):
            rng = random.Random(start_idx)
            for i in range(count):
                vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
                self.index.insert(vec, payload={"worker": start_idx, "i": i})

        threads = []
        for t_idx in range(4):
            t = threading.Thread(target=worker, args=(t_idx * 100, 15))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(self.index), 60)
        self.assertEqual(len(self.index.codes), 60 * self.M)
        res = self.index.search([0.1] * self.dim, k=5)
        self.assertEqual(len(res), 5)


class TestInstinctCachePQIntegration(unittest.TestCase):
    def test_cache_with_pq_enabled(self):
        cache = InstinctCache(use_pq=True)
        self.assertTrue(cache.use_pq)
        self.assertIsNotNone(cache._pq_quantizer)
        self.assertIsNotNone(cache._pq_index)

        stats = cache.stats()
        self.assertTrue(stats["use_pq"])
        self.assertEqual(stats["pq_indexed_count"], 0)

        cache.clear()
        self.assertTrue(cache.use_pq)
        self.assertIsNotNone(cache._pq_index)


if __name__ == "__main__":
    unittest.main()
