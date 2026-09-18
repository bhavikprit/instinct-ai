"""
Unit tests for Reflex InstinctCache (L1 exact & L2 semantic vector memory).
"""

import os
import tempfile
import time
import unittest

from reflex.cache import InstinctCache
from reflex.client import Reflex
from reflex.primitives import Noul, Choice, DecisionResult


class TestInstinctCache(unittest.TestCase):

    def setUp(self):
        self.cache = InstinctCache(max_size=5, similarity_threshold=0.85)

    def test_exact_hit(self):
        questions = {"is_urgent": Noul("Is this an urgent production issue?")}
        mock_result = DecisionResult(
            decisions={"is_urgent": Noul(instructions="Is this an urgent issue?", probability=0.95)},
            latency_ms=1.2,
            backend="local",
        )
        state = "Database CPU spike to 99%"
        self.cache.set(state, questions, mock_result)

        hit = self.cache.get(state, questions)
        self.assertIsNotNone(hit)
        self.assertTrue(hit.cached)
        self.assertEqual(hit.backend, "local (cached)")
        self.assertAlmostEqual(hit.decisions["is_urgent"].probability, 0.95)
        self.assertEqual(self.cache.exact_hits, 1)

    def test_semantic_hit(self):
        questions = {"is_urgent": Noul("Is this an urgent production issue?")}
        mock_result = DecisionResult(
            decisions={"is_urgent": Noul(instructions="Is this an urgent issue?", probability=0.92)},
            latency_ms=2.0,
            backend="local",
        )
        state_original = "Critical PostgreSQL replica timeout on port 5432"
        self.cache.set(state_original, questions, mock_result)

        # Semantically overlapping query with high similarity
        state_rephrased = "Critical PostgreSQL replica timeout on port 5432! Server down"
        hit = self.cache.get(state_rephrased, questions)

        self.assertIsNotNone(hit)
        self.assertTrue(hit.cached)
        self.assertEqual(self.cache.semantic_hits, 1)
        self.assertAlmostEqual(hit.decisions["is_urgent"].probability, 0.92)

    def test_cache_miss(self):
        questions = {"is_urgent": Noul("Is this an urgent production issue?")}
        mock_result = DecisionResult(
            decisions={"is_urgent": Noul(instructions="Is this an urgent issue?", probability=0.9)},
            latency_ms=1.0,
            backend="local",
        )
        self.cache.set("System outage and crash", questions, mock_result)

        unrelated = "How do I bake chocolate chip cookies?"
        miss = self.cache.get(unrelated, questions)
        self.assertIsNone(miss)
        self.assertEqual(self.cache.misses, 1)

    def test_lru_eviction(self):
        small_cache = InstinctCache(max_size=2)
        q = {"is_spam": Noul("Is this spam?")}
        res = DecisionResult(decisions={}, latency_ms=0.5, backend="local")

        small_cache.set("query 1", q, res)
        small_cache.set("query 2", q, res)
        small_cache.set("query 3", q, res)

        self.assertEqual(len(small_cache._entries), 2)
        self.assertEqual(small_cache.evictions, 1)
        # "query 1" should have been evicted
        self.assertIsNone(small_cache.get("query 1", q))

    def test_ttl_expiration(self):
        ttl_cache = InstinctCache(ttl_seconds=0.05)
        q = {"q": Noul("Question?")}
        res = DecisionResult(decisions={}, latency_ms=0.1, backend="local")

        ttl_cache.set("temporary state", q, res)
        self.assertIsNotNone(ttl_cache.get("temporary state", q))

        time.sleep(0.06)
        self.assertIsNone(ttl_cache.get("temporary state", q))

    def test_persistence_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            q = {"is_refund": Noul("Is refund requested?")}
            res = DecisionResult(
                decisions={"is_refund": Noul(instructions="Is refund?", probability=0.88)},
                latency_ms=0.8,
                backend="semantic",
            )
            self.cache.set("I want my money back for order 99", q, res)
            self.cache.save_to_file(tmp_path)

            new_cache = InstinctCache()
            new_cache.load_from_file(tmp_path)
            self.assertEqual(len(new_cache._entries), 1)

            hit = new_cache.get("I want my money back for order 99", q)
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(hit.decisions["is_refund"].probability, 0.88)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_reflex_client_cache_integration(self):
        rx = Reflex(backend="local", cache=True)
        q = "Is this an emergency incident?"
        state = "Datacenter on fire, servers unresponsive!"

        # First pass -> cache miss, evaluates on backend
        res1 = rx.evaluate(state, {"alert": Noul(q)})
        self.assertFalse(res1.cached)

        # Second pass with exact state -> cache hit (<0.02ms)
        res2 = rx.evaluate(state, {"alert": Noul(q)})
        self.assertTrue(res2.cached)
        self.assertIn("(cached)", res2.backend)
        self.assertEqual(rx.cache.exact_hits, 1)


if __name__ == "__main__":
    unittest.main()
