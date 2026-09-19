"""
Unit Tests for Hardware-Accelerated Micro-Embedding SIMD Kernel & Quantization (Phase 28).
Verifies CPU feature detection, FP32 SIMD dot products, INT8 symmetric quantization,
1-bit binary sign quantization, 4-bit nibble packing, pure-Python fallback, and model integration.
Zero external dependencies (Python standard library only).
"""

import math
import os
import unittest

from reflex import SemanticVectorEncoder, PromptSpec, InstinctCompiler
from reflex.embeddings import cosine_similarity
from reflex.simd import (
    SimdEngine,
    get_simd_engine,
    CPUFeatures,
    detect_cpu_features,
)


class TestSimdDetection(unittest.TestCase):
    """Verifies CPU capability detection across platforms."""

    def test_detect_cpu_features(self):
        features = detect_cpu_features()
        self.assertIsInstance(features, CPUFeatures)
        self.assertTrue(len(features.arch) > 0)
        d = features.to_dict()
        self.assertIn("arch", d)
        self.assertIn("has_neon", d)
        self.assertIn("has_avx2", d)
        self.assertIn("native_lib_loaded", d)

    def test_global_simd_engine_singleton(self):
        engine1 = get_simd_engine()
        engine2 = get_simd_engine()
        self.assertIs(engine1, engine2)
        self.assertIsInstance(engine1, SimdEngine)


class TestFP32DotProduct(unittest.TestCase):
    """Verifies FP32 SIMD vector math and cosine similarity."""

    def setUp(self):
        self.simd = get_simd_engine()

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            self.simd.dot_product_f32([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_numerical_accuracy(self):
        dim = 384
        v1 = [math.sin(i * 0.1) for i in range(dim)]
        v2 = [math.cos(i * 0.1) for i in range(dim)]

        expected = sum(a * b for a, b in zip(v1, v2))
        actual = self.simd.dot_product_f32(v1, v2)
        self.assertAlmostEqual(actual, expected, places=4)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0] * 192
        v2 = [0.0, 1.0] * 192
        dot = self.simd.dot_product_f32(v1, v2)
        self.assertAlmostEqual(dot, 0.0, places=5)

    def test_identical_normalized_vectors(self):
        dim = 384
        v = [1.0 / math.sqrt(dim)] * dim
        sim = self.simd.cosine_similarity(v, v)
        self.assertAlmostEqual(sim, 1.0, places=4)


class TestINT8Quantization(unittest.TestCase):
    """Verifies INT8 symmetric quantization and dot product."""

    def setUp(self):
        self.simd = get_simd_engine()

    def test_quantize_and_dot(self):
        dim = 384
        v1 = [0.02 * (i % 21 - 10) for i in range(dim)]
        v2 = [0.03 * (i % 17 - 8) for i in range(dim)]

        fp32_dot = self.simd.dot_product_f32(v1, v2)

        q1, s1 = self.simd.quantize_i8(v1)
        q2, s2 = self.simd.quantize_i8(v2)

        self.assertEqual(len(q1), dim)
        self.assertEqual(len(q2), dim)
        self.assertGreater(s1, 0.0)
        self.assertGreater(s2, 0.0)

        i8_dot = self.simd.dot_product_i8(q1, s1, q2, s2)
        # Quantization error should be very small (< 0.02)
        self.assertLess(abs(fp32_dot - i8_dot), 0.02)

    def test_zero_vector_quantization(self):
        zeros = [0.0] * 384
        q, scale = self.simd.quantize_i8(zeros)
        self.assertEqual(len(q), 384)
        self.assertEqual(bytes(q), bytes([0] * 384))


class Test1BitBinaryQuantization(unittest.TestCase):
    """Verifies 1-bit sign binarization and Hamming distance."""

    def setUp(self):
        self.simd = get_simd_engine()

    def test_binarize_and_hamming(self):
        dim = 384
        v1 = [(1.0 if i % 2 == 0 else -1.0) for i in range(dim)]
        v2 = [(-1.0 if i % 2 == 0 else 1.0) for i in range(dim)]

        b1 = self.simd.binarize_384(v1)
        b2 = self.simd.binarize_384(v2)

        # 384 dimensions = 48 bytes (6 x 64-bit words)
        self.assertEqual(len(b1), 48)
        self.assertEqual(len(b2), 48)

        # Distance to self must be 0
        dist_self = self.simd.hamming_distance_384(b1, b1)
        self.assertEqual(dist_self, 0)
        self.assertAlmostEqual(self.simd.binary_similarity_384(b1, b1), 1.0, places=4)

        # Opposite signs must yield maximum distance 384
        dist_opp = self.simd.hamming_distance_384(b1, b2)
        self.assertEqual(dist_opp, 384)
        self.assertAlmostEqual(self.simd.binary_similarity_384(b1, b2), -1.0, places=4)


class Test4BitNibbleQuantization(unittest.TestCase):
    """Verifies 4-bit nibble packing [-8, 7]."""

    def setUp(self):
        self.simd = get_simd_engine()

    def test_quantize_and_dot_i4(self):
        dim = 384
        v1 = [0.05 * (i % 15 - 7) for i in range(dim)]
        v2 = [0.03 * (i % 13 - 6) for i in range(dim)]

        fp32_dot = self.simd.dot_product_f32(v1, v2)

        packed_w, scale_w = self.simd.quantize_i4(v1)
        q2, scale_q = self.simd.quantize_i8(v2)

        # 384 dimensions pack into 192 bytes
        self.assertEqual(len(packed_w), 192)

        i4_dot = self.simd.dot_product_i4(packed_w, scale_w, q2, scale_q)
        self.assertLess(abs(fp32_dot - i4_dot), 0.05)


class TestPurePythonFallback(unittest.TestCase):
    """Verifies that all operations work identically when native C library is detached."""

    def setUp(self):
        # Force fallback by pointing to non-existent library
        self.fallback = SimdEngine(lib_path="/non_existent_library_libreflex.so")
        self.assertFalse(self.fallback.features.native_lib_loaded)

    def test_fallback_fp32_dot(self):
        v1 = [1.0, 2.0, 3.0, 4.0]
        v2 = [0.5, 1.5, 2.5, 3.5]
        dot = self.fallback.dot_product_f32(v1, v2)
        self.assertAlmostEqual(dot, 25.0, places=5)

    def test_fallback_int8(self):
        v1 = [0.1 * i for i in range(384)]
        v2 = [0.05 * i for i in range(384)]
        q1, s1 = self.fallback.quantize_i8(v1)
        q2, s2 = self.fallback.quantize_i8(v2)
        res = self.fallback.dot_product_i8(q1, s1, q2, s2)
        self.assertGreater(res, 0.0)

    def test_fallback_hamming(self):
        v = [1.0] * 384
        b = self.fallback.binarize_384(v)
        self.assertEqual(len(b), 48)
        self.assertEqual(self.fallback.hamming_distance_384(b, b), 0)


class TestEncoderAndModelIntegration(unittest.TestCase):
    """Verifies SIMD integration into SemanticVectorEncoder and CompiledInstinct."""

    def test_semantic_vector_encoder_quantization(self):
        encoder = SemanticVectorEncoder()
        text = "Unauthorized access attempt detected on port 22"

        # INT8 Quantization
        q_bytes, scale = encoder.encode_quantized_i8(text)
        self.assertEqual(len(q_bytes), 384)
        self.assertGreater(scale, 0.0)

        # 1-Bit Binary
        b_bytes = encoder.encode_binary(text)
        self.assertEqual(len(b_bytes), 48)

    def test_cosine_similarity_simd_routing(self):
        encoder = SemanticVectorEncoder()
        v1 = encoder.encode("System alert: database failure")
        v2 = encoder.encode("System alert: database crash")
        v3 = encoder.encode("Unrelated cooking recipe for apple pie")

        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)

        self.assertGreater(sim_related, sim_unrelated)
        self.assertGreater(sim_related, 0.5)

    def test_compiled_instinct_quantization(self):
        compiler = InstinctCompiler()
        spec = PromptSpec(
            name="urgency_router",
            prompt="Triage incident urgency level",
            decision_type="choice",
            options=["low", "medium", "critical"],
            guidelines={
                "low": "Documentation typo or non-urgent suggestion",
                "medium": "Slow response time or UI visual glitch",
                "critical": "Complete production outage, server down, data leak",
            },
        )
        model = compiler.compile(spec, samples_per_class=10, epochs=15)

        # Baseline prediction
        res_fp32 = model.predict("Production cluster is down and database is corrupted!")
        self.assertEqual(res_fp32.decisions["choice"].selected, "critical")

        # Quantize to INT8
        model.quantize("int8")
        self.assertEqual(model.quantization, "int8")

        # Quantized prediction should match
        res_int8 = model.predict("Production cluster is down and database is corrupted!")
        self.assertEqual(res_int8.decisions["choice"].selected, "critical")


if __name__ == "__main__":
    unittest.main()
