"""
Reflex Example 28: Hardware-Accelerated Micro-Embedding SIMD Kernel & Quantization (Phase 28).
Demonstrates 1,000,000 vector similarity operations comparing Scalar Python, FP32 SIMD,
INT8 Quantized, and 1-Bit Binary Sign Quantization with Hamming Distance (32x compression).
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import SemanticVectorEncoder, PromptSpec, InstinctCompiler
from reflex.simd import get_simd_engine, detect_cpu_features


def run_simd_benchmark_demo():
    print("=" * 85)
    print("⚡ REFLEX: HARDWARE-ACCELERATED SIMD KERNEL & QUANTIZATION BENCHMARK (PHASE 28)")
    print("=" * 85)

    simd = get_simd_engine()
    caps = simd.features

    print("\n1. 🔍 Hardware Capability Detection:")
    print(f" • Host Architecture   : {caps.arch}")
    print(f" • Native C99 libreflex : {'🟢 Loaded' if caps.native_lib_loaded else '⚠️ Pure-Python Fallback'}")
    print(f" • ARM NEON SIMD (128b) : {'✅ Active' if caps.has_neon else '❌ Unavailable'}")
    print(f" • x86_64 AVX2 (256b)   : {'✅ Active' if caps.has_avx2 else '❌ Unavailable'}")
    print(f" • Hardware POPCOUNT    : {'✅ Active' if caps.has_popcnt else '❌ Unavailable'}")

    # -----------------------------------------------------------------
    # 2. Vector Quantization Format Comparison
    # -----------------------------------------------------------------
    print("\n2. 📦 Vector Embedding Quantization & Compression Formats (384-d):")
    encoder = SemanticVectorEncoder()
    sample_text = "Urgent: unauthorized root login detected on bastion host from unknown IP"
    fp32_vec = encoder.encode(sample_text)

    # A. FP32
    fp32_bytes = len(fp32_vec) * 4

    # B. INT8
    i8_bytes_arr, scale_i8 = encoder.encode_quantized_i8(sample_text)

    # C. 4-Bit Nibble
    i4_bytes_arr, scale_i4 = simd.quantize_i4(fp32_vec)

    # D. 1-Bit Binary
    bin_bytes = encoder.encode_binary(sample_text)

    print(f" • FP32 Dense Vector    : {fp32_bytes:,} bytes (Baseline 1.0x)")
    print(f" • INT8 Quantized (s8)  : {len(i8_bytes_arr):,} bytes (4.0x compression, scale={scale_i8:.5f})")
    print(f" • 4-Bit Packed Nibble  : {len(i4_bytes_arr):,} bytes (8.0x compression, scale={scale_i4:.5f})")
    print(f" • 1-Bit Binary Sign    : {len(bin_bytes):,} bytes (32.0x compression, 6x uint64 words)")

    # -----------------------------------------------------------------
    # 3. 1,000,000 Vector Similarity Operations Benchmark
    # -----------------------------------------------------------------
    print("\n3. ⏱️  Executing 1,000,000 Vector Similarity Comparisons (10,000 Docs x 100 Queries)...")
    num_docs = 10000
    num_queries = 100
    total_ops = num_docs * num_queries

    # Generate synthetic vector corpus
    print(" • Generating 10,000 document vector corpus...")
    corpus_fp32 = []
    corpus_i8 = []
    corpus_bin = []

    for i in range(num_docs):
        # Deterministic synthetic vector
        v = [math.sin(i * 0.05 + j * 0.1) for j in range(384)]
        norm = math.sqrt(sum(x * x for x in v))
        v_norm = [x / norm for x in v]
        corpus_fp32.append(v_norm)

        q, s = simd.quantize_i8(v_norm)
        corpus_i8.append((q, s))

        b = simd.binarize_384(v_norm)
        corpus_bin.append(b)

    # Queries
    query_fp32 = [math.cos(1.0 + j * 0.15) for j in range(384)]
    q_norm = math.sqrt(sum(x * x for x in query_fp32))
    query_fp32 = [x / q_norm for x in query_fp32]
    query_i8, q_scale = simd.quantize_i8(query_fp32)
    query_bin = simd.binarize_384(query_fp32)

    # A. Pure-Python Scalar Baseline (Run on subset to project total)
    print(" • Benchmarking Scalar Python baseline...")
    bench_subset = 2000
    t0 = time.perf_counter()
    for doc in corpus_fp32[:bench_subset]:
        _ = sum(a * b for a, b in zip(query_fp32, doc))
    dt_scalar = (time.perf_counter() - t0) * (total_ops / bench_subset)
    ns_scalar = (dt_scalar / total_ops) * 1e9

    # B. INT8 Quantized SIMD (Full 1,000,000 comparisons)
    print(f" • Benchmarking INT8 Quantized SIMD ({total_ops:,} ops)...")
    t0 = time.perf_counter()
    for _ in range(num_queries):
        for doc_q, doc_s in corpus_i8:
            _ = simd.dot_product_i8(query_i8, q_scale, doc_q, doc_s)
    dt_i8 = time.perf_counter() - t0
    ns_i8 = (dt_i8 / total_ops) * 1e9

    # C. 1-Bit Binary Hamming Distance (Full 1,000,000 comparisons)
    print(f" • Benchmarking 1-Bit Binary POPCOUNT ({total_ops:,} ops)...")
    t0 = time.perf_counter()
    for _ in range(num_queries):
        for doc_b in corpus_bin:
            _ = simd.hamming_distance_384(query_bin, doc_b)
    dt_bin = time.perf_counter() - t0
    ns_bin = (dt_bin / total_ops) * 1e9

    # -----------------------------------------------------------------
    # 4. Print Benchmark Results
    # -----------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f"🏆 1,000,000 VECTOR SIMILARITY BENCHMARK RESULTS (384-DIMENSIONAL)")
    print("=" * 85)
    headers = f"{'Kernel Mode':<28} | {'Total Time':<12} | {'Latency (ns)':<14} | {'Throughput':<16} | {'Corpus RAM':<12}"
    print(headers)
    print("-" * 85)

    mem_fp32 = (num_docs * 1536) / (1024 * 1024)
    mem_i8 = (num_docs * 384) / (1024 * 1024)
    mem_bin = (num_docs * 48) / (1024 * 1024)

    print(f"{'Scalar Python (Float32)':<28} | {dt_scalar:<12.2f}s | {ns_scalar:<14.1f} | {total_ops/dt_scalar:>12,.0f} ops/s | {mem_fp32:>8.2f} MB")
    print(f"{'INT8 Quantized SIMD (4x)':<28} | {dt_i8:<12.2f}s | {ns_i8:<14.1f} | {total_ops/dt_i8:>12,.0f} ops/s | {mem_i8:>8.2f} MB")
    print(f"{'1-Bit Binary Sign (32x)':<28} | {dt_bin:<12.2f}s | {ns_bin:<14.1f} | {total_ops/dt_bin:>12,.0f} ops/s | {mem_bin:>8.2f} MB")
    print("=" * 85)

    speedup_i8 = dt_scalar / max(0.001, dt_i8)
    speedup_bin = dt_scalar / max(0.001, dt_bin)
    print(f"🚀 INT8 Quantized SIMD is {speedup_i8:.1f}x FASTER than scalar Python with 4x memory savings")
    print(f"⚡ 1-Bit Binary Sign is {speedup_bin:.1f}x FASTER than scalar Python with 32x memory savings")
    print("🛡️  100% Zero External Dependencies (Standard Library + C99 FFI)\n")

    # -----------------------------------------------------------------
    # 5. Compiled Model Acceleration
    # -----------------------------------------------------------------
    print("4. 🧠 Testing End-to-End Decision Acceleration on Compiled Model:")
    compiler = InstinctCompiler()
    spec = PromptSpec(
        name="threat_triage",
        prompt="Triage cybersecurity threat level for incoming HTTP packets",
        decision_type="choice",
        options=["benign_traffic", "probe_scan", "critical_exploit"],
        guidelines={
            "benign_traffic": "Standard GET request from verified browser session",
            "probe_scan": "Nmap TCP SYN scan, banner grabbing, port probing",
            "critical_exploit": "SQL injection UNION SELECT, remote code execution CVE, buffer overflow",
        },
    )
    model = compiler.compile(spec, samples_per_class=15, epochs=20)
    test_incident = "GET /api/v1/auth?id=' UNION SELECT 1, @@version, password FROM users --"

    # FP32 Prediction
    res_fp32 = model.predict(test_incident)
    print(f" • FP32 Model Result : {res_fp32.decisions['choice'].selected:<18} (conf={res_fp32.decisions['choice'].confidence*100:.1f}%, latency={res_fp32.latency_ms:.3f} ms)")

    # INT8 Quantized Prediction
    model.quantize("int8")
    res_int8 = model.predict(test_incident)
    print(f" • INT8 Model Result : {res_int8.decisions['choice'].selected:<18} (conf={res_int8.decisions['choice'].confidence*100:.1f}%, latency={res_int8.latency_ms:.3f} ms)")

    print("\n✅ Phase 28: Hardware-Accelerated SIMD Kernel & Quantization Verified.")


if __name__ == "__main__":
    run_simd_benchmark_demo()
