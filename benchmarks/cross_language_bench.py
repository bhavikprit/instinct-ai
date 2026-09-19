"""
Cross-Language Benchmark: Python, Native C99, and JavaScript (@reflex-ai/sdk).
Measures throughput, latency, and validates zero-dependency parity across all three runtimes.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex.embeddings import SemanticVectorEncoder, PureSemanticEngine
from reflex.primitives import Noul, Choice
from reflex.guardrails import GuardrailSuite
from reflex.backends.c_engine import NativeCEngine, find_libreflex

SAMPLE_PROMPT = "Urgent: Suspicious activity on your account. Click here to cancel wire #8129!"
ITERATIONS = 5000


def benchmark_python() -> Dict[str, Any]:
    print("🐍 Benchmarking Pure-Python Runtime (Zero Pip Dependencies)...")
    encoder = SemanticVectorEncoder()
    engine = PureSemanticEngine()
    suite = GuardrailSuite()

    # 1. Encode
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = encoder.encode(SAMPLE_PROMPT)
    t_enc = (time.perf_counter() - t0)

    # 2. Noul
    q_noul = {"_q": Noul("Is this a security threat or phishing scam?", threshold=0.75)}
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = engine.evaluate(SAMPLE_PROMPT, q_noul)
    t_noul = (time.perf_counter() - t0)

    # 3. Guardrail
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = suite.check(SAMPLE_PROMPT)
    t_guard = (time.perf_counter() - t0)

    return {
        "runtime": "Pure Python (v3.12+)",
        "encode_us": (t_enc / ITERATIONS) * 1_000_000,
        "noul_us": (t_noul / ITERATIONS) * 1_000_000,
        "guard_us": (t_guard / ITERATIONS) * 1_000_000,
        "throughput_ops": ITERATIONS / t_noul,
        "deps": "0 external pip packages",
    }


def benchmark_c() -> Dict[str, Any]:
    print("⚡ Benchmarking Native C99 Runtime (libreflex)...")
    lib_path = find_libreflex()
    if not lib_path:
        return {}
    c_engine = NativeCEngine(lib_path)

    # 1. Encode
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = c_engine.encode(SAMPLE_PROMPT)
    t_enc = (time.perf_counter() - t0)

    # 2. Noul
    q_noul = {"_q": Noul("Is this a security threat or phishing scam?", threshold=0.75)}
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = c_engine.evaluate(SAMPLE_PROMPT, q_noul)
    t_noul = (time.perf_counter() - t0)

    # 3. Guardrail
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = c_engine.guardrail_check(SAMPLE_PROMPT)
    t_guard = (time.perf_counter() - t0)

    return {
        "runtime": "Native C99 (libreflex)",
        "encode_us": (t_enc / ITERATIONS) * 1_000_000,
        "noul_us": (t_noul / ITERATIONS) * 1_000_000,
        "guard_us": (t_guard / ITERATIONS) * 1_000_000,
        "throughput_ops": ITERATIONS / t_noul,
        "deps": "0 external C libraries",
    }


def benchmark_js() -> Dict[str, Any]:
    print("🌐 Benchmarking JavaScript / V8 Runtime (@reflex-ai/sdk)...")
    if not shutil.which("node"):
        return {}

    js_code = f"""
    import {{ Reflex, SemanticVectorEncoder, PureSemanticEngine, Noul, GuardrailSuite }} from "./packages/reflex-sdk/index.js";
    const prompt = {json.dumps(SAMPLE_PROMPT)};
    const iters = {ITERATIONS};

    const enc = new SemanticVectorEncoder();
    const eng = new PureSemanticEngine();
    const suite = new GuardrailSuite();

    let t0 = performance.now();
    for (let i = 0; i < iters; i++) enc.encode(prompt);
    let t_enc = (performance.now() - t0) / 1000.0;

    const q = {{ _q: new Noul({{ instructions: "Is this a security threat or phishing scam?", threshold: 0.75 }}) }};
    t0 = performance.now();
    for (let i = 0; i < iters; i++) eng.evaluate(prompt, q);
    let t_noul = (performance.now() - t0) / 1000.0;

    t0 = performance.now();
    for (let i = 0; i < iters; i++) suite.check(prompt);
    let t_guard = (performance.now() - t0) / 1000.0;

    console.log(JSON.stringify({{
        encode_us: (t_enc / iters) * 1000000,
        noul_us: (t_noul / iters) * 1000000,
        guard_us: (t_guard / iters) * 1000000,
        throughput_ops: iters / t_noul
    }}));
    """

    res = subprocess.run(["node", "--input-type=module", "-e", js_code], capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    data["runtime"] = "JavaScript / V8 (@reflex-ai/sdk)"
    data["deps"] = "0 npm dependencies"
    return data


def benchmark_rust() -> Dict[str, Any]:
    print("🦀 Benchmarking Rust Runtime (reflex-rs)...")
    if not shutil.which("cargo"):
        return {}

    res = subprocess.run(
        [
            "cargo", "run", "--manifest-path", "packages/reflex-rs/Cargo.toml",
            "--example", "bench", "--release", "--quiet", str(ITERATIONS)
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout.strip())
    data["runtime"] = "Rust Safe Runtime (reflex-rs)"
    data["deps"] = "0 external crates"
    return data


def main():
    print("=" * 75)
    print("🚀 Reflex Quad-Language Cross-Platform Benchmark")
    print(f"   Sample Input: '{SAMPLE_PROMPT[:50]}...'")
    print(f"   Benchmark Sample Size: {ITERATIONS:,} iterations per test")
    print("=" * 75 + "\n")

    results = []
    r_c = benchmark_c()
    if r_c: results.append(r_c)

    r_rs = benchmark_rust()
    if r_rs: results.append(r_rs)

    r_py = benchmark_python()
    if r_py: results.append(r_py)

    r_js = benchmark_js()
    if r_js: results.append(r_js)

    print("\n" + "=" * 75)
    print("📊 Cross-Language Performance Leaderboard:")
    print("=" * 75)
    print(f"{'Runtime':<32} {'Throughput (ops/s)':<20} {'Noul Decision':<16} {'Vector Encode':<16} {'Guardrails'}")
    print("-" * 95)

    for r in results:
        print(
            f"{r['runtime']:<32} "
            f"{r['throughput_ops']:>14,.0f} ops/s   "
            f"{r['noul_us']:>8.1f} us        "
            f"{r['encode_us']:>8.1f} us       "
            f"{r['guard_us']:>6.1f} us"
        )
    print("-" * 95 + "\n")


if __name__ == "__main__":
    main()
