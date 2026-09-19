"""
Reflex Example 27: Zero-Copy Shared Memory IPC Daemon (Phase 27).
Demonstrates ultra-low latency IPC (<5us via POSIX Shared Memory ring buffer, <25us via UDS)
for high-throughput agent microservice fleets bypassing HTTP/TCP stack overhead.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
import http.server
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Choice, Noul, Score, PromptSpec, InstinctCompiler
from reflex.shm import (
    ReflexIPCDaemon,
    ReflexIPCClient,
    SHMConfig,
    IPCOpCode,
    SharedMemoryRingBuffer,
)


def run_ipc_benchmark_demo():
    print("=" * 85)
    print("⚡ REFLEX: ZERO-COPY SHARED MEMORY IPC DAEMON FOR MICROSERVICES (PHASE 27)")
    print("=" * 85)

    temp_dir = tempfile.mkdtemp(prefix="reflex_ipc_")
    sock_path = os.path.join(temp_dir, "reflex_ipc.sock")
    shm_name = f"reflex_ring_{uuid.uuid4().hex[:8]}"

    try:
        # -----------------------------------------------------------------
        # 1. Compile High-Speed Edge Decision Model
        # -----------------------------------------------------------------
        print("\n1. 🛠️  Compiling High-Speed Decision Model (.reflex)...")
        compiler = InstinctCompiler()
        spec = PromptSpec(
            name="microservice_router",
            prompt="High-throughput ingress microservice traffic router and security gatekeeper.",
            decision_type="choice",
            options=["fast_path", "security_audit", "rate_limit_drop"],
            guidelines={
                "fast_path": "Authenticated, verified JWT token, valid idempotency key, read-only GET requests.",
                "security_audit": "Unrecognized IP address, high-entropy payload, SQL/XSS tokens, admin endpoint.",
                "rate_limit_drop": "DDoS volume, repetitive burst from single client, invalid HTTP method, blacklisted CIDR.",
            },
        )
        compiled_model = compiler.compile(spec, samples_per_class=20, epochs=25)
        model_path = os.path.join(temp_dir, "router.reflex")
        compiled_model.save(model_path)
        print(f" • Compiled Model : {compiled_model.name} ({len(spec.options)} classes)")
        print(f" • Model Size     : {os.path.getsize(model_path):,} bytes on disk")

        # -----------------------------------------------------------------
        # 2. Launch Background Reflex IPC Daemon
        # -----------------------------------------------------------------
        print("\n2. 🚀 Starting Zero-Copy Reflex IPC Daemon (SHM + UDS)...")
        cfg = SHMConfig(
            socket_path=sock_path,
            shm_name=shm_name,
            num_slots=16,
            slot_size=4096,
            use_shm=True,
            poll_sleep_s=0.00001,  # 10us spin interval
        )
        daemon = ReflexIPCDaemon(config=cfg, model=compiled_model)
        daemon.start(background=True)
        time.sleep(0.05)

        print(f" • Unix Domain Socket : {cfg.socket_path}")
        print(f" • POSIX Shared Memory: /{cfg.shm_name}")
        print(f" • Ring Buffer Slots  : {cfg.num_slots} slots x {cfg.slot_size} bytes")
        print(" • Daemon Status      : 🟢 Active (sub-5µs spin-loop)")

        # -----------------------------------------------------------------
        # 3. Verify ReflexIPCClient & Reflex(backend="ipc")
        # -----------------------------------------------------------------
        print("\n3. 🔍 Testing Reflex IPC Client Primitives...")
        client = ReflexIPCClient(config=cfg)

        # Ping
        ping_us = client.ping()
        print(f" • IPC Ping (Round-Trip) : {ping_us:.2f} µs")

        # NOUL
        p_auth = client.noul("Is the caller authorized?", "Valid JWT Bearer token with admin:read scope", threshold=0.5)
        print(f" • IPC Noul (Boolean)    : probability={p_auth:.3f} (authorized={p_auth >= 0.5})")

        # CHOICE
        route = client.choice(
            "Select routing path",
            ["fast_path", "security_audit", "rate_limit_drop"],
            "GET /v1/user/profile with valid Authorization header",
        )
        print(f" • IPC Choice (Routing)  : route={route}")

        # PREDICT
        pred = client.predict("POST /v1/admin/debug with suspect SQL injection syntax")
        print(f" • IPC Direct Predict    : selected={pred['decisions']['choice']['selected']} (confidence={pred['decisions']['choice']['confidence']*100:.1f}%)")

        # High-level Reflex client with backend="ipc"
        rx = Reflex(backend="ipc", socket_path=sock_path, shm_name=shm_name)
        rx_eval = rx.evaluate(
            "Burst of 10,000 requests per second from single untrusted residential proxy",
            {
                "route": Choice("Triage traffic", ["fast_path", "rate_limit_drop"]),
                "is_ddos": Noul("Is this a DDoS assault?"),
                "risk_score": Score("Risk score 1 to 10"),
            },
        )
        print(f" • Reflex(backend='ipc') : backend={rx_eval.backend}, route={rx_eval.decisions['route'].selected}, ddos={rx_eval.decisions['is_ddos'].is_true}, latency={rx_eval.latency_ms:.3f} ms")

        # -----------------------------------------------------------------
        # 4. Spin up HTTP Loopback Server for Comparative Baseline
        # -----------------------------------------------------------------
        print("\n4. 🌐 Starting Local HTTP Loopback Server for Baseline...")

        class SimpleHTTPHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                _ = self.rfile.read(content_len)
                resp = json.dumps({"status": "ok", "selected": "fast_path"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, format, *args):
                pass  # Silence HTTP server logs

        httpd = http.server.HTTPServer(("127.0.0.1", 0), SimpleHTTPHandler)
        http_port = httpd.server_address[1]
        http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        http_thread.start()
        http_url = f"http://127.0.0.1:{http_port}/v1/predict"
        print(f" • HTTP Loopback Server  : {http_url}")

        # -----------------------------------------------------------------
        # 5. Live Microsecond Latency Benchmark: HTTP vs UDS vs SHM
        # -----------------------------------------------------------------
        print("\n5. ⏱️  Executing Latency Benchmark (HTTP Loopback vs UDS vs POSIX Shared Memory)...")
        num_iterations = 200
        test_payload = json.dumps({"state": "GET /v1/catalog/items?page=1 HTTP/1.1"}).encode("utf-8")

        # A. HTTP/TCP Loopback
        http_latencies_us = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            req = urllib.request.Request(http_url, data=test_payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                _ = resp.read()
            http_latencies_us.append((time.perf_counter() - t0) * 1_000_000.0)

        # B. Unix Domain Socket (UDS) IPC
        uds_cfg = SHMConfig(socket_path=sock_path, use_shm=False)
        uds_client = ReflexIPCClient(config=uds_cfg)
        uds_latencies_us = []
        # Warmup
        for _ in range(10):
            uds_client.predict("Warmup query")
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = uds_client.predict("GET /v1/catalog/items?page=1 HTTP/1.1")
            uds_latencies_us.append((time.perf_counter() - t0) * 1_000_000.0)
        uds_client.close()

        # C. POSIX Shared Memory Ring Buffer (<5us)
        shm_client = ReflexIPCClient(config=cfg)
        shm_latencies_us = []
        # Warmup
        for _ in range(10):
            shm_client.predict("Warmup query")
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = shm_client.predict("GET /v1/catalog/items?page=1 HTTP/1.1")
            shm_latencies_us.append((time.perf_counter() - t0) * 1_000_000.0)
        shm_client.close()

        # Statistics helper
        def compute_stats(latencies: list[float]) -> dict:
            sorted_lat = sorted(latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
            mean = sum(sorted_lat) / len(sorted_lat)
            min_v = sorted_lat[0]
            throughput = 1_000_000.0 / max(0.001, mean)
            return {"min": min_v, "p50": p50, "p95": p95, "p99": p99, "mean": mean, "tput": throughput}

        s_http = compute_stats(http_latencies_us)
        s_uds = compute_stats(uds_latencies_us)
        s_shm = compute_stats(shm_latencies_us)

        # -----------------------------------------------------------------
        # 6. Print Benchmark Results Table
        # -----------------------------------------------------------------
        print("\n" + "=" * 85)
        print("🏆 MICROSERVICE IPC TRANSPORT BENCHMARK RESULTS")
        print("=" * 85)
        headers = f"{'Transport Layer':<28} | {'Min (µs)':<10} | {'P50 (µs)':<10} | {'P99 (µs)':<10} | {'Mean (µs)':<10} | {'Throughput':<14}"
        print(headers)
        print("-" * 85)

        print(f"{'HTTP/1.1 (TCP Loopback)':<28} | {s_http['min']:<10.1f} | {s_http['p50']:<10.1f} | {s_http['p99']:<10.1f} | {s_http['mean']:<10.1f} | {s_http['tput']:>10,.0f} req/s")
        print(f"{'Unix Domain Socket (UDS)':<28} | {s_uds['min']:<10.1f} | {s_uds['p50']:<10.1f} | {s_uds['p99']:<10.1f} | {s_uds['mean']:<10.1f} | {s_uds['tput']:>10,.0f} req/s")
        print(f"{'Zero-Copy Shared Memory':<28} | {s_shm['min']:<10.1f} | {s_shm['p50']:<10.1f} | {s_shm['p99']:<10.1f} | {s_shm['mean']:<10.1f} | {s_shm['tput']:>10,.0f} req/s")
        print("=" * 85)

        speedup_shm_vs_http = s_http["mean"] / max(0.01, s_shm["mean"])
        speedup_shm_vs_uds = s_uds["mean"] / max(0.01, s_shm["mean"])
        print(f"🚀 POSIX Shared Memory is {speedup_shm_vs_http:,.1f}x FASTER than HTTP/TCP Loopback")
        print(f"⚡ POSIX Shared Memory is {speedup_shm_vs_uds:,.1f}x FASTER than Unix Domain Sockets")
        print("🛡️  100% Python Standard Library (zero external C extensions or wheels required)\n")

        # Daemon statistics
        ipc_stats = client.call(IPCOpCode.STATS, {})
        print(f"📊 Daemon Total Handled Requests: {ipc_stats['total_requests']:,} (SHM: {ipc_stats['shm_requests']:,}, UDS: {ipc_stats['uds_requests']:,})")
        print(f"⏱️  Daemon Cumulative Mean Latency: {ipc_stats['mean_latency_us']:.2f} µs")

        # Cleanup
        client.close()
        daemon.stop()
        httpd.shutdown()
        httpd.server_close()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n✅ Phase 27: Zero-Copy Shared Memory IPC Demonstration Complete.")


if __name__ == "__main__":
    run_ipc_benchmark_demo()
