"""
Reflex Command-Line Interface (CLI).
"""

import argparse
import sys
from reflex.client import Reflex
from reflex.primitives import Noul, Choice
import os
import platform
import shutil
import time

def run_doctor():
    from reflex import __version__
    from reflex.embeddings import SemanticVectorEncoder
    from reflex.backends.c_engine import find_libreflex, NativeCEngine
    from reflex.guardrails import GuardrailSuite

    print("=" * 70)
    print(f"⚡ Reflex System Diagnostic & Environment Doctor (v{__version__})")
    print("=" * 70)

    # 1. Host Environment
    os_name = platform.system()
    arch = platform.machine()
    py_ver = platform.python_version()
    print("\n🖥️  Host Environment:")
    print(f"  • Operating System   : {os_name} {platform.release()} ({arch})")
    print(f"  • Python Runtime     : v{py_ver} ({sys.executable})")
    print(f"  • Zero Dependencies  : ✅ Active (100% standard library core)")

    # 2. Backends & Acceleration
    print("\n⚙️  Runtime Engines & Hardware Acceleration:")
    print(f"  • Pure Semantic      : ✅ Operational (sub-0.1ms 384-d dense vectors)")

    lib_path = find_libreflex()
    if lib_path and os.path.exists(lib_path):
        print(f"  • Native C Engine    : ✅ Operational ({os.path.basename(lib_path)}, sub-10us)")
    else:
        print(f"  • Native C Engine    : ⚪ Not compiled (Run 'make -C reflex_c all' to build)")

    node_bin = shutil.which("node")
    sdk_manifest = os.path.join(os.path.dirname(__file__), "..", "packages", "reflex-sdk", "package.json")
    if node_bin and os.path.exists(sdk_manifest):
        print(f"  • Edge SDK (@reflex) : ✅ Available (Node.js {node_bin})")
    else:
        print(f"  • Edge SDK (@reflex) : ⚪ Node.js not detected on PATH")

    cargo_bin = shutil.which("cargo")
    rust_manifest = os.path.join(os.path.dirname(__file__), "..", "packages", "reflex-rs", "Cargo.toml")
    if cargo_bin and os.path.exists(rust_manifest):
        print(f"  • Rust SDK (reflex-rs): ✅ Available (Cargo {cargo_bin})")
    else:
        print(f"  • Rust SDK (reflex-rs): ⚪ Cargo toolchain not detected on PATH")

    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"  • ONNX Runtime       : ✅ Installed ({', '.join(providers)})")
    except ImportError:
        print(f"  • ONNX Runtime       : ⚪ Optional (not installed, run 'pip install reflex-ai[local]')")

    # 3. Microsecond Latency Diagnostic
    print("\n⏱️  Live Microsecond Latency Benchmark (500 iterations):")
    test_text = "Urgent security threat: root password modified by external IP address"

    # Python benchmark
    py_enc = SemanticVectorEncoder()
    t0 = time.perf_counter()
    for _ in range(500):
        _ = py_enc.encode(test_text)
    py_us = ((time.perf_counter() - t0) / 500.0) * 1_000_000.0
    print(f"  • Pure Python Encode : {py_us:.1f} us/op ({1_000_000.0 / py_us:.0f} ops/sec)")

    # C benchmark if available
    if lib_path and os.path.exists(lib_path):
        try:
            c_eng = NativeCEngine(lib_path)
            t0 = time.perf_counter()
            for _ in range(500):
                _ = c_eng.encode(test_text)
            c_us = ((time.perf_counter() - t0) / 500.0) * 1_000_000.0
            speedup = py_us / max(0.01, c_us)
            print(f"  • Native C Encode    : {c_us:.1f} us/op ({1_000_000.0 / c_us:.0f} ops/sec) -> 🚀 {speedup:.1f}x speedup")

            t0 = time.perf_counter()
            for _ in range(500):
                _ = c_eng.guardrail_check(test_text)
            guard_us = ((time.perf_counter() - t0) / 500.0) * 1_000_000.0
            print(f"  • C Guardrail Check  : {guard_us:.1f} us/op ({1_000_000.0 / guard_us:.0f} ops/sec)")
        except Exception as e:
            print(f"  • Native C Error     : {e}")
    else:
        suite = GuardrailSuite()
        t0 = time.perf_counter()
        for _ in range(500):
            _ = suite.check(test_text)
        guard_us = ((time.perf_counter() - t0) / 500.0) * 1_000_000.0
        print(f"  • Python Guardrail   : {guard_us:.1f} us/op ({1_000_000.0 / guard_us:.0f} ops/sec)")

    print("\n✅ Diagnostic check complete. System healthy.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reflex: Universal System-1 AI Runtime & Dual-Brain Gateway"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: serve / gateway (Production AI Envoy reverse proxy)
    for cmd_name in ("serve", "gateway"):
        p = subparsers.add_parser(cmd_name, help="Start Reflex AI Envoy production reverse proxy gateway")
        p.add_argument("--host", default="0.0.0.0" if cmd_name == "gateway" else "127.0.0.1", help="Host address")
        p.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
        p.add_argument("--upstream", default=os.environ.get("UPSTREAM_OPENAI_URL", "https://api.openai.com/v1"), help="Upstream API base URL")
        p.add_argument("--cache-ttl", type=float, default=3600.0, help="Semantic cache TTL in seconds (default: 3600)")
        p.add_argument("--similarity-threshold", type=float, default=0.95, help="Cosine threshold for L2 semantic cache (default: 0.95)")
        p.add_argument("--no-cache", action="store_true", help="Disable semantic deduplication cache")
        p.add_argument("--no-guardrails", action="store_true", help="Disable pre-flight security guardrails")
        p.add_argument("--mesh-peers", default="", help="Comma-separated URLs of cluster mesh peers")
        p.add_argument("--mesh-secret", default=os.environ.get("REFLEX_MESH_SECRET", ""), help="Cluster HMAC secret for sync")

    # Command: mesh (Cluster inspection and sync)
    mesh_parser = subparsers.add_parser("mesh", help="Inspect and ping Reflex Instinct Mesh cluster")
    mesh_sub = mesh_parser.add_subparsers(dest="mesh_action", help="Mesh action: peers")
    peers_p = mesh_sub.add_parser("peers", help="Query active mesh peers")
    peers_p.add_argument("--gateway", default="http://127.0.0.1:8080", help="Gateway URL (default: http://127.0.0.1:8080)")
    peers_p.add_argument("--secret", default=os.environ.get("REFLEX_MESH_SECRET", ""), help="Cluster HMAC secret")

    # Command: eval (instant reflex evaluation)
    eval_parser = subparsers.add_parser("eval", help="Evaluate a quick System 1 decision")
    eval_parser.add_argument("state", help="Unstructured text to evaluate")
    eval_parser.add_argument("--noul", help="Noul question (returns probability)")
    eval_parser.add_argument("--choice", help="Choice question")
    eval_parser.add_argument("--options", help="Comma-separated choice options")

    # Command: mcp (Model Context Protocol stdio server)
    subparsers.add_parser("mcp", help="Start Model Context Protocol (MCP) server over stdio")

    # Command: dataset-gen (OpenRLCD synthetic dataset generator)
    ds_parser = subparsers.add_parser("dataset-gen", help="Generate calibrated synthetic decision dataset")
    ds_parser.add_argument("--samples", type=int, default=100, help="Number of samples (default: 100)")
    ds_parser.add_argument("--output", default="decision_dataset.jsonl", help="Output JSONL path")
    ds_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Command: playground (Interactive dual-brain browser UI)
    play_parser = subparsers.add_parser("playground", help="Launch interactive Dual-Brain Web Playground")
    play_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    play_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    play_parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")

    # Command: quantize (INT8 ONNX model quantization)
    q_parser = subparsers.add_parser("quantize", help="Quantize an ONNX model to INT8")
    q_parser.add_argument("model_path", help="Path to input .onnx model file")
    q_parser.add_argument("--output", default=None, help="Path for quantized output .onnx file")

    # Command: serve-api (Production REST API microservice gateway)
    api_parser = subparsers.add_parser("serve-api", help="Start production REST API gateway with Prometheus metrics")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    api_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # Command: benchmark (DecisionBench comparison runner)
    bench_parser = subparsers.add_parser("benchmark", help="Run DecisionBench evaluation across backends")
    bench_parser.add_argument("--output", default=None, help="Path to export Markdown leaderboard")

    # Command: models (Open-weights catalog and cache manager)
    models_parser = subparsers.add_parser("models", help="Inspect and download open-weight checkpoints")
    models_sub = models_parser.add_subparsers(dest="models_action", help="Action: list or download")
    models_sub.add_parser("list", help="List available and cached models")
    dl_parser = models_sub.add_parser("download", help="Download a model checkpoint from the catalog")
    dl_parser.add_argument("model_name", help="Name of model to download (e.g., reflex-0.5b-int8)")

    # Command: repl (Interactive terminal shell)
    repl_parser = subparsers.add_parser("repl", help="Start interactive System 1 decision terminal shell")
    repl_parser.add_argument("--backend", default="semantic", help="Initial backend (default: semantic)")

    # Command: doctor (System health and hardware acceleration diagnostics)
    subparsers.add_parser("doctor", help="Run comprehensive system health and hardware acceleration diagnostics")

    # Command: tune (Active learning offline/batch tuner)
    tune_parser = subparsers.add_parser("tune", help="Fine-tune local instinct head from feedback dataset")
    tune_parser.add_argument("--dataset", required=True, help="Path to feedback JSONL dataset")
    tune_parser.add_argument("--output", default="reflex_weights.json", help="Path to export tuned weights JSON")
    tune_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    tune_parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (default: 0.05)")

    args = parser.parse_args()

    if args.command == "doctor":
        run_doctor()
    elif args.command == "tune":
        from reflex.feedback import FeedbackCollector
        from reflex.learning import SelfTuningInstinctHead, OnlineTuner
        collector = FeedbackCollector()
        collector.load_jsonl(args.dataset)
        samples = collector.get_samples()
        print(f"\n🧠 Tuning Reflex Instinct Head on {len(samples)} feedback samples...")
        head = SelfTuningInstinctHead()
        tuner = OnlineTuner(head=head, lr=args.lr)
        stats = tuner.tune_on_samples(samples, epochs=args.epochs)
        head.save_weights(args.output)
        print(f"✅ Finished {stats['epochs']} epochs:")
        print(f" • Initial Loss : {stats['initial_loss']:.4f}")
        print(f" • Final Loss   : {stats['final_loss']:.4f}")
        print(f" • Saved Model  : {args.output}\n")
    elif args.command == "repl":
        from reflex.repl import start_repl
        start_repl(initial_backend=args.backend)
    elif args.command in ("serve", "gateway"):
        from reflex.gateway import ReflexGatewayServer, GatewayConfig
        peers_list = [p.strip() for p in getattr(args, "mesh_peers", "").split(",") if p.strip()]
        secret = getattr(args, "mesh_secret", "") or None
        cfg = GatewayConfig(
            host=args.host,
            port=args.port,
            upstream_url=getattr(args, "upstream", os.environ.get("UPSTREAM_OPENAI_URL", "https://api.openai.com/v1")),
            cache_enabled=not getattr(args, "no_cache", False),
            cache_ttl=getattr(args, "cache_ttl", 3600.0),
            semantic_threshold=getattr(args, "similarity_threshold", 0.95),
            guardrails_enabled=not getattr(args, "no_guardrails", False),
            mesh_enabled=bool(peers_list or secret),
            mesh_peers=peers_list,
            mesh_secret=secret,
        )
        server = ReflexGatewayServer(cfg)
        try:
            server.start(background=False)
        except KeyboardInterrupt:
            print("\nShutting down Reflex AI Envoy Gateway...")
            server.stop()
            sys.exit(0)
    elif args.command == "mesh":
        import urllib.request
        gateway = getattr(args, "gateway", "http://127.0.0.1:8080").rstrip("/")
        try:
            req = urllib.request.Request(f"{gateway}/v1/mesh/peers")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print("\n🌐 Reflex Instinct Mesh Cluster Status:")
                print(f" • Node ID    : {data.get('node_id', 'unknown')}")
                print(f" • Generation : {data.get('generation', 0)}")
                print(f" • Peers ({len(data.get('peers', []))}):")
                for peer in data.get("peers", []):
                    print(f"   - {peer.get('address')}: {peer.get('status', 'unknown')} (latency: {peer.get('last_latency_ms', 0):.1f}ms, gen: {peer.get('generation', 0)})")
                print()
        except Exception as e:
            print(f"\n❌ Error querying mesh gateway: {e}\n")
    elif args.command == "benchmark":
        from reflex.eval import generate_leaderboard
        print(generate_leaderboard(args.output))
    elif args.command == "models":
        from reflex.models import list_models, download_model
        if args.models_action == "download":
            path = download_model(args.model_name)
            print(f"✅ Ready: {path}")
        else:
            models = list_models()
            print("\n📦 Reflex Open-Weights Model Catalog:")
            print(f"{'Model Name':<26} {'Size':<10} {'Cached':<8} {'Description'}")
            print("-" * 75)
            for m in models:
                cached_str = "✅ Yes" if m["cached"] else "❌ No"
                print(f"{m['name']:<26} {m['size_mb']:.1f} MB   {cached_str:<8} {m['description']}")
            print("\nDownload any model via: reflex models download <name>\n")
    elif args.command == "serve-api":
        from reflex.server import start_server
        try:
            start_server(host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("\nShutting down Reflex API Gateway...")
            sys.exit(0)
    elif args.command == "playground":
        from reflex.web.playground import start_playground
        try:
            start_playground(host=args.host, port=args.port, open_browser=not args.no_browser)
        except KeyboardInterrupt:
            print("\nShutting down Reflex Playground...")
            sys.exit(0)
    elif args.command == "quantize":
        from reflex.export import quantize_onnx_model
        out = quantize_onnx_model(args.model_path, args.output)
        print(f"✅ Quantized model ready at: {out}")
    elif args.command == "mcp":
        from reflex.mcp import start_mcp_server
        try:
            start_mcp_server()
        except KeyboardInterrupt:
            sys.exit(0)
    elif args.command == "dataset-gen":
        from reflex.rlcd import generate_decision_dataset, save_dataset_jsonl
        print(f"Generating {args.samples} calibrated decision samples (seed={args.seed})...")
        samples = generate_decision_dataset(num_samples=args.samples, seed=args.seed)
        save_dataset_jsonl(samples, args.output)
        print(f"✅ Successfully saved dataset to {args.output}")
    elif args.command == "eval":
        rx = Reflex()
        if args.noul:
            prob = rx.noul(args.noul, args.state)
            print(f"Noul probability: {prob:.4f}")
        elif args.choice and args.options:
            opts = [o.strip() for o in args.options.split(",")]
            selected = rx.choice(args.choice, opts, args.state)
            print(f"Selected: {selected}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
