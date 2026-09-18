"""
Reflex Command-Line Interface (CLI).
"""

import argparse
import sys
from reflex.client import Reflex
from reflex.primitives import Noul, Choice
from reflex.proxy import start_proxy


def main():
    parser = argparse.ArgumentParser(
        description="Reflex: Universal System-1 AI Runtime & Dual-Brain Gateway"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: serve (proxy)
    serve_parser = subparsers.add_parser("serve", help="Start drop-in OpenAI-compatible proxy")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")

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

    args = parser.parse_args()

    if args.command == "serve":
        try:
            start_proxy(host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("\nShutting down Reflex Proxy...")
            sys.exit(0)
    elif args.command == "playground":
        from reflex.web.playground import start_playground
        try:
            start_playground(host=args.host, port=args.port, open_browser=not args.no_browser)
        except KeyboardInterrupt:
            print("\nShutting down Reflex Playground...")
            sys.exit(0)
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
