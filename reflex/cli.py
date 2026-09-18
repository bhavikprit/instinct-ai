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

    args = parser.parse_args()

    if args.command == "serve":
        try:
            start_proxy(host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("\nShutting down Reflex Proxy...")
            sys.exit(0)
    elif args.command == "mcp":
        from reflex.mcp import start_mcp_server
        try:
            start_mcp_server()
        except KeyboardInterrupt:
            sys.exit(0)
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
