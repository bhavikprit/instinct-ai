"""
Reflex Interactive Command-Line REPL.
Provides a lightning-fast terminal shell for real-time System 1 decisions and guardrail evaluation.
"""

from __future__ import annotations
import sys
import os
import time
from typing import Optional

from reflex import Reflex, Noul, Choice, Score, GuardrailSuite
from reflex.models import list_models


def _supports_color() -> bool:
    """Checks whether the current terminal supports ANSI color output."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


CYAN = "\033[1;36m" if _supports_color() else ""
GREEN = "\033[1;32m" if _supports_color() else ""
YELLOW = "\033[1;33m" if _supports_color() else ""
RED = "\033[1;31m" if _supports_color() else ""
MAGENTA = "\033[1;35m" if _supports_color() else ""
BOLD = "\033[1m" if _supports_color() else ""
RESET = "\033[0m" if _supports_color() else ""


def _render_bar(prob: float, width: int = 15) -> str:
    """Renders a text progress bar representing probability."""
    filled = int(round(prob * width))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def start_repl(initial_backend: str = "semantic"):
    """Starts the interactive Reflex REPL shell."""
    current_backend = initial_backend
    rx = Reflex(backend=current_backend)
    guard = GuardrailSuite()

    print(f"\n{CYAN}⚡ Reflex Interactive System 1 Shell (v0.2.0){RESET}")
    print(f"Backend: {GREEN}{current_backend}{RESET} | Type {YELLOW}/help{RESET} for commands, {YELLOW}exit{RESET} to quit.\n")

    while True:
        try:
            line = input(f"{BOLD}reflex ({current_backend})>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Reflex REPL...")
            break

        if not line:
            continue

        cmd = line.lower()
        if cmd in ("exit", "quit", "/exit", "/quit"):
            print("Goodbye!")
            break
        elif cmd in ("help", "/help"):
            _print_help()
        elif cmd.startswith("/backend"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                target = parts[1].strip().lower()
                try:
                    rx = Reflex(backend=target)
                    current_backend = target
                    print(f"✅ Switched active backend to: {GREEN}{current_backend}{RESET}")
                except Exception as e:
                    print(f"{RED}Error switching backend:{RESET} {e}")
            else:
                print(f"Active backend: {GREEN}{current_backend}{RESET}")
        elif cmd in ("/models", "models"):
            models = list_models()
            print(f"\n{BOLD}Open-Weights Model Catalog:{RESET}")
            for m in models:
                st = f"{GREEN}Cached{RESET}" if m["cached"] else f"{YELLOW}Remote{RESET}"
                print(f" • {m['name']:<24} {m['size_mb']:.1f} MB  [{st}]")
            print()
        elif cmd.startswith("guard "):
            text = line[6:].strip()
            verdict = guard.check(text)
            if verdict.blocked:
                print(f"{RED}✖ BLOCKED{RESET} ({verdict.latency_ms:.3f} ms): {verdict.reason}")
            else:
                print(f"{GREEN}✔ SAFE{RESET} ({verdict.latency_ms:.3f} ms) - Zero violations detected")
        elif "::" in line:
            _handle_colon_syntax(rx, line)
        else:
            # Default to Noul question on the input text
            start_t = time.perf_counter()
            prob = rx.noul("Is this prompt high-priority or demanding immediate action?", line)
            lat = (time.perf_counter() - start_t) * 1000.0
            verdict_str = f"{GREEN}TRUE{RESET}" if prob >= 0.5 else f"{YELLOW}FALSE{RESET}"
            bar = _render_bar(prob)
            print(f"  [{bar}] {prob*100:.1f}% -> {verdict_str} ({lat:.2f} ms)")


def _handle_colon_syntax(rx: Reflex, line: str):
    """Handles syntax like: instructions :: state text"""
    prefix, state = line.split("::", 1)
    prefix = prefix.strip()
    state = state.strip()

    if prefix.lower().startswith("choice"):
        # e.g., choice [support, sales, billing] :: I need an invoice
        opts_str = prefix[6:].strip().strip("[]()")
        options = [o.strip() for o in opts_str.split(",") if o.strip()]
        if not options:
            options = ["yes", "no"]
        start_t = time.perf_counter()
        selected = rx.choice("Select route", options, state)
        lat = (time.perf_counter() - start_t) * 1000.0
        print(f"  Selected: {GREEN}{selected}{RESET} ({lat:.2f} ms)")
    else:
        # e.g., Is this spam? :: Click here to win a car!
        start_t = time.perf_counter()
        prob = rx.noul(prefix, state)
        lat = (time.perf_counter() - start_t) * 1000.0
        verdict_str = f"{GREEN}TRUE{RESET}" if prob >= 0.5 else f"{YELLOW}FALSE{RESET}"
        bar = _render_bar(prob)
        print(f"  [{bar}] {prob*100:.1f}% -> {verdict_str} ({lat:.2f} ms)")


def _print_help():
    print(f"""
{BOLD}Reflex REPL Commands & Syntax:{RESET}
  {CYAN}<text>{RESET}                                 Evaluates priority Noul on text
  {CYAN}<question> :: <text>{RESET}                   Evaluates custom Noul question on text
  {CYAN}choice [opt1, opt2, ...] :: <text>{RESET}     Evaluates multi-option Choice routing
  {CYAN}guard <text>{RESET}                             Scans text for prompt injections and PII
  {CYAN}/backend [semantic|local|onnx]{RESET}         Switches active runtime backend
  {CYAN}/models{RESET}                                Lists catalog models
  {CYAN}exit{RESET}                                   Quit REPL
""")
