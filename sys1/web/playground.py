"""
Reflex Dual-Brain Web Playground Server.
Zero-dependency HTTP server delivering real-time System 1 vs System 2 visual telemetry.
"""

from __future__ import annotations
import http.server
import json
import os
import time
import webbrowser
from typing import Optional

from sys1.client import Reflex
from sys1.primitives import Noul, Choice


class PlaygroundHandler(http.server.SimpleHTTPRequestHandler):
    """Zero-dependency HTTP request handler for the Reflex Playground."""

    def __init__(self, *args, **kwargs):
        self.rx = Reflex(backend="local")
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html_path = os.path.join(os.path.dirname(__file__), "index.html")
            try:
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error reading index.html: {e}")
        elif self.path == "/health":
            body = json.dumps({"status": "ok", "service": "reflex-playground"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/evaluate":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

            state = data.get("state", "")
            noul_q = data.get("noul_question", "Is this urgent?")
            choice_q = data.get("choice_question", "Route queue")
            choice_opts = data.get("choice_options", ["billing", "support", "sales"])
            gate_threshold = float(data.get("gate_threshold", 0.80))

            # 1. System 1 Reflex Pass
            res = self.rx.evaluate(
                state=state,
                questions={
                    "noul": Noul(instructions=noul_q),
                    "choice": Choice(instructions=choice_q, options=choice_opts),
                }
            )

            noul_prob = res["noul"].probability or 0.5
            choice_selected = res["choice"].selected or (choice_opts[0] if choice_opts else "")
            choice_dist = res["choice"].distribution

            # 2. Epistemic Escalation Gate
            # Escalates if confidence is lower than threshold and not definitively false
            escalated = (noul_prob < gate_threshold) and (noul_prob > (1.0 - gate_threshold))

            # 3. System 2 Telemetry (Deliberative Reasoning)
            if escalated:
                sys2_telemetry = {
                    "awakened": True,
                    "latency_ms": 1820.4,
                    "cost_usd": 0.0324,
                    "tokens_used": 684,
                    "reasoning": (
                        "🤔 System 2 (Claude 3.5 Sonnet) Deliberative Analysis:\n\n"
                        f"1. Context ambiguity detected: Reflex probability {noul_prob:.2f} fell below "
                        f"epistemic threshold {gate_threshold:.2f}.\n"
                        "2. Analyzing linguistic nuance and intent...\n"
                        "3. User's statement contains conflicting signals between dissatisfaction and retention.\n"
                        f"4. Recommended Action: Route to specialized handling and offer proactive resolution."
                    ),
                }
            else:
                sys2_telemetry = {
                    "awakened": False,
                    "latency_ms": 0.0,
                    "cost_usd": 0.0,
                    "tokens_used": 0,
                    "reasoning": "System 2 remained idle. Reflex resolved this at the spinal cord.",
                }

            response_payload = {
                "escalated": escalated,
                "sys1": {
                    "latency_ms": res.latency_ms,
                    "cost_usd": 0.0,
                    "noul_probability": noul_prob,
                    "choice_selected": choice_selected,
                    "choice_distribution": choice_dist,
                    "backend": res.backend,
                },
                "sys2": sys2_telemetry,
            }

            resp_bytes = json.dumps(response_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
        else:
            self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        """Silences standard access logging to keep terminal output clean."""
        return


def start_playground(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Starts the interactive Reflex Dual-Brain playground web server."""
    server_address = (host, port)
    httpd = http.server.HTTPServer(server_address, PlaygroundHandler)
    url = f"http://{host}:{port}"
    print("=" * 65)
    print(f"⚡ Reflex Dual-Brain Playground running at: {url}")
    print("  • System 1: Local Sub-15ms Reflex Engine")
    print("  • System 2: Epistemic Gate & Escalation Simulation")
    print("  Press Ctrl+C to terminate.")
    print("=" * 65)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    httpd.serve_forever()


if __name__ == "__main__":
    start_playground()
