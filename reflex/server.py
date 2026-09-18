"""
Reflex Enterprise Production Microservice & Gateway.
High-throughput REST API server with Prometheus-compatible telemetry.
"""

from __future__ import annotations
import http.server
import json
import socketserver
import threading
import time
from typing import Any, Dict, List, Optional

from reflex.client import Reflex
from reflex.guardrails import GuardrailSuite
from reflex.tool_router import FastToolRouter
from reflex.primitives import Noul, Choice, Score


class MetricsCollector:
    """Thread-safe Prometheus metrics collector for Reflex."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests_total: Dict[str, int] = {
            "/v1/evaluate": 0,
            "/v1/guardrails": 0,
            "/v1/tools/route": 0,
        }
        self.decisions_total = 0
        self.guardrails_blocked_total = 0
        self.tools_pruned_total = 0
        self.cost_saved_usd = 0.0
        self.latencies: List[float] = []

    def record_request(self, endpoint: str):
        with self._lock:
            self.requests_total[endpoint] = self.requests_total.get(endpoint, 0) + 1

    def record_decision(self, latency_ms: float, count: int = 1):
        with self._lock:
            self.decisions_total += count
            # Estimated $0.03 saved per decision compared to calling GPT-4o / Claude 3.5
            self.cost_saved_usd += 0.03 * count
            self.latencies.append(latency_ms)
            if len(self.latencies) > 1000:
                self.latencies.pop(0)

    def record_guardrail(self, blocked: bool):
        with self._lock:
            if blocked:
                self.guardrails_blocked_total += 1

    def record_tools_pruned(self, original_count: int, pruned_count: int):
        with self._lock:
            self.tools_pruned_total += max(0, original_count - pruned_count)

    def to_prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# HELP reflex_requests_total Total number of HTTP requests processed by endpoint",
                "# TYPE reflex_requests_total counter",
            ]
            for ep, count in self.requests_total.items():
                lines.append(f'reflex_requests_total{{endpoint="{ep}"}} {count}')

            lines.extend([
                "# HELP reflex_decisions_total Total System 1 decisions resolved",
                "# TYPE reflex_decisions_total counter",
                f"reflex_decisions_total {self.decisions_total}",
                "# HELP reflex_guardrails_blocked_total Malicious or PII prompts blocked",
                "# TYPE reflex_guardrails_blocked_total counter",
                f"reflex_guardrails_blocked_total {self.guardrails_blocked_total}",
                "# HELP reflex_tools_pruned_total Cumulative candidate tools pruned",
                "# TYPE reflex_tools_pruned_total counter",
                f"reflex_tools_pruned_total {self.tools_pruned_total}",
                "# HELP reflex_cost_saved_usd Estimated cumulative dollars saved vs System 2 cloud LLMs",
                "# TYPE reflex_cost_saved_usd gauge",
                f"reflex_cost_saved_usd {self.cost_saved_usd:.2f}",
            ])

            if self.latencies:
                sorted_l = sorted(self.latencies)
                p50 = sorted_l[int(len(sorted_l) * 0.50)]
                p90 = sorted_l[int(len(sorted_l) * 0.90)]
                p99 = sorted_l[int(len(sorted_l) * 0.99)]
            else:
                p50, p90, p99 = 0.0, 0.0, 0.0

            lines.extend([
                "# HELP reflex_latency_ms System 1 evaluation latency in milliseconds",
                "# TYPE reflex_latency_ms gauge",
                f'reflex_latency_ms{{quantile="0.50"}} {p50:.3f}',
                f'reflex_latency_ms{{quantile="0.90"}} {p90:.3f}',
                f'reflex_latency_ms{{quantile="0.99"}} {p99:.3f}',
            ])

            return "\n".join(lines) + "\n"


GLOBAL_METRICS = MetricsCollector()


class ReflexAPIHandler(http.server.SimpleHTTPRequestHandler):
    """Production REST API handler for Reflex."""

    def __init__(self, *args, **kwargs):
        self.rx = Reflex(backend="local")
        self.guardrail_suite = GuardrailSuite()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "healthy", "service": "reflex-gateway", "version": "0.1.0"})
        elif self.path == "/metrics":
            metrics_text = GLOBAL_METRICS.to_prometheus_text().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(metrics_text)))
            self.end_headers()
            self.wfile.write(metrics_text)
        else:
            self.send_error(404, f"Path {self.path} not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Malformed JSON payload"})
            return

        if self.path == "/v1/evaluate":
            GLOBAL_METRICS.record_request(self.path)
            state = payload.get("state", "")
            raw_questions = payload.get("questions", {})

            # Reconstruct typed primitives
            typed_questions = {}
            for k, q_def in raw_questions.items():
                q_type = q_def.get("type", "noul")
                instr = q_def.get("instructions", "")
                if q_type == "noul":
                    typed_questions[k] = Noul(instructions=instr, threshold=q_def.get("threshold", 0.85))
                elif q_type == "choice":
                    typed_questions[k] = Choice(instructions=instr, options=q_def.get("options", []))
                elif q_type == "score":
                    typed_questions[k] = Score(instructions=instr, min_val=q_def.get("min_val", 1.0), max_val=q_def.get("max_val", 10.0))

            res = self.rx.evaluate(state, typed_questions)
            GLOBAL_METRICS.record_decision(res.latency_ms, count=len(typed_questions))
            self._send_json(200, res.to_dict())

        elif self.path == "/v1/guardrails":
            GLOBAL_METRICS.record_request(self.path)
            text = payload.get("text", "")
            result = self.guardrail_suite.check(text)
            GLOBAL_METRICS.record_guardrail(result.blocked)
            self._send_json(200, result.to_dict())

        elif self.path == "/v1/tools/route":
            GLOBAL_METRICS.record_request(self.path)
            prompt = payload.get("prompt", "")
            tools = payload.get("tools", [])
            top_k = payload.get("top_k", 2)

            pruned = FastToolRouter.filter_openai_tools(prompt=prompt, tools=tools, top_k=top_k)
            GLOBAL_METRICS.record_tools_pruned(len(tools), len(pruned))
            self._send_json(200, {
                "pruned_tools": pruned,
                "original_count": len(tools),
                "pruned_count": len(pruned),
                "token_savings_pct": round((1.0 - (len(pruned) / max(1, len(tools)))) * 100, 1),
            })

        else:
            self.send_error(404, f"Endpoint {self.path} not found")

    def _send_json(self, status: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server for high concurrency."""
    daemon_threads = True
    allow_reuse_address = True


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Starts the production Reflex API Gateway."""
    server = ThreadedHTTPServer((host, port), ReflexAPIHandler)
    print("=" * 65)
    print(f"🚀 Reflex Production Gateway running at: http://{host}:{port}")
    print(f"  • REST API:    POST /v1/evaluate")
    print(f"  • Guardrails:  POST /v1/guardrails")
    print(f"  • Tool Router: POST /v1/tools/route")
    print(f"  • Prometheus:  GET  /metrics")
    print(f"  • Health:      GET  /health")
    print("  Press Ctrl+C to terminate.")
    print("=" * 65)
    server.serve_forever()


if __name__ == "__main__":
    start_server()
