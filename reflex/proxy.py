"""
Reflex Reverse Proxy: The Drop-in OpenAI-compatible System 1 Interceptor.
Runs with zero third-party dependencies using Python standard library.
"""

from __future__ import annotations
import json
import os
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

from reflex.client import Reflex
from reflex.primitives import Noul, Choice, Score


class ReflexProxyHandler(BaseHTTPRequestHandler):
    """
    HTTP Handler that intercepts classification/routing calls and fulfills them
    via Reflex System 1 in <15ms, passing complex generative prompts upstream.
    """

    rx = Reflex(backend="auto")
    upstream_url = os.environ.get("UPSTREAM_OPENAI_URL", "https://api.openai.com/v1")
    upstream_key = os.environ.get("OPENAI_API_KEY", "")

    def do_POST(self):
        if self.path.endswith("/chat/completions"):
            self.handle_chat_completions()
        else:
            self.send_error(404, "Endpoint not supported by Reflex Proxy")

    def handle_chat_completions(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        req_json = json.loads(body)

        # 1. Analyze if request is a decision / classification candidate
        messages = req_json.get("messages", [])
        combined_prompt = " ".join(m.get("content", "") for m in messages)
        response_format = req_json.get("response_format", {})

        is_decision = self._is_decision_candidate(combined_prompt, response_format)

        if is_decision:
            # Fulfill via System 1 Reflex!
            response_json = self._fulfill_via_system1(req_json, combined_prompt)
            self._send_json(200, response_json)
        else:
            # Pass through to upstream OpenAI / Anthropic
            self._passthrough_upstream(req_json)

    def _is_decision_candidate(self, prompt: str, response_format: dict) -> bool:
        """Determines if the prompt is an if/else, routing, or classification task."""
        if response_format.get("type") == "json_object":
            return True
        keywords = ["classify", "route", "category", "is_", "should_", "select", "options:", "triage"]
        return any(k in prompt.lower() for k in keywords)

    def _fulfill_via_system1(self, req_json: dict, prompt: str) -> dict:
        start_time = time.perf_counter()
        
        # Extract question or default to triage
        res = self.rx.evaluate(
            state=prompt,
            questions={
                "is_urgent": Noul("Is this message urgent or high priority?"),
                "category": Choice("Select category", options=["support", "billing", "security", "general"])
            }
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Build clean JSON content
        content_dict = {
            "is_urgent": res["is_urgent"].probability > 0.85, # type: ignore
            "confidence": res["is_urgent"].probability, # type: ignore
            "category": res["category"].selected # type: ignore
        }

        return {
            "id": f"chatcmpl-reflex-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "reflex-system1",
            "system_fingerprint": "fp_reflex_0_1_0",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content_dict)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 0, # Reflex output tokens are free
                "total_tokens": len(prompt.split())
            },
            "reflex_meta": {
                "intercepted": True,
                "latency_ms": round(elapsed_ms, 2),
                "backend": res.backend
            }
        }

    def _passthrough_upstream(self, req_json: dict):
        if not self.upstream_key:
            self._send_json(500, {"error": "Upstream OPENAI_API_KEY not configured for non-decision prompts"})
            return

        upstream_req = urllib.request.Request(
            f"{self.upstream_url}/chat/completions",
            data=json.dumps(req_json).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.upstream_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(upstream_req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._send_json(200, data)
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"Upstream proxy error: {e}"})

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Clean terminal logging
        print(f"[Reflex Proxy] {args[0]} {args[1]} -> {args[2]}")


def start_proxy(host: str = "127.0.0.1", port: int = 8080):
    """Starts the drop-in OpenAI-compatible Reflex proxy server."""
    server = HTTPServer((host, port), ReflexProxyHandler)
    print("=" * 65)
    print(f"⚡ Reflex Proxy running at http://{host}:{port}/v1")
    print(f"Point your OpenAI SDK here:")
    print(f"  client = OpenAI(base_url='http://{host}:{port}/v1')")
    print("=" * 65)
    server.serve_forever()
