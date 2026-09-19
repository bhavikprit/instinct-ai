"""
Reflex AI Envoy: Production Reverse Proxy Gateway & Dynamic Cost Arbitrage (Phase 18).
Delivers sub-millisecond semantic deduplication caching, pre-flight security guardrails,
System-1 short-circuiting, and real-time financial ROI telemetry.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
from collections import OrderedDict
import copy
import json
import os
import re
import socketserver
import threading
import time
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error
import uuid

from reflex.client import Reflex
from reflex.embeddings import SemanticVectorEncoder, cosine_similarity
from reflex.guardrails import GuardrailSuite
from reflex.primitives import Noul, Choice
from reflex.mesh import InstinctMeshNode, MeshConfig


@dataclass
class GatewayConfig:
    """Configuration options for Reflex AI Envoy Gateway."""
    host: str = "0.0.0.0"
    port: int = 8080
    upstream_url: str = os.environ.get("UPSTREAM_OPENAI_URL", "https://api.openai.com/v1")
    upstream_key: Optional[str] = os.environ.get("OPENAI_API_KEY", None)
    cache_enabled: bool = True
    cache_ttl: float = 3600.0
    semantic_threshold: float = 0.95
    guardrails_enabled: bool = True
    system1_routing_enabled: bool = True
    estimated_upstream_latency_ms: float = 1850.0
    estimated_upstream_cost_usd: float = 0.005
    mesh_enabled: bool = False
    mesh_peers: List[str] = field(default_factory=list)
    mesh_secret: Optional[str] = os.environ.get("REFLEX_MESH_SECRET", None)
    mesh_node_id: Optional[str] = None


@dataclass
class GatewayCacheEntry:
    """Entry stored in the Gateway Semantic Cache."""
    prompt: str
    vector: List[float]
    model: str
    response_data: dict
    created_at: float
    last_accessed: float
    similarity: float = 1.0


class GatewaySemanticCache:
    """Multi-tier L1 exact and L2 cosine semantic cache for OpenAI chat completions."""

    def __init__(
        self,
        max_size: int = 2000,
        ttl_seconds: float = 3600.0,
        similarity_threshold: float = 0.95,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.encoder = SemanticVectorEncoder()
        self._entries: OrderedDict[str, GatewayCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, prompt: str, model: str) -> Optional[GatewayCacheEntry]:
        now = time.time()
        exact_key = f"{model}##{prompt.strip().lower()}"
        with self._lock:
            # 1. L1 Exact Match
            if exact_key in self._entries:
                entry = self._entries[exact_key]
                if self.ttl_seconds and (now - entry.created_at > self.ttl_seconds):
                    del self._entries[exact_key]
                else:
                    self._entries.move_to_end(exact_key)
                    entry.last_accessed = now
                    entry.similarity = 1.0
                    return entry

            # 2. L2 Semantic Cosine Match
            query_vec = self.encoder.encode(prompt)
            best_entry: Optional[GatewayCacheEntry] = None
            best_key: Optional[str] = None
            best_sim = -1.0

            for key, entry in self._entries.items():
                if entry.model != model:
                    continue
                if self.ttl_seconds and (now - entry.created_at > self.ttl_seconds):
                    continue
                sim = cosine_similarity(query_vec, entry.vector)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
                    best_key = key

            if best_entry is not None and best_sim >= self.similarity_threshold:
                self._entries.move_to_end(best_key)
                best_entry.last_accessed = now
                best_entry.similarity = best_sim
                return best_entry

            return None

    def set(self, prompt: str, response_data: dict, model: str):
        now = time.time()
        exact_key = f"{model}##{prompt.strip().lower()}"
        vec = self.encoder.encode(prompt)
        entry = GatewayCacheEntry(
            prompt=prompt,
            vector=vec,
            model=model,
            response_data=copy.deepcopy(response_data),
            created_at=now,
            last_accessed=now,
            similarity=1.0,
        )
        with self._lock:
            if exact_key in self._entries:
                self._entries.move_to_end(exact_key)
            self._entries[exact_key] = entry
            if len(self._entries) > self.max_size:
                self._entries.popitem(last=False)


@dataclass
class GatewayMetrics:
    """Thread-safe real-time telemetry tracking cost, token, and latency savings."""
    total_requests: int = 0
    l1_cache_hits: int = 0
    l2_semantic_hits: int = 0
    guardrail_blocks: int = 0
    system1_shortcircuits: int = 0
    upstream_requests: int = 0
    tokens_saved: int = 0
    dollars_saved: float = 0.0
    total_saved_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_l1_hit(self, tokens: int, cost: float, latency_saved: float):
        with self._lock:
            self.total_requests += 1
            self.l1_cache_hits += 1
            self.tokens_saved += tokens
            self.dollars_saved += cost
            self.total_saved_latency_ms += latency_saved

    def record_l2_hit(self, tokens: int, cost: float, latency_saved: float):
        with self._lock:
            self.total_requests += 1
            self.l2_semantic_hits += 1
            self.tokens_saved += tokens
            self.dollars_saved += cost
            self.total_saved_latency_ms += latency_saved

    def record_guardrail_block(self, tokens: int, cost: float):
        with self._lock:
            self.total_requests += 1
            self.guardrail_blocks += 1
            self.tokens_saved += tokens
            self.dollars_saved += cost

    def record_system1_shortcircuit(self, tokens: int, cost: float, latency_saved: float):
        with self._lock:
            self.total_requests += 1
            self.system1_shortcircuits += 1
            self.tokens_saved += tokens
            self.dollars_saved += cost
            self.total_saved_latency_ms += latency_saved

    def record_upstream(self):
        with self._lock:
            self.total_requests += 1
            self.upstream_requests += 1

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            total_cache_hits = self.l1_cache_hits + self.l2_semantic_hits
            total_intercepted = total_cache_hits + self.system1_shortcircuits + self.guardrail_blocks
            cache_hit_rate = (total_cache_hits / self.total_requests) if self.total_requests > 0 else 0.0
            interception_rate = (total_intercepted / self.total_requests) if self.total_requests > 0 else 0.0

            return {
                "total_requests": self.total_requests,
                "l1_cache_hits": self.l1_cache_hits,
                "l2_semantic_hits": self.l2_semantic_hits,
                "total_cache_hits": total_cache_hits,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "guardrail_blocks": self.guardrail_blocks,
                "system1_shortcircuits": self.system1_shortcircuits,
                "upstream_requests": self.upstream_requests,
                "interception_rate": round(interception_rate, 4),
                "tokens_saved": self.tokens_saved,
                "dollars_saved": round(self.dollars_saved, 4),
                "total_saved_latency_ms": round(self.total_saved_latency_ms, 2),
            }


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server supporting high-concurrency requests."""
    daemon_threads = True


class GatewayRequestHandler(BaseHTTPRequestHandler):
    """
    High-performance request handler providing OpenAI-compatible chat completions
    with intelligent semantic caching, pre-flight security, and System-1 shortcuts.
    """

    config: GatewayConfig = GatewayConfig()
    metrics: GatewayMetrics = GatewayMetrics()
    cache: Optional[GatewaySemanticCache] = None
    guardrails: GuardrailSuite = GuardrailSuite()
    rx: Reflex = Reflex()
    mesh_node: Optional[InstinctMeshNode] = None

    @classmethod
    def initialize(cls, config: GatewayConfig):
        cls.config = config
        cls.metrics = GatewayMetrics()
        cls.guardrails = GuardrailSuite()
        cls.rx = Reflex(learning=True)
        if config.cache_enabled:
            cls.cache = GatewaySemanticCache(
                max_size=2000,
                ttl_seconds=config.cache_ttl,
                similarity_threshold=config.semantic_threshold,
            )
        else:
            cls.cache = None

        if config.mesh_enabled or config.mesh_peers:
            mesh_cfg = MeshConfig(
                node_id=config.mesh_node_id or f"gateway-{uuid.uuid4().hex[:8]}",
                host=config.host,
                port=config.port,
                peers=config.mesh_peers,
                cluster_secret=config.mesh_secret,
            )
            cls.mesh_node = InstinctMeshNode(config=mesh_cfg, instinct_head=cls.rx.instinct_head)
        else:
            cls.mesh_node = None

    def do_GET(self):
        norm_path = self.path.split("?")[0]
        if norm_path in ("/healthz", "/health"):
            self._send_json(200, {"status": "healthy", "service": "reflex-gateway", "version": "0.2.0"})
        elif norm_path in ("/v1/gateway/stats", "/stats"):
            self._send_json(200, self.metrics.to_dict())
        elif norm_path in ("/v1/mesh/peers", "/mesh/peers"):
            if self.mesh_node is not None:
                status, res = self.mesh_node.handle_peers_request()
                self._send_json(status, res)
            else:
                self._send_json(400, {"error": "Instinct Mesh is not enabled on this gateway"})
        elif norm_path in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": "reflex-system1", "object": "model", "owned_by": "reflex-ai"},
                    {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
                    {"id": "claude-3-5-sonnet-20241022", "object": "model", "owned_by": "anthropic"},
                ]
            })
        else:
            self.send_error(404, f"Endpoint '{self.path}' not found")

    def do_POST(self):
        norm_path = self.path.split("?")[0]
        if norm_path in ("/v1/chat/completions", "/chat/completions"):
            self.handle_chat_completions()
        elif norm_path in ("/v1/mesh/sync", "/mesh/sync"):
            if self.mesh_node is not None:
                content_len = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_len)
                status, res = self.mesh_node.handle_sync_request(body_bytes, dict(self.headers))
                self._send_json(status, res)
            else:
                self._send_json(400, {"error": "Instinct Mesh is not enabled on this gateway"})
        elif norm_path in ("/v1/mesh/heartbeat", "/mesh/heartbeat"):
            if self.mesh_node is not None:
                content_len = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_len)
                status, res = self.mesh_node.handle_heartbeat_request(body_bytes, dict(self.headers))
                self._send_json(status, res)
            else:
                self._send_json(400, {"error": "Instinct Mesh is not enabled on this gateway"})
        else:
            self.send_error(404, f"Endpoint '{self.path}' not supported by Reflex Gateway")

    def handle_chat_completions(self):
        t0 = time.perf_counter()
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            req_json = json.loads(body)
        except Exception as e:
            self._send_json(400, {
                "error": {
                    "message": f"Invalid JSON payload: {str(e)}",
                    "type": "invalid_request_error",
                    "code": 400
                }
            })
            return

        messages = req_json.get("messages", [])
        combined_prompt = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        tokens_est = max(1, len(combined_prompt.split()))
        model = req_json.get("model", "gpt-4o")

        # -------------------------------------------------------------
        # 1. Pre-Flight Security Guardrail Check (<1ms)
        # -------------------------------------------------------------
        if self.config.guardrails_enabled:
            guard_res = self.guardrails.check(combined_prompt)
            if not guard_res.is_safe:
                self.metrics.record_guardrail_block(
                    tokens=tokens_est,
                    cost=self.config.estimated_upstream_cost_usd,
                )
                self._send_json(400, {
                    "error": {
                        "message": f"Blocked by Reflex Security Guardrail: {guard_res.reason}",
                        "type": "guardrail_violation",
                        "category": guard_res.category,
                        "code": 400
                    }
                }, extra_headers={"X-Reflex-Guardrail": "BLOCKED"})
                return

        # -------------------------------------------------------------
        # 2. Intelligent Semantic Deduplication Cache (<1ms)
        # -------------------------------------------------------------
        if self.cache is not None:
            cache_entry = self.cache.get(combined_prompt, model=model)
            if cache_entry is not None:
                cached_data = cache_entry.response_data
                hit_type = "HIT-L1" if cache_entry.similarity >= 0.999 else "HIT-L2"
                latency_ms = (time.perf_counter() - t0) * 1000.0

                if hit_type == "HIT-L1":
                    self.metrics.record_l1_hit(
                        tokens=tokens_est,
                        cost=self.config.estimated_upstream_cost_usd,
                        latency_saved=self.config.estimated_upstream_latency_ms,
                    )
                else:
                    self.metrics.record_l2_hit(
                        tokens=tokens_est,
                        cost=self.config.estimated_upstream_cost_usd,
                        latency_saved=self.config.estimated_upstream_latency_ms,
                    )

                self._send_json(200, cached_data, extra_headers={
                    "X-Reflex-Cache": hit_type,
                    "X-Reflex-Similarity": f"{cache_entry.similarity:.4f}",
                    "X-Reflex-Latency-Ms": f"{latency_ms:.2f}",
                    "X-Reflex-Cost-Saved-USD": f"{self.config.estimated_upstream_cost_usd:.4f}",
                })
                return

        # -------------------------------------------------------------
        # 3. System-1 Instant Short-Circuit (<15ms)
        # -------------------------------------------------------------
        response_format = req_json.get("response_format", {})
        is_decision = self._is_decision_candidate(combined_prompt, response_format)

        if self.config.system1_routing_enabled and is_decision:
            res_payload = self._fulfill_system1(req_json, combined_prompt)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            
            self.metrics.record_system1_shortcircuit(
                tokens=tokens_est,
                cost=self.config.estimated_upstream_cost_usd,
                latency_saved=self.config.estimated_upstream_latency_ms,
            )

            # Store in cache for subsequent calls
            if self.cache is not None:
                self.cache.set(combined_prompt, res_payload, model=model)

            self._send_json(200, res_payload, extra_headers={
                "X-Reflex-Cache": "SHORTCIRCUIT-SYSTEM1",
                "X-Reflex-Latency-Ms": f"{latency_ms:.2f}",
                "X-Reflex-Cost-Saved-USD": f"{self.config.estimated_upstream_cost_usd:.4f}",
            })
            return

        # -------------------------------------------------------------
        # 4. Upstream Forwarding & Passthrough
        # -------------------------------------------------------------
        self.metrics.record_upstream()
        self._forward_upstream(req_json, combined_prompt, model, t0)

    def _is_decision_candidate(self, prompt: str, response_format: dict) -> bool:
        """Heuristically identify if a prompt is an if/else, routing, or classification task."""
        if response_format.get("type") == "json_object":
            return True
        keywords = [
            "classify", "route", "category", "is_", "should_", "select",
            "options:", "triage", "boolean", "intent"
        ]
        return any(k in prompt.lower() for k in keywords)

    def _fulfill_system1(self, req_json: dict, prompt: str) -> dict:
        """Resolve a structured decision request locally in <15ms with $0 cost."""
        eval_res = self.rx.evaluate(
            state=prompt,
            questions={
                "is_urgent": Noul("Is this message urgent or high priority?"),
                "category": Choice("Select category", options=["support", "billing", "security", "general"]),
            }
        )

        content_dict = {
            "is_urgent": eval_res["is_urgent"].is_true,
            "confidence": round(eval_res["is_urgent"].confidence, 3),
            "category": eval_res["category"].selected,
        }

        return {
            "id": f"chatcmpl-reflex-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "reflex-system1",
            "system_fingerprint": "fp_reflex_0_2_0",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content_dict),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(json.dumps(content_dict).split()),
                "total_tokens": len(prompt.split()) + len(json.dumps(content_dict).split()),
            },
        }

    def _forward_upstream(self, req_json: dict, prompt: str, model: str, t0: float):
        """Forward generative request upstream with header injection and caching."""
        target_url = f"{self.config.upstream_url.rstrip('/')}/chat/completions"
        api_key = self.config.upstream_key or self.headers.get("Authorization", "")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Reflex-AI-Envoy/0.2.0",
        }
        if api_key:
            if not api_key.startswith("Bearer "):
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                headers["Authorization"] = api_key

        data = json.dumps(req_json).encode("utf-8")
        req = urllib.request.Request(target_url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                resp_bytes = resp.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))
                latency_ms = (time.perf_counter() - t0) * 1000.0

                # Cache successful response
                if self.cache is not None:
                    self.cache.set(prompt, resp_json, model=model)

                self._send_json(resp.status, resp_json, extra_headers={
                    "X-Reflex-Cache": "MISS",
                    "X-Reflex-Latency-Ms": f"{latency_ms:.2f}",
                })
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
            except Exception:
                err_json = {"error": {"message": err_body, "code": e.code}}
            self._send_json(e.code, err_json)
        except Exception as e:
            # Standalone fallback if upstream unreachable
            fallback_json = {
                "id": f"chatcmpl-fallback-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "reflex-fallback",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[Reflex Local Fallback: Upstream provider unavailable: {str(e)}]"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 10, "total_tokens": len(prompt.split()) + 10}
            }
            self._send_json(200, fallback_json, extra_headers={"X-Reflex-Fallback": "TRUE"})

    def _send_json(self, status_code: int, data: dict, extra_headers: Optional[Dict[str, str]] = None):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Reflex-Runtime", "reflex-gateway/0.2.0")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any):
        """Suppress default HTTP request access logs unless verbose debugging."""
        pass


class ReflexGatewayServer:
    """
    High-level orchestrator for starting and stopping the Reflex AI Envoy gateway server.
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

        # Dynamically create an isolated handler subclass per server instance
        class IsolatedGatewayHandler(GatewayRequestHandler):
            config = self.config

        self.handler_class = IsolatedGatewayHandler

    @property
    def mesh_node(self) -> Optional[InstinctMeshNode]:
        return self.handler_class.mesh_node

    @property
    def rx(self) -> Reflex:
        return self.handler_class.rx

    @property
    def metrics(self) -> GatewayMetrics:
        return self.handler_class.metrics

    def start(self, background: bool = False):
        """Start the gateway server."""
        self.handler_class.initialize(self.config)
        self.server = ThreadingHTTPServer((self.config.host, self.config.port), self.handler_class)
        
        if self.handler_class.mesh_node is not None:
            self.handler_class.mesh_node.start_background_gossip()

        if background:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
        else:
            print(f"⚡ Reflex AI Envoy Gateway running at http://{self.config.host}:{self.config.port}")
            print(f"   • Upstream Provider: {self.config.upstream_url}")
            print(f"   • Semantic Cache   : {'Enabled (TTL: ' + str(self.config.cache_ttl) + 's)' if self.config.cache_enabled else 'Disabled'}")
            print(f"   • Pre-flight Shield: {'Active (sub-1ms)' if self.config.guardrails_enabled else 'Disabled'}")
            if self.handler_class.mesh_node is not None:
                print(f"   • Instinct Mesh    : Connected ({len(self.config.mesh_peers)} peers)")
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        """Stop the gateway server."""
        if self.handler_class.mesh_node is not None:
            self.handler_class.mesh_node.stop()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            self.thread = None
