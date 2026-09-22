"""
Reflex Instinct Mesh: Distributed Fleet Sync & Peer-to-Peer Active Learning (Phase 19).
Synchronizes learned instinct heads, decision boundaries, and active learning deltas across
fleets of Reflex pods and AI Envoy Gateways without external databases or coordinators.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
import socketserver
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.request
import uuid

from sys1.learning import SelfTuningInstinctHead


def compute_mesh_signature(secret: str, body: bytes) -> str:
    """Compute HMAC-SHA256 signature for mesh payloads."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_mesh_signature(secret: str, body: bytes, signature: str) -> bool:
    """Constant-time verification of HMAC-SHA256 signature."""
    expected = compute_mesh_signature(secret, body)
    return hmac.compare_digest(expected, signature)


def blend_instinct_weights(
    local_head: SelfTuningInstinctHead,
    remote_weights: Dict[str, List[float]],
    remote_biases: Dict[str, float],
    remote_steps: int,
) -> int:
    """
    Federated Weight Blending: Combines local and remote instinct heads
    proportionally weighted by their training sample counts.
    Returns the new combined sample count.
    """
    local_steps = local_head.training_steps
    total_steps = local_steps + remote_steps

    if total_steps <= 0:
        total_steps = 1
    
    if local_steps == 0:
        # Local head has no experience; directly adopt remote
        local_head.weights = copy.deepcopy(remote_weights)
        local_head.biases = copy.deepcopy(remote_biases)
        local_head.training_steps = remote_steps
        return remote_steps

    if remote_steps == 0:
        # Remote head has no experience; preserve local
        return local_steps

    all_keys = set(local_head.weights.keys()) | set(remote_weights.keys())

    for k in all_keys:
        if k in local_head.weights and k in remote_weights:
            lw = local_head.weights[k]
            rw = remote_weights[k]
            lb = local_head.biases.get(k, 0.0)
            rb = remote_biases.get(k, 0.0)

            # Mathematically weighted blending: (n1*w1 + n2*w2) / (n1 + n2)
            blended_w = [
                (local_steps * wi + remote_steps * rwi) / total_steps
                for wi, rwi in zip(lw, rw)
            ]
            blended_b = (local_steps * lb + remote_steps * rb) / total_steps

            local_head.weights[k] = blended_w
            local_head.biases[k] = blended_b
        elif k in remote_weights:
            # Remote introduced a new decision rubric
            local_head.weights[k] = copy.deepcopy(remote_weights[k])
            local_head.biases[k] = remote_biases.get(k, 0.0)

    local_head.training_steps = total_steps
    return total_steps


@dataclass
class MeshConfig:
    """Configuration options for Reflex Instinct Mesh Node."""
    node_id: str = field(default_factory=lambda: f"node-{uuid.uuid4().hex[:8]}")
    host: str = "0.0.0.0"
    port: int = 8080
    peers: List[str] = field(default_factory=list)
    cluster_secret: Optional[str] = os.environ.get("REFLEX_MESH_SECRET", None)
    sync_interval_s: float = 30.0
    auto_broadcast: bool = True
    max_clock_skew_s: float = 60.0


@dataclass
class MeshPeerState:
    """Runtime tracking of a known cluster peer."""
    address: str
    node_id: Optional[str] = None
    last_seen: float = 0.0
    status: str = "unknown"  # "healthy", "unreachable", "unknown"
    last_latency_ms: float = 0.0
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "node_id": self.node_id,
            "last_seen": self.last_seen,
            "status": self.status,
            "last_latency_ms": round(self.last_latency_ms, 2),
            "generation": self.generation,
        }


class InstinctMeshNode:
    """
    Peer-to-peer distributed synchronization node for Reflex instinct heads.
    Handles HMAC validation, background gossip, and federated weight blending.
    """

    def __init__(
        self,
        config: Optional[MeshConfig] = None,
        instinct_head: Optional[SelfTuningInstinctHead] = None,
    ):
        self.config = config or MeshConfig()
        self.head = instinct_head or SelfTuningInstinctHead()
        self.generation: int = 0
        self.peers: Dict[str, MeshPeerState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._gossip_thread: Optional[threading.Thread] = None

        for p in self.config.peers:
            norm_addr = p.rstrip("/")
            self.peers[norm_addr] = MeshPeerState(address=norm_addr)

    def add_peer(self, peer_address: str):
        """Add a peer address to the known mesh topology."""
        norm_addr = peer_address.rstrip("/")
        with self._lock:
            if norm_addr not in self.peers:
                self.peers[norm_addr] = MeshPeerState(address=norm_addr)

    def handle_sync_request(self, body_bytes: bytes, headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        """
        Process incoming POST /v1/mesh/sync request:
        Validates HMAC signature, freshness, and performs federated weight blending.
        """
        # 1. HMAC Security Verification
        if self.config.cluster_secret:
            sig = headers.get("X-Reflex-Signature", "")
            if not sig or not verify_mesh_signature(self.config.cluster_secret, body_bytes, sig):
                return 401, {"error": "Unauthorized: Invalid or missing mesh signature"}

        # 2. Parse payload
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            return 400, {"error": f"Invalid JSON payload: {str(e)}"}

        origin_id = payload.get("origin_node_id", "unknown")
        ts = payload.get("timestamp", 0.0)
        peer_gen = payload.get("generation", 0)

        # 3. Anti-Replay Clock Skew Check
        now = time.time()
        if abs(now - ts) > self.config.max_clock_skew_s:
            return 400, {"error": f"Payload rejected: clock skew exceeds {self.config.max_clock_skew_s}s"}

        # 4. Federated Weight Blending
        remote_w = payload.get("weights", {})
        remote_b = payload.get("biases", {})
        remote_steps = payload.get("training_steps", 0)

        with self._lock:
            new_steps = blend_instinct_weights(
                local_head=self.head,
                remote_weights=remote_w,
                remote_biases=remote_b,
                remote_steps=remote_steps,
            )
            self.generation = max(self.generation + 1, peer_gen + 1)

            # Update peer state
            origin_addr = headers.get("X-Reflex-Origin-Addr", "")
            if origin_addr and origin_addr in self.peers:
                self.peers[origin_addr].last_seen = now
                self.peers[origin_addr].status = "healthy"
                self.peers[origin_addr].generation = peer_gen
                self.peers[origin_addr].node_id = origin_id

        return 200, {
            "status": "synchronized",
            "node_id": self.config.node_id,
            "generation": self.generation,
            "training_steps": new_steps,
        }

    def handle_heartbeat_request(self, body_bytes: bytes, headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        """Process incoming POST /v1/mesh/heartbeat request."""
        if self.config.cluster_secret:
            sig = headers.get("X-Reflex-Signature", "")
            if not sig or not verify_mesh_signature(self.config.cluster_secret, body_bytes, sig):
                return 401, {"error": "Unauthorized: Invalid mesh signature"}

        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            payload = {}

        origin_id = payload.get("origin_node_id", "unknown")
        now = time.time()

        origin_addr = headers.get("X-Reflex-Origin-Addr", "")
        with self._lock:
            if origin_addr and origin_addr in self.peers:
                self.peers[origin_addr].last_seen = now
                self.peers[origin_addr].status = "healthy"
                self.peers[origin_addr].node_id = origin_id

        return 200, {
            "status": "alive",
            "node_id": self.config.node_id,
            "generation": self.generation,
            "timestamp": now,
        }

    def handle_peers_request(self) -> Tuple[int, Dict[str, Any]]:
        """Process GET /v1/mesh/peers request."""
        with self._lock:
            peer_list = [p.to_dict() for p in self.peers.values()]
            return 200, {
                "node_id": self.config.node_id,
                "generation": self.generation,
                "peers": peer_list,
            }

    def broadcast_teach_update(self, sample_delta: int = 1, asynchronous: bool = True):
        """Broadcasts the latest weights to all configured mesh peers."""
        with self._lock:
            self.generation += 1
            payload = {
                "origin_node_id": self.config.node_id,
                "timestamp": time.time(),
                "generation": self.generation,
                "sample_delta": sample_delta,
                "weights": copy.deepcopy(self.head.weights),
                "biases": copy.deepcopy(self.head.biases),
                "training_steps": self.head.training_steps,
            }
            peer_addrs = list(self.peers.keys())

        if not peer_addrs:
            return

        if asynchronous:
            t = threading.Thread(target=self._send_broadcast, args=(peer_addrs, payload), daemon=True)
            t.start()
        else:
            self._send_broadcast(peer_addrs, payload)

    def _send_broadcast(self, peers: List[str], payload: dict):
        """Sends signed sync payload to peer endpoints."""
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Reflex-Origin-Addr": f"http://{self.config.host}:{self.config.port}",
            "X-Reflex-Node-Id": self.config.node_id,
        }
        if self.config.cluster_secret:
            headers["X-Reflex-Signature"] = compute_mesh_signature(self.config.cluster_secret, body)

        for peer in peers:
            url = f"{peer}/v1/mesh/sync"
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    with self._lock:
                        if peer in self.peers:
                            self.peers[peer].last_seen = time.time()
                            self.peers[peer].status = "healthy"
                            self.peers[peer].last_latency_ms = latency_ms
                            self.peers[peer].generation = data.get("generation", 0)
                            self.peers[peer].node_id = data.get("node_id", self.peers[peer].node_id)
            except Exception:
                with self._lock:
                    if peer in self.peers:
                        self.peers[peer].status = "unreachable"

    def ping_peers(self):
        """Sends heartbeat pings to all peers to refresh connectivity metrics."""
        with self._lock:
            peer_addrs = list(self.peers.keys())

        payload = {
            "origin_node_id": self.config.node_id,
            "timestamp": time.time(),
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Reflex-Origin-Addr": f"http://{self.config.host}:{self.config.port}",
        }
        if self.config.cluster_secret:
            headers["X-Reflex-Signature"] = compute_mesh_signature(self.config.cluster_secret, body)

        for peer in peer_addrs:
            url = f"{peer}/v1/mesh/heartbeat"
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    with self._lock:
                        if peer in self.peers:
                            self.peers[peer].last_seen = time.time()
                            self.peers[peer].status = "healthy"
                            self.peers[peer].last_latency_ms = lat_ms
                            self.peers[peer].generation = data.get("generation", 0)
                            self.peers[peer].node_id = data.get("node_id", self.peers[peer].node_id)
            except Exception:
                with self._lock:
                    if peer in self.peers:
                        self.peers[peer].status = "unreachable"

    def start_background_gossip(self):
        """Spawns background gossip thread for periodic heartbeat and anti-entropy."""
        if self._gossip_thread is not None and self._gossip_thread.is_alive():
            return

        self._stop_event.clear()

        def _gossip_loop():
            while not self._stop_event.is_set():
                self.ping_peers()
                self._stop_event.wait(self.config.sync_interval_s)

        self._gossip_thread = threading.Thread(target=_gossip_loop, daemon=True)
        self._gossip_thread.start()

    def stop(self):
        """Stops background gossip workers."""
        self._stop_event.set()
        if self._gossip_thread is not None and self._gossip_thread.is_alive():
            self._gossip_thread.join(timeout=2.0)
            self._gossip_thread = None
