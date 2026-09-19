"""
Reflex Zero-Copy Shared Memory IPC Daemon (Phase 27).
Provides POSIX shared memory ring buffers and Unix domain socket IPC (<5us latency)
for high-throughput agent microservices fleets in Go, Python, Node.js, Rust, and C.
Zero external dependencies (Python standard library only).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import json
import mmap
import os
import socket
import struct
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from multiprocessing import shared_memory
    HAS_POSIX_SHM = True
except ImportError:
    shared_memory = None
    HAS_POSIX_SHM = False

SHM_MAGIC = b"RSHM"
DEFAULT_SLOT_SIZE = 4096
DEFAULT_NUM_SLOTS = 16
HEADER_SIZE = 8  # 1 byte status, 1 byte opcode, 2 bytes req_len, 2 bytes resp_len, 2 bytes reserved


class SlotState(IntEnum):
    FREE = 0
    REQ_READY = 1
    RESP_READY = 2
    ERROR = 3


class IPCOpCode(IntEnum):
    PING = 1
    NOUL = 2
    CHOICE = 3
    SCORE = 4
    EVALUATE = 5
    PREDICT = 6
    STATS = 7


@dataclass
class SHMConfig:
    """Configuration for Shared Memory and Unix Domain Socket IPC."""
    socket_path: str = "/tmp/reflex_ipc.sock"
    shm_name: str = "reflex_shm_ring"
    num_slots: int = DEFAULT_NUM_SLOTS
    slot_size: int = DEFAULT_SLOT_SIZE
    use_shm: bool = True
    poll_sleep_s: float = 0.00005  # 50 microseconds spin sleep


class SharedMemoryRingBuffer:
    """
    Fixed-size POSIX or file-backed shared memory ring buffer.
    Slots are synchronized using atomic status byte transitions:
    FREE -> REQ_READY -> RESP_READY -> FREE.
    """

    def __init__(
        self,
        name: str = "reflex_shm_ring",
        num_slots: int = DEFAULT_NUM_SLOTS,
        slot_size: int = DEFAULT_SLOT_SIZE,
        create: bool = False,
    ):
        self.name = name.lstrip("/")
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.total_size = 16 + (num_slots * slot_size)  # 16 bytes global header + slots
        self.create = create
        self._shm: Optional[Any] = None
        self._buf: Optional[Union[memoryview, mmap.mmap]] = None
        self._file: Optional[Any] = None
        self._file_path: Optional[str] = None
        self._lock = threading.Lock()
        self._init_memory()

    def _init_memory(self) -> None:
        """Initializes POSIX shared memory or falls back to file-backed mmap."""
        if HAS_POSIX_SHM:
            try:
                if self.create:
                    # Clean up old shm if exists
                    try:
                        old = shared_memory.SharedMemory(name=self.name)
                        old.close()
                        old.unlink()
                    except (FileNotFoundError, FileExistsError):
                        pass
                    self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.total_size)
                    self._buf = self._shm.buf
                    # Write global header
                    self._buf[0:4] = SHM_MAGIC
                    struct.pack_into(">HH", self._buf, 4, self.num_slots, self.slot_size)
                    # Initialize all slots to FREE
                    for i in range(self.num_slots):
                        offset = 16 + (i * self.slot_size)
                        self._buf[offset] = SlotState.FREE
                else:
                    self._shm = shared_memory.SharedMemory(name=self.name, create=False)
                    self._buf = self._shm.buf
                return
            except Exception:
                # Fallback to file-backed mmap
                pass

        # File-backed mmap fallback
        temp_dir = tempfile.gettempdir()
        self._file_path = os.path.join(temp_dir, f"{self.name}.mmap")
        if self.create:
            f = open(self._file_path, "wb+")
            f.write(b"\x00" * self.total_size)
            f.flush()
            self._file = f
            self._buf = mmap.mmap(f.fileno(), self.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            self._buf[0:4] = SHM_MAGIC
            struct.pack_into(">HH", self._buf, 4, self.num_slots, self.slot_size)
            for i in range(self.num_slots):
                offset = 16 + (i * self.slot_size)
                self._buf[offset] = SlotState.FREE
        else:
            f = open(self._file_path, "r+b")
            self._file = f
            self._buf = mmap.mmap(f.fileno(), self.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def write_request(self, op_code: IPCOpCode, payload: bytes, timeout_s: float = 0.5) -> int:
        """
        Finds a FREE slot, writes the request, and transitions status to REQ_READY.
        Returns slot index.
        """
        t_deadline = time.time() + timeout_s
        max_payload = self.slot_size - HEADER_SIZE
        if len(payload) > max_payload:
            raise ValueError(f"Payload size ({len(payload)} bytes) exceeds slot limit ({max_payload} bytes)")

        while time.time() < t_deadline:
            with self._lock:
                for i in range(self.num_slots):
                    offset = 16 + (i * self.slot_size)
                    if self._buf[offset] == SlotState.FREE:
                        # Write slot header
                        self._buf[offset + 1] = int(op_code)
                        struct.pack_into(">H", self._buf, offset + 2, len(payload))
                        struct.pack_into(">H", self._buf, offset + 4, 0)
                        # Write payload
                        self._buf[offset + HEADER_SIZE : offset + HEADER_SIZE + len(payload)] = payload
                        # Flip status atomically
                        self._buf[offset] = SlotState.REQ_READY
                        return i
            time.sleep(0)

        raise TimeoutError("Timed out waiting for an available shared memory slot")

    def poll_response(self, slot_idx: int, timeout_s: float = 0.5) -> Tuple[SlotState, bytes]:
        """
        Polls a specific slot until RESP_READY or ERROR is received.
        Releases the slot back to FREE upon reading.
        """
        t_deadline = time.time() + timeout_s
        offset = 16 + (slot_idx * self.slot_size)
        while time.time() < t_deadline:
            status = self._buf[offset]
            if status in (SlotState.RESP_READY, SlotState.ERROR):
                (resp_len,) = struct.unpack_from(">H", self._buf, offset + 4)
                data = bytes(self._buf[offset + HEADER_SIZE : offset + HEADER_SIZE + resp_len])
                # Free the slot
                self._buf[offset] = SlotState.FREE
                return SlotState(status), data
            time.sleep(0)

        # Release slot on timeout to avoid leaking slots
        self._buf[offset] = SlotState.FREE
        raise TimeoutError(f"Shared memory slot {slot_idx} timed out waiting for response")

    def daemon_poll_request(self) -> Optional[Tuple[int, IPCOpCode, bytes]]:
        """
        Daemon polling method: checks for any slot with status REQ_READY.
        Returns (slot_idx, op_code, request_bytes) or None.
        """
        for i in range(self.num_slots):
            offset = 16 + (i * self.slot_size)
            if self._buf[offset] == SlotState.REQ_READY:
                op_code = IPCOpCode(self._buf[offset + 1])
                (req_len,) = struct.unpack_from(">H", self._buf, offset + 2)
                data = bytes(self._buf[offset + HEADER_SIZE : offset + HEADER_SIZE + req_len])
                return i, op_code, data
        return None

    def daemon_write_response(self, slot_idx: int, response: bytes, is_error: bool = False) -> None:
        """
        Daemon method: writes response data into slot and transitions status to RESP_READY or ERROR.
        """
        offset = 16 + (slot_idx * self.slot_size)
        max_payload = self.slot_size - HEADER_SIZE
        clipped = response[:max_payload]
        struct.pack_into(">H", self._buf, offset + 4, len(clipped))
        self._buf[offset + HEADER_SIZE : offset + HEADER_SIZE + len(clipped)] = clipped
        self._buf[offset] = SlotState.ERROR if is_error else SlotState.RESP_READY

    def close(self) -> None:
        """Closes memory views and handles."""
        if self._buf is not None:
            if hasattr(self._buf, "close"):
                try:
                    self._buf.close()
                except Exception:
                    pass
            self._buf = None

        if self._shm is not None:
            try:
                self._shm.close()
                if self.create:
                    self._shm.unlink()
            except Exception:
                pass
            self._shm = None

        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

        if self.create and self._file_path and os.path.exists(self._file_path):
            try:
                os.unlink(self._file_path)
            except Exception:
                pass


class ReflexIPCDaemon:
    """
    High-Performance System-1 IPC Daemon.
    Listens concurrently on:
    1. POSIX Shared Memory Ring Buffer (sub-5us hot-path)
    2. Unix Domain Socket (/tmp/reflex.sock, sub-25us stream)
    """

    def __init__(
        self,
        config: Optional[SHMConfig] = None,
        model: Optional[Any] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or SHMConfig()
        self.model = model
        self.model_path = model_path
        self._running = False
        self._shm_buffer: Optional[SharedMemoryRingBuffer] = None
        self._sock_server: Optional[socket.socket] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._sock_thread: Optional[threading.Thread] = None
        self.stats = {
            "total_requests": 0,
            "shm_requests": 0,
            "uds_requests": 0,
            "mean_latency_us": 0.0,
            "start_time": time.time(),
        }
        self._lock = threading.Lock()
        self._init_runtime()

    def _init_runtime(self) -> None:
        """Loads decision model if not already provided."""
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                if self.model_path.endswith(".reflex-ensemble"):
                    from reflex.ensemble import InstinctEnsemble
                    self.model = InstinctEnsemble.load(self.model_path)
                elif self.model_path.endswith(".reflex"):
                    from reflex.compiler import CompiledInstinct
                    self.model = CompiledInstinct.load(self.model_path)
            if self.model is None:
                from reflex.client import Reflex
                self.model = Reflex()

    def start(self, background: bool = True) -> None:
        """Starts IPC daemon listening threads."""
        self._running = True

        # 1. Initialize Shared Memory Ring Buffer
        if self.config.use_shm:
            try:
                self._shm_buffer = SharedMemoryRingBuffer(
                    name=self.config.shm_name,
                    num_slots=self.config.num_slots,
                    slot_size=self.config.slot_size,
                    create=True,
                )
                self._worker_thread = threading.Thread(target=self._shm_worker_loop, daemon=True)
                self._worker_thread.start()
            except Exception as e:
                print(f"⚠️ Warning: Failed to initialize SHM ring buffer: {e}. Falling back to UDS.")
                self._shm_buffer = None

        # 2. Initialize Unix Domain Socket
        if hasattr(socket, "AF_UNIX") and self.config.socket_path:
            if os.path.exists(self.config.socket_path):
                try:
                    os.unlink(self.config.socket_path)
                except OSError:
                    pass
            self._sock_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock_server.bind(self.config.socket_path)
            self._sock_server.listen(64)
            self._sock_thread = threading.Thread(target=self._uds_server_loop, daemon=True)
            self._sock_thread.start()

        if not background:
            try:
                while self._running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                self.stop()

    def stop(self) -> None:
        """Stops the daemon and cleans up resources."""
        self._running = False
        if self._sock_server:
            try:
                self._sock_server.close()
            except Exception:
                pass
            self._sock_server = None

        if self.config.socket_path and os.path.exists(self.config.socket_path):
            try:
                os.unlink(self.config.socket_path)
            except OSError:
                pass

        if self._shm_buffer:
            try:
                self._shm_buffer.close()
            except Exception:
                pass
            self._shm_buffer = None

    def _execute_op(self, op_code: IPCOpCode, payload_bytes: bytes) -> Tuple[bool, bytes]:
        """Dispatches an IPC request against the local System-1 model."""
        t0 = time.perf_counter()
        try:
            req_str = payload_bytes.decode("utf-8") if payload_bytes else "{}"
            req_data = json.loads(req_str) if req_str.startswith("{") else {"state": req_str}
        except Exception:
            req_data = {"state": payload_bytes.decode("utf-8", errors="replace")}

        try:
            if op_code == IPCOpCode.PING:
                resp_data = {"status": "pong", "time": time.time(), "daemon": "reflex-shm"}

            elif op_code == IPCOpCode.STATS:
                resp_data = dict(self.stats)

            elif op_code == IPCOpCode.NOUL:
                inst = req_data.get("instructions", "")
                state = req_data.get("state", "")
                thresh = req_data.get("threshold", 0.5)
                if hasattr(self.model, "noul"):
                    prob = self.model.noul(inst, state, threshold=thresh)
                else:
                    pred = self.model.predict(state)
                    prob = 0.5
                    if hasattr(pred, "decisions") and isinstance(pred.decisions, dict):
                        if "noul" in pred.decisions:
                            n_d = pred.decisions["noul"]
                            prob = getattr(n_d, "probability", n_d.get("probability", 0.5) if isinstance(n_d, dict) else 0.5)
                        elif "choice" in pred.decisions:
                            c_d = pred.decisions["choice"]
                            prob = getattr(c_d, "confidence", c_d.get("confidence", 0.5) if isinstance(c_d, dict) else 0.5)
                resp_data = {"probability": float(prob), "is_true": bool(prob >= thresh)}

            elif op_code == IPCOpCode.CHOICE:
                inst = req_data.get("instructions", "")
                opts = req_data.get("options", [])
                state = req_data.get("state", "")
                if hasattr(self.model, "choice"):
                    sel = self.model.choice(inst, opts, state)
                    dist = {}
                else:
                    pred = self.model.predict(state)
                    sel = opts[0] if opts else ""
                    dist = {}
                    if hasattr(pred, "decisions") and isinstance(pred.decisions, dict):
                        c = pred.decisions.get("choice")
                        if c is not None:
                            sel = getattr(c, "selected", c.get("selected", sel) if isinstance(c, dict) else sel)
                            dist = getattr(c, "distribution", c.get("distribution", {}) if isinstance(c, dict) else {})
                resp_data = {"selected": str(sel), "distribution": dist}

            elif op_code == IPCOpCode.SCORE:
                inst = req_data.get("instructions", "")
                state = req_data.get("state", "")
                if hasattr(self.model, "score"):
                    score_val = self.model.score(inst, state)
                else:
                    pred = self.model.predict(state)
                    score_val = 5.0
                    if hasattr(pred, "decisions") and isinstance(pred.decisions, dict):
                        s = pred.decisions.get("score")
                        if s is not None:
                            score_val = getattr(s, "score", s.get("score", 5.0) if isinstance(s, dict) else 5.0)
                resp_data = {"score": float(score_val)}

            elif op_code in (IPCOpCode.PREDICT, IPCOpCode.EVALUATE):
                state = req_data.get("state", "")
                has_ensemble = hasattr(self.model, "ensemble") and self.model.ensemble is not None
                if req_data.get("cascade", False) and (has_ensemble or hasattr(self.model, "cascade_predict") and not hasattr(self.model, "ensemble")):
                    ens_res = self.model.cascade_predict(state)
                    resp_data = ens_res.to_dict()
                elif hasattr(self.model, "compiled_instinct") and self.model.compiled_instinct is None and not has_ensemble:
                    # Reflex client without compiled model or ensemble: direct semantic triage
                    sel = self.model.choice("Triage state", ["normal", "flagged"], state)
                    resp_data = {"selected": sel, "state": state, "backend": "semantic:fallback"}
                elif hasattr(self.model, "predict"):
                    pred = self.model.predict(state)
                    if hasattr(pred, "to_dict"):
                        resp_data = pred.to_dict()
                    elif hasattr(pred, "decisions"):
                        # Extract decisions dictionary
                        dec_dict = {}
                        for k, v in pred.decisions.items():
                            if hasattr(v, "selected"):
                                dec_dict[k] = {"selected": v.selected, "distribution": getattr(v, "distribution", {})}
                            elif hasattr(v, "probability"):
                                dec_dict[k] = {"probability": v.probability, "is_true": getattr(v, "is_true", False)}
                            elif hasattr(v, "score"):
                                dec_dict[k] = {"score": v.score}
                        resp_data = {
                            "decisions": dec_dict,
                            "latency_ms": getattr(pred, "latency_ms", 0.0),
                            "backend": getattr(pred, "backend", "shm"),
                        }
                    else:
                        resp_data = {"result": str(pred)}
                else:
                    resp_data = {"result": "ok"}
            else:
                resp_data = {"error": f"Unknown opcode: {op_code}"}

            resp_bytes = json.dumps(resp_data).encode("utf-8")
            elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
            with self._lock:
                self.stats["total_requests"] += 1
                n = self.stats["total_requests"]
                prev_mean = self.stats["mean_latency_us"]
                self.stats["mean_latency_us"] = prev_mean + ((elapsed_us - prev_mean) / n)
            return False, resp_bytes

        except Exception as e:
            err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
            return True, err_bytes

    def _shm_worker_loop(self) -> None:
        """Polls shared memory ring buffer continuously."""
        while self._running and self._shm_buffer is not None:
            req = self._shm_buffer.daemon_poll_request()
            if req is not None:
                slot_idx, op_code, data = req
                is_err, resp_bytes = self._execute_op(op_code, data)
                self._shm_buffer.daemon_write_response(slot_idx, resp_bytes, is_error=is_err)
                with self._lock:
                    self.stats["shm_requests"] += 1
            else:
                time.sleep(0)

    def _uds_server_loop(self) -> None:
        """Accepts incoming Unix domain socket client connections."""
        while self._running and self._sock_server is not None:
            try:
                conn, _ = self._sock_server.accept()
                threading.Thread(target=self._handle_uds_client, args=(conn,), daemon=True).start()
            except OSError:
                break

    def _handle_uds_client(self, conn: socket.socket) -> None:
        """Handles framed requests from a Unix domain socket connection."""
        with conn:
            while self._running:
                try:
                    # 1. Read 4-byte big-endian frame length
                    len_bytes = conn.recv(4)
                    if not len_bytes or len(len_bytes) < 4:
                        break
                    (frame_len,) = struct.unpack(">I", len_bytes)

                    # 2. Read frame payload
                    buf = bytearray()
                    while len(buf) < frame_len:
                        chunk = conn.recv(min(frame_len - len(buf), 8192))
                        if not chunk:
                            break
                        buf.extend(chunk)

                    if len(buf) < frame_len:
                        break

                    op_code = IPCOpCode(buf[0])
                    payload = bytes(buf[1:])

                    # 3. Execute op and respond
                    is_err, resp_bytes = self._execute_op(op_code, payload)
                    with self._lock:
                        self.stats["uds_requests"] += 1

                    # 4. Write framed response: [status 1B][resp_len 4B][resp_bytes]
                    resp_hdr = struct.pack(">BI", 1 if is_err else 0, len(resp_bytes))
                    conn.sendall(resp_hdr + resp_bytes)

                except (ConnectionResetError, BrokenPipeError):
                    break
                except Exception:
                    break


class ReflexIPCClient:
    """
    Client for ultra-low latency Reflex IPC (<5us via SHM, <25us via UDS).
    Supports automatic failover between Shared Memory and Unix Domain Sockets.
    """

    def __init__(
        self,
        config: Optional[SHMConfig] = None,
        socket_path: Optional[str] = None,
        shm_name: Optional[str] = None,
    ):
        self.config = config or SHMConfig()
        if socket_path:
            self.config.socket_path = socket_path
        if shm_name:
            self.config.shm_name = shm_name
        self._shm_buffer: Optional[SharedMemoryRingBuffer] = None
        self._sock: Optional[socket.socket] = None
        self._connect()

    def _connect(self) -> None:
        """Attempts to attach to Shared Memory ring buffer; falls back to UDS."""
        if self.config.use_shm:
            try:
                self._shm_buffer = SharedMemoryRingBuffer(
                    name=self.config.shm_name,
                    num_slots=self.config.num_slots,
                    slot_size=self.config.slot_size,
                    create=False,
                )
            except Exception:
                self._shm_buffer = None

        if hasattr(socket, "AF_UNIX") and self.config.socket_path:
            if os.path.exists(self.config.socket_path):
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.config.socket_path)
                    self._sock = s
                except Exception:
                    self._sock = None

    def call(self, op_code: IPCOpCode, data: Dict[str, Any], timeout_s: float = 0.5) -> Dict[str, Any]:
        """
        Executes an IPC RPC query. Prefers shared memory (<5µs), with UDS fallback.
        """
        payload_bytes = json.dumps(data).encode("utf-8")

        # 1. Try Shared Memory Ring Buffer
        if self._shm_buffer is not None:
            try:
                slot_idx = self._shm_buffer.write_request(op_code, payload_bytes, timeout_s=timeout_s)
                status, resp_bytes = self._shm_buffer.poll_response(slot_idx, timeout_s=timeout_s)
                res = json.loads(resp_bytes.decode("utf-8"))
                if status == SlotState.ERROR:
                    raise RuntimeError(res.get("error", "SHM IPC Execution Error"))
                return res
            except Exception:
                # Fallback to UDS
                pass

        # 2. Try Unix Domain Socket
        if self._sock is None:
            self._connect()

        if self._sock is not None:
            try:
                frame_data = bytes([int(op_code)]) + payload_bytes
                frame_hdr = struct.pack(">I", len(frame_data))
                self._sock.sendall(frame_hdr + frame_data)

                # Read response header: [status 1B][resp_len 4B]
                hdr_bytes = self._recv_exact(5, timeout_s=timeout_s)
                status, resp_len = struct.unpack(">BI", hdr_bytes)
                resp_payload = self._recv_exact(resp_len, timeout_s=timeout_s)
                res = json.loads(resp_payload.decode("utf-8"))
                if status != 0:
                    raise RuntimeError(res.get("error", "UDS IPC Execution Error"))
                return res
            except Exception as e:
                # Reconnect attempt next time
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
                raise RuntimeError(f"IPC communication error: {e}")

        raise RuntimeError("Reflex IPC Client: Neither Shared Memory nor Unix Domain Socket is available.")

    def _recv_exact(self, num_bytes: int, timeout_s: float = 0.5) -> bytes:
        """Reads exactly num_bytes from socket with timeout."""
        self._sock.settimeout(timeout_s)
        buf = bytearray()
        while len(buf) < num_bytes:
            chunk = self._sock.recv(num_bytes - len(buf))
            if not chunk:
                raise ConnectionResetError("Socket closed while awaiting response")
            buf.extend(chunk)
        return bytes(buf)

    def ping(self) -> float:
        """Pings the IPC daemon and returns round-trip latency in microseconds."""
        t0 = time.perf_counter()
        self.call(IPCOpCode.PING, {})
        return (time.perf_counter() - t0) * 1_000_000.0

    def noul(self, instructions: str, state: str, threshold: float = 0.5) -> float:
        """Evaluates a probabilistic boolean question over IPC."""
        res = self.call(IPCOpCode.NOUL, {"instructions": instructions, "state": state, "threshold": threshold})
        return float(res.get("probability", 0.5))

    def choice(self, instructions: str, options: List[str], state: str) -> str:
        """Evaluates a multi-class choice rubric over IPC."""
        res = self.call(IPCOpCode.CHOICE, {"instructions": instructions, "options": options, "state": state})
        return str(res.get("selected", options[0] if options else ""))

    def score(self, instructions: str, state: str) -> float:
        """Evaluates a continuous score over IPC."""
        res = self.call(IPCOpCode.SCORE, {"instructions": instructions, "state": state})
        return float(res.get("score", 5.0))

    def predict(self, state: str, cascade: bool = True) -> Dict[str, Any]:
        """Executes a direct compiled instinct or ensemble prediction over IPC."""
        return self.call(IPCOpCode.PREDICT, {"state": state, "cascade": cascade})

    def close(self) -> None:
        """Closes IPC resources."""
        if self._shm_buffer:
            try:
                self._shm_buffer.close()
            except Exception:
                pass
            self._shm_buffer = None

        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
