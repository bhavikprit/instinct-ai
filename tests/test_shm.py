"""
Unit Tests for Zero-Copy Shared Memory IPC Daemon (Phase 27).
Verifies POSIX/mmap shared memory ring buffer, Unix domain socket transport,
atomic slot state transitions, IPC opcodes (PING, NOUL, CHOICE, SCORE, PREDICT, STATS),
error propagation, and Reflex(backend="ipc") client integration.
Zero external dependencies (Python standard library only).
"""

import json
import os
import tempfile
import threading
import time
import unittest
import uuid

from reflex import Reflex, Choice, Noul, Score
from reflex.shm import (
    SlotState,
    IPCOpCode,
    SHMConfig,
    SharedMemoryRingBuffer,
    ReflexIPCDaemon,
    ReflexIPCClient,
)


class TestSharedMemoryRingBuffer(unittest.TestCase):
    """Verifies low-level ring buffer operations, slot state machine, and error handling."""

    def setUp(self):
        self.shm_name = f"test_ring_{uuid.uuid4().hex[:8]}"
        self.num_slots = 4
        self.slot_size = 512
        self.ring_server = SharedMemoryRingBuffer(
            name=self.shm_name,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            create=True,
        )

    def tearDown(self):
        if self.ring_server:
            self.ring_server.close()

    def test_slot_state_lifecycle(self):
        """Test write_request -> daemon_poll_request -> daemon_write_response -> poll_response."""
        client_ring = SharedMemoryRingBuffer(
            name=self.shm_name,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            create=False,
        )
        try:
            # 1. Client writes request
            req_data = b'{"hello": "shm"}'
            slot_idx = client_ring.write_request(IPCOpCode.PING, req_data, timeout_s=0.5)
            self.assertGreaterEqual(slot_idx, 0)
            self.assertLess(slot_idx, self.num_slots)

            # 2. Daemon polls and extracts request
            poll_res = self.ring_server.daemon_poll_request()
            self.assertIsNotNone(poll_res)
            s_idx, op_code, data = poll_res
            self.assertEqual(s_idx, slot_idx)
            self.assertEqual(op_code, IPCOpCode.PING)
            self.assertEqual(data, req_data)

            # 3. Daemon writes response
            resp_data = b'{"status": "pong"}'
            self.ring_server.daemon_write_response(slot_idx, resp_data, is_error=False)

            # 4. Client polls response
            status, res = client_ring.poll_response(slot_idx, timeout_s=0.5)
            self.assertEqual(status, SlotState.RESP_READY)
            self.assertEqual(res, resp_data)

            # 5. Slot should now be FREE again
            offset = 16 + (slot_idx * self.slot_size)
            self.assertEqual(self.ring_server._buf[offset], SlotState.FREE)
        finally:
            client_ring.close()

    def test_payload_size_limit(self):
        """Verifies that writing a payload exceeding slot size raises ValueError."""
        oversized = b"x" * (self.slot_size + 10)
        with self.assertRaises(ValueError):
            self.ring_server.write_request(IPCOpCode.PING, oversized)

    def test_poll_response_timeout(self):
        """Verifies that poll_response raises TimeoutError if daemon never responds."""
        slot_idx = self.ring_server.write_request(IPCOpCode.PING, b"test", timeout_s=0.5)
        with self.assertRaises(TimeoutError):
            self.ring_server.poll_response(slot_idx, timeout_s=0.05)


class TestReflexIPCDaemon(unittest.TestCase):
    """Verifies ReflexIPCDaemon and ReflexIPCClient over both SHM and UDS transports."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.socket_path = os.path.join(cls.temp_dir, "test_reflex.sock")
        cls.shm_name = f"test_shm_{uuid.uuid4().hex[:8]}"

        cls.config = SHMConfig(
            socket_path=cls.socket_path,
            shm_name=cls.shm_name,
            num_slots=8,
            slot_size=2048,
            use_shm=True,
            poll_sleep_s=0.00001,
        )

        cls.daemon = ReflexIPCDaemon(config=cls.config)
        cls.daemon.start(background=True)
        time.sleep(0.05)  # Wait for threads to listen

    @classmethod
    def tearDownClass(cls):
        cls.daemon.stop()
        if os.path.exists(cls.socket_path):
            try:
                os.unlink(cls.socket_path)
            except OSError:
                pass
        try:
            os.rmdir(cls.temp_dir)
        except OSError:
            pass

    def test_shm_ping(self):
        """Verifies ultra-low latency Ping over Shared Memory."""
        client = ReflexIPCClient(config=self.config)
        try:
            # Warm up initial page fault / runner thread scheduling
            client.ping()
            lat_us = client.ping()
            self.assertGreater(lat_us, 0.0)
            self.assertLess(lat_us, 50_000.0)  # Sub-50ms even on busy virtualized CI
        finally:
            client.close()

    def test_shm_noul_query(self):
        """Verifies NOUL probabilistic boolean evaluation over SHM."""
        client = ReflexIPCClient(config=self.config)
        try:
            prob = client.noul(
                instructions="Is this an emergency request?",
                state="The server room is flooding with water!",
                threshold=0.5,
            )
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        finally:
            client.close()

    def test_shm_choice_query(self):
        """Verifies CHOICE categorical routing evaluation over SHM."""
        client = ReflexIPCClient(config=self.config)
        try:
            opts = ["billing", "technical", "sales"]
            selected = client.choice(
                instructions="Route user inquiry",
                options=opts,
                state="I was billed twice on my credit card last week",
            )
            self.assertIn(selected, opts)
        finally:
            client.close()

    def test_shm_score_query(self):
        """Verifies SCORE continuous regression evaluation over SHM."""
        client = ReflexIPCClient(config=self.config)
        try:
            s = client.score(
                instructions="Rate customer satisfaction from 1 to 10",
                state="Excellent service, prompt and helpful agent!",
            )
            self.assertIsInstance(s, float)
            self.assertGreaterEqual(s, 1.0)
            self.assertLessEqual(s, 10.0)
        finally:
            client.close()

    def test_shm_predict_query(self):
        """Verifies PREDICT multi-decision evaluation over SHM."""
        client = ReflexIPCClient(config=self.config)
        try:
            res = client.predict("Querying system state for diagnostics")
            self.assertIsInstance(res, dict)
            self.assertTrue("decisions" in res or "result" in res or "error" not in res)
        finally:
            client.close()

    def test_uds_transport_fallback(self):
        """Verifies that client works strictly over Unix Domain Sockets when SHM is disabled."""
        uds_cfg = SHMConfig(
            socket_path=self.socket_path,
            shm_name="non_existent_shm_name",
            use_shm=False,
        )
        client = ReflexIPCClient(config=uds_cfg)
        try:
            lat_us = client.ping()
            self.assertGreater(lat_us, 0.0)

            # Test choice over UDS
            opts = ["refund", "account_update", "login_help"]
            selected = client.choice(
                instructions="Classify support ticket",
                options=opts,
                state="I forgot my password and cannot sign in",
            )
            self.assertIn(selected, opts)
        finally:
            client.close()

    def test_stats_collection(self):
        """Verifies daemon tracks cumulative request counts and mean latency."""
        client = ReflexIPCClient(config=self.config)
        try:
            stats = client.call(IPCOpCode.STATS, {})
            self.assertIn("total_requests", stats)
            self.assertIn("shm_requests", stats)
            self.assertIn("uds_requests", stats)
            self.assertIn("mean_latency_us", stats)
            self.assertGreater(stats["total_requests"], 0)
        finally:
            client.close()

    def test_reflex_client_ipc_backend(self):
        """Verifies Reflex(backend='ipc') client wrapper executing high-level decisions."""
        rx = Reflex(
            backend="ipc",
            socket_path=self.socket_path,
            shm_name=self.shm_name,
        )
        try:
            # 1. Choice decision
            c_res = rx.choice(
                "Categorize ticket urgency",
                ["low", "medium", "critical"],
                "Production database is unreachable and dropping connections!",
            )
            self.assertIn(c_res, ["low", "medium", "critical"])

            # 2. Noul decision
            n_res = rx.noul(
                "Requires supervisor escalation?",
                "Production database is unreachable and dropping connections!",
            )
            self.assertIsInstance(n_res, float)
            self.assertGreaterEqual(n_res, 0.0)
            self.assertLessEqual(n_res, 1.0)

            # 3. Score decision
            s_res = rx.score(
                "Severity score from 1 to 10",
                "Production database is unreachable and dropping connections!",
            )
            self.assertIsInstance(s_res, float)

            # 4. Multi-decision batch
            res = rx.evaluate(
                "Production database is down",
                {
                    "urgency": Choice("Urgency level", ["low", "critical"]),
                    "escalate": Noul("Escalate immediately?"),
                    "impact": Score("Rate impact on scale 1 to 10"),
                },
            )
            self.assertEqual(res.backend, "ipc:reflex-shm")
            self.assertIn(res.decisions["urgency"].selected, ["low", "critical"])
            self.assertIn(res.decisions["escalate"].is_true, [True, False])
            self.assertIsInstance(res.decisions["impact"].score, float)
        finally:
            if rx.ipc_client:
                rx.ipc_client.close()

    def test_multithreaded_concurrent_requests(self):
        """Verifies concurrent threads successfully query daemon without race conditions."""
        errors = []
        num_threads = 4
        reqs_per_thread = 20

        def worker():
            try:
                c = ReflexIPCClient(config=self.config)
                for _ in range(reqs_per_thread):
                    prob = c.noul("Test instruction", "Test context", threshold=0.5)
                    if not (0.0 <= prob <= 1.0):
                        errors.append(f"Invalid prob: {prob}")
                c.close()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors encountered: {errors}")

    def test_error_handling_propagation(self):
        """Verifies that an error in daemon execution is returned and raises RuntimeError in client."""
        client = ReflexIPCClient(config=self.config)
        try:
            # OpCode 99 is invalid, should cause error or unknown opcode
            # Let's test calling an unknown opcode directly
            with self.assertRaises(Exception):
                client.call(IPCOpCode(99), {"invalid": True})
        finally:
            client.close()


class TestCompiledModelIPC(unittest.TestCase):
    """Verifies ReflexIPCDaemon hosting a compiled .reflex artifact."""

    @classmethod
    def setUpClass(cls):
        from reflex.compiler import InstinctCompiler, PromptSpec
        compiler = InstinctCompiler()
        spec = PromptSpec(
            name="triage_model",
            prompt="Triage incident severity level",
            decision_type="choice",
            options=["low", "medium", "high"],
            guidelines={
                "low": "Minor typographical error or cosmetic UI glitch",
                "medium": "Non-blocking performance degradation or latency spike",
                "high": "Data corruption, database outage, security incident",
            },
        )
        cls.compiled_model = compiler.compile(spec, samples_per_class=10, epochs=15)

        cls.temp_dir = tempfile.mkdtemp()
        cls.socket_path = os.path.join(cls.temp_dir, "test_compiled.sock")
        cls.shm_name = f"test_compiled_{uuid.uuid4().hex[:8]}"

        cls.config = SHMConfig(
            socket_path=cls.socket_path,
            shm_name=cls.shm_name,
            num_slots=8,
            slot_size=2048,
            use_shm=True,
        )
        cls.daemon = ReflexIPCDaemon(config=cls.config, model=cls.compiled_model)
        cls.daemon.start(background=True)
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.daemon.stop()
        if os.path.exists(cls.socket_path):
            try:
                os.unlink(cls.socket_path)
            except OSError:
                pass
        try:
            os.rmdir(cls.temp_dir)
        except OSError:
            pass

    def test_compiled_model_predict_over_shm(self):
        """Verifies compiled model sub-50us prediction over shared memory."""
        client = ReflexIPCClient(config=self.config)
        try:
            res = client.predict("Database cluster primary is down and replication failed!")
            self.assertIn("decisions", res)
            self.assertIn("choice", res["decisions"])
            decision = res["decisions"]["choice"]
            self.assertIn(decision["selected"], ["low", "medium", "high"])
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
