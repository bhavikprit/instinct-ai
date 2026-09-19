"""
Unit and integration tests for Reflex Instinct Mesh (Phase 19).
Validates HMAC-SHA256 authentication, anti-replay protection,
federated weight blending, and multi-node peer-to-peer synchronization.
"""

import copy
import json
import time
import unittest
import urllib.request
import urllib.error

from reflex.learning import SelfTuningInstinctHead
from reflex.mesh import (
    MeshConfig,
    MeshPeerState,
    InstinctMeshNode,
    compute_mesh_signature,
    verify_mesh_signature,
    blend_instinct_weights,
)
from reflex.gateway import ReflexGatewayServer, GatewayConfig
from reflex.client import Reflex


class TestInstinctMesh(unittest.TestCase):
    """Unit tests for core cryptographic and mathematical mesh primitives."""

    def test_hmac_signatures(self):
        secret = "super-secret-cluster-key-42"
        body = b'{"origin": "pod-1", "generation": 1}'

        sig = compute_mesh_signature(secret, body)
        self.assertIsInstance(sig, str)
        self.assertEqual(len(sig), 64)  # SHA-256 hex string

        # Valid verification
        self.assertTrue(verify_mesh_signature(secret, body, sig))

        # Tampered body verification
        tampered_body = b'{"origin": "pod-1", "generation": 2}'
        self.assertFalse(verify_mesh_signature(secret, tampered_body, sig))

        # Tampered secret
        self.assertFalse(verify_mesh_signature("wrong-secret", body, sig))

    def test_federated_weight_blending(self):
        head_local = SelfTuningInstinctHead()
        head_local.training_steps = 10
        head_local.weights = {"is_spam": [1.0] * 384}
        head_local.biases = {"is_spam": 0.5}

        remote_weights = {"is_spam": [4.0] * 384, "is_fraud": [2.0] * 384}
        remote_biases = {"is_spam": 2.0, "is_fraud": 1.0}
        remote_steps = 20

        new_steps = blend_instinct_weights(head_local, remote_weights, remote_biases, remote_steps)
        self.assertEqual(new_steps, 30)
        self.assertEqual(head_local.training_steps, 30)

        # Expected blended weight: (10*1.0 + 20*4.0) / 30 = 90 / 30 = 3.0
        self.assertAlmostEqual(head_local.weights["is_spam"][0], 3.0, places=5)
        # Expected blended bias: (10*0.5 + 20*2.0) / 30 = 45 / 30 = 1.5
        self.assertAlmostEqual(head_local.biases["is_spam"], 1.5, places=5)

        # Newly introduced rubric from remote
        self.assertIn("is_fraud", head_local.weights)
        self.assertEqual(head_local.weights["is_fraud"][0], 2.0)

    def test_adoption_when_local_steps_zero(self):
        head_local = SelfTuningInstinctHead()
        self.assertEqual(head_local.training_steps, 0)

        remote_weights = {"is_vip": [3.14] * 384}
        remote_biases = {"is_vip": -0.5}
        remote_steps = 5

        new_steps = blend_instinct_weights(head_local, remote_weights, remote_biases, remote_steps)
        self.assertEqual(new_steps, 5)
        self.assertEqual(head_local.weights["is_vip"][0], 3.14)
        self.assertEqual(head_local.biases["is_vip"], -0.5)

    def test_mesh_node_authentication_and_clock_skew(self):
        cfg = MeshConfig(
            node_id="node-test-1",
            cluster_secret="alpha-secret",
            max_clock_skew_s=30.0,
        )
        node = InstinctMeshNode(config=cfg)

        payload = {
            "origin_node_id": "node-test-2",
            "timestamp": time.time(),
            "generation": 1,
            "training_steps": 5,
            "weights": {"is_triage": [0.5] * 384},
            "biases": {"is_triage": 0.1},
        }
        body = json.dumps(payload).encode("utf-8")

        # 1. Missing signature -> 401
        code, resp = node.handle_sync_request(body, {})
        self.assertEqual(code, 401)
        self.assertIn("Unauthorized", resp.get("error", ""))

        # 2. Invalid signature -> 401
        code, resp = node.handle_sync_request(body, {"X-Reflex-Signature": "invalid-hex"})
        self.assertEqual(code, 401)

        # 3. Valid signature -> 200
        valid_sig = compute_mesh_signature("alpha-secret", body)
        code, resp = node.handle_sync_request(body, {"X-Reflex-Signature": valid_sig})
        self.assertEqual(code, 200)
        self.assertEqual(resp.get("status"), "synchronized")
        self.assertGreaterEqual(resp.get("generation", 0), 2)

        # 4. Expired clock skew -> 400
        stale_payload = copy.deepcopy(payload)
        stale_payload["timestamp"] = time.time() - 100.0  # 100 seconds ago
        stale_body = json.dumps(stale_payload).encode("utf-8")
        stale_sig = compute_mesh_signature("alpha-secret", stale_body)
        code, resp = node.handle_sync_request(stale_body, {"X-Reflex-Signature": stale_sig})
        self.assertEqual(code, 400)
        self.assertIn("clock skew", resp.get("error", ""))

    def test_heartbeat_and_peers_table(self):
        cfg = MeshConfig(
            node_id="node-test-hb",
            cluster_secret="hb-secret",
            peers=["http://10.0.0.2:8080", "http://10.0.0.3:8080"],
        )
        node = InstinctMeshNode(config=cfg)

        code, data = node.handle_peers_request()
        self.assertEqual(code, 200)
        self.assertEqual(len(data["peers"]), 2)

        hb_payload = json.dumps({"origin_node_id": "node-peer-1", "timestamp": time.time()}).encode("utf-8")
        hb_sig = compute_mesh_signature("hb-secret", hb_payload)
        code, resp = node.handle_heartbeat_request(hb_payload, {
            "X-Reflex-Signature": hb_sig,
            "X-Reflex-Origin-Addr": "http://10.0.0.2:8080",
        })
        self.assertEqual(code, 200)
        self.assertEqual(resp["status"], "alive")

        # Peer 10.0.0.2 should now be healthy
        self.assertEqual(node.peers["http://10.0.0.2:8080"].status, "healthy")


class TestMultiNodeLiveMesh(unittest.TestCase):
    """Integration test: Spawns two gateways on loopback and synchronizes learned instincts."""

    @classmethod
    def setUpClass(cls):
        cls.port_a = 18881
        cls.port_b = 18882
        cls.secret = "cluster-test-mesh-token"

        cls.cfg_a = GatewayConfig(
            host="127.0.0.1",
            port=cls.port_a,
            cache_enabled=False,
            guardrails_enabled=False,
            mesh_enabled=True,
            mesh_peers=[f"http://127.0.0.1:{cls.port_b}"],
            mesh_secret=cls.secret,
            mesh_node_id="pod-alpha",
        )
        cls.server_a = ReflexGatewayServer(cls.cfg_a)
        cls.server_a.start(background=True)

        cls.cfg_b = GatewayConfig(
            host="127.0.0.1",
            port=cls.port_b,
            cache_enabled=False,
            guardrails_enabled=False,
            mesh_enabled=True,
            mesh_peers=[f"http://127.0.0.1:{cls.port_a}"],
            mesh_secret=cls.secret,
            mesh_node_id="pod-beta",
        )
        cls.server_b = ReflexGatewayServer(cls.cfg_b)
        cls.server_b.start(background=True)

        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server_a.stop()
        cls.server_b.stop()

    def test_live_peer_sync(self):
        # 1. Verify Node A can see Node B in peers
        req = urllib.request.Request(f"http://127.0.0.1:{self.port_a}/v1/mesh/peers")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["node_id"], "pod-alpha")
            self.assertEqual(len(data["peers"]), 1)
            self.assertEqual(data["peers"][0]["address"], f"http://127.0.0.1:{self.port_b}")

        # 2. Node A creates a Reflex client connected to its mesh node
        rx_a = Reflex(
            learning=True,
            mesh_node=self.server_a.mesh_node,
        )

        # Teach Node A a specific ground truth
        test_state = "Customer urgent request: emergency account suspension after physical robbery"
        loss = rx_a.teach(
            state=test_state,
            question_key="is_emergency",
            ground_truth=True,
            lr=0.2,
        )
        self.assertIsInstance(loss, float)

        # Broadcast sync from Node A to Node B synchronously for verification
        self.server_a.mesh_node.broadcast_teach_update(sample_delta=1, asynchronous=False)

        # 3. Check that Node B received and blended the weights
        head_b = self.server_b.rx.instinct_head
        self.assertIn("is_emergency", head_b.weights)
        self.assertGreater(head_b.training_steps, 0)

        # Verify Node B evaluates the prompt with high confidence
        prob_b = head_b.predict_noul(test_state, question_key="is_emergency")
        self.assertGreater(prob_b, 0.5)


if __name__ == "__main__":
    unittest.main()
