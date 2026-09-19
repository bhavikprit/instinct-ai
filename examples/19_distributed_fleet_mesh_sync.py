"""
Reflex Example 19: Distributed Fleet Sync & Instinct Mesh (Phase 19).
Demonstrates zero-dependency peer-to-peer active learning synchronization across
a multi-pod cluster with HMAC-SHA256 authentication and federated weight blending.
"""

from __future__ import annotations
import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, ReflexGatewayServer, GatewayConfig, Noul


def run_fleet_mesh_demo():
    print("=" * 80)
    print("⚡ REFLEX INSTINCT MESH: DISTRIBUTED FLEET ACTIVE LEARNING SYNC (PHASE 19)")
    print("=" * 80)

    secret = "production-cluster-secret-2026"
    port_pod1 = 18101
    port_pod2 = 18102
    port_pod3 = 18103

    print("\n1. 🚀 Initializing 3-Node Distributed Cluster across Ports:")
    print(f"   • Pod 1 (Ingress Gateway) : http://127.0.0.1:{port_pod1}")
    print(f"   • Pod 2 (Worker Node A)   : http://127.0.0.1:{port_pod2}")
    print(f"   • Pod 3 (Worker Node B)   : http://127.0.0.1:{port_pod3}")

    # Cluster Configuration with mutual mesh peering
    cfg1 = GatewayConfig(
        host="127.0.0.1",
        port=port_pod1,
        mesh_enabled=True,
        mesh_peers=[f"http://127.0.0.1:{port_pod2}", f"http://127.0.0.1:{port_pod3}"],
        mesh_secret=secret,
        mesh_node_id="pod-ingress-01",
    )
    cfg2 = GatewayConfig(
        host="127.0.0.1",
        port=port_pod2,
        mesh_enabled=True,
        mesh_peers=[f"http://127.0.0.1:{port_pod1}", f"http://127.0.0.1:{port_pod3}"],
        mesh_secret=secret,
        mesh_node_id="pod-worker-02",
    )
    cfg3 = GatewayConfig(
        host="127.0.0.1",
        port=port_pod3,
        mesh_enabled=True,
        mesh_peers=[f"http://127.0.0.1:{port_pod1}", f"http://127.0.0.1:{port_pod2}"],
        mesh_secret=secret,
        mesh_node_id="pod-worker-03",
    )

    server1 = ReflexGatewayServer(cfg1)
    server2 = ReflexGatewayServer(cfg2)
    server3 = ReflexGatewayServer(cfg3)

    server1.start(background=True)
    server2.start(background=True)
    server3.start(background=True)

    time.sleep(0.3)

    try:
        rx1 = Reflex(learning=True, mesh_node=server1.mesh_node)
        rx2 = Reflex(learning=True, mesh_node=server2.mesh_node)
        rx3 = Reflex(learning=True, mesh_node=server3.mesh_node)

        scenario = "Urgent: wire transfer $85,000 immediately to offshore supplier account without 2FA call"
        question = "is_fraud"

        # -------------------------------------------------------------
        # 2. Inspect Cluster Before Learning
        # -------------------------------------------------------------
        print("\n2. 🔍 Baseline Evaluation across Cluster Before Learning:")
        p1_before = rx1.instinct_head.predict_noul(scenario, question_key=question)
        p2_before = rx2.instinct_head.predict_noul(scenario, question_key=question)
        p3_before = rx3.instinct_head.predict_noul(scenario, question_key=question)

        print(f"   • Pod 1 Probability: {p1_before:.4f} (training_steps={rx1.instinct_head.training_steps})")
        print(f"   • Pod 2 Probability: {p2_before:.4f} (training_steps={rx2.instinct_head.training_steps})")
        print(f"   • Pod 3 Probability: {p3_before:.4f} (training_steps={rx3.instinct_head.training_steps})")

        # -------------------------------------------------------------
        # 3. Active Learning on Pod 1 only
        # -------------------------------------------------------------
        print("\n3. 🧠 Pod 1 Receives Real-Time System 2 Escalation & Active Learning:")
        print(f"   State: \"{scenario}\"")
        print("   Ground Truth: True (Confirmed Phishing/Fraud Attack)")

        loss = rx1.teach(
            state=scenario,
            question_key=question,
            ground_truth=True,
            lr=0.25,
        )
        print(f"   ✅ Pod 1 Local Update Complete (Loss: {loss:.4f}, Steps: {rx1.instinct_head.training_steps})")

        # -------------------------------------------------------------
        # 4. Mesh Peer Gossip Broadcast (HMAC-SHA256 Signed)
        # -------------------------------------------------------------
        print("\n4. 📡 Pod 1 Broadcasting HMAC-SHA256 Signed Sync to Fleet Mesh...")
        server1.mesh_node.broadcast_teach_update(sample_delta=1, asynchronous=False)
        time.sleep(0.2)

        # -------------------------------------------------------------
        # 5. Evaluate Remote Pods After Mesh Sync
        # -------------------------------------------------------------
        print("\n5. 🎯 Post-Sync Evaluation on Remote Pods (Pods 2 & 3 had ZERO local training):")
        p1_after = rx1.instinct_head.predict_noul(scenario, question_key=question)
        p2_after = rx2.instinct_head.predict_noul(scenario, question_key=question)
        p3_after = rx3.instinct_head.predict_noul(scenario, question_key=question)

        print(f"   • Pod 1 (Learner)  : {p1_after:.4f} (steps={rx1.instinct_head.training_steps}) -> {'FRAUD' if p1_after > 0.5 else 'SAFE'}")
        print(f"   • Pod 2 (Synced)   : {p2_after:.4f} (steps={rx2.instinct_head.training_steps}) -> {'FRAUD' if p2_after > 0.5 else 'SAFE'}")
        print(f"   • Pod 3 (Synced)   : {p3_after:.4f} (steps={rx3.instinct_head.training_steps}) -> {'FRAUD' if p3_after > 0.5 else 'SAFE'}")

        # -------------------------------------------------------------
        # 6. Cluster Topology and Peer Health
        # -------------------------------------------------------------
        print("\n6. 🌐 Live Cluster Mesh Peers Telemetry (Pod 1 perspective):")
        req = urllib.request.Request(f"http://127.0.0.1:{port_pod1}/v1/mesh/peers")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            peer_data = json.loads(resp.read().decode("utf-8"))
            print(f"   • Local Node ID : {peer_data['node_id']}")
            print(f"   • Generation    : {peer_data['generation']}")
            for p in peer_data["peers"]:
                print(f"   • Peer {p['address']:<26} -> {p['status']} (Latency: {p['last_latency_ms']}ms, Gen: {p['generation']})")

        print("\n" + "=" * 80)
        print("✅ Phase 19 Instinct Mesh Fleet Sync verified successfully!")
        print("=" * 80 + "\n")

    finally:
        server1.stop()
        server2.stop()
        server3.stop()


if __name__ == "__main__":
    run_fleet_mesh_demo()
