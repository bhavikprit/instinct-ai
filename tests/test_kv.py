"""
Unit and Integration Tests for Reflex KV: Semantic KV-Cache Alignment & Deduplication (Phase 42).
"""

from collections import OrderedDict
import copy
import json
import os
import struct
import tempfile
import time
import unittest
import urllib.request
import urllib.error
import zlib

from reflex import (
    Reflex,
    ReflexGatewayServer,
    GatewayConfig,
    KVConfig,
    PrefixTreeNode,
    PrefixTree,
    PromptAligner,
    KVCachePredictor,
    KVCacheEngine,
    KV_MAGIC,
)
from reflex.kv import _simple_tokenize, PROVIDER_DEFAULTS


class TestKVConfig(unittest.TestCase):
    """Tests for KVConfig configuration validation and defaults."""

    def test_default_config(self):
        cfg = KVConfig()
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.min_cache_tokens, 1024)
        self.assertEqual(cfg.chunk_size, 16)
        self.assertTrue(cfg.align_tools)
        self.assertTrue(cfg.segregate_dynamic_variables)
        self.assertTrue(cfg.inject_provider_breakpoints)
        self.assertEqual(cfg.cached_discount_rate, 0.50)

    def test_invalid_provider(self):
        with self.assertRaises(ValueError):
            KVConfig(provider="unsupported_vendor")

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            KVConfig(min_cache_tokens=0)
        with self.assertRaises(ValueError):
            KVConfig(max_tree_nodes=50)
        with self.assertRaises(ValueError):
            KVConfig(cached_discount_rate=1.5)
        with self.assertRaises(ValueError):
            KVConfig(chunk_size=0)


class TestPrefixTree(unittest.TestCase):
    """Tests for Radix / Prefix Tree token indexing, matching, and pruning."""

    def test_empty_match(self):
        tree = PrefixTree(chunk_size=4)
        matched, meta = tree.match(["hello", "world"])
        self.assertEqual(matched, 0)
        self.assertIsNone(meta)

    def test_insert_and_match(self):
        tree = PrefixTree(chunk_size=2)
        tokens = ["you", "are", "a", "helpful", "coding", "agent"]
        matched_on_insert = tree.insert(tokens, metadata={"test": 123})
        self.assertEqual(matched_on_insert, 0)

        # Match exact same tokens
        matched, meta = tree.match(tokens)
        self.assertEqual(matched, 6)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("test"), 123)

    def test_partial_prefix_match(self):
        tree = PrefixTree(chunk_size=2)
        prompt1 = ["system", "prompt", "rule", "one", "user", "turn"]
        prompt2 = ["system", "prompt", "rule", "one", "different", "turn"]
        tree.insert(prompt1)
        matched, _ = tree.match(prompt2)
        # First 4 tokens match: ["system", "prompt"] and ["rule", "one"] = 2 chunks = 4 tokens
        self.assertEqual(matched, 4)

    def test_pruning(self):
        tree = PrefixTree(chunk_size=2, max_nodes=10)
        # Insert multiple diverging branches to exceed max_nodes
        for i in range(15):
            tree.insert([f"unique_tok_{i}", f"val_{i}", "extra", "word"])
        self.assertLessEqual(tree.total_nodes, 15)
        pruned = tree.prune(target_nodes=5)
        self.assertGreater(pruned, 0)
        self.assertLessEqual(tree.total_nodes, 8)

    def test_tree_stats(self):
        tree = PrefixTree(chunk_size=2)
        tree.insert(["a", "b", "c", "d"])
        tree.insert(["a", "b", "e", "f"])  # Hits chunk 1
        stats = tree.stats()
        self.assertEqual(stats["total_inserts"], 2)
        self.assertEqual(stats["total_hits"], 1)
        self.assertEqual(stats["total_misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)
        self.assertEqual(stats["total_saved_tokens"], 2)


class TestPromptAligner(unittest.TestCase):
    """Tests for Canonical Normalization and Dynamic Header Segregation."""

    def test_canonicalize_tools_sorting(self):
        tools = [
            {"type": "function", "function": {"name": "zeta_tool", "parameters": {"z": 1, "a": 2}}},
            {"type": "function", "function": {"name": "alpha_tool", "parameters": {"b": 2, "a": 1}}},
        ]
        canonical = PromptAligner.canonicalize_tools(tools)
        # First tool should be alpha_tool
        self.assertEqual(canonical[0]["function"]["name"], "alpha_tool")
        self.assertEqual(canonical[1]["function"]["name"], "zeta_tool")
        # Internal dictionary keys should be sorted
        self.assertEqual(list(canonical[0]["function"]["parameters"].keys()), ["a", "b"])

    def test_segregate_dynamic_headers(self):
        text = (
            "Current time: 2026-09-22 14:00:00 UTC\n"
            "Request ID: req-8392104\n"
            "Session ID: sess-94821\n"
            "You are an enterprise AI assistant.\n"
            "Always follow corporate security policies."
        )
        static_body, dynamic_block = PromptAligner.segregate_dynamic_headers(text)
        self.assertIn("You are an enterprise AI assistant.", static_body)
        self.assertIn("Always follow corporate security policies.", static_body)
        self.assertNotIn("Current time:", static_body)
        self.assertNotIn("Request ID:", static_body)
        self.assertIn("Current time: 2026-09-22 14:00:00 UTC", dynamic_block)
        self.assertIn("Request ID: req-8392104", dynamic_block)
        self.assertIn("Session ID: sess-94821", dynamic_block)

    def test_segregate_no_dynamic_headers(self):
        text = "You are an enterprise AI assistant.\nAlways follow corporate security policies."
        static_body, dynamic_block = PromptAligner.segregate_dynamic_headers(text)
        self.assertEqual(static_body, text)
        self.assertEqual(dynamic_block, "")

    def test_align_messages_openai(self):
        messages = [
            {
                "role": "system",
                "content": "Current time: 2026-09-22T10:00:00Z\nYou are a financial auditor.",
            },
            {"role": "user", "content": "Audit transaction 994"},
        ]
        aligned, tools, telemetry = PromptAligner.align_messages(messages, provider="openai")
        self.assertTrue(telemetry["dynamic_segregated"])
        self.assertEqual(len(aligned), 3)
        self.assertEqual(aligned[0]["role"], "system")
        self.assertEqual(aligned[0]["content"], "You are a financial auditor.")
        self.assertEqual(aligned[1]["role"], "system")
        self.assertIn("[Runtime Context Variables]", aligned[1]["content"])
        self.assertEqual(aligned[2]["role"], "user")
        self.assertEqual(aligned[2]["content"], "Audit transaction 994")

    def test_align_messages_anthropic_breakpoints(self):
        messages = [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": "Review contract clause 4."},
        ]
        tools = [{"type": "function", "function": {"name": "search_law"}}]
        aligned, aligned_tools, telemetry = PromptAligner.align_messages(
            messages, tools=tools, provider="anthropic", inject_breakpoints=True
        )
        self.assertGreater(telemetry["breakpoints_injected"], 0)
        # Check system message format for Anthropic
        self.assertIsInstance(aligned[0]["content"], list)
        self.assertEqual(aligned[0]["content"][0]["cache_control"], {"type": "ephemeral"})
        # Check tool format
        self.assertEqual(aligned_tools[-1]["cache_control"], {"type": "ephemeral"})


class TestKVCachePredictor(unittest.TestCase):
    """Tests for KV Cache hit prediction, cost savings, and TTFT latency reduction."""

    def test_predict_hit_and_miss(self):
        cfg = KVConfig(provider="openai", min_cache_tokens=16, input_cost_per_million=2.50, cached_discount_rate=0.50)
        tree = PrefixTree(chunk_size=4)
        tokens = [f"tok_{i}" for i in range(32)]
        tree.insert(tokens)

        predictor = KVCachePredictor(cfg, tree)

        # Above threshold: Hit
        res = predictor.predict(tokens)
        self.assertTrue(res["is_cache_hit"])
        self.assertEqual(res["saved_tokens"], 32)
        self.assertGreater(res["cost_saved_usd"], 0.0)
        self.assertEqual(res["cost_savings_pct"], 50.0)
        self.assertGreater(res["estimated_ttft_saved_ms"], 0.0)

        # Below threshold: Miss
        short_tokens = ["different", "token", "stream"]
        res_miss = predictor.predict(short_tokens)
        self.assertFalse(res_miss["is_cache_hit"])
        self.assertEqual(res_miss["saved_tokens"], 0)
        self.assertEqual(res_miss["cost_saved_usd"], 0.0)


class TestKVCacheEngine(unittest.TestCase):
    """Tests for KVCacheEngine end-to-end processing and binary persistence."""

    def test_process_simulation(self):
        cfg = KVConfig(provider="openai", min_cache_tokens=16, chunk_size=4)
        engine = KVCacheEngine(config=cfg)

        base_prompt = "You are a reliable dual-brain System-1 reflex runtime."
        req1 = {
            "messages": [
                {"role": "system", "content": f"Current time: 2026-09-22 10:00:00\n{base_prompt}"},
                {"role": "user", "content": "Ping 1"},
            ]
        }
        req2 = {
            "messages": [
                {"role": "system", "content": f"Current time: 2026-09-22 10:05:00\n{base_prompt}"},
                {"role": "user", "content": "Ping 2"},
            ]
        }

        # Turn 1: Cold start miss
        opt1, tele1 = engine.process(req1)
        self.assertFalse(tele1["is_cache_hit"])

        # Turn 2: Warm prefix hit
        opt2, tele2 = engine.process(req2)
        self.assertTrue(tele2["is_cache_hit"])
        self.assertGreaterEqual(tele2["matched_prefix_tokens"], 16)
        self.assertGreater(tele2["cost_saved_usd"], 0.0)

    def test_ascii_prefix_tree(self):
        engine = KVCacheEngine(KVConfig(provider="anthropic", min_cache_tokens=16, chunk_size=4))
        engine.process({
            "messages": [{"role": "system", "content": "System instruction alpha"}, {"role": "user", "content": "hello"}]
        })
        tree_ascii = engine.ascii_prefix_tree()
        self.assertIn("ANTHROPIC", tree_ascii)
        self.assertIn("Radix Prefix Tree Structure", tree_ascii)

    def test_save_and_load_binary(self):
        cfg = KVConfig(provider="deepseek", min_cache_tokens=64, chunk_size=8)
        engine = KVCacheEngine(config=cfg)
        for i in range(5):
            engine.process({
                "messages": [
                    {"role": "system", "content": f"Session ID: sess_{i}\nUniversal invariant prompt rules for agent."},
                    {"role": "user", "content": f"Query {i}"},
                ]
            })

        with tempfile.NamedTemporaryFile(suffix=".reflex-kv", delete=False) as f:
            filepath = f.name

        try:
            engine.save(filepath)
            self.assertTrue(os.path.exists(filepath))

            loaded = KVCacheEngine.load(filepath)
            self.assertEqual(loaded.config.provider, "deepseek")
            self.assertEqual(loaded.config.min_cache_tokens, 64)
            self.assertEqual(loaded.config.chunk_size, 8)
            self.assertEqual(loaded.tree.total_inserts, engine.tree.total_inserts)
            self.assertEqual(loaded.tree.total_nodes, engine.tree.total_nodes)

            # Test corrupted file CRC
            with open(filepath, "r+b") as f:
                f.seek(70)
                orig_byte = f.read(1)
                f.seek(70)
                f.write(bytes([orig_byte[0] ^ 0xFF]))

            with self.assertRaises(ValueError) as ctx:
                KVCacheEngine.load(filepath)
            self.assertIn("CRC32 checksum mismatch", str(ctx.exception))
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_corrupt_magic(self):
        with tempfile.NamedTemporaryFile(suffix=".reflex-kv", delete=False) as f:
            filepath = f.name
            f.write(b"BADM" + b"\x00" * 80)

        try:
            # Checksum matches or fails
            with self.assertRaises(ValueError):
                KVCacheEngine.load(filepath)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            KVCacheEngine.load("/nonexistent/path/model.reflex-kv")


class TestReflexClientAndGatewayIntegration(unittest.TestCase):
    """Tests for Reflex client helper method and Gateway KV-Cache alignment."""

    def test_reflex_client_align_prompt(self):
        rx = Reflex(kv_engine=True)
        self.assertIsNotNone(rx.kv_engine)
        messages = [
            {"role": "system", "content": "Timestamp: 2026-09-22T12:00:00Z\nCore prompt rules."},
            {"role": "user", "content": "Execute trade"},
        ]
        aligned, tools, meta = rx.align_prompt(messages)
        self.assertTrue(meta["dynamic_segregated"])
        self.assertEqual(aligned[0]["content"], "Core prompt rules.")

    def test_gateway_chat_completions_with_kv(self):
        import io
        from reflex.gateway import GatewayRequestHandler

        cfg = GatewayConfig(
            kv_alignment_enabled=True,
            kv_provider="openai",
            kv_min_cache_tokens=16,
            upstream_url="http://127.0.0.1:19999",
            cache_enabled=False,
            guardrails_enabled=False,
            system1_routing_enabled=False,
        )
        server = ReflexGatewayServer(config=cfg)
        self.assertIsNotNone(server)
        GatewayRequestHandler.initialize(cfg)
        self.assertIsNotNone(GatewayRequestHandler.kv_engine)

        class DummyHandler(GatewayRequestHandler):
            def __init__(self, request_bytes=b"", path="/v1/chat/completions"):
                self.rfile = io.BytesIO(request_bytes)
                self.wfile = io.BytesIO()
                self.headers = {"Content-Length": str(len(request_bytes))}
                self.path = path
                self.client_address = ("127.0.0.1", 12345)
                self._headers_sent = []

            def send_response(self, code):
                self.response_code = code

            def send_header(self, k, v):
                self._headers_sent.append((k, v))

            def end_headers(self):
                pass

        # Query 1: Cold start miss
        body1 = json.dumps({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Current time: 2026-09-22 10:00:00\nSystem instruction invariant across requests."},
                {"role": "user", "content": "Hello 1"},
            ]
        }).encode("utf-8")
        h1 = DummyHandler(request_bytes=body1, path="/v1/chat/completions")
        h1.handle_chat_completions()
        self.assertEqual(h1.response_code, 200)
        hdrs1 = dict(h1._headers_sent)
        self.assertEqual(hdrs1.get("X-Reflex-KV-Aligned"), "TRUE")
        self.assertEqual(hdrs1.get("X-Reflex-KV-Hit"), "FALSE")

        # Query 2: Same system invariant, different timestamp -> prefix hit
        body2 = json.dumps({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Current time: 2026-09-22 10:01:00\nSystem instruction invariant across requests."},
                {"role": "user", "content": "Hello 2"},
            ]
        }).encode("utf-8")
        h2 = DummyHandler(request_bytes=body2, path="/v1/chat/completions")
        h2.handle_chat_completions()
        self.assertEqual(h2.response_code, 200)
        hdrs2 = dict(h2._headers_sent)
        self.assertEqual(hdrs2.get("X-Reflex-KV-Aligned"), "TRUE")
        self.assertEqual(hdrs2.get("X-Reflex-KV-Hit"), "TRUE")
        self.assertGreaterEqual(int(hdrs2.get("X-Reflex-KV-Matched-Tokens", 0)), 16)

        # GET /v1/kv/stats
        h_stats = DummyHandler(path="/v1/kv/stats")
        h_stats.do_GET()
        self.assertEqual(h_stats.response_code, 200)
        data = json.loads(h_stats.wfile.getvalue().decode("utf-8"))
        self.assertEqual(data["provider"], "openai")
        self.assertEqual(data["tree_stats"]["total_hits"], 1)

        # POST /v1/kv/align
        body_align = json.dumps({
            "messages": [
                {"role": "system", "content": "Request ID: 12345\nAnalyze security log."},
                {"role": "user", "content": "Log item"},
            ]
        }).encode("utf-8")
        h_align = DummyHandler(request_bytes=body_align, path="/v1/kv/align")
        h_align.do_POST()
        self.assertEqual(h_align.response_code, 200)
        align_data = json.loads(h_align.wfile.getvalue().decode("utf-8"))
        self.assertTrue(align_data["kv_telemetry"]["alignment"]["dynamic_segregated"])

    def test_gateway_metrics_kv_tracking(self):
        from reflex.gateway import GatewayMetrics
        metrics = GatewayMetrics()
        metrics.record_kv_alignment(is_hit=False, tokens_saved=0, cost_saved=0.0)
        metrics.record_kv_alignment(is_hit=True, tokens_saved=256, cost_saved=0.0012)
        d = metrics.to_dict()
        self.assertEqual(d["kv_aligned_requests"], 2)
        self.assertEqual(d["kv_cache_hits"], 1)
        self.assertEqual(d["kv_hit_rate"], 0.5)
        self.assertEqual(d["kv_tokens_saved"], 256)
        self.assertEqual(d["kv_dollars_saved"], 0.0012)

    def test_provider_id_mapping(self):
        from reflex.kv import PROVIDER_NAME_TO_ID, PROVIDER_ID_TO_NAME
        for name, pid in PROVIDER_NAME_TO_ID.items():
            self.assertEqual(PROVIDER_ID_TO_NAME[pid], name)

    def test_provider_defaults(self):
        from reflex.kv import PROVIDER_DEFAULTS
        self.assertEqual(PROVIDER_DEFAULTS["deepseek"]["min_cache_tokens"], 64)
        self.assertEqual(PROVIDER_DEFAULTS["vllm"]["min_cache_tokens"], 16)
        self.assertEqual(PROVIDER_DEFAULTS["anthropic"]["discount"], 0.90)

    def test_prefix_tree_pruning_noop(self):
        tree = PrefixTree(chunk_size=4)
        tree.insert(["one", "two", "three", "four"])
        pruned = tree.prune(target_nodes=100)
        self.assertEqual(pruned, 0)

    def test_prefix_tree_multiple_branches(self):
        tree = PrefixTree(chunk_size=2)
        tree.insert(["sys", "a", "user", "1"])
        tree.insert(["sys", "b", "user", "2"])
        tree.insert(["sys", "c", "user", "3"])
        # Root should have 3 distinct child chunks
        self.assertEqual(len(tree.root.children), 3)

    def test_prompt_aligner_nested_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "complex_fn",
                    "parameters": {
                        "z_param": {"inner_b": 2, "inner_a": 1},
                        "a_param": [3, 2, 1],
                    },
                },
            }
        ]
        res = PromptAligner.canonicalize_tools(tools)
        params = res[0]["function"]["parameters"]
        self.assertEqual(list(params.keys()), ["a_param", "z_param"])
        self.assertEqual(list(params["z_param"].keys()), ["inner_a", "inner_b"])

    def test_prompt_aligner_empty_tools(self):
        self.assertEqual(PromptAligner.canonicalize_tools([]), [])
        self.assertEqual(PromptAligner.canonicalize_tools(None), [])

    def test_simple_tokenize(self):
        tokens = _simple_tokenize("Hello, world! 2026-09-22 runtime_v2.")
        self.assertIn("Hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("2026", tokens)
        self.assertIn("runtime_v2", tokens)

    def test_save_and_load_anthropic_provider(self):
        cfg = KVConfig(provider="anthropic", min_cache_tokens=1024)
        engine = KVCacheEngine(config=cfg)
        engine.process({
            "messages": [{"role": "system", "content": "You are Claude."}],
            "tools": [{"type": "function", "function": {"name": "test_tool"}}],
        })
        with tempfile.NamedTemporaryFile(suffix=".reflex-kv", delete=False) as f:
            filepath = f.name
        try:
            engine.save(filepath)
            loaded = KVCacheEngine.load(filepath)
            self.assertEqual(loaded.config.provider, "anthropic")
            self.assertEqual(loaded.config.min_cache_tokens, 1024)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    unittest.main()

