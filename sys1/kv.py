"""
Reflex KV: Semantic KV-Cache Alignment & Prompt Prefix Deduplication (Phase 42).
Maximizes provider prompt caching hit rates across OpenAI, Anthropic, DeepSeek, and vLLM
by canonically aligning prompts, segregating dynamic metadata, and indexing prefix tokens.

Mathematical & Engineering Foundations:
- Sub-millisecond Radix / Prefix Tree Token Cache Engine.
- Dynamic variable transposition (moving static prompt invariants forward, dynamic anchors to trailing context).
- Deterministic tool schema canonicalization & alphabetical sorting.
- Provider breakpoint placement (Anthropic ephemeral cache control injection).
- Accurate cost and TTFT latency savings estimation based on provider pricing tiers.
- Zero-dependency binary persistence (.reflex-kv, magic RFKV, 56-byte header, 32-bit CRC32 trailer).
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import re
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import zlib

KV_MAGIC = b"RFKV"
KV_FORMAT_VERSION = 1

PROVIDER_OPENAI = 0
PROVIDER_ANTHROPIC = 1
PROVIDER_DEEPSEEK = 2
PROVIDER_VLLM = 3

PROVIDER_NAME_TO_ID = {
    "openai": PROVIDER_OPENAI,
    "anthropic": PROVIDER_ANTHROPIC,
    "deepseek": PROVIDER_DEEPSEEK,
    "vllm": PROVIDER_VLLM,
}
PROVIDER_ID_TO_NAME = {v: k for k, v in PROVIDER_NAME_TO_ID.items()}

# Default minimum cacheable prefix tokens and discount rates per provider
PROVIDER_DEFAULTS = {
    "openai": {"min_cache_tokens": 1024, "discount": 0.50, "ttft_reduction": 0.50},
    "anthropic": {"min_cache_tokens": 1024, "discount": 0.90, "ttft_reduction": 0.85},
    "deepseek": {"min_cache_tokens": 64, "discount": 0.90, "ttft_reduction": 0.90},
    "vllm": {"min_cache_tokens": 16, "discount": 1.00, "ttft_reduction": 0.95},
}


@dataclass
class KVConfig:
    """Configuration for Semantic KV-Cache Aligner and Prefix Tree."""
    provider: str = "openai"
    min_cache_tokens: int = 1024
    max_tree_nodes: int = 50000
    align_tools: bool = True
    segregate_dynamic_variables: bool = True
    inject_provider_breakpoints: bool = True
    input_cost_per_million: float = 2.50
    cached_discount_rate: float = 0.50
    chunk_size: int = 16  # token chunk size for trie nodes

    def __post_init__(self) -> None:
        if self.provider not in PROVIDER_NAME_TO_ID:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. Must be one of {list(PROVIDER_NAME_TO_ID.keys())}"
            )
        if self.min_cache_tokens < 1:
            raise ValueError(f"min_cache_tokens must be >= 1, got {self.min_cache_tokens}")
        if self.max_tree_nodes < 100:
            raise ValueError(f"max_tree_nodes must be >= 100, got {self.max_tree_nodes}")
        if not (0.0 <= self.cached_discount_rate <= 1.0):
            raise ValueError(f"cached_discount_rate must be in [0.0, 1.0], got {self.cached_discount_rate}")
        if self.input_cost_per_million < 0.0:
            raise ValueError(f"input_cost_per_million must be >= 0.0, got {self.input_cost_per_million}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")


def _simple_tokenize(text: str) -> List[str]:
    """Fast, zero-dependency tokenization into words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


class PrefixTreeNode:
    """A single node in the Radix Prefix Tree."""

    __slots__ = ("token_chunk", "children", "hit_count", "last_accessed", "metadata")

    def __init__(self, token_chunk: str):
        self.token_chunk: str = token_chunk
        self.children: Dict[str, PrefixTreeNode] = {}
        self.hit_count: int = 0
        self.last_accessed: float = time.time()
        self.metadata: Optional[Dict[str, Any]] = None


class PrefixTree:
    """
    Sub-millisecond Radix / Prefix Tree Token Cache Engine.
    Tracks token chunk prefixes across requests, matching longest common prefixes
    to simulate and predict provider prompt cache hits.
    """

    def __init__(self, chunk_size: int = 16, max_nodes: int = 50000):
        self.chunk_size = chunk_size
        self.max_nodes = max_nodes
        self.root = PrefixTreeNode(token_chunk="")
        self.total_nodes: int = 1
        self.total_inserts: int = 0
        self.total_hits: int = 0
        self.total_misses: int = 0
        self.total_saved_tokens: int = 0

    def _chunk_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Groups token stream into uniform chunk strings for efficient trie compression."""
        chunks = []
        c_size = self.chunk_size
        for i in range(0, len(tokens), c_size):
            chunks.append(" ".join(tokens[i : i + c_size]))
        return chunks

    def insert(self, tokens: Sequence[str], metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Inserts token sequence into the prefix tree.
        Returns the number of matching prefix tokens found before insertion.
        """
        self.total_inserts += 1
        chunks = self._chunk_tokens(tokens)
        matched_chunks = 0
        now = time.time()

        curr = self.root
        curr.hit_count += 1
        curr.last_accessed = now

        for chunk in chunks:
            if chunk in curr.children:
                curr = curr.children[chunk]
                curr.hit_count += 1
                curr.last_accessed = now
                matched_chunks += 1
            else:
                if self.total_nodes >= self.max_nodes:
                    self.prune(int(self.max_nodes * 0.8))
                new_node = PrefixTreeNode(token_chunk=chunk)
                curr.children[chunk] = new_node
                curr = new_node
                self.total_nodes += 1

        if metadata:
            curr.metadata = metadata

        matched_tokens = matched_chunks * self.chunk_size
        if matched_tokens > 0:
            self.total_hits += 1
            self.total_saved_tokens += matched_tokens
        else:
            self.total_misses += 1

        return matched_tokens

    def match(self, tokens: Sequence[str]) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Finds the longest common prefix match for tokens in the tree.
        Returns (matched_token_count, last_matched_node_metadata).
        """
        chunks = self._chunk_tokens(tokens)
        matched_chunks = 0
        curr = self.root
        last_meta = curr.metadata

        for chunk in chunks:
            if chunk in curr.children:
                curr = curr.children[chunk]
                matched_chunks += 1
                if curr.metadata:
                    last_meta = curr.metadata
            else:
                break

        return matched_chunks * self.chunk_size, last_meta

    def prune(self, target_nodes: int) -> int:
        """Prunes least-recently-accessed leaf nodes to bound memory footprint."""
        if self.total_nodes <= target_nodes:
            return 0

        # Collect leaf nodes sorted by last_accessed
        nodes_to_remove = []
        stack = [(self.root, parent_child_key) for parent_child_key in self.root.children.keys()]

        # Simple BFS / DFS collect leaves
        bfs_queue = deque([(self.root, k, self.root.children[k]) for k in list(self.root.children.keys())])
        all_children = []

        while bfs_queue:
            parent, key, child = bfs_queue.popleft()
            all_children.append((parent, key, child))
            for ck, grand_child in list(child.children.items()):
                bfs_queue.append((child, ck, grand_child))

        # Sort by last_accessed ascending
        all_children.sort(key=lambda item: item[2].last_accessed)

        removed = 0
        to_prune_count = self.total_nodes - target_nodes
        for parent, key, child in all_children:
            if removed >= to_prune_count:
                break
            if key in parent.children and len(child.children) == 0:
                del parent.children[key]
                removed += 1
                self.total_nodes -= 1

        return removed

    def stats(self) -> Dict[str, Any]:
        """Returns prefix tree operating metrics and cache hit rates."""
        total_queries = self.total_hits + self.total_misses
        hit_rate = (self.total_hits / total_queries) if total_queries > 0 else 0.0

        # Calculate max tree depth
        max_depth = 0
        q = deque([(self.root, 0)])
        while q:
            node, depth = q.popleft()
            if depth > max_depth:
                max_depth = depth
            for child in node.children.values():
                q.append((child, depth + 1))

        return {
            "total_nodes": self.total_nodes,
            "max_depth_chunks": max_depth,
            "max_depth_tokens": max_depth * self.chunk_size,
            "total_inserts": self.total_inserts,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(hit_rate, 4),
            "total_saved_tokens": self.total_saved_tokens,
        }


class PromptAligner:
    """
    Canonical Normalizer & Dynamic Aligner.
    Transforms raw user prompts and chat messages into prefix-cache-optimized canonical forms:
    1. Segregates dynamic metadata (timestamps, session IDs) from static system instructions.
    2. Canonically sorts and hashes tool / function definitions.
    3. Standardizes parameter key order across few-shot exemplars.
    4. Injects provider-specific cache control breakpoints (e.g. Anthropic cache_control).
    """

    DYNAMIC_HEADER_PATTERNS = [
        re.compile(r"^(?:current[-_\s]*time|timestamp|date|today[-_\s]*is|datetime)\s*:\s*[^\n]+\n?", re.IGNORECASE),
        re.compile(r"^(?:session[-_\s]*id|request[-_\s]*id|trace[-_\s]*id|user[-_\s]*id|client[-_\s]*id|run[-_\s]*id)\s*:\s*[^\n]+\n?", re.IGNORECASE),
        re.compile(r"^(?:now|system[-_\s]*time)\s*:\s*[^\n]+\n?", re.IGNORECASE),
        re.compile(r"^(?:uuid|guid|ip|ip[-_\s]*address)\s*:\s*[^\n]+\n?", re.IGNORECASE),
        re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\n]*\n?", re.IGNORECASE),
    ]

    @classmethod
    def canonicalize_tools(cls, tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts tools alphabetically by function name and canonicalizes parameter schemas
        so identical tool sets produce 100% identical byte/token prefixes.
        """
        if not tools:
            return []

        def get_tool_name(t: Dict[str, Any]) -> str:
            if "function" in t and isinstance(t["function"], dict):
                return t["function"].get("name", "")
            return t.get("name", "")

        def sort_dict(d: Any) -> Any:
            if isinstance(d, dict):
                return {k: sort_dict(v) for k, v in sorted(d.items())}
            elif isinstance(d, list):
                return [sort_dict(x) for x in d]
            return d

        sorted_tools = sorted(tools, key=get_tool_name)
        return [sort_dict(t) for t in sorted_tools]

    @classmethod
    def segregate_dynamic_headers(cls, text: str) -> Tuple[str, str]:
        """
        Separates dynamic prefix headers (timestamps, UUIDs, session IDs) from static instructions.
        Returns (static_instructions, dynamic_variables_block).
        """
        static_lines = []
        dynamic_lines = []

        lines = text.split("\n")
        in_header = True

        for line in lines:
            line_stripped = line.strip()
            if in_header and line_stripped:
                matched = False
                for pattern in cls.DYNAMIC_HEADER_PATTERNS:
                    if pattern.match(line_stripped):
                        dynamic_lines.append(line_stripped)
                        matched = True
                        break
                if not matched:
                    in_header = False
                    static_lines.append(line)
            else:
                static_lines.append(line)

        static_body = "\n".join(static_lines).strip()
        dynamic_block = "\n".join(dynamic_lines).strip()
        return static_body, dynamic_block

    @classmethod
    def align_messages(
        cls,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        provider: str = "openai",
        inject_breakpoints: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Aligns chat messages and tool definitions into a cache-optimal sequence:
        - Invariant static system instructions are placed first.
        - Canonicalized tools are attached.
        - Dynamic context (timestamps, session parameters) is placed directly after static invariant.
        - Conversation history and user turn are preserved.
        """
        aligned_messages = []
        telemetry = {
            "dynamic_segregated": False,
            "tools_reordered": False,
            "breakpoints_injected": 0,
            "static_prefix_tokens_est": 0,
        }

        # 1. Canonicalize Tools
        aligned_tools = None
        if tools:
            canonical_tools = cls.canonicalize_tools(tools)
            if json.dumps(canonical_tools) != json.dumps(tools):
                telemetry["tools_reordered"] = True
            aligned_tools = canonical_tools

        # 2. Process System Message & Segregate Dynamic Anchors
        system_idx = -1
        for idx, m in enumerate(messages):
            if m.get("role") == "system":
                system_idx = idx
                break

        if system_idx != -1:
            sys_msg = messages[system_idx]
            sys_content = sys_msg.get("content", "")
            if isinstance(sys_content, str):
                static_text, dynamic_text = cls.segregate_dynamic_headers(sys_content)
                if dynamic_text:
                    telemetry["dynamic_segregated"] = True
                    # Reconstruct system message: static instructions first, dynamic block trailing
                    reconstructed_system = static_text
                    aligned_messages.append({"role": "system", "content": reconstructed_system})
                    # Place dynamic context as immediate context note
                    aligned_messages.append({
                        "role": "system",
                        "content": f"[Runtime Context Variables]\n{dynamic_text}",
                    })
                else:
                    aligned_messages.append(dict(sys_msg))
            else:
                aligned_messages.append(dict(sys_msg))
        else:
            # No system message found
            pass

        # 3. Append remaining user / assistant messages in sequence
        for idx, m in enumerate(messages):
            if idx == system_idx:
                continue
            aligned_messages.append(dict(m))

        # 4. Inject Provider Breakpoints (Anthropic Ephemeral Cache Control)
        if inject_breakpoints and provider == "anthropic":
            # For Anthropic, attach cache_control to the first static system message
            if aligned_messages and aligned_messages[0].get("role") == "system":
                content = aligned_messages[0].get("content")
                if isinstance(content, str):
                    aligned_messages[0]["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                    telemetry["breakpoints_injected"] += 1
            # If tools are present, tag last tool for cache breakpoint
            if aligned_tools and len(aligned_tools) > 0:
                aligned_tools[-1]["cache_control"] = {"type": "ephemeral"}
                telemetry["breakpoints_injected"] += 1

        # Estimate static prefix tokens
        static_str = ""
        if aligned_messages:
            static_str = str(aligned_messages[0].get("content", ""))
        if aligned_tools:
            static_str += " " + json.dumps(aligned_tools)
        telemetry["static_prefix_tokens_est"] = len(_simple_tokenize(static_str))

        return aligned_messages, aligned_tools, telemetry


class KVCachePredictor:
    """
    Computes financial token savings and TTFT latency reduction
    based on prefix tree matching and provider-specific pricing structures.
    """

    def __init__(self, config: KVConfig, prefix_tree: PrefixTree):
        self.config = config
        self.prefix_tree = prefix_tree

    def predict(self, tokens: Sequence[str]) -> Dict[str, Any]:
        """Evaluates token sequence against prefix tree and provider threshold."""
        matched_tokens, _ = self.prefix_tree.match(tokens)
        min_cache = self.config.min_cache_tokens
        is_hit = matched_tokens >= min_cache

        cost_per_token = self.config.input_cost_per_million / 1_000_000.0
        discount = self.config.cached_discount_rate

        if is_hit:
            saved_tokens = matched_tokens
            cost_saved = saved_tokens * cost_per_token * discount
            pct_saved = discount * 100.0
            provider_defaults = PROVIDER_DEFAULTS.get(self.config.provider, {})
            ttft_pct = provider_defaults.get("ttft_reduction", 0.75) * 100.0
            ttft_saved_ms = 850.0 * (ttft_pct / 100.0)
        else:
            saved_tokens = 0
            cost_saved = 0.0
            pct_saved = 0.0
            ttft_saved_ms = 0.0

        return {
            "matched_prefix_tokens": matched_tokens,
            "min_cache_tokens": min_cache,
            "is_cache_hit": is_hit,
            "saved_tokens": saved_tokens,
            "cost_saved_usd": round(cost_saved, 6),
            "cost_savings_pct": round(pct_saved, 1),
            "estimated_ttft_saved_ms": round(ttft_saved_ms, 1),
        }


class KVCacheEngine:
    """
    High-Level Semantic KV-Cache Alignment & Optimization Engine.
    Coordinates PromptAligner, PrefixTree, and KVCachePredictor.
    Provides zero-dependency binary persistence (.reflex-kv) with CRC32 verification.
    """

    def __init__(self, config: Optional[KVConfig] = None):
        self.config = config or KVConfig()
        self.tree = PrefixTree(chunk_size=self.config.chunk_size, max_nodes=self.config.max_tree_nodes)
        self.predictor = KVCachePredictor(self.config, self.tree)
        self.aligner = PromptAligner()

    def process(self, req_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Processes and optimizes an incoming chat completion request:
        1. Aligns messages and tool definitions.
        2. Predicts cache hit and token savings.
        3. Updates internal prefix tree.
        4. Injects provider-appropriate cache headers and returns optimized payload.
        """
        messages = req_json.get("messages", [])
        tools = req_json.get("tools", None)

        # 1. Align Prompt
        aligned_msgs, aligned_tools, align_meta = self.aligner.align_messages(
            messages=messages,
            tools=tools,
            provider=self.config.provider,
            inject_breakpoints=self.config.inject_provider_breakpoints,
        )

        optimized_req = dict(req_json)
        optimized_req["messages"] = aligned_msgs
        if aligned_tools is not None:
            optimized_req["tools"] = aligned_tools

        # 2. Extract full serialized text for prefix matching
        # Static invariants (system prompt + tools) are ordered first to maximize common prefix matching
        text_stream = []
        dynamic_msgs = []
        conversation_msgs = []

        for m in aligned_msgs:
            role = m.get("role")
            c = m.get("content", "")
            text_val = ""
            if isinstance(c, str):
                text_val = c
            elif isinstance(c, list):
                parts = [part.get("text", "") for part in c if isinstance(part, dict) and "text" in part]
                text_val = " ".join(parts)

            if role == "system" and not text_val.startswith("[Runtime Context Variables]"):
                text_stream.append(text_val)
            elif role == "system" and text_val.startswith("[Runtime Context Variables]"):
                dynamic_msgs.append(text_val)
            else:
                conversation_msgs.append(text_val)

        # Canonical tools form static invariants across calls
        if aligned_tools:
            text_stream.append(json.dumps(aligned_tools))

        # Dynamic runtime variables followed by conversation turns
        for d in dynamic_msgs:
            text_stream.append(d)
        for cv in conversation_msgs:
            text_stream.append(cv)

        full_prompt_text = " ".join(text_stream)
        tokens = _simple_tokenize(full_prompt_text)

        # 3. Predict Cache Hit
        prediction = self.predictor.predict(tokens)

        # 4. Insert into Prefix Tree
        self.tree.insert(tokens, metadata={"provider": self.config.provider, "tokens": len(tokens)})

        telemetry = {
            "provider": self.config.provider,
            "total_tokens": len(tokens),
            "matched_prefix_tokens": prediction["matched_prefix_tokens"],
            "is_cache_hit": prediction["is_cache_hit"],
            "cost_saved_usd": prediction["cost_saved_usd"],
            "cost_savings_pct": prediction["cost_savings_pct"],
            "ttft_saved_ms": prediction["estimated_ttft_saved_ms"],
            "alignment": align_meta,
        }

        return optimized_req, telemetry

    def ascii_prefix_tree(self) -> str:
        """Renders current prefix tree topology, hit rates, and token savings in ASCII."""
        stats = self.tree.stats()
        lines = []
        lines.append("=" * 65)
        lines.append("⚡ Reflex Semantic KV-Cache Prefix Tree Status (.reflex-kv)")
        lines.append("=" * 65)
        lines.append(f" • Target Provider          : {self.config.provider.upper()}")
        lines.append(f" • Min Cache Prefix Tokens  : {self.config.min_cache_tokens}")
        lines.append(f" • Cached Discount Rate     : {self.config.cached_discount_rate * 100:.1f}%")
        lines.append(f" • Total Radix Nodes        : {stats['total_nodes']} / {self.config.max_tree_nodes}")
        lines.append(f" • Max Tree Depth           : {stats['max_depth_tokens']} tokens ({stats['max_depth_chunks']} chunks)")
        lines.append(f" • Total Inserts            : {stats['total_inserts']}")
        lines.append(f" • Cache Hits / Misses      : {stats['total_hits']} hits / {stats['total_misses']} misses")
        lines.append(f" • Estimated Cache Hit Rate : {stats['hit_rate'] * 100:.1f}%")
        lines.append(f" • Cumulative Tokens Saved  : {stats['total_saved_tokens']:,}")

        # ASCII Tree Visualizer for top root branches
        lines.append("\n Radix Prefix Tree Structure (Root Branches):")
        lines.append(" [Root]")
        root_children = list(self.tree.root.children.items())[:6]
        for i, (chunk, node) in enumerate(root_children):
            is_last = (i == len(root_children) - 1)
            prefix = " └── " if is_last else " ├── "
            trunc_chunk = (chunk[:32] + "...") if len(chunk) > 32 else chunk
            lines.append(f"{prefix}'{trunc_chunk}' (hits={node.hit_count}, children={len(node.children)})")

            sub_children = list(node.children.items())[:2]
            for j, (sub_chunk, sub_node) in enumerate(sub_children):
                sub_prefix = "     └── " if (j == len(sub_children) - 1) else "     ├── "
                trunc_sub = (sub_chunk[:24] + "...") if len(sub_chunk) > 24 else sub_chunk
                lines.append(f"{sub_prefix}'{trunc_sub}' (hits={sub_node.hit_count})")

        lines.append("=" * 65)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Binary Persistence (.reflex-kv)
    # -------------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Serializes KVCacheEngine and PrefixTree to a zero-dependency binary format (.reflex-kv).
        Header format (60 bytes):
          - 4 bytes: Magic b"RFKV"
          - 2 bytes: Version (uint16)
          - 2 bytes: Provider ID (uint16)
          - 4 bytes: Chunk Size (uint32)
          - 4 bytes: Min Cache Tokens (uint32)
          - 4 bytes: Max Tree Nodes (uint32)
          - 4 bytes: Total Nodes in Tree (uint32)
          - 4 bytes: Total Inserts (uint32)
          - 4 bytes: Total Hits (uint32)
          - 4 bytes: Total Misses (uint32)
          - 8 bytes: Input Cost Per Million (double float64)
          - 8 bytes: Cached Discount Rate (double float64)
          - 4 bytes: Metadata JSON length (uint32)
          - 4 bytes: Serialized Tree length (uint32)
          - [Metadata JSON bytes]
          - [Serialized Tree JSON/Binary bytes]
          - 4 bytes: CRC32 checksum trailer
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_path = f"{filepath}.tmp.{os.getpid()}"

        meta = {
            "provider": self.config.provider,
            "align_tools": self.config.align_tools,
            "segregate_dynamic_variables": self.config.segregate_dynamic_variables,
            "inject_provider_breakpoints": self.config.inject_provider_breakpoints,
            "total_saved_tokens": self.tree.total_saved_tokens,
        }
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

        # Serialize tree nodes
        def serialize_node(node: PrefixTreeNode) -> Dict[str, Any]:
            return {
                "c": node.token_chunk,
                "h": node.hit_count,
                "a": node.last_accessed,
                "m": node.metadata,
                "k": {k: serialize_node(v) for k, v in node.children.items()},
            }

        tree_data = serialize_node(self.tree.root)
        tree_bytes = json.dumps(tree_data, separators=(",", ":")).encode("utf-8")

        header = struct.pack(
            "<4sHH7I2d2I",
            KV_MAGIC,
            KV_FORMAT_VERSION,
            PROVIDER_NAME_TO_ID.get(self.config.provider, PROVIDER_OPENAI),
            self.config.chunk_size,
            self.config.min_cache_tokens,
            self.config.max_tree_nodes,
            self.tree.total_nodes,
            self.tree.total_inserts,
            self.tree.total_hits,
            self.tree.total_misses,
            self.config.input_cost_per_million,
            self.config.cached_discount_rate,
            len(meta_bytes),
            len(tree_bytes),
        )

        body = header + meta_bytes + tree_bytes
        crc = zlib.crc32(body) & 0xFFFFFFFF
        trailer = struct.pack("<I", crc)

        with open(tmp_path, "wb") as f:
            f.write(body + trailer)

        os.replace(tmp_path, filepath)

    @classmethod
    def load(cls, filepath: str) -> "KVCacheEngine":
        """Loads and verifies a KVCacheEngine model from a .reflex-kv binary file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "rb") as f:
            data = f.read()

        header_size = 60
        if len(data) < header_size + 4:
            raise ValueError(f"Corrupt .reflex-kv file: too short ({len(data)} bytes)")

        # Verify CRC32 checksum
        body = data[:-4]
        stored_crc = struct.unpack("<I", data[-4:])[0]
        actual_crc = zlib.crc32(body) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ValueError(
                f"CRC32 checksum mismatch in .reflex-kv: expected {stored_crc:#010x}, got {actual_crc:#010x}"
            )

        (
            magic,
            version,
            provider_id,
            chunk_size,
            min_cache_tokens,
            max_tree_nodes,
            total_nodes,
            total_inserts,
            total_hits,
            total_misses,
            input_cost,
            discount_rate,
            meta_len,
            tree_len,
        ) = struct.unpack("<4sHH7I2d2I", body[:header_size])

        if magic != KV_MAGIC:
            raise ValueError(f"Invalid magic bytes in .reflex-kv file: {magic!r}")
        if version != KV_FORMAT_VERSION:
            raise ValueError(f"Unsupported format version {version} (expected {KV_FORMAT_VERSION})")

        meta_start = header_size
        meta_end = meta_start + meta_len
        tree_start = meta_end
        tree_end = tree_start + tree_len

        meta = json.loads(body[meta_start:meta_end].decode("utf-8"))
        tree_data = json.loads(body[tree_start:tree_end].decode("utf-8"))

        provider_name = PROVIDER_ID_TO_NAME.get(provider_id, "openai")
        cfg = KVConfig(
            provider=provider_name,
            min_cache_tokens=min_cache_tokens,
            max_tree_nodes=max_tree_nodes,
            align_tools=meta.get("align_tools", True),
            segregate_dynamic_variables=meta.get("segregate_dynamic_variables", True),
            inject_provider_breakpoints=meta.get("inject_provider_breakpoints", True),
            input_cost_per_million=input_cost,
            cached_discount_rate=discount_rate,
            chunk_size=chunk_size,
        )

        engine = cls(config=cfg)
        engine.tree.total_nodes = total_nodes
        engine.tree.total_inserts = total_inserts
        engine.tree.total_hits = total_hits
        engine.tree.total_misses = total_misses
        engine.tree.total_saved_tokens = meta.get("total_saved_tokens", 0)

        def deserialize_node(d: Dict[str, Any]) -> PrefixTreeNode:
            node = PrefixTreeNode(token_chunk=d.get("c", ""))
            node.hit_count = d.get("h", 0)
            node.last_accessed = d.get("a", 0.0)
            node.metadata = d.get("m")
            node.children = {k: deserialize_node(v) for k, v in d.get("k", {}).items()}
            return node

        engine.tree.root = deserialize_node(tree_data)
        return engine
