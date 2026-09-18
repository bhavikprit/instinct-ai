"""
Reflex LlamaIndex Integration.
Provides sub-millisecond query routing and zero-cost node post-processing for RAG pipelines.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

from reflex.client import Reflex
from reflex.primitives import Choice, Noul


class ReflexQueryRouter:
    """
    Drop-in Query Router for LlamaIndex RAG pipelines.
    
    Instead of burning 1.5–3.0 seconds calling an LLM selector to pick between
    a Vector Index, SQL Engine, or Summary Index, ReflexQueryRouter resolves
    the routing decision in <0.1ms for $0.00.
    
    Usage:
        router = ReflexQueryRouter(
            choices={
                "vector_engine": "Detailed knowledge base and API documentation search",
                "sql_engine": "Structured SQL queries for customer accounts and financial orders",
                "summary_engine": "High-level summaries and release note overviews",
            }
        )
        selected = router.route("What was our Q3 gross margin?")
        print(selected) # -> "sql_engine"
    """

    def __init__(
        self,
        choices: Union[List[str], Dict[str, str]],
        client: Optional[Reflex] = None,
        default_choice: Optional[str] = None,
    ):
        self.client = client or Reflex()
        if isinstance(choices, dict):
            self.options = list(choices.keys())
            self.descriptions = choices
        else:
            self.options = choices
            self.descriptions = None
        self.default_choice = default_choice or (self.options[0] if self.options else "")

    def route(self, query: Union[str, Any]) -> str:
        """Evaluates query and returns selected target engine name."""
        query_str = self._extract_query_str(query)
        if not query_str:
            return self.default_choice

        res = self.client.evaluate(
            state=query_str,
            questions={
                "route": Choice(
                    instructions="Select the best query engine for answering this question",
                    options=self.options,
                    criteria=self.descriptions,
                )
            },
        )
        return res.get_choice("route").selected or self.default_choice

    def route_with_metadata(self, query: Union[str, Any]) -> Dict[str, Any]:
        """Returns selected route alongside latency and confidence telemetry."""
        query_str = self._extract_query_str(query)
        res = self.client.evaluate(
            state=query_str,
            questions={
                "route": Choice(
                    instructions="Select the best query engine for answering this question",
                    options=self.options,
                    criteria=self.descriptions,
                )
            },
        )
        choice = res.get_choice("route")
        return {
            "selected": choice.selected or self.default_choice,
            "distribution": choice.distribution,
            "latency_ms": res.latency_ms,
            "backend": res.backend,
            "cost_usd": res.cost_usd,
        }

    def _extract_query_str(self, query: Union[str, Any]) -> str:
        if isinstance(query, str):
            return query
        # Handles LlamaIndex QueryBundle objects
        if hasattr(query, "query_str"):
            return str(query.query_str)
        return str(query)


class ReflexNodePostprocessor:
    """
    Sub-1ms Node Postprocessor for LlamaIndex.
    Evaluates retrieved chunks for relevance or safety before passing them
    to heavy synthesis LLMs, trimming prompt tokens and discarding low-relevance noise.
    """

    def __init__(
        self,
        relevance_threshold: float = 0.5,
        client: Optional[Reflex] = None,
    ):
        self.threshold = relevance_threshold
        self.client = client or Reflex()

    def postprocess_nodes(
        self,
        nodes: List[Any],
        query: Union[str, Any],
    ) -> List[Any]:
        """Filters retrieved nodes by relevance to query."""
        if not nodes:
            return []

        query_str = str(getattr(query, "query_str", query))
        filtered = []

        for node in nodes:
            text = self._get_node_text(node)
            prob = self.client.noul(
                instructions=f"Is this retrieved context relevant to answering: '{query_str}'?",
                state=text,
                threshold=self.threshold,
            )
            if prob >= self.threshold:
                filtered.append(node)

        return filtered

    def _get_node_text(self, node: Any) -> str:
        if isinstance(node, str):
            return node
        if hasattr(node, "get_content"):
            return node.get_content()
        if hasattr(node, "text"):
            return str(node.text)
        if hasattr(node, "node") and hasattr(node.node, "text"):
            return str(node.node.text)
        return str(node)
