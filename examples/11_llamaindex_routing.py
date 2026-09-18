"""
Example 11: LlamaIndex Sub-Millisecond Query Routing & Node Postprocessing.

Demonstrates:
1. ReflexQueryRouter: Sub-0.1ms routing between Vector, SQL, and Summary indices.
2. ReflexNodePostprocessor: Rapid relevance filtering of retrieved chunks before LLM synthesis.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex.integrations.llamaindex import ReflexQueryRouter, ReflexNodePostprocessor


class DocumentNode:
    """Mock LlamaIndex NodeWithScore / TextNode."""
    def __init__(self, text: str, doc_id: str):
        self.text = text
        self.doc_id = doc_id


def main():
    print("=" * 70)
    print("⚡ Reflex LlamaIndex Sub-Millisecond Query Routing")
    print("=" * 70)

    # 1. Initialize ReflexQueryRouter
    router = ReflexQueryRouter(
        choices={
            "sql_financial_db": "SQL queries on transactions, invoices, gross margins, revenues",
            "vector_api_docs": "Developer documentation, Python SDK guides, code examples, API references",
            "summary_quarterly": "Executive summaries, shareholder letters, high-level quarterly commentary",
        }
    )

    test_queries = [
        "What was our net operating cashflow in Q2 2024?",
        "How do I configure AsyncReflex client with custom timeout in Python?",
        "Provide an executive overview of the latest annual shareholder letter.",
    ]

    print("\n1. Routing Queries to Target Engines (<0.1ms each):")
    for q in test_queries:
        res = router.route_with_metadata(q)
        print(f"\nQuery: \"{q}\"")
        print(f" -> Selected Engine : {res['selected']}")
        print(f" -> Latency         : {res['latency_ms']} ms")
        print(f" -> Cost            : ${res['cost_usd']:.4f}")

    # 2. Node Postprocessor Filtering
    print("\n" + "=" * 70)
    print("2. ReflexNodePostprocessor: Sub-Millisecond Chunk Filtering")
    print("=" * 70)

    postprocessor = ReflexNodePostprocessor(relevance_threshold=0.4)

    retrieved_nodes = [
        DocumentNode(
            text="Reflex provides sub-15ms local spinal decisions with zero external dependencies.",
            doc_id="doc_1"
        ),
        DocumentNode(
            text="The best way to make espresso at home requires fresh beans ground at 18 clicks.",
            doc_id="doc_2"
        ),
        DocumentNode(
            text="AsyncReflex enables non-blocking evaluation in FastAPI and asyncio event loops.",
            doc_id="doc_3"
        ),
    ]

    rag_query = "Tell me about Reflex local execution and asyncio performance."
    print(f"\nRAG Query: \"{rag_query}\"")
    print(f"Retrieved Chunks: {len(retrieved_nodes)}")

    filtered_nodes = postprocessor.postprocess_nodes(retrieved_nodes, query=rag_query)
    print(f"Kept Relevant Chunks: {len(filtered_nodes)}")
    for node in filtered_nodes:
        print(f" • [{node.doc_id}]: {node.text}")

    print("\n✅ LlamaIndex integration demonstration complete!")


if __name__ == "__main__":
    main()
