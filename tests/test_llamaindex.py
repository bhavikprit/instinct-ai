"""
Unit tests for Reflex LlamaIndex QueryRouter and NodePostprocessor.
"""

import unittest
from reflex.integrations.llamaindex import ReflexQueryRouter, ReflexNodePostprocessor


class MockNode:
    def __init__(self, text: str):
        self.text = text


class MockQueryBundle:
    def __init__(self, query_str: str):
        self.query_str = query_str


class TestLlamaIndexIntegration(unittest.TestCase):

    def setUp(self):
        self.router = ReflexQueryRouter(
            choices={
                "sql_engine": "Revenue, sales transactions, SQL database queries",
                "vector_docs": "Developer documentation, Python SDK guides, API references",
                "summary_engine": "Annual company reports and executive high-level summaries",
            }
        )

    def test_query_routing_sql(self):
        selected = self.router.route("SELECT sum(amount) from transactions WHERE quarter = 'Q3'")
        self.assertEqual(selected, "sql_engine")

    def test_query_routing_bundle(self):
        bundle = MockQueryBundle("Where can I find the Python SDK API documentation?")
        selected = self.router.route(bundle)
        self.assertEqual(selected, "vector_docs")

    def test_route_with_metadata(self):
        res = self.router.route_with_metadata("Query accounting database for gross receipts")
        self.assertIn("selected", res)
        self.assertIn("distribution", res)
        self.assertIn("latency_ms", res)
        self.assertLess(res["latency_ms"], 20.0)

    def test_node_postprocessor_filter(self):
        postprocessor = ReflexNodePostprocessor(relevance_threshold=0.3)
        nodes = [
            MockNode("Database index maintenance and query optimization techniques"),
            MockNode("The recipe for homemade Italian pasta carbonara with eggs"),
        ]
        filtered = postprocessor.postprocess_nodes(
            nodes=nodes,
            query="How to optimize SQL query execution and database indexes?",
        )
        self.assertGreaterEqual(len(filtered), 1)
        self.assertEqual(filtered[0].text, nodes[0].text)


if __name__ == "__main__":
    unittest.main()
