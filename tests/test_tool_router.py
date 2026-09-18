"""
Unit tests for Reflex Fast Tool Router.
"""

import unittest
from reflex.tool_router import FastToolRouter, ToolDefinition


class TestFastToolRouter(unittest.TestCase):

    def setUp(self):
        self.router = FastToolRouter()
        self.router.register_tool("calc", "Performs mathematical arithmetic calculations")
        self.router.register_tool("sql", "Executes SQL queries on database")
        self.router.register_tool("search", "Searches the web for latest online news")
        self.router.register_tool("refund", "Processes customer payment refunds and billing adjustments")

    def test_route_math_query(self):
        ranked = self.router.route("What is 482 multiplied by 19?", top_k=2)
        self.assertEqual(len(ranked), 2)
        top_tool, prob = ranked[0]
        self.assertEqual(top_tool.name, "calc")
        self.assertGreater(prob, 0.0)

    def test_route_refund_query(self):
        ranked = self.router.route("Customer wants refund for invoice charge", top_k=1)
        self.assertEqual(len(ranked), 1)
        top_tool, _ = ranked[0]
        self.assertEqual(top_tool.name, "refund")

    def test_filter_openai_tools(self):
        tools = [
            {"type": "function", "function": {"name": "calc", "description": "Math calculator"}},
            {"type": "function", "function": {"name": "sql", "description": "Database queries"}},
            {"type": "function", "function": {"name": "refund", "description": "Refund processor"}},
            {"type": "function", "function": {"name": "email", "description": "Send email"}},
        ]
        pruned = FastToolRouter.filter_openai_tools(
            prompt="Compute the square root of 144",
            tools=tools,
            top_k=2,
        )
        self.assertEqual(len(pruned), 2)
        names = [t["function"]["name"] for t in pruned]
        self.assertIn("calc", names)

    def test_empty_tools(self):
        empty_router = FastToolRouter()
        self.assertEqual(empty_router.route("hello"), [])

    def test_single_tool_bypass(self):
        single_router = FastToolRouter()
        single_router.register_tool("only_tool", "Does everything")
        res = single_router.route("anything", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0].name, "only_tool")


if __name__ == "__main__":
    unittest.main()
