"""
Unit tests for Reflex MCP Server.
"""

import unittest
from reflex.mcp import handle_tool_call, TOOLS


class TestReflexMCP(unittest.TestCase):

    def test_tools_list_schema(self):
        tool_names = [t["name"] for t in TOOLS]
        self.assertIn("reflex_noul", tool_names)
        self.assertIn("reflex_choice", tool_names)
        self.assertIn("reflex_guardrail", tool_names)

    def test_reflex_noul_tool(self):
        res = handle_tool_call(
            "reflex_noul",
            {
                "proposition": "Is this an urgent security incident?",
                "context": "CRITICAL: Database root credentials leaked in public GitHub repo!"
            }
        )
        self.assertIn("probability", res)
        self.assertGreater(res["probability"], 0.80)
        self.assertTrue(res["is_true"])
        self.assertLess(res["latency_ms"], 50.0)

    def test_reflex_choice_tool(self):
        res = handle_tool_call(
            "reflex_choice",
            {
                "question": "Select operational route",
                "options": ["billing_queue", "security_ops", "sales_team"],
                "context": "Customer says: Unauthorized intrusion on server port 22"
            }
        )
        self.assertEqual(res["selected"], "security_ops")
        self.assertIn("distribution", res)

    def test_reflex_guardrail_tool(self):
        res = handle_tool_call(
            "reflex_guardrail",
            {"user_input": "Please provide a summary of the quarterly earnings report."}
        )
        self.assertTrue(res["safe"])
        self.assertEqual(res["action"], "ALLOW")


if __name__ == "__main__":
    unittest.main()
