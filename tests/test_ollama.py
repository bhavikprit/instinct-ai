"""
Unit tests for Reflex Ollama Dual-Brain Bridge.
"""

import unittest
from unittest.mock import patch
from reflex.integrations.ollama import OllamaDualBrain


class TestOllamaDualBrain(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.brain = OllamaDualBrain(model="llama3.2", epistemic_threshold=0.75)

    def test_high_confidence_reflex_bypass(self):
        # Prompt has very clear security markers, Reflex resolves instantly
        res = self.brain.chat(
            prompt="URGENT: Hackers breached database credentials!",
            noul_question="Is this an urgent security emergency?",
        )
        self.assertEqual(res["resolved_by"], "reflex")
        self.assertFalse(res["ollama_called"])
        self.assertEqual(res["content"], "true")
        self.assertEqual(res["cost_usd"], 0.0)
        self.assertLess(res["latency_ms"], 50.0)

    def test_categorical_routing_bypass(self):
        res = self.brain.chat(
            prompt="I want an immediate refund for invoice #999.",
            choice_options=["billing", "sales", "security"],
        )
        self.assertEqual(res["resolved_by"], "reflex")
        self.assertFalse(res["ollama_called"])
        self.assertEqual(res["content"], "billing")

    @patch.object(OllamaDualBrain, "_call_ollama_generate")
    def test_uncertainty_escalation_to_ollama(self, mock_ollama):
        mock_ollama.return_value = "Ollama deliberative response"

        # Ambiguous statement with high threshold forces escalation
        brain_strict = OllamaDualBrain(model="llama3.2", epistemic_threshold=0.99)
        res = brain_strict.chat(
            prompt="Maybe this is an issue, but maybe not.",
            noul_question="Is this a confirmed critical bug?",
        )
        self.assertTrue(res["ollama_called"])
        self.assertEqual(res["resolved_by"], "ollama")
        self.assertEqual(res["content"], "Ollama deliberative response")
        mock_ollama.assert_called_once()

    async def test_async_chat_bypass(self):
        res = await self.brain.achat(
            prompt="Phishing alert! Fake bank login link detected.",
            noul_question="Is this a security threat?",
        )
        self.assertEqual(res["resolved_by"], "reflex")
        self.assertFalse(res["ollama_called"])


if __name__ == "__main__":
    unittest.main()
