"""
Unit tests for Reflex Primitives and Client.
"""

import unittest
from reflex.primitives import Noul, Choice, Score
from reflex.client import Reflex
from reflex.backends.local import LocalEngine


class TestReflexPrimitives(unittest.TestCase):

    def test_noul_thresholds(self):
        n = Noul("Is this urgent?")
        resolved_high = n.resolve(0.95)
        self.assertTrue(resolved_high.is_true)
        self.assertFalse(resolved_high.is_uncertain)

        resolved_uncertain = n.resolve(0.52)
        self.assertTrue(resolved_uncertain.is_uncertain)
        self.assertFalse(resolved_uncertain.is_true)

        resolved_low = n.resolve(0.05)
        self.assertTrue(resolved_low.is_false)

    def test_choice_probabilities(self):
        c = Choice("Select category", options=["billing", "support", "sales"])
        resolved = c.resolve("billing", {"billing": 0.88, "support": 0.08, "sales": 0.04})
        self.assertEqual(resolved.selected, "billing")
        self.assertAlmostEqual(resolved.get_prob("billing"), 0.88)

    def test_score_range(self):
        s = Score("Urgency scale", min_val=1.0, max_val=10.0)
        resolved = s.resolve(8.7, confidence=0.95)
        self.assertEqual(resolved.score, 8.7)
        self.assertEqual(resolved.confidence, 0.95)


class TestReflexClient(unittest.TestCase):

    def setUp(self):
        self.rx = Reflex(backend="local")

    def test_multi_primitive_evaluation(self):
        result = self.rx.evaluate(
            state="URGENT: Suspicious unauthorized login attempt from foreign IP",
            questions={
                "is_fraud": Noul("Is this a security incident?"),
                "action": Choice("Action", options=["quarantine", "allow", "log"]),
                "threat": Score("Threat level 1-10")
            }
        )
        self.assertIn("is_fraud", result.decisions)
        self.assertIn("action", result.decisions)
        self.assertIn("threat", result.decisions)
        self.assertLess(result.latency_ms, 50.0) # Sub-50ms local latency
        self.assertEqual(result.cost_usd, 0.0)

    def test_shortcuts(self):
        prob = self.rx.noul("Is this an emergency?", "Critical security alert on primary database!")
        self.assertGreater(prob, 0.70)

        choice = self.rx.choice("Select route", ["billing", "security", "sales"], "Someone hacked my account password!")
        self.assertEqual(choice, "security")


if __name__ == "__main__":
    unittest.main()
