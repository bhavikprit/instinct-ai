"""
Unit tests for Pure-Python Semantic Vector Encoder and Engine.
"""

import math
import unittest
from reflex.embeddings import (
    SemanticVectorEncoder,
    cosine_similarity,
    PureSemanticEngine,
)
from reflex.client import Reflex
from reflex.primitives import Noul, Choice, Score


class TestSemanticEmbeddings(unittest.TestCase):

    def setUp(self):
        self.encoder = SemanticVectorEncoder()
        self.engine = PureSemanticEngine()
        self.rx = Reflex(backend="semantic")

    def test_vector_dimension_and_normalization(self):
        vec = self.encoder.encode("Hello world this is a test")
        self.assertEqual(len(vec), 384)
        l2_norm = math.sqrt(sum(x * x for x in vec))
        self.assertAlmostEqual(l2_norm, 1.0, places=4)

    def test_empty_string_vector(self):
        vec = self.encoder.encode("")
        self.assertEqual(len(vec), 384)
        self.assertEqual(sum(vec), 0.0)

    def test_cosine_similarity(self):
        v1 = self.encoder.encode("PostgreSQL database crash")
        v2 = self.encoder.encode("Database engine failure")
        v3 = self.encoder.encode("Fresh baked apple pie")

        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)

        self.assertGreater(sim_related, sim_unrelated)

    def test_pure_semantic_engine_evaluate(self):
        res = self.engine.evaluate(
            state="Invoice #889 was billed twice, please issue a refund immediately.",
            questions={
                "is_refund": Noul("Does the user demand a refund?"),
                "queue": Choice("Route department", options=["billing", "engineering", "sales"]),
                "distress": Score("Rate user distress 1-10", min_val=1.0, max_val=10.0),
            }
        )
        self.assertEqual(res.backend, "semantic-pure")
        self.assertEqual(res.cost_usd, 0.0)
        self.assertIn("is_refund", res.decisions)
        self.assertGreater(res["is_refund"].probability, 0.60)
        self.assertEqual(res["queue"].selected, "billing")
        self.assertIn("distress", res.decisions)

    def test_reflex_client_integration(self):
        prob = self.rx.noul("Is this a security hazard?", "Ransomware warning: files locked!")
        self.assertGreater(prob, 0.55)

        choice = self.rx.choice("Select route", ["billing", "security", "support"], "Someone hacked my account!")
        self.assertEqual(choice, "security")


if __name__ == "__main__":
    unittest.main()
