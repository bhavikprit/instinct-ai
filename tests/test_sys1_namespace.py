"""
Tests for sys1 namespace and backward compatibility with reflex.
"""

import unittest


class TestSys1Namespace(unittest.TestCase):
    def test_import_sys1_core(self):
        import sys1
        self.assertEqual(sys1.__version__, "0.2.0")
        
        from sys1 import Reflex, Noul, Choice, Score, DecisionResult
        self.assertTrue(callable(Reflex))
        self.assertTrue(callable(Noul))
        self.assertTrue(callable(Choice))
        self.assertTrue(callable(Score))

    def test_sys1_evaluation(self):
        import sys1
        rx = sys1.Reflex(backend="local")
        res = rx.evaluate(
            state="User asks for refund on duplicate charge",
            questions={
                "refund": sys1.Noul("Is refund requested?"),
                "dept": sys1.Choice("Department", ["billing", "tech", "sales"]),
                "distress": sys1.Score("Distress 1-10", min_val=1.0, max_val=10.0),
            }
        )
        self.assertTrue(res["refund"].is_true)
        self.assertEqual(res["dept"].selected, "billing")
        self.assertLess(res.latency_ms, 50.0)
        self.assertEqual(res.cost_usd, 0.0)

    def test_sys1_submodules_import(self):
        from sys1.compiler import InstinctCompiler
        from sys1.index import HNSWIndex
        from sys1.conformal import ConformalPredictor
        from sys1.pq import ProductQuantizer
        from sys1.simd import get_simd_engine
        from sys1.guardrails import GuardrailSuite
        
        self.assertTrue(callable(InstinctCompiler))
        self.assertTrue(callable(HNSWIndex))
        self.assertTrue(callable(ConformalPredictor))
        self.assertTrue(callable(ProductQuantizer))
        self.assertTrue(callable(get_simd_engine))
        self.assertTrue(callable(GuardrailSuite))

    def test_backward_compatibility_reflex_import(self):
        import reflex
        self.assertEqual(reflex.__version__, "0.2.0")
        
        from reflex import Reflex, Noul, Choice, Score
        rx = Reflex(backend="local")
        res = rx.evaluate("test", {"q": Noul("Is this a test?")})
        self.assertIn("q", res)

    def test_sys1_cli_entrypoint(self):
        import sys1.cli
        self.assertTrue(hasattr(sys1.cli, "main"))
        self.assertTrue(callable(sys1.cli.main))


if __name__ == "__main__":
    unittest.main()
