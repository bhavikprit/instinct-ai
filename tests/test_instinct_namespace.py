"""
Unit tests for instinct-ai namespace and cross-alias interoperability.
"""

import unittest


class TestInstinctNamespace(unittest.TestCase):
    def test_import_instinct_core(self):
        import instinct
        self.assertEqual(instinct.__version__, "0.2.0")

        from instinct import Instinct, Reflex, Noul, Choice, Score, DecisionResult
        self.assertTrue(callable(Instinct))
        self.assertTrue(callable(Reflex))
        self.assertTrue(callable(Noul))
        self.assertTrue(callable(Choice))
        self.assertTrue(callable(Score))

    def test_instinct_evaluation(self):
        import instinct
        ins = instinct.Instinct(backend="local")
        res = ins.evaluate(
            state="User asks for refund on duplicate charge",
            questions={
                "refund": instinct.Noul("Is refund requested?"),
                "dept": instinct.Choice("Department", ["billing", "tech", "sales"]),
                "distress": instinct.Score("Distress 1-10", min_val=1.0, max_val=10.0),
            },
        )
        self.assertTrue(res["refund"].is_true)
        self.assertEqual(res["dept"].selected, "billing")
        self.assertLess(res.latency_ms, 50.0)
        self.assertEqual(res.cost_usd, 0.0)

    def test_instinct_submodules_import(self):
        from instinct.compiler import InstinctCompiler
        from instinct.index import HNSWIndex
        from instinct.conformal import ConformalPredictor
        from instinct.pq import ProductQuantizer
        from instinct.simd import get_simd_engine
        from instinct.guardrails import GuardrailSuite
        from instinct.cascade import CascadeRouter
        from instinct.drift import DriftGuard
        from instinct.kv import KVCacheEngine

        self.assertTrue(callable(InstinctCompiler))
        self.assertTrue(callable(HNSWIndex))
        self.assertTrue(callable(ConformalPredictor))
        self.assertTrue(callable(ProductQuantizer))
        self.assertTrue(callable(get_simd_engine))
        self.assertTrue(callable(GuardrailSuite))
        self.assertTrue(callable(CascadeRouter))
        self.assertTrue(callable(DriftGuard))
        self.assertTrue(callable(KVCacheEngine))

    def test_backward_compatibility_reflex(self):
        import reflex
        self.assertEqual(reflex.__version__, "0.2.0")

        rx = reflex.Reflex(backend="local")
        res_rx = rx.evaluate("test", {"q": reflex.Noul("Is this a test?")})
        self.assertIn("q", res_rx)

    def test_instinct_cli_entrypoint(self):
        import instinct.cli
        self.assertTrue(hasattr(instinct.cli, "main"))
        self.assertTrue(callable(instinct.cli.main))

    def test_instinct_integrations(self):
        from instinct.integrations.langchain import InstinctRouterNode, InstinctGuardrailNode
        from instinct.integrations.llamaindex import InstinctQueryRouter, InstinctNodePostprocessor
        self.assertTrue(callable(InstinctRouterNode))
        self.assertTrue(callable(InstinctGuardrailNode))
        self.assertTrue(callable(InstinctQueryRouter))
        self.assertTrue(callable(InstinctNodePostprocessor))


if __name__ == "__main__":
    unittest.main()
