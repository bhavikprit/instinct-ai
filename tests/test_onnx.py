"""
Unit tests for Reflex ONNX Neural Engine.
"""

import unittest
from reflex.primitives import Noul, Choice, Score
from reflex.backends.onnx_engine import ONNXEngine
from reflex.client import Reflex


class TestONNXEngine(unittest.TestCase):

    def setUp(self):
        # Use mock_mode=True for zero-external-dependency test stability
        self.engine = ONNXEngine(mock_mode=True, temperature=1.0)
        self.rx = Reflex(backend="onnx", mock_mode=True)

    def test_mock_mode_evaluation(self):
        res = self.engine.evaluate(
            state="URGENT SECURITY ALERT: Database breached!",
            questions={
                "is_danger": Noul("Is this a critical security threat?"),
                "route": Choice("Action", options=["quarantine", "allow", "ignore"]),
                "severity": Score("Severity rating 1-10", min_val=1.0, max_val=10.0),
            }
        )
        self.assertEqual(res.backend, "onnx-local")
        self.assertEqual(res.cost_usd, 0.0)
        self.assertIn("is_danger", res.decisions)
        self.assertTrue(res.decisions["is_danger"].probability > 0.6)
        self.assertIn("route", res.decisions)
        self.assertIn("severity", res.decisions)

    def test_client_integration(self):
        prob = self.rx.noul("Is this an urgent phishing attack?", "Click link now to avoid account lock!")
        self.assertGreater(prob, 0.65)

    def test_missing_dependencies_error(self):
        # Test that without mock_mode, it validates environment
        try:
            import onnxruntime
            import numpy
            # If installed, verify it can initialize
            engine = ONNXEngine(model_path="dummy.onnx")
            self.assertIsNotNone(engine)
        except ImportError:
            with self.assertRaises(ImportError):
                ONNXEngine(model_path="dummy.onnx")


if __name__ == "__main__":
    unittest.main()
