"""
Unit tests for Reflex CLI commands.
"""

import io
import sys
import unittest
from unittest.mock import patch

from reflex.cli import run_doctor, main


class TestCLI(unittest.TestCase):
    def test_run_doctor(self):
        f = io.StringIO()
        with patch("sys.stdout", f):
            run_doctor()
        out = f.getvalue()
        self.assertIn("Reflex System Diagnostic & Environment Doctor", out)
        self.assertIn("Host Environment:", out)
        self.assertIn("Runtime Engines & Hardware Acceleration:", out)
        self.assertIn("Live Microsecond Latency Benchmark", out)
        self.assertIn("Diagnostic check complete", out)

    def test_cli_models_list(self):
        f = io.StringIO()
        with patch("sys.stdout", f), patch.object(sys, "argv", ["reflex", "models", "list"]):
            main()
        out = f.getvalue()
        self.assertIn("Reflex Open-Weights Model Catalog:", out)
        self.assertIn("reflex-0.5b-int8", out)


if __name__ == "__main__":
    unittest.main()
