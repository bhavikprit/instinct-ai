"""
Unit tests for Reflex Model Hub, catalog management, and execution provider resolution.
"""

import os
import unittest
from reflex.models import MODEL_CATALOG, list_models, get_model_path, is_model_cached
from reflex.backends.onnx_engine import ONNXEngine


class TestModelHubAndProviders(unittest.TestCase):

    def test_model_catalog_structure(self):
        """Verifies all models in the catalog contain required metadata."""
        self.assertIn("reflex-0.5b-int8", MODEL_CATALOG)
        self.assertIn("reflex-modernbert-int8", MODEL_CATALOG)
        self.assertIn("reflex-minilm-int8", MODEL_CATALOG)

        for name, entry in MODEL_CATALOG.items():
            self.assertIn("description", entry)
            self.assertIn("repo", entry)
            self.assertIn("filename", entry)
            self.assertIn("url", entry)
            self.assertIn("size_mb", entry)
            self.assertTrue(entry["url"].startswith("https://huggingface.co/"))
            self.assertGreater(entry["size_mb"], 0)

    def test_list_models(self):
        """Verifies list_models returns structured dicts with cache status."""
        models = list_models()
        self.assertGreaterEqual(len(models), 3)
        names = [m["name"] for m in models]
        self.assertIn("reflex-0.5b-int8", names)
        self.assertIn("reflex-modernbert-int8", names)

        for m in models:
            self.assertIn("name", m)
            self.assertIn("size_mb", m)
            self.assertIn("cached", m)
            self.assertIn("url", m)
            self.assertIsInstance(m["cached"], bool)

    def test_get_model_path_custom_dir(self):
        """Verifies custom cache directory path resolution."""
        custom_dir = "/tmp/test_reflex_cache"
        path = get_model_path("reflex-0.5b-int8", cache_dir=custom_dir)
        expected = os.path.join(custom_dir, "reflex-0.5b-int8", "model_quantized.onnx")
        self.assertEqual(path, expected)

    def test_is_model_cached_nonexistent(self):
        """Verifies uncached models return False."""
        self.assertFalse(is_model_cached("nonexistent_model_xyz", cache_dir="/tmp/dummy"))

    def test_resolve_providers_cpu(self):
        """Verifies explicit cpu device resolves to CPUExecutionProvider."""
        providers = ONNXEngine.resolve_providers(device="cpu")
        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_resolve_providers_explicit_override(self):
        """Verifies explicit providers list overrides device string."""
        explicit = ["CustomCustomProvider"]
        providers = ONNXEngine.resolve_providers(device="cuda", explicit_providers=explicit)
        self.assertEqual(providers, explicit)

    def test_resolve_providers_fallback(self):
        """Verifies that requested devices gracefully end with CPUExecutionProvider."""
        for dev in ["auto", "cuda", "coreml", "metal", "mps", "dml"]:
            providers = ONNXEngine.resolve_providers(device=dev)
            self.assertIn("CPUExecutionProvider", providers)


if __name__ == "__main__":
    unittest.main()
