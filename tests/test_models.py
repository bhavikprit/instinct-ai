"""
Unit tests for Reflex Model Downloader and Catalog Manager.
"""

import os
import tempfile
import unittest
from reflex.models import (
    MODEL_CATALOG,
    get_model_path,
    is_model_cached,
    download_model,
)


class TestModelManager(unittest.TestCase):

    def test_catalog_entries(self):
        self.assertIn("reflex-modernbert-int8", MODEL_CATALOG)
        self.assertIn("reflex-minilm-int8", MODEL_CATALOG)
        for key, meta in MODEL_CATALOG.items():
            self.assertTrue(meta["url"].startswith("https://huggingface.co/"))
            self.assertTrue(meta["filename"].endswith(".onnx"))

    def test_get_model_path(self):
        custom_dir = "/tmp/reflex_test_models"
        path = get_model_path("reflex-modernbert-int8", cache_dir=custom_dir)
        self.assertTrue(path.startswith(custom_dir))
        self.assertTrue(path.endswith("model_quantized.onnx"))

    def test_is_model_cached_false(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertFalse(is_model_cached("reflex-modernbert-int8", cache_dir=tmp_dir))

    def test_download_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            download_model("nonexistent-invalid-model-name")

    def test_download_local_path(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            resolved = download_model(tmp.name)
            self.assertEqual(resolved, os.path.abspath(tmp.name))


if __name__ == "__main__":
    unittest.main()
