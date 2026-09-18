"""
Model Manager for Reflex System 1 open-weight checkpoints.
Zero-dependency model downloading, caching, and catalog resolution.
"""

from __future__ import annotations
import os
import urllib.request
import urllib.error
from typing import Dict, Optional, Any

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/reflex/models")

# Curated catalog of lightweight open-weights decision models
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "reflex-modernbert-int8": {
        "description": "ModernBERT quantized INT8 decision model (~45MB)",
        "repo": "reflex-ai/reflex-modernbert-int8",
        "filename": "model_quantized.onnx",
        "url": "https://huggingface.co/reflex-ai/reflex-modernbert-int8/resolve/main/model_quantized.onnx",
        "size_mb": 45.2,
    },
    "reflex-minilm-int8": {
        "description": "MiniLM-L6-v2 quantized INT8 decision model (~22MB)",
        "repo": "reflex-ai/reflex-minilm-int8",
        "filename": "model_quantized.onnx",
        "url": "https://huggingface.co/reflex-ai/reflex-minilm-int8/resolve/main/model_quantized.onnx",
        "size_mb": 22.8,
    },
}


def get_model_path(model_name: str, cache_dir: Optional[str] = None) -> str:
    """Returns the absolute path where the model is or should be cached."""
    base_dir = cache_dir or DEFAULT_CACHE_DIR
    target_dir = os.path.join(base_dir, model_name)
    filename = MODEL_CATALOG.get(model_name, {}).get("filename", "model.onnx")
    return os.path.join(target_dir, filename)


def is_model_cached(model_name: str, cache_dir: Optional[str] = None) -> bool:
    """Checks whether the specified model is downloaded and cached."""
    path = get_model_path(model_name, cache_dir)
    return os.path.exists(path) and os.path.getsize(path) > 0


def download_model(
    model_name_or_url: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Downloads an ONNX model into the local cache directory.
    
    Args:
        model_name_or_url: Known catalog key or direct HTTPS URL.
        cache_dir: Custom cache directory (defaults to ~/.cache/reflex/models).
        force: If True, re-downloads even if already cached.
        
    Returns:
        Absolute path to the downloaded .onnx file.
    """
    base_dir = cache_dir or DEFAULT_CACHE_DIR

    if model_name_or_url in MODEL_CATALOG:
        info = MODEL_CATALOG[model_name_or_url]
        url = info["url"]
        target_path = get_model_path(model_name_or_url, cache_dir)
    elif model_name_or_url.startswith("http://") or model_name_or_url.startswith("https://"):
        url = model_name_or_url
        filename = os.path.basename(url.split("?")[0]) or "custom_model.onnx"
        target_path = os.path.join(base_dir, "custom", filename)
    else:
        # Assume it's a direct local file path
        if os.path.exists(model_name_or_url):
            return os.path.abspath(model_name_or_url)
        raise ValueError(
            f"Unknown model '{model_name_or_url}'. "
            f"Choose from: {list(MODEL_CATALOG.keys())} or provide a direct URL or local path."
        )

    if os.path.exists(target_path) and not force and os.path.getsize(target_path) > 0:
        return target_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".tmp"

    print(f"Downloading {model_name_or_url} from {url}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "reflex-ai/0.1.0"},
        )
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
            chunk_size = 64 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)

        os.replace(temp_path, target_path)
        print(f"✅ Saved model to: {target_path}")
        return target_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e
