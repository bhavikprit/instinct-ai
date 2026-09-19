"""
Unit and integration tests for Reflex Multimodal Decision Primitives & Vision (Phase 20).
Validates zero-dependency PNG encode/decode, difference hashing (dHash),
Hamming distance invariance, VisualNoul/VisualChoice, and gateway visual caching.
"""

import base64
import json
import time
import unittest
import urllib.request

from reflex.vision import (
    RawImage,
    ZeroDepImageDecoder,
    PerceptualHasher,
    VisualNoul,
    VisualChoice,
)
from reflex.client import Reflex
from reflex.gateway import ReflexGatewayServer, GatewayConfig


class TestVisionPrimitives(unittest.TestCase):
    """Test pure-Python image processing, hashing, and visual decision primitives."""

    def test_png_encode_and_decode_pure_python(self):
        # 1. Create a synthetic test pattern
        w, h = 32, 24
        rgb_data = bytearray(w * h * 3)
        for y in range(h):
            for x in range(w):
                idx = (y * w + x) * 3
                rgb_data[idx] = (x * 8) & 0xFF
                rgb_data[idx + 1] = (y * 10) & 0xFF
                rgb_data[idx + 2] = 128

        # 2. Encode to PNG bytes
        png_bytes = ZeroDepImageDecoder.encode_png(w, h, bytes(rgb_data))
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

        # 3. Decode back using native zero-dependency decoder
        decoded = ZeroDepImageDecoder.decode(png_bytes)
        self.assertEqual(decoded.width, w)
        self.assertEqual(decoded.height, h)
        self.assertEqual(decoded.channels, 3)
        self.assertEqual(decoded.data, bytes(rgb_data))

    def test_grayscale_and_resampling(self):
        # Pure red image (255, 0, 0)
        red_rgb = bytes([255, 0, 0] * 16)
        img = RawImage(width=4, height=4, channels=3, data=red_rgb)

        gray = img.to_grayscale()
        self.assertEqual(gray.channels, 1)
        self.assertEqual(len(gray.data), 16)
        # 0.299 * 255 = 76
        self.assertEqual(gray.data[0], 76)

        # Resize to 2x2
        resized = gray.resize(2, 2)
        self.assertEqual(resized.width, 2)
        self.assertEqual(resized.height, 2)
        self.assertEqual(len(resized.data), 4)

    def test_perceptual_dhash_and_hamming_distance(self):
        receipt_img = ZeroDepImageDecoder.create_synthetic_image(64, 64, pattern="receipt")
        screenshot_img = ZeroDepImageDecoder.create_synthetic_image(64, 64, pattern="screenshot")

        # Compute dHash
        hash_receipt = PerceptualHasher.dhash(receipt_img)
        hash_screenshot = PerceptualHasher.dhash(screenshot_img)

        self.assertIsInstance(hash_receipt, int)
        self.assertIsInstance(hash_screenshot, int)

        # Create slightly resized receipt (60x60)
        resized_receipt = receipt_img.resize(60, 60)
        hash_receipt_resized = PerceptualHasher.dhash(resized_receipt)

        # Perceptual stability: Hamming distance between original and resized receipt should be small (<= 4)
        dist_same = PerceptualHasher.hamming_distance(hash_receipt, hash_receipt_resized)
        self.assertLessEqual(dist_same, 4)

        # Invariance: Receipt vs Dark Screenshot should have large Hamming distance (> 10)
        dist_diff = PerceptualHasher.hamming_distance(hash_receipt, hash_screenshot)
        self.assertGreater(dist_diff, 10)

        # Test hex format
        hex_str = PerceptualHasher.dhash_hex(receipt_img)
        self.assertEqual(len(hex_str), 16)  # 64 bits = 16 hex chars

    def test_visual_features_extraction(self):
        img = ZeroDepImageDecoder.create_synthetic_image(64, 64, pattern="receipt")
        feats = PerceptualHasher.extract_features(img, dim=128)

        self.assertEqual(len(feats), 128)
        # Unit norm verification
        norm = sum(x * x for x in feats) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_visual_noul_and_choice_shortcuts(self):
        rx = Reflex()

        receipt = ZeroDepImageDecoder.create_synthetic_image(64, 64, pattern="receipt")
        screenshot = ZeroDepImageDecoder.create_synthetic_image(64, 64, pattern="screenshot")
        id_card = ZeroDepImageDecoder.create_synthetic_image(64, 64, pattern="id_card")

        # 1. Visual Noul
        p_receipt = rx.visual_noul("Is this a bright white receipt paper?", receipt)
        self.assertGreater(p_receipt, 0.6)

        p_dark = rx.visual_noul("Is this a dark IDE code terminal?", screenshot)
        self.assertGreater(p_dark, 0.6)

        # 2. Visual Choice
        opt_receipt = rx.visual_choice(
            instructions="Select document category",
            options=["receipt", "screenshot", "id_card"],
            image=receipt,
        )
        self.assertEqual(opt_receipt, "receipt")

        opt_screenshot = rx.visual_choice(
            instructions="Select document category",
            options=["receipt", "screenshot", "id_card"],
            image=screenshot,
        )
        self.assertEqual(opt_screenshot, "screenshot")


class TestGatewayMultimodalCaching(unittest.TestCase):
    """Test multimodal image deduplication at the AI Envoy Gateway reverse proxy."""

    @classmethod
    def setUpClass(cls):
        cls.port = 18991
        cls.config = GatewayConfig(
            host="127.0.0.1",
            port=cls.port,
            cache_enabled=True,
            guardrails_enabled=True,
            system1_routing_enabled=True,
        )
        cls.server = ReflexGatewayServer(cls.config)
        cls.server.start(background=True)
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_gateway_image_cache_hit(self):
        # Create a test image and encode to base64 data URI
        receipt_raw = ZeroDepImageDecoder.create_synthetic_image(48, 48, pattern="receipt")
        png_bytes = ZeroDepImageDecoder.encode_png(48, 48, receipt_raw.data)
        b64_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"

        req_payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this expense document: route and verify"},
                        {"type": "image_url", "image_url": {"url": b64_url}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(req_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        # Request 1: Evaluated via System 1 / Cached
        req1 = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req1, timeout=5.0) as resp:
            self.assertEqual(resp.status, 200)
            data1 = json.loads(resp.read().decode("utf-8"))
            self.assertIn("choices", data1)

        # Request 2: Send identical image + prompt -> should HIT the cache (<1ms)
        t0 = time.perf_counter()
        req2 = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req2, timeout=5.0) as resp:
            self.assertEqual(resp.status, 200)
            cache_header = resp.headers.get("X-Reflex-Cache")
            self.assertIn(cache_header, ("HIT-L1", "HIT-L2", "SHORTCIRCUIT-SYSTEM1"))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.assertLess(elapsed_ms, 25.0)  # Sub-25ms loopback roundtrip


if __name__ == "__main__":
    unittest.main()
