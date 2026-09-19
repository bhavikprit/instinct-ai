"""
Reflex Multimodal Decision Primitives & Perceptual Image Vision (Phase 20).
Enables zero-dependency visual classification, structural feature extraction,
and perceptual image deduplication without calling costly multimodal vision LLMs.
Pure Python standard library only (struct, zlib, math, hashlib, io, base64).
"""

from __future__ import annotations
import base64
from dataclasses import dataclass
import hashlib
import io
import math
import os
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import zlib

from reflex.primitives import PrimitiveType, DecisionResult


@dataclass
class RawImage:
    """Zero-dependency in-memory image container."""
    width: int
    height: int
    channels: int  # 1 for Grayscale, 3 for RGB, 4 for RGBA
    data: bytes    # Raw byte buffer (width * height * channels)

    def to_grayscale(self) -> RawImage:
        """Converts RGB/RGBA image to 1-channel luminance."""
        if self.channels == 1:
            return self

        out = bytearray(self.width * self.height)
        step = self.channels
        data = self.data

        for i in range(self.width * self.height):
            idx = i * step
            r = data[idx]
            g = data[idx + 1]
            b = data[idx + 2]
            # Standard Rec. 601 luma formula
            out[i] = int(0.299 * r + 0.587 * g + 0.114 * b)

        return RawImage(width=self.width, height=self.height, channels=1, data=bytes(out))

    def resize(self, target_w: int, target_h: int) -> RawImage:
        """Fast nearest-neighbor spatial resampling."""
        if self.width == target_w and self.height == target_h:
            return self

        out = bytearray(target_w * target_h * self.channels)
        x_ratio = self.width / target_w
        y_ratio = self.height / target_h
        ch = self.channels
        src = self.data

        for y in range(target_h):
            src_y = min(self.height - 1, int(y * y_ratio))
            y_offset = y * target_w * ch
            src_y_offset = src_y * self.width * ch

            for x in range(target_w):
                src_x = min(self.width - 1, int(x * x_ratio))
                src_idx = src_y_offset + src_x * ch
                dst_idx = y_offset + x * ch
                for c in range(ch):
                    out[dst_idx + c] = src[src_idx + c]

        return RawImage(width=target_w, height=target_h, channels=self.channels, data=bytes(out))


class ZeroDepImageDecoder:
    """
    Pure-Python image parser for PNG, PPM, BMP, and raw image buffers.
    Zero external pip dependencies (no Pillow/OpenCV required).
    """

    @staticmethod
    def decode(source: Union[str, bytes, bytearray, RawImage]) -> RawImage:
        """Decode image from file path, raw bytes, base64 data URI, or RawImage."""
        if isinstance(source, RawImage):
            return source

        if isinstance(source, str):
            if source.startswith("data:image/"):
                # Base64 data URL: data:image/png;base64,iVBORw...
                comma = source.find(",")
                if comma != -1:
                    raw_b64 = source[comma + 1:]
                    return ZeroDepImageDecoder.decode(base64.b64decode(raw_b64))
            elif os.path.exists(source):
                with open(source, "rb") as f:
                    return ZeroDepImageDecoder.decode(f.read())
            else:
                # Try raw base64 decode
                try:
                    return ZeroDepImageDecoder.decode(base64.b64decode(source))
                except Exception:
                    raise ValueError(f"Cannot resolve image source from string: {source[:60]}...")

        raw_bytes = bytes(source)

        # 1. Check PNG signature: \x89PNG\r\n\x1a\n
        if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ZeroDepImageDecoder._decode_png(raw_bytes)

        # 2. Check PPM signature: P6 or P3
        if raw_bytes.startswith(b"P6") or raw_bytes.startswith(b"P3"):
            return ZeroDepImageDecoder._decode_ppm(raw_bytes)

        # 3. Check BMP signature: BM
        if raw_bytes.startswith(b"BM"):
            return ZeroDepImageDecoder._decode_bmp(raw_bytes)

        # 4. Optional Pillow fallback if installed
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            return RawImage(width=img.width, height=img.height, channels=3, data=img.tobytes())
        except Exception:
            pass

        raise ValueError("Unsupported image format. ZeroDepImageDecoder natively supports PNG, PPM, and BMP.")

    @staticmethod
    def _decode_png(data: bytes) -> RawImage:
        """Native zero-dependency PNG chunk parser and scanline reconstruction."""
        offset = 8
        width = 0
        height = 0
        bit_depth = 8
        color_type = 2  # 2: RGB, 6: RGBA, 0: Grayscale
        idat_chunks = []

        while offset < len(data):
            if offset + 8 > len(data):
                break
            length, chunk_type = struct.unpack(">I4s", data[offset:offset + 8])
            offset += 8
            chunk_data = data[offset:offset + length]
            offset += length + 4  # skip CRC

            if chunk_type == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
            elif chunk_type == b"IDAT":
                idat_chunks.append(chunk_data)
            elif chunk_type == b"IEND":
                break

        if width == 0 or height == 0:
            raise ValueError("Corrupt PNG: missing or invalid IHDR")

        # Determine channels
        channels = 1 if color_type == 0 else (3 if color_type == 2 else 4)
        bytes_per_pixel = channels

        # Decompress zlib stream
        raw_decompressed = zlib.decompress(b"".join(idat_chunks))
        stride = width * bytes_per_pixel
        output = bytearray(width * height * bytes_per_pixel)

        src_offset = 0
        dst_offset = 0
        prev_row = bytearray(stride)

        for _ in range(height):
            if src_offset >= len(raw_decompressed):
                break
            filter_type = raw_decompressed[src_offset]
            src_offset += 1
            curr_row = bytearray(raw_decompressed[src_offset:src_offset + stride])
            src_offset += stride

            # Scanline un-filtering
            if filter_type == 1:  # Sub
                for x in range(bytes_per_pixel, stride):
                    curr_row[x] = (curr_row[x] + curr_row[x - bytes_per_pixel]) & 0xFF
            elif filter_type == 2:  # Up
                for x in range(stride):
                    curr_row[x] = (curr_row[x] + prev_row[x]) & 0xFF
            elif filter_type == 3:  # Average
                for x in range(stride):
                    left = curr_row[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
                    up = prev_row[x]
                    curr_row[x] = (curr_row[x] + ((left + up) >> 1)) & 0xFF
            elif filter_type == 4:  # Paeth
                for x in range(stride):
                    a = curr_row[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
                    b = prev_row[x]
                    c = prev_row[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
                    p = a + b - c
                    pa = abs(p - a)
                    pb = abs(p - b)
                    pc = abs(p - c)
                    pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    curr_row[x] = (curr_row[x] + pr) & 0xFF

            output[dst_offset:dst_offset + stride] = curr_row
            prev_row = curr_row
            dst_offset += stride

        return RawImage(width=width, height=height, channels=channels, data=bytes(output))

    @staticmethod
    def _decode_ppm(data: bytes) -> RawImage:
        """Decodes binary PPM P6."""
        header_end = 0
        lines = []
        for line in data.split(b"\n"):
            header_end += len(line) + 1
            s = line.strip()
            if not s or s.startswith(b"#"):
                continue
            lines.append(s)
            if len(lines) >= 3:
                break

        dims = lines[1].split()
        width, height = int(dims[0]), int(dims[1])
        pixel_data = data[header_end:header_end + (width * height * 3)]
        return RawImage(width=width, height=height, channels=3, data=pixel_data)

    @staticmethod
    def _decode_bmp(data: bytes) -> RawImage:
        """Decodes uncompressed 24-bit Windows BMP."""
        if len(data) < 54:
            raise ValueError("Invalid BMP buffer")
        pixel_offset, = struct.unpack("<I", data[10:14])
        width, height = struct.unpack("<ii", data[18:26])
        height = abs(height)
        row_size = ((width * 3 + 3) // 4) * 4

        raw = bytearray(width * height * 3)
        dst = 0
        for y in range(height):
            # BMP rows are bottom-to-top
            src_row = pixel_offset + (height - 1 - y) * row_size
            for x in range(width):
                b = data[src_row + x * 3]
                g = data[src_row + x * 3 + 1]
                r = data[src_row + x * 3 + 2]
                raw[dst] = r
                raw[dst + 1] = g
                raw[dst + 2] = b
                dst += 3

        return RawImage(width=width, height=height, channels=3, data=bytes(raw))

    @staticmethod
    def encode_png(width: int, height: int, rgb_bytes: bytes) -> bytes:
        """Pure-Python standard library PNG encoder."""
        header = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_chunk = struct.pack(">I4s", len(ihdr), b"IHDR") + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))

        # Scanlines with filter 0 (None)
        stride = width * 3
        raw_scanlines = bytearray()
        for y in range(height):
            raw_scanlines.append(0)  # Filter 0
            raw_scanlines.extend(rgb_bytes[y * stride:(y + 1) * stride])

        compressed = zlib.compress(bytes(raw_scanlines), level=6)
        idat_chunk = struct.pack(">I4s", len(compressed), b"IDAT") + compressed + struct.pack(">I", zlib.crc32(b"IDAT" + compressed))

        iend_chunk = struct.pack(">I4s", 0, b"IEND") + struct.pack(">I", zlib.crc32(b"IEND"))
        return header + ihdr_chunk + idat_chunk + iend_chunk

    @staticmethod
    def create_synthetic_image(width: int, height: int, pattern: str = "receipt") -> RawImage:
        """Creates synthetic structural test patterns in pure Python."""
        rgb = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 3
                if pattern == "receipt":
                    # Light background with horizontal text bands
                    is_band = (y % 12 < 4) and (10 < x < width - 10)
                    val = 40 if is_band else 245
                    rgb[idx] = val
                    rgb[idx + 1] = val
                    rgb[idx + 2] = val
                elif pattern == "screenshot":
                    # Dark IDE theme with colored code lines
                    if y < 15:
                        # Window titlebar
                        rgb[idx], rgb[idx + 1], rgb[idx + 2] = 45, 45, 50
                    elif x < 20:
                        # Line numbers column
                        rgb[idx], rgb[idx + 1], rgb[idx + 2] = 30, 30, 35
                    else:
                        # Code editor background with syntax highlights
                        is_token = (y % 8 < 3) and (25 < x < width - 15)
                        if is_token:
                            rgb[idx], rgb[idx + 1], rgb[idx + 2] = 80, 200, 120
                        else:
                            rgb[idx], rgb[idx + 1], rgb[idx + 2] = 24, 24, 28
                elif pattern == "id_card":
                    # Card frame with avatar photo square
                    is_photo = (15 < y < height - 15) and (15 < x < width // 3)
                    if is_photo:
                        rgb[idx], rgb[idx + 1], rgb[idx + 2] = 180, 140, 100
                    else:
                        rgb[idx], rgb[idx + 1], rgb[idx + 2] = 230, 235, 240
                else:
                    # Checkerboard
                    val = 255 if ((x // 16) + (y // 16)) % 2 == 0 else 0
                    rgb[idx] = val
                    rgb[idx + 1] = val
                    rgb[idx + 2] = val

        return RawImage(width=width, height=height, channels=3, data=bytes(rgb))


class PerceptualHasher:
    """
    Sub-millisecond perceptual image hashing and structural visual feature extraction.
    Generates difference hashes (dHash) and average hashes (aHash) for visual deduplication.
    """

    @staticmethod
    def dhash(image: Union[RawImage, bytes, str], hash_size: int = 8) -> int:
        """
        Computes 64-bit Difference Hash (dHash).
        Resizes to hash_size * (hash_size + 1), computes horizontal gradient bitmask.
        """
        raw = ZeroDepImageDecoder.decode(image)
        gray = raw.to_grayscale()
        resized = gray.resize(target_w=hash_size + 1, target_h=hash_size)

        diff_bits = 0
        data = resized.data
        w = hash_size + 1

        for y in range(hash_size):
            y_offset = y * w
            for x in range(hash_size):
                left = data[y_offset + x]
                right = data[y_offset + x + 1]
                diff_bits = (diff_bits << 1) | (1 if left > right else 0)

        return diff_bits

    @staticmethod
    def ahash(image: Union[RawImage, bytes, str], hash_size: int = 8) -> int:
        """Computes 64-bit Average Hash (aHash)."""
        raw = ZeroDepImageDecoder.decode(image)
        gray = raw.to_grayscale()
        resized = gray.resize(target_w=hash_size, target_h=hash_size)

        data = resized.data
        avg = sum(data) / len(data)

        bits = 0
        for val in data:
            bits = (bits << 1) | (1 if val >= avg else 0)

        return bits

    @staticmethod
    def dhash_hex(image: Union[RawImage, bytes, str], hash_size: int = 8) -> str:
        """Returns hex-formatted string representation of dHash."""
        val = PerceptualHasher.dhash(image, hash_size=hash_size)
        fmt = f"0{hash_size * hash_size // 4}x"
        return format(val, fmt)

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Computes bit-difference count between two hashes (0 = identical)."""
        diff = hash1 ^ hash2
        if hasattr(diff, "bit_count"):
            return diff.bit_count()
        return bin(diff).count("1")

    @staticmethod
    def extract_features(image: Union[RawImage, bytes, str], dim: int = 128) -> List[float]:
        """
        Extracts structural visual feature vector (spatial grid energy + color variance).
        Produces a normalized vector suitable for System-1 Decision heads.
        """
        raw = ZeroDepImageDecoder.decode(image)
        gray = raw.to_grayscale()

        # 8x8 spatial luminance distribution (64 dimensions)
        g8 = gray.resize(8, 8).data
        feats = [p / 255.0 for p in g8]

        # Horizontal edge gradients (32 dimensions)
        g8x9 = gray.resize(9, 8).data
        for y in range(8):
            for x in range(4):
                d = abs(int(g8x9[y * 9 + x * 2]) - int(g8x9[y * 9 + x * 2 + 1])) / 255.0
                feats.append(d)

        # Aspect ratio & color balance (remaining dimensions to reach dim)
        aspect = min(3.0, raw.width / max(1, raw.height)) / 3.0
        feats.append(aspect)

        if raw.channels >= 3:
            # Sample coarse color means
            rgb16 = raw.resize(4, 4).data
            r_mean = sum(rgb16[i] for i in range(0, len(rgb16), raw.channels)) / (16.0 * 255.0)
            g_mean = sum(rgb16[i] for i in range(1, len(rgb16), raw.channels)) / (16.0 * 255.0)
            b_mean = sum(rgb16[i] for i in range(2, len(rgb16), raw.channels)) / (16.0 * 255.0)
            feats.extend([r_mean, g_mean, b_mean])
        else:
            feats.extend([0.5, 0.5, 0.5])

        # Pad or trim to target dim
        while len(feats) < dim:
            feats.append(0.0)
        feats = feats[:dim]

        # L2 unit normalization
        norm = math.sqrt(sum(x * x for x in feats))
        if norm > 0:
            feats = [x / norm for x in feats]

        return feats


@dataclass
class VisualNoul:
    """Zero-dependency visual boolean decision primitive."""
    instructions: str
    threshold: float = 0.5
    confidence: float = 0.0
    probability: float = 0.5
    is_true: bool = False

    def resolve(self, prob: float) -> VisualNoul:
        conf = abs(prob - 0.5) * 2.0
        return VisualNoul(
            instructions=self.instructions,
            threshold=self.threshold,
            confidence=round(conf, 4),
            probability=round(prob, 4),
            is_true=prob >= self.threshold,
        )


@dataclass
class VisualChoice:
    """Zero-dependency visual multi-class categorization primitive."""
    instructions: str
    options: List[str]
    selected: str = ""
    distribution: Dict[str, float] = None  # type: ignore

    def resolve(self, selected: str, dist: Optional[Dict[str, float]] = None) -> VisualChoice:
        return VisualChoice(
            instructions=self.instructions,
            options=self.options,
            selected=selected,
            distribution=dist or {selected: 1.0},
        )
