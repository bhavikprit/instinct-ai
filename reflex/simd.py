"""
Reflex Hardware-Accelerated Micro-Embedding SIMD Kernel & Quantization (Phase 28).
Provides sub-microsecond vector operations (ARM NEON & x86_64 AVX2/FMA), INT8 quantization,
4-bit nibble weight packing, and 1-bit binary Hamming distance embeddings.
Zero external dependencies (Python standard library + ctypes FFI with pure-Python fallback).
"""

from __future__ import annotations
import ctypes
from dataclasses import dataclass
import math
import os
import platform
import struct
import subprocess
import sys
from typing import List, Optional, Tuple, Union

from reflex.backends.c_engine import find_libreflex


class ReflexSimdCaps(ctypes.Structure):
    """C struct for CPU hardware capabilities matching reflex_simd.h."""
    _fields_ = [
        ("has_neon", ctypes.c_int),
        ("has_avx2", ctypes.c_int),
        ("has_avx512", ctypes.c_int),
        ("has_fma", ctypes.c_int),
        ("has_popcnt", ctypes.c_int),
        ("arch_name", ctypes.c_char * 32),
    ]


@dataclass
class CPUFeatures:
    """Detected CPU instruction sets and hardware SIMD capabilities."""
    arch: str
    has_neon: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_fma: bool = False
    has_popcnt: bool = False
    native_lib_loaded: bool = False

    def to_dict(self) -> dict:
        return {
            "arch": self.arch,
            "has_neon": self.has_neon,
            "has_avx2": self.has_avx2,
            "has_avx512": self.has_avx512,
            "has_fma": self.has_fma,
            "has_popcnt": self.has_popcnt,
            "native_lib_loaded": self.native_lib_loaded,
        }


def detect_cpu_features() -> CPUFeatures:
    """
    Detects hardware SIMD vector instruction support on macOS, Linux, and Windows.
    """
    machine = platform.machine().lower()
    system = platform.system()

    has_neon = False
    has_avx2 = False
    has_avx512 = False
    has_fma = False
    has_popcnt = False

    if "arm" in machine or "aarch64" in machine:
        has_neon = True
        has_popcnt = True
    elif system == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-a"], stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
            has_neon = "hw.optional.neon: 1" in out or "hw.optional.arm" in out
            has_avx2 = "hw.optional.avx2_0: 1" in out
            has_fma = "hw.optional.fma: 1" in out
            has_popcnt = True
        except Exception:
            if "arm" in machine:
                has_neon = True
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lower()
                has_neon = "neon" in content or "asimd" in content
                has_avx2 = "avx2" in content
                has_avx512 = "avx512" in content
                has_fma = "fma" in content
                has_popcnt = "popcnt" in content
        except Exception:
            pass

    return CPUFeatures(
        arch=machine,
        has_neon=has_neon,
        has_avx2=has_avx2,
        has_avx512=has_avx512,
        has_fma=has_fma,
        has_popcnt=has_popcnt,
    )


class SimdEngine:
    """
    Reflex Hardware-Accelerated SIMD Vector & Quantization Engine.
    Dispatches to libreflex C99 NEON/AVX2 kernels when available,
    with zero-dependency pure-Python fallback.
    """

    def __init__(self, lib_path: Optional[str] = None):
        self.features = detect_cpu_features()
        self._lib = None
        self._load_native_library(lib_path)

    def _load_native_library(self, lib_path: Optional[str] = None) -> None:
        target_path = lib_path or find_libreflex()
        if target_path and os.path.exists(target_path):
            try:
                lib = ctypes.CDLL(target_path)

                # Capability detection
                if hasattr(lib, "reflex_detect_simd_capabilities"):
                    lib.reflex_detect_simd_capabilities.argtypes = [ctypes.POINTER(ReflexSimdCaps)]
                    lib.reflex_detect_simd_capabilities.restype = None

                # FP32 SIMD
                if hasattr(lib, "reflex_dot_product_f32_simd"):
                    lib.reflex_dot_product_f32_simd.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                    ]
                    lib.reflex_dot_product_f32_simd.restype = ctypes.c_float

                if hasattr(lib, "reflex_cosine_similarity_f32_simd"):
                    lib.reflex_cosine_similarity_f32_simd.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                    ]
                    lib.reflex_cosine_similarity_f32_simd.restype = ctypes.c_float

                # INT8 Quantization
                if hasattr(lib, "reflex_quantize_i8"):
                    lib.reflex_quantize_i8.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_int8),
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                    ]
                    lib.reflex_quantize_i8.restype = None

                if hasattr(lib, "reflex_dot_product_i8_simd"):
                    lib.reflex_dot_product_i8_simd.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                        ctypes.c_int,
                    ]
                    lib.reflex_dot_product_i8_simd.restype = ctypes.c_int32

                if hasattr(lib, "reflex_quantized_similarity_i8"):
                    lib.reflex_quantized_similarity_i8.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_float,
                        ctypes.c_char_p,
                        ctypes.c_float,
                        ctypes.c_int,
                    ]
                    lib.reflex_quantized_similarity_i8.restype = ctypes.c_float

                # 1-Bit Binary Sign Quantization
                if hasattr(lib, "reflex_binarize_384"):
                    lib.reflex_binarize_384.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_uint64),
                    ]
                    lib.reflex_binarize_384.restype = None

                if hasattr(lib, "reflex_hamming_distance_384"):
                    lib.reflex_hamming_distance_384.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                    ]
                    lib.reflex_hamming_distance_384.restype = ctypes.c_int

                if hasattr(lib, "reflex_binary_similarity_384"):
                    lib.reflex_binary_similarity_384.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                    ]
                    lib.reflex_binary_similarity_384.restype = ctypes.c_float

                # 4-Bit Nibble Quantization
                if hasattr(lib, "reflex_quantize_i4"):
                    lib.reflex_quantize_i4.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                    ]
                    lib.reflex_quantize_i4.restype = None

                if hasattr(lib, "reflex_dot_product_i4"):
                    lib.reflex_dot_product_i4.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                        ctypes.c_int,
                    ]
                    lib.reflex_dot_product_i4.restype = ctypes.c_int32

                if hasattr(lib, "reflex_quantized_similarity_i4"):
                    lib.reflex_quantized_similarity_i4.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_float,
                        ctypes.c_char_p,
                        ctypes.c_float,
                        ctypes.c_int,
                    ]
                    lib.reflex_quantized_similarity_i4.restype = ctypes.c_float

                self._lib = lib
                self.features.native_lib_loaded = True

                # Query capabilities from C ABI
                if hasattr(lib, "reflex_detect_simd_capabilities"):
                    caps = ReflexSimdCaps()
                    lib.reflex_detect_simd_capabilities(ctypes.byref(caps))
                    self.features.has_neon = bool(caps.has_neon)
                    self.features.has_avx2 = bool(caps.has_avx2)
                    self.features.has_avx512 = bool(caps.has_avx512)
                    self.features.has_fma = bool(caps.has_fma)
                    self.features.has_popcnt = bool(caps.has_popcnt)
            except Exception:
                self._lib = None
                self.features.native_lib_loaded = False

    # -----------------------------------------------------------------
    # 1. FP32 SIMD Operations
    # -----------------------------------------------------------------

    def dot_product_f32(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Vectorized float32 dot product."""
        dim = len(vec_a)
        if dim != len(vec_b):
            raise ValueError(f"Vector dimensions must match ({dim} != {len(vec_b)})")

        if self._lib and hasattr(self._lib, "reflex_dot_product_f32_simd"):
            if hasattr(vec_a, "_type_") and hasattr(vec_b, "_type_"):
                return float(self._lib.reflex_dot_product_f32_simd(vec_a, vec_b, dim))

        # Pure-Python fallback
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Vectorized cosine similarity (assumes unit-normalized vectors)."""
        return self.dot_product_f32(vec_a, vec_b)

    # -----------------------------------------------------------------
    # 2. INT8 Quantization
    # -----------------------------------------------------------------

    def quantize_i8(self, vec: List[float]) -> Tuple[bytes, float]:
        """
        Quantizes FP32 vector to signed 8-bit integers [-127, 127] + float scale factor.
        Returns (bytes, scale).
        """
        dim = len(vec)
        if self._lib and hasattr(self._lib, "reflex_quantize_i8"):
            arr_in = (ctypes.c_float * dim)(*vec)
            arr_out = (ctypes.c_int8 * dim)()
            scale_out = ctypes.c_float(1.0)
            self._lib.reflex_quantize_i8(arr_in, arr_out, dim, ctypes.byref(scale_out))
            return bytes(arr_out), float(scale_out.value)

        # Pure-Python fallback
        max_abs = max((abs(x) for x in vec), default=0.0)
        if max_abs < 1e-9:
            return bytes([0] * dim), 1.0

        scale = max_abs / 127.0
        inv_scale = 127.0 / max_abs
        q_bytes = bytearray(dim)
        for i, x in enumerate(vec):
            v = int(round(x * inv_scale))
            v = max(-127, min(127, v))
            q_bytes[i] = v if v >= 0 else (256 + v)
        return bytes(q_bytes), scale

    def dot_product_i8(
        self,
        q_a: Union[bytes, bytearray],
        scale_a: float,
        q_b: Union[bytes, bytearray],
        scale_b: float,
    ) -> float:
        """Computes reconstructed dot product from two INT8 quantized vectors."""
        dim = len(q_a)
        if dim != len(q_b):
            raise ValueError(f"Quantized vector dimensions must match ({dim} != {len(q_b)})")

        if self._lib and hasattr(self._lib, "reflex_quantized_similarity_i8"):
            bytes_a = q_a if isinstance(q_a, bytes) else bytes(q_a)
            bytes_b = q_b if isinstance(q_b, bytes) else bytes(q_b)
            return float(self._lib.reflex_quantized_similarity_i8(
                bytes_a, ctypes.c_float(scale_a),
                bytes_b, ctypes.c_float(scale_b),
                dim
            ))

        # Pure-Python fallback
        int_a = struct.unpack(f"{dim}b", q_a)
        int_b = struct.unpack(f"{dim}b", q_b)
        raw_dot = sum(a * b for a, b in zip(int_a, int_b))
        return (scale_a * scale_b) * float(raw_dot)

    # -----------------------------------------------------------------
    # 3. 1-Bit Binary Sign Quantization & Hamming Distance
    # -----------------------------------------------------------------

    def binarize_384(self, vec: List[float]) -> bytes:
        """
        Binarizes 384-dimensional float vector into 384 bits (48 bytes).
        Bit i = 1 if vec[i] >= 0, else 0.
        """
        if len(vec) != 384:
            raise ValueError(f"Expected 384 dimensions for binarize_384, got {len(vec)}")

        if self._lib and hasattr(self._lib, "reflex_binarize_384"):
            arr_in = (ctypes.c_float * 384)(*vec)
            arr_out = (ctypes.c_uint64 * 6)()
            self._lib.reflex_binarize_384(arr_in, arr_out)
            return bytes(arr_out)

        # Pure-Python fallback
        words = []
        for w in range(6):
            word = 0
            base = w * 64
            for b in range(64):
                if vec[base + b] >= 0.0:
                    word |= (1 << b)
            words.append(word)
        return struct.pack("<6Q", *words)

    def hamming_distance_384(self, bits_a: bytes, bits_b: bytes) -> int:
        """Computes Hamming distance (number of bit flips) between two 48-byte binary vectors."""
        if len(bits_a) != 48 or len(bits_b) != 48:
            raise ValueError(f"Expected 48-byte vectors for hamming_distance_384")

        if self._lib and hasattr(self._lib, "reflex_hamming_distance_384"):
            bytes_a = bits_a if isinstance(bits_a, bytes) else bytes(bits_a)
            bytes_b = bits_b if isinstance(bits_b, bytes) else bytes(bits_b)
            return int(self._lib.reflex_hamming_distance_384(bytes_a, bytes_b))

        # Pure-Python fallback
        words_a = struct.unpack("<6Q", bits_a)
        words_b = struct.unpack("<6Q", bits_b)
        dist = 0
        for wa, wb in zip(words_a, words_b):
            dist += (wa ^ wb).bit_count()
        return dist

    def binary_similarity_384(self, bits_a: bytes, bits_b: bytes) -> float:
        """Approximates cosine similarity from 1-bit Hamming distance: cos(pi * dist / 384)."""
        if self._lib and hasattr(self._lib, "reflex_binary_similarity_384"):
            bytes_a = bits_a if isinstance(bits_a, bytes) else bytes(bits_a)
            bytes_b = bits_b if isinstance(bits_b, bytes) else bytes(bits_b)
            return float(self._lib.reflex_binary_similarity_384(bytes_a, bytes_b))
        dist = self.hamming_distance_384(bits_a, bits_b)
        norm_dist = dist / 384.0
        return math.cos(math.pi * norm_dist)

    # -----------------------------------------------------------------
    # 4. 4-Bit Nibble Quantization
    # -----------------------------------------------------------------

    def quantize_i4(self, vec: List[float]) -> Tuple[bytes, float]:
        """
        Quantizes FP32 vector into packed 4-bit integers [-8, 7].
        384 dimensions compress to 192 bytes (2 values per byte).
        Returns (packed_bytes, scale).
        """
        dim = len(vec)
        out_len = (dim + 1) // 2

        if self._lib and hasattr(self._lib, "reflex_quantize_i4"):
            arr_in = (ctypes.c_float * dim)(*vec)
            arr_out = (ctypes.c_uint8 * out_len)()
            scale_out = ctypes.c_float(1.0)
            self._lib.reflex_quantize_i4(arr_in, arr_out, dim, ctypes.byref(scale_out))
            return bytes(arr_out), float(scale_out.value)

        # Pure-Python fallback
        max_abs = max((abs(x) for x in vec), default=0.0)
        if max_abs < 1e-9:
            return bytes([0] * out_len), 1.0

        scale = max_abs / 7.0
        inv_scale = 7.0 / max_abs
        packed = bytearray(out_len)
        for i in range(0, dim, 2):
            v0 = int(round(vec[i] * inv_scale))
            v0 = max(-8, min(7, v0)) & 0x0F
            v1 = 0
            if i + 1 < dim:
                v1 = int(round(vec[i + 1] * inv_scale))
                v1 = max(-8, min(7, v1)) & 0x0F
            packed[i // 2] = (v1 << 4) | v0
        return bytes(packed), scale

    def dot_product_i4(
        self,
        packed_w: bytes,
        scale_w: float,
        q_vec: bytes,
        scale_q: float,
    ) -> float:
        """Computes dot product between packed 4-bit weights and an INT8 quantized vector."""
        dim = len(q_vec)
        if self._lib and hasattr(self._lib, "reflex_quantized_similarity_i4"):
            bytes_w = packed_w if isinstance(packed_w, bytes) else bytes(packed_w)
            bytes_q = q_vec if isinstance(q_vec, bytes) else bytes(q_vec)
            return float(self._lib.reflex_quantized_similarity_i4(
                bytes_w, ctypes.c_float(scale_w),
                bytes_q, ctypes.c_float(scale_q),
                dim
            ))

        # Pure-Python fallback
        int_q = struct.unpack(f"{dim}b", q_vec)
        raw_total = 0
        for i in range(0, dim, 2):
            byte = packed_w[i // 2]
            # Sign extend 4-bit values
            nib0 = byte & 0x0F
            w0 = nib0 if nib0 < 8 else nib0 - 16
            nib1 = (byte >> 4) & 0x0F
            w1 = nib1 if nib1 < 8 else nib1 - 16

            raw_total += w0 * int_q[i]
            if i + 1 < dim:
                raw_total += w1 * int_q[i + 1]

        return (scale_w * scale_q) * float(raw_total)


# Global singleton instance
_GLOBAL_SIMD_ENGINE: Optional[SimdEngine] = None


def get_simd_engine() -> SimdEngine:
    """Returns global cached SimdEngine instance."""
    global _GLOBAL_SIMD_ENGINE
    if _GLOBAL_SIMD_ENGINE is None:
        _GLOBAL_SIMD_ENGINE = SimdEngine()
    return _GLOBAL_SIMD_ENGINE
