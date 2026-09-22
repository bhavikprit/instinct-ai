"""
Native C Engine Backend for Reflex using ctypes FFI.
Provides sub-10 microsecond System-1 evaluation backed by libreflex.
"""

from __future__ import annotations
import ctypes
import os
import platform
import sys
import time
from typing import Dict, List, Optional

from sys1.backends.base import BaseBackend
from sys1.primitives import (
    PrimitiveType,
    Noul,
    Choice,
    Score,
    DecisionResult,
)


# C ABI Structures matching reflex.h
class ReflexNoulResult(ctypes.Structure):
    _fields_ = [
        ("probability", ctypes.c_float),
        ("is_true", ctypes.c_int),
        ("is_false", ctypes.c_int),
        ("is_uncertain", ctypes.c_int),
    ]


class ReflexChoiceResult(ctypes.Structure):
    _fields_ = [
        ("selected", ctypes.c_char * 128),
        ("confidence", ctypes.c_float),
        ("num_options", ctypes.c_int),
        ("option_names", (ctypes.c_char * 128) * 32),
        ("distribution", ctypes.c_float * 32),
    ]


class ReflexScoreResult(ctypes.Structure):
    _fields_ = [
        ("score", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]


class ReflexGuardrailResult(ctypes.Structure):
    _fields_ = [
        ("is_safe", ctypes.c_int),
        ("blocked", ctypes.c_int),
        ("risk_score", ctypes.c_float),
        ("category", ctypes.c_char * 128),
        ("reason", ctypes.c_char * 256),
        ("latency_us", ctypes.c_float),
    ]


def find_libreflex() -> Optional[str]:
    """Finds libreflex shared library in workspace or system paths."""
    system = platform.system()
    if system == "Darwin":
        lib_name = "libreflex.dylib"
    elif system == "Windows":
        lib_name = "reflex.dll"
    else:
        lib_name = "libreflex.so"

    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "reflex_c", "build", lib_name),
        os.path.join(os.getcwd(), "reflex_c", "build", lib_name),
        os.path.join(os.getcwd(), lib_name),
        f"/usr/local/lib/{lib_name}",
        f"/usr/lib/{lib_name}",
    ]

    for p in candidates:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p
    return None


class NativeCEngine(BaseBackend):
    """
    High-Performance Native C99 Decision Engine.
    Executes in sub-10 microseconds (<0.01ms) via C ABI FFI.
    """

    def __init__(self, lib_path: Optional[str] = None, temperature: float = 0.25):
        self.name = "native-c99"
        self.temperature = max(0.01, temperature)
        resolved_path = lib_path or find_libreflex()

        if not resolved_path or not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"libreflex shared library not found. Run 'make -C reflex_c' to build it."
            )

        self.lib_path = resolved_path
        self._lib = ctypes.CDLL(self.lib_path)
        self._setup_ffi()

    def _setup_ffi(self):
        # reflex_encode_384(const char* text, float* out_vec)
        self._lib.reflex_encode_384.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.reflex_encode_384.restype = None

        # reflex_cosine_similarity(const float* a, const float* b, int dim)
        self._lib.reflex_cosine_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._lib.reflex_cosine_similarity.restype = ctypes.c_float

        # reflex_evaluate_noul
        self._lib.reflex_evaluate_noul.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ReflexNoulResult),
        ]
        self._lib.reflex_evaluate_noul.restype = None

        # reflex_evaluate_choice
        self._lib.reflex_evaluate_choice.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ReflexChoiceResult),
        ]
        self._lib.reflex_evaluate_choice.restype = None

        # reflex_evaluate_score
        self._lib.reflex_evaluate_score.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ReflexScoreResult),
        ]
        self._lib.reflex_evaluate_score.restype = None

        # reflex_guardrail_check
        self._lib.reflex_guardrail_check.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ReflexGuardrailResult),
        ]
        self._lib.reflex_guardrail_check.restype = None

    def encode(self, text: str) -> List[float]:
        """Encodes text into a 384-dimensional vector via C ABI."""
        out = (ctypes.c_float * 384)()
        self._lib.reflex_encode_384(text.encode("utf-8"), out)
        return [float(x) for x in out]

    def evaluate(self, state: str, questions: Dict[str, PrimitiveType]) -> DecisionResult:
        start_time = time.perf_counter()
        decisions: Dict[str, PrimitiveType] = {}
        state_bytes = state.encode("utf-8")

        for key, q in questions.items():
            if isinstance(q, Noul):
                noul_out = ReflexNoulResult()
                self._lib.reflex_evaluate_noul(
                    state_bytes,
                    q.instructions.encode("utf-8"),
                    ctypes.c_float(q.threshold),
                    ctypes.c_float(self.temperature),
                    ctypes.byref(noul_out),
                )
                decisions[key] = q.resolve(float(noul_out.probability))

            elif isinstance(q, Choice):
                choice_out = ReflexChoiceResult()
                opts_bytes = [opt.encode("utf-8") for opt in q.options]
                c_opts = (ctypes.c_char_p * len(opts_bytes))(*opts_bytes)
                self._lib.reflex_evaluate_choice(
                    state_bytes,
                    q.instructions.encode("utf-8"),
                    c_opts,
                    ctypes.c_int(len(opts_bytes)),
                    ctypes.c_float(self.temperature),
                    ctypes.byref(choice_out),
                )
                dist = {}
                for i in range(choice_out.num_options):
                    opt_name = choice_out.option_names[i].value.decode("utf-8")
                    dist[opt_name] = round(float(choice_out.distribution[i]), 4)
                decisions[key] = q.resolve(
                    selected=choice_out.selected.decode("utf-8"),
                    distribution=dist,
                )

            elif isinstance(q, Score):
                score_out = ReflexScoreResult()
                self._lib.reflex_evaluate_score(
                    state_bytes,
                    q.instructions.encode("utf-8"),
                    ctypes.c_float(q.min_val),
                    ctypes.c_float(q.max_val),
                    ctypes.byref(score_out),
                )
                decisions[key] = q.resolve(
                    score=round(float(score_out.score), 2),
                    confidence=round(float(score_out.confidence), 3),
                )
            else:
                raise TypeError(f"Unsupported primitive: {type(q)}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DecisionResult(
            decisions=decisions,
            latency_ms=round(elapsed_ms, 3),
            backend=self.name,
            input_tokens=len(state.split()),
            output_tokens=0,
            cost_usd=0.0,
        )

    def guardrail_check(self, text: str) -> Dict[str, any]:
        """Runs native C sub-microsecond guardrail check."""
        out = ReflexGuardrailResult()
        self._lib.reflex_guardrail_check(text.encode("utf-8"), ctypes.byref(out))
        return {
            "is_safe": bool(out.is_safe),
            "blocked": bool(out.blocked),
            "risk_score": round(float(out.risk_score), 4),
            "category": out.category.decode("utf-8"),
            "reason": out.reason.decode("utf-8"),
            "latency_us": round(float(out.latency_us), 2),
        }
