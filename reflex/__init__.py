"""
Reflex: Universal System-1 AI Runtime & Dual-Brain Gateway.
"""

from reflex.primitives import Noul, Choice, Score, DecisionResult
from reflex.client import Reflex
from reflex.async_client import AsyncReflex
from reflex.embeddings import SemanticVectorEncoder, PureSemanticEngine
from reflex.models import list_models, download_model, MODEL_CATALOG
from reflex.guardrails import (
    GuardrailResult,
    BaseGuardrail,
    PromptInjectionGuardrail,
    PIIGuardrail,
    GuardrailSuite,
)
from reflex.tool_router import FastToolRouter, ToolDefinition
from reflex.streaming import (
    TokenStreamInterceptor,
    StreamBlockedError,
    StreamingDecisionGate,
)
from reflex.cache import InstinctCache
from reflex.telemetry import OpenTelemetryTracer
from reflex.proxy import start_proxy

__version__ = "0.2.0"
__all__ = [
    "Reflex",
    "AsyncReflex",
    "Noul",
    "Choice",
    "Score",
    "DecisionResult",
    "SemanticVectorEncoder",
    "PureSemanticEngine",
    "InstinctCache",
    "OpenTelemetryTracer",
    "list_models",
    "download_model",
    "MODEL_CATALOG",
    "GuardrailResult",
    "BaseGuardrail",
    "PromptInjectionGuardrail",
    "PIIGuardrail",
    "GuardrailSuite",
    "FastToolRouter",
    "ToolDefinition",
    "TokenStreamInterceptor",
    "StreamBlockedError",
    "StreamingDecisionGate",
    "start_proxy",
]
