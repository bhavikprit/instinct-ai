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
from reflex.feedback import FeedbackCollector
from reflex.learning import SelfTuningInstinctHead, OnlineTuner
from reflex.backends.c_engine import NativeCEngine
from reflex.proxy import start_proxy
from reflex.gateway import ReflexGatewayServer, GatewayConfig, GatewayMetrics
from reflex.mesh import InstinctMeshNode, MeshConfig, MeshPeerState
from reflex.vision import (
    VisualNoul,
    VisualChoice,
    PerceptualHasher,
    ZeroDepImageDecoder,
    RawImage,
)
from reflex.flow import (
    StateGraph,
    Flow,
    START,
    END,
    FlowStep,
    FlowResult,
    FlowError,
    MaxStepsExceededError,
    InvalidTransitionError,
)
from reflex.shadow import (
    ShadowStage,
    ShadowConfig,
    ShadowEvaluationRecord,
    DivergenceTracker,
    DecisionShadowRouter,
)
from reflex.speculative import (
    SpeculativeEngine,
    SpeculativeAction,
    SpeculativeSession,
    SpeculativeStatus,
    SpeculativeMetrics,
)
from reflex.policy import (
    PolicyAction,
    PolicyViolationError,
    PolicyRule,
    PolicyRuleSet,
    PolicyVerdict,
    PolicyEngine,
    AuditEntry,
    MerkleTree,
    MerkleAuditLog,
)
from reflex.compiler import (
    PromptSpec,
    CompiledInstinct,
    SyntheticDataGenerator,
    InstinctCompiler,
    CalibrationMetrics,
)
from reflex.ensemble import (
    SpecialistModel,
    MoRGatingNetwork,
    EnsembleResult,
    InstinctEnsemble,
    HierarchicalCascade,
)
from reflex.shm import (
    ReflexIPCDaemon,
    ReflexIPCClient,
    SharedMemoryRingBuffer,
    SHMConfig,
    SlotState,
    IPCOpCode,
)

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
    "FeedbackCollector",
    "SelfTuningInstinctHead",
    "OnlineTuner",
    "NativeCEngine",
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
    "ReflexGatewayServer",
    "GatewayConfig",
    "GatewayMetrics",
    "InstinctMeshNode",
    "MeshConfig",
    "MeshPeerState",
    "VisualNoul",
    "VisualChoice",
    "PerceptualHasher",
    "ZeroDepImageDecoder",
    "RawImage",
    "StateGraph",
    "Flow",
    "START",
    "END",
    "FlowStep",
    "FlowResult",
    "FlowError",
    "MaxStepsExceededError",
    "InvalidTransitionError",
    "ShadowStage",
    "ShadowConfig",
    "ShadowEvaluationRecord",
    "DivergenceTracker",
    "DecisionShadowRouter",
    "SpeculativeEngine",
    "SpeculativeAction",
    "SpeculativeSession",
    "SpeculativeStatus",
    "SpeculativeMetrics",
    "PolicyAction",
    "PolicyViolationError",
    "PolicyRule",
    "PolicyRuleSet",
    "PolicyVerdict",
    "PolicyEngine",
    "AuditEntry",
    "MerkleTree",
    "MerkleAuditLog",
    "PromptSpec",
    "CompiledInstinct",
    "SyntheticDataGenerator",
    "InstinctCompiler",
    "CalibrationMetrics",
    "SpecialistModel",
    "MoRGatingNetwork",
    "EnsembleResult",
    "InstinctEnsemble",
    "HierarchicalCascade",
    "ReflexIPCDaemon",
    "ReflexIPCClient",
    "SharedMemoryRingBuffer",
    "SHMConfig",
    "SlotState",
    "IPCOpCode",
]



