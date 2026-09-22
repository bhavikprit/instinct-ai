"""
Reflex: Universal System-1 AI Runtime & Dual-Brain Gateway.
"""

from sys1.primitives import Noul, Choice, Score, DecisionResult
from sys1.client import Reflex
from sys1.async_client import AsyncReflex
from sys1.embeddings import SemanticVectorEncoder, PureSemanticEngine
from sys1.models import list_models, download_model, MODEL_CATALOG
from sys1.guardrails import (
    GuardrailResult,
    BaseGuardrail,
    PromptInjectionGuardrail,
    PIIGuardrail,
    GuardrailSuite,
)
from sys1.tool_router import FastToolRouter, ToolDefinition
from sys1.streaming import (
    TokenStreamInterceptor,
    StreamBlockedError,
    StreamingDecisionGate,
)
from sys1.cache import InstinctCache
from sys1.telemetry import OpenTelemetryTracer
from sys1.feedback import FeedbackCollector
from sys1.learning import SelfTuningInstinctHead, OnlineTuner
from sys1.backends.c_engine import NativeCEngine
from sys1.proxy import start_proxy
from sys1.gateway import ReflexGatewayServer, GatewayConfig, GatewayMetrics
from sys1.mesh import InstinctMeshNode, MeshConfig, MeshPeerState
from sys1.vision import (
    VisualNoul,
    VisualChoice,
    PerceptualHasher,
    ZeroDepImageDecoder,
    RawImage,
)
from sys1.flow import (
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
from sys1.shadow import (
    ShadowStage,
    ShadowConfig,
    ShadowEvaluationRecord,
    DivergenceTracker,
    DecisionShadowRouter,
)
from sys1.speculative import (
    SpeculativeEngine,
    SpeculativeAction,
    SpeculativeSession,
    SpeculativeStatus,
    SpeculativeMetrics,
)
from sys1.policy import (
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
from sys1.compiler import (
    PromptSpec,
    CompiledInstinct,
    SyntheticDataGenerator,
    InstinctCompiler,
    CalibrationMetrics,
)
from sys1.ensemble import (
    SpecialistModel,
    MoRGatingNetwork,
    EnsembleResult,
    InstinctEnsemble,
    HierarchicalCascade,
)
from sys1.shm import (
    ReflexIPCDaemon,
    ReflexIPCClient,
    SharedMemoryRingBuffer,
    SHMConfig,
    SlotState,
    IPCOpCode,
)
from sys1.simd import (
    SimdEngine,
    get_simd_engine,
    CPUFeatures,
    detect_cpu_features,
)
from sys1.distill import (
    DistillationTrace,
    DistillationBuffer,
    MinedCluster,
    ClusterMiner,
    IntentSynthesizer,
    DistillationResult,
    AutonomousDistiller,
    DistillationWorker,
)
from sys1.index import (
    HNSWIndex,
    HNSWConfig,
    HNSWNode,
    SearchResult,
)
from sys1.pq import (
    PQConfig,
    ProductQuantizer,
    PQIndex,
    PQSearchResult,
)
from sys1.ivfpq import (
    IVFPQConfig,
    IVFPQIndex,
    IVFPQSearchResult,
)
from sys1.conformal import (
    ConformalConfig,
    ConformalPredictor,
    ConformalNoulResult,
    ConformalChoiceResult,
)
from sys1.crc import (
    CRCConfig,
    ConformalRiskController,
    ScoreRiskBound,
    DecisionRiskBound,
)
from sys1.aci import (
    ACIConfig,
    AdaptiveConformalTracker,
    ACIStatus,
)
from sys1.cqr import (
    CQRConfig,
    CQRInterval,
    QuantileInstinctHead,
    ConformalizedQuantileRegressor,
)
from sys1.calib import (
    CalibConfig,
    CalibrationStatus,
    OnlineProbabilityCalibrator,
)
from sys1.venn_abers import (
    VennAbersConfig,
    VennAbersNoulResult,
    VennAbersChoiceResult,
    VennAbersPredictor,
)
from sys1.reject import (
    SelectiveRejectConfig,
    SelectiveDecision,
    RiskCoveragePoint,
    SelectiveClassifier,
)
from sys1.cascade import (
    CascadeTier,
    CascadeConfig,
    CascadeDecision,
    CascadeFrontierPoint,
    CascadeRouter,
)
from sys1.drift import (
    DriftConfig,
    DriftResult,
    DriftGuard,
    DRIFT_MAGIC,
)
from sys1.kv import (
    KVConfig,
    PrefixTreeNode,
    PrefixTree,
    PromptAligner,
    KVCachePredictor,
    KVCacheEngine,
    KV_MAGIC,
)

# Canonical aliases
Sys1 = Reflex
AsyncSys1 = AsyncReflex
Sys1GatewayServer = ReflexGatewayServer

__version__ = "0.2.0"
__all__ = [
    "Sys1",
    "AsyncSys1",
    "Sys1GatewayServer",
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
    "SimdEngine",
    "get_simd_engine",
    "CPUFeatures",
    "detect_cpu_features",
    "DistillationTrace",
    "DistillationBuffer",
    "MinedCluster",
    "ClusterMiner",
    "IntentSynthesizer",
    "DistillationResult",
    "AutonomousDistiller",
    "DistillationWorker",
    "HNSWIndex",
    "HNSWConfig",
    "HNSWNode",
    "SearchResult",
    "PQConfig",
    "ProductQuantizer",
    "PQIndex",
    "PQSearchResult",
    "IVFPQConfig",
    "IVFPQIndex",
    "IVFPQSearchResult",
    "ConformalConfig",
    "ConformalPredictor",
    "ConformalNoulResult",
    "ConformalChoiceResult",
    "CRCConfig",
    "ConformalRiskController",
    "ScoreRiskBound",
    "DecisionRiskBound",
    "ACIConfig",
    "AdaptiveConformalTracker",
    "ACIStatus",
    "CQRConfig",
    "CQRInterval",
    "QuantileInstinctHead",
    "ConformalizedQuantileRegressor",
    "CalibConfig",
    "CalibrationStatus",
    "OnlineProbabilityCalibrator",
    "VennAbersConfig",
    "VennAbersNoulResult",
    "VennAbersChoiceResult",
    "VennAbersPredictor",
    "SelectiveRejectConfig",
    "SelectiveDecision",
    "RiskCoveragePoint",
    "SelectiveClassifier",
    "CascadeTier",
    "CascadeConfig",
    "CascadeDecision",
    "CascadeFrontierPoint",
    "CascadeRouter",
    "DriftConfig",
    "DriftResult",
    "DriftGuard",
    "DRIFT_MAGIC",
    "KVConfig",
    "PrefixTreeNode",
    "PrefixTree",
    "PromptAligner",
    "KVCachePredictor",
    "KVCacheEngine",
    "KV_MAGIC",
]



