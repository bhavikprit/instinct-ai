/**
 * @reflex-ai/sdk: Universal System-1 AI Runtime & Dual-Brain Gateway
 * Isomorphic, zero-dependency library for Cloudflare Workers, Edge, Node.js, and Browsers.
 */

export { Noul, Choice, Score, DecisionResult } from "./src/primitives.js";
export { SemanticVectorEncoder, cosineSimilarity, md5 } from "./src/encoder.js";
export { PureSemanticEngine } from "./src/engine.js";
export { InstinctCache } from "./src/cache.js";
export {
  GuardrailResult,
  PromptInjectionGuardrail,
  PIIGuardrail,
  GuardrailSuite,
} from "./src/guardrails.js";
export { CompiledInstinct, crc32 } from "./src/compiler.js";
export { Reflex, Reflex as Sys1 } from "./src/client.js";

// Default export
import { Reflex } from "./src/client.js";
export default Reflex;
