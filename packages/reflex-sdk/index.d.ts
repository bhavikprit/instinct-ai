export interface NoulOptions {
  instructions: string;
  probability?: number | null;
  threshold?: number;
  uncertaintyLow?: number;
  uncertaintyHigh?: number;
}

export declare class Noul {
  instructions: string;
  probability: number | null;
  threshold: number;
  uncertaintyLow: number;
  uncertaintyHigh: number;
  constructor(options: NoulOptions);
  resolve(prob: number): Noul;
  get isTrue(): boolean;
  get isFalse(): boolean;
  get isUncertain(): boolean;
  toJSON(): Record<string, any>;
}

export interface ChoiceOptions {
  instructions: string;
  options?: string[];
  criteria?: Record<string, string> | null;
  selected?: string | null;
  distribution?: Record<string, number>;
}

export declare class Choice {
  instructions: string;
  options: string[];
  criteria: Record<string, string> | null;
  selected: string | null;
  distribution: Record<string, number>;
  constructor(options: ChoiceOptions);
  resolve(selected: string, distribution?: Record<string, number> | null): Choice;
  getProb(option: string): number;
  toJSON(): Record<string, any>;
}

export interface ScoreOptions {
  instructions: string;
  minVal?: number;
  maxVal?: number;
  score?: number | null;
  confidence?: number | null;
}

export declare class Score {
  instructions: string;
  minVal: number;
  maxVal: number;
  score: number | null;
  confidence: number | null;
  constructor(options: ScoreOptions);
  resolve(score: number, confidence?: number | null): Score;
  toJSON(): Record<string, any>;
}

export type PrimitiveType = Noul | Choice | Score;

export declare class DecisionResult {
  decisions: Record<string, PrimitiveType>;
  latencyMs: number;
  backend: string;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  constructor(options?: {
    decisions?: Record<string, PrimitiveType>;
    latencyMs?: number;
    backend?: string;
    inputTokens?: number;
    outputTokens?: number;
    costUsd?: number;
  });
  get isUncertain(): boolean;
  toJSON(): Record<string, any>;
}

export declare class SemanticVectorEncoder {
  DIM: number;
  encode(text: string): number[];
}

export declare function cosineSimilarity(vecA: number[], vecB: number[]): number;
export declare function md5(string: string): string;

export declare class PureSemanticEngine {
  name: string;
  encoder: SemanticVectorEncoder;
  temperature: number;
  constructor(options?: { temperature?: number });
  evaluate(state: string, questions: Record<string, PrimitiveType>): DecisionResult;
}

export declare class InstinctCache {
  maxSize: number;
  similarityThreshold: number;
  ttlSeconds: number | null;
  exactHits: number;
  semanticHits: number;
  misses: number;
  evictions: number;
  constructor(options?: {
    maxSize?: number;
    similarityThreshold?: number;
    ttlSeconds?: number | null;
    encoder?: SemanticVectorEncoder | null;
  });
  get(state: string, questions: Record<string, PrimitiveType>): DecisionResult | null;
  set(state: string, questions: Record<string, PrimitiveType>, result: DecisionResult): void;
  get stats(): Record<string, any>;
  clear(): void;
}

export declare class GuardrailResult {
  isSafe: boolean;
  blocked: boolean;
  riskScore: number;
  category: string | null;
  reason: string | null;
  detectedEntities: string[];
  latencyMs: number;
  constructor(options?: {
    isSafe?: boolean;
    blocked?: boolean;
    riskScore?: number;
    category?: string | null;
    reason?: string | null;
    detectedEntities?: string[];
    latencyMs?: number;
  });
  toJSON(): Record<string, any>;
}

export declare class PromptInjectionGuardrail {
  name: string;
  threshold: number;
  constructor(options?: { threshold?: number });
  check(text: string): GuardrailResult;
}

export declare class PIIGuardrail {
  name: string;
  check(text: string): GuardrailResult;
}

export declare class GuardrailSuite {
  failFast: boolean;
  constructor(options?: {
    guardrails?: Array<{ check(text: string): GuardrailResult }> | null;
    failFast?: boolean;
  });
  check(text: string): GuardrailResult;
}

export interface ReflexOptions {
  backend?: string;
  temperature?: number;
  cache?: boolean | InstinctCache;
  guardrails?: boolean | GuardrailSuite;
  baseUrl?: string | null;
  learning?: boolean;
  compiledInstinct?: CompiledInstinct | null;
}

export declare function crc32(bytes: Uint8Array): number;

export interface CompiledInstinctOptions {
  name: string;
  decisionType: "choice" | "noul" | "score";
  options?: string[];
  weights?: Record<string, number[]>;
  biases?: Record<string, number>;
  temperature?: number;
  metrics?: Record<string, any> | null;
  spec?: Record<string, any> | null;
  compiledAt?: string | null;
  version?: string;
}

export declare class CompiledInstinct {
  name: string;
  decisionType: "choice" | "noul" | "score";
  options: string[];
  weights: Record<string, number[]>;
  biases: Record<string, number>;
  temperature: number;
  metrics: Record<string, any>;
  spec: Record<string, any> | null;
  compiledAt: string | null;
  version: string;

  constructor(options: CompiledInstinctOptions);
  static fromBinary(buffer: Uint8Array | ArrayBuffer | Buffer): CompiledInstinct;
  static fromFile(filePath: string): Promise<CompiledInstinct>;
  static fromFileSync(filePath: string): CompiledInstinct;
  predict(state: string): DecisionResult;
}

export declare class Reflex {
  backendName: string;
  temperature: number;
  baseUrl: string | null;
  engine: PureSemanticEngine;
  cache: InstinctCache | null;
  guardrails: GuardrailSuite | null;
  learning: boolean;
  compiledInstinct: CompiledInstinct | null;

  constructor(options?: ReflexOptions);

  evaluate(state: string, questions: Record<string, PrimitiveType>): Promise<DecisionResult>;
  predict(state: string): DecisionResult;
  loadCompiledModel(bufferOrPath: Uint8Array | ArrayBuffer | string): Promise<CompiledInstinct>;
  noul(instructions: string, state: string, threshold?: number): Promise<number>;
  choice(instructions: string, options: string[], state: string, criteria?: Record<string, string> | null): Promise<string>;
  score(instructions: string, state: string, minVal?: number, maxVal?: number): Promise<number>;
  guardrail(text: string): GuardrailResult;
  teach(
    state: string,
    questionKey: string,
    groundTruth: boolean | string | number,
    options?: { lr?: number; candidateOptions?: string[] | null }
  ): number;
}

export { Reflex as Instinct, Reflex as Sys1 };
export default Reflex;
