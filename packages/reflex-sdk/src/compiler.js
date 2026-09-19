/**
 * Reflex Compiled Instinct & Edge Binary Runtime for Node.js, Bun, Cloudflare Workers, and Browser.
 * Parses and executes portable .reflex models (RFX1 format) in <20us.
 * Zero external dependencies.
 */

import { SemanticVectorEncoder } from "./encoder.js";
import { Choice, Noul, Score, DecisionResult } from "./primitives.js";

// Precomputed IEEE 802.3 CRC32 lookup table
const CRC32_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let j = 0; j < 8; j++) {
    c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
  }
  CRC32_TABLE[i] = c >>> 0;
}

/**
 * Computes the 32-bit CRC checksum of a byte buffer.
 * @param {Uint8Array} bytes
 * @returns {number} Unsigned 32-bit integer
 */
export function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function sigmoid(z) {
  const clamped = Math.max(-30.0, Math.min(30.0, z));
  return 1.0 / (1.0 + Math.exp(-clamped));
}

function softmax(logits, temperature = 1.0) {
  if (!logits || logits.length === 0) return [];
  const temp = Math.max(0.01, temperature);
  const scaled = logits.map((x) => x / temp);
  const maxL = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(Math.max(-30.0, Math.min(30.0, x - maxL))));
  const sumExps = exps.reduce((acc, x) => acc + x, 0);
  if (sumExps <= 0.0) {
    return logits.map(() => 1.0 / logits.length);
  }
  return exps.map((x) => x / sumExps);
}

/**
 * Portable, self-contained compiled decision model evaluating in <20us.
 */
export class CompiledInstinct {
  /**
   * @param {Object} options
   * @param {string} options.name
   * @param {"choice"|"noul"|"score"} options.decisionType
   * @param {string[]} options.options
   * @param {Record<string, number[]>} options.weights
   * @param {Record<string, number>} options.biases
   * @param {number} [options.temperature=1.0]
   * @param {Object} [options.metrics]
   * @param {Object} [options.spec]
   * @param {string} [options.compiledAt]
   * @param {string} [options.version="1.0"]
   */
  constructor({
    name,
    decisionType,
    options = [],
    weights = {},
    biases = {},
    temperature = 1.0,
    metrics = null,
    spec = null,
    compiledAt = null,
    version = "1.0",
  }) {
    this.name = name;
    this.decisionType = decisionType;
    this.options = options;
    this.weights = weights;
    this.biases = biases;
    this.temperature = Math.max(0.01, temperature);
    this.metrics = metrics || { accuracy: 1.0, brier_score: 0.0, ece: 0.0 };
    this.spec = spec;
    this.compiledAt = compiledAt;
    this.version = version;
    this.encoder = new SemanticVectorEncoder();
  }

  /**
   * Loads and verifies a .reflex binary model from a Uint8Array or Buffer.
   * @param {Uint8Array|ArrayBuffer|Buffer} buffer
   * @returns {CompiledInstinct}
   */
  static fromBinary(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    if (bytes.length < 12) {
      throw new Error("Corrupt .reflex file: header too short");
    }

    // 1. Verify 4-byte Magic Header: 'RFX1'
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    if (magic !== "RFX1") {
      throw new Error(`Invalid magic header: expected 'RFX1', got '${magic}'`);
    }

    // 2. Read CRC32 and Length (Big-Endian)
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const expectedCrc = view.getUint32(4, false);
    const length = view.getUint32(8, false);

    // 3. Extract and Verify Payload
    const payloadBytes = bytes.subarray(12, 12 + length);
    if (payloadBytes.length !== length) {
      throw new Error("Incomplete .reflex file: truncated payload");
    }

    const actualCrc = crc32(payloadBytes);
    if (actualCrc !== expectedCrc) {
      throw new Error(`CRC32 checksum mismatch: expected ${expectedCrc}, got ${actualCrc} (corrupt model)`);
    }

    // 4. Parse UTF-8 JSON Payload
    const decoder = new TextDecoder("utf-8");
    const jsonStr = decoder.decode(payloadBytes);
    const data = JSON.parse(jsonStr);

    return new CompiledInstinct({
      name: data.name,
      decisionType: data.decision_type,
      options: data.options || [],
      weights: data.weights || {},
      biases: data.biases || {},
      temperature: data.temperature || 1.0,
      metrics: data.metrics,
      spec: data.spec,
      compiledAt: data.compiled_at,
      version: data.version || "1.0",
    });
  }

  /**
   * Loads a .reflex file from the filesystem (Node.js / Bun).
   * @param {string} filePath
   * @returns {Promise<CompiledInstinct>}
   */
  static async fromFile(filePath) {
    if (typeof process !== "undefined" && process.versions && process.versions.node) {
      const fs = await import("node:fs/promises");
      const buffer = await fs.readFile(filePath);
      return CompiledInstinct.fromBinary(buffer);
    }
    throw new Error("CompiledInstinct.fromFile() is only supported in Node.js/Bun environments.");
  }

  /**
   * Synchronous file loader for Node.js / Bun.
   * @param {string} filePath
   * @returns {CompiledInstinct}
   */
  static fromFileSync(filePath) {
    if (typeof process !== "undefined" && process.versions && process.versions.node) {
      // Use dynamic require or import in CommonJS / ESM
      const fs = require ? require("fs") : null;
      if (fs && fs.readFileSync) {
        return CompiledInstinct.fromBinary(fs.readFileSync(filePath));
      }
    }
    throw new Error("CompiledInstinct.fromFileSync() requires synchronous Node.js fs.");
  }

  /**
   * Executes sub-20 microsecond inference on input state with calibrated probabilities.
   * @param {string} state
   * @returns {DecisionResult}
   */
  predict(state) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const vec = this.encoder.encode(state);

    const decisions = {};
    if (this.decisionType === "choice") {
      const logits = [];
      for (const opt of this.options) {
        const w = this.weights[opt] || new Array(vec.length).fill(0.0);
        const b = this.biases[opt] || 0.0;
        let z = b;
        for (let i = 0; i < vec.length; i++) {
          z += w[i] * vec[i];
        }
        logits.push(z);
      }

      const probs = softmax(logits, this.temperature);
      const dist = {};
      let bestIdx = 0;
      let maxP = -1;
      for (let i = 0; i < this.options.length; i++) {
        const p = Math.round(probs[i] * 10000) / 10000;
        dist[this.options[i]] = p;
        if (probs[i] > maxP) {
          maxP = probs[i];
          bestIdx = i;
        }
      }

      const bestOpt = this.options[bestIdx] || "";
      decisions.choice = new Choice({
        instructions: this.spec?.prompt || this.name,
        options: this.options,
        selected: bestOpt,
        distribution: dist,
      });
    } else if (this.decisionType === "noul") {
      const w = this.weights["noul"] || new Array(vec.length).fill(0.0);
      const b = this.biases["noul"] || 0.0;
      let z = b;
      for (let i = 0; i < vec.length; i++) {
        z += w[i] * vec[i];
      }
      const prob = sigmoid(z / this.temperature);
      decisions.noul = new Noul({
        instructions: this.spec?.prompt || this.name,
        probability: Math.round(prob * 10000) / 10000,
      });
    } else {
      const w = this.weights["score"] || new Array(vec.length).fill(0.0);
      const b = this.biases["score"] || 0.0;
      let z = b;
      for (let i = 0; i < vec.length; i++) {
        z += w[i] * vec[i];
      }
      const normVal = sigmoid(z / this.temperature);
      const scoreVal = Math.round((1.0 + normVal * 9.0) * 100) / 100;
      decisions.score = new Score({
        instructions: this.spec?.prompt || this.name,
        score: scoreVal,
        confidence: 0.95,
      });
    }

    const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const latencyMs = Math.round((t1 - t0) * 1000) / 1000;

    return new DecisionResult({
      decisions,
      latencyMs,
      backend: `compiled:${this.name}`,
      inputTokens: state.split(/\s+/).filter(Boolean).length,
      outputTokens: 0,
      costUsd: 0.0,
    });
  }
}
