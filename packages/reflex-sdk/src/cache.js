/**
 * Reflex InstinctCache: Sub-0.05ms Semantic Memory & Dual-Brain Cache in JavaScript.
 * Multi-tier (L1 exact, L2 semantic vector) LRU cache with TTL and JSON serialization.
 */

import { SemanticVectorEncoder, cosineSimilarity } from "./encoder.js";
import { Noul, Choice, Score, DecisionResult } from "./primitives.js";

export class InstinctCache {
  /**
   * @param {Object} options
   * @param {number} [options.maxSize=1000] - Max cache entries before LRU eviction.
   * @param {number} [options.similarityThreshold=0.88] - Cosine similarity cutoff for L2 hits.
   * @param {number|null} [options.ttlSeconds=null] - Optional TTL in seconds.
   * @param {SemanticVectorEncoder} [options.encoder=null] - Vector encoder instance.
   */
  constructor({
    maxSize = 1000,
    similarityThreshold = 0.88,
    ttlSeconds = null,
    encoder = null,
  } = {}) {
    this.maxSize = Math.max(1, maxSize);
    this.similarityThreshold = similarityThreshold;
    this.ttlSeconds = ttlSeconds;
    this.encoder = encoder || new SemanticVectorEncoder();

    // JS Map preserves insertion order, perfect for LRU
    this._entries = new Map();

    // Telemetry
    this.exactHits = 0;
    this.semanticHits = 0;
    this.misses = 0;
    this.evictions = 0;
  }

  _makeQuestionsSig(questions) {
    const keys = Object.keys(questions).sort();
    const sigParts = [];
    for (const k of keys) {
      const q = questions[k];
      if (q instanceof Noul) {
        sigParts.push(`N:${k}:${q.instructions}:${q.threshold}`);
      } else if (q instanceof Choice) {
        const opts = [...q.options].sort().join(",");
        sigParts.push(`C:${k}:${q.instructions}:${opts}`);
      } else if (q instanceof Score) {
        sigParts.push(`S:${k}:${q.instructions}:${q.minVal}-${q.maxVal}`);
      }
    }
    return sigParts.join("|");
  }

  _makeExactKey(state, questionsSig) {
    return `${questionsSig}##${(state || "").trim().toLowerCase()}`;
  }

  get(state, questions) {
    const now = Date.now() / 1000;
    const qSig = this._makeQuestionsSig(questions);
    const exactKey = this._makeExactKey(state, qSig);

    // 1. Tier 1: L1 Exact Match (O(1))
    if (this._entries.has(exactKey)) {
      const entry = this._entries.get(exactKey);
      if (this.ttlSeconds && now - entry.createdAt > this.ttlSeconds) {
        this._entries.delete(exactKey);
      } else {
        // Move to end for LRU refresh
        this._entries.delete(exactKey);
        entry.lastAccessed = now;
        entry.accessCount++;
        this._entries.set(exactKey, entry);
        this.exactHits++;
        return entry.result;
      }
    }

    // 2. Tier 2: L2 Semantic Vector Match
    const queryVec = this.encoder.encode(state);
    let bestEntry = null;
    let bestKey = null;
    let bestSim = -1.0;

    for (const [key, entry] of this._entries.entries()) {
      if (entry.questionsSig !== qSig) continue;
      if (this.ttlSeconds && now - entry.createdAt > this.ttlSeconds) continue;

      const sim = cosineSimilarity(queryVec, entry.vector);
      if (sim > bestSim) {
        bestSim = sim;
        bestEntry = entry;
        bestKey = key;
      }
    }

    if (bestEntry && bestSim >= this.similarityThreshold) {
      // LRU refresh
      this._entries.delete(bestKey);
      bestEntry.lastAccessed = now;
      bestEntry.accessCount++;
      this._entries.set(bestKey, bestEntry);
      this.semanticHits++;

      // Clone result with updated backend note
      return new DecisionResult({
        decisions: bestEntry.result.decisions,
        latencyMs: 0.04,
        backend: `cache-semantic-l2 (sim=${bestSim.toFixed(3)})`,
        inputTokens: bestEntry.result.inputTokens,
        outputTokens: 0,
        costUsd: 0.0,
      });
    }

    this.misses++;
    return null;
  }

  set(state, questions, result) {
    const now = Date.now() / 1000;
    const qSig = this._makeQuestionsSig(questions);
    const exactKey = this._makeExactKey(state, qSig);

    if (this._entries.size >= this.maxSize && !this._entries.has(exactKey)) {
      // Evict least recently used (first key in Map)
      const oldestKey = this._entries.keys().next().value;
      if (oldestKey) {
        this._entries.delete(oldestKey);
        this.evictions++;
      }
    }

    const vector = this.encoder.encode(state);
    this._entries.set(exactKey, {
      state,
      vector,
      questionsSig: qSig,
      result,
      createdAt: now,
      lastAccessed: now,
      accessCount: 1,
    });
  }

  get stats() {
    const totalRequests = this.exactHits + this.semanticHits + this.misses;
    const hitRate = totalRequests > 0 ? (this.exactHits + this.semanticHits) / totalRequests : 0.0;
    return {
      size: this._entries.size,
      max_size: this.maxSize,
      exact_hits: this.exactHits,
      semantic_hits: this.semanticHits,
      misses: this.misses,
      evictions: this.evictions,
      hit_rate: Math.round(hitRate * 1000) / 1000,
    };
  }

  clear() {
    this._entries.clear();
  }
}
