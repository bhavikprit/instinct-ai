/**
 * Pure JavaScript System 1 Decision Engine powered by SemanticVectorEncoder.
 * Zero external dependencies: sub-0.05ms execution in Node.js, Browsers, and Edge Workers.
 */

import { SemanticVectorEncoder, cosineSimilarity } from "./encoder.js";
import { Noul, Choice, Score, DecisionResult } from "./primitives.js";

export class PureSemanticEngine {
  constructor({ temperature = 0.25 } = {}) {
    this.name = "semantic-pure";
    this.encoder = new SemanticVectorEncoder();
    this.temperature = Math.max(0.01, temperature);
  }

  evaluate(state, questions) {
    const startTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const stateVec = this.encoder.encode(state);
    const decisions = {};

    for (const [key, q] of Object.entries(questions)) {
      if (q instanceof Noul) {
        decisions[key] = this._evaluateNoul(state, stateVec, q);
      } else if (q instanceof Choice) {
        decisions[key] = this._evaluateChoice(state, stateVec, q);
      } else if (q instanceof Score) {
        decisions[key] = this._evaluateScore(state, stateVec, q);
      } else {
        throw new TypeError(`Unsupported primitive: ${typeof q}`);
      }
    }

    const endTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const elapsedMs = Math.round((endTime - startTime) * 100) / 100;

    return new DecisionResult({
      decisions,
      latencyMs: elapsedMs,
      backend: this.name,
      inputTokens: (state || "").trim().split(/\s+/).filter(Boolean).length,
      outputTokens: 0,
      costUsd: 0.0,
    });
  }

  _evaluateNoul(state, stateVec, noul) {
    const queryVec = this.encoder.encode(noul.instructions);
    const sim = cosineSimilarity(stateVec, queryVec);

    const stateTokens = new Set((state || "").toLowerCase().match(/\w+/g) || []);
    const queryTokens = new Set((noul.instructions || "").toLowerCase().match(/\w+/g) || []);

    // Contrast polarity adjustment
    const negWords = new Set(["not", "never", "safe", "normal", "routine", "false", "ignore"]);
    let hasNeg = false;
    for (const w of negWords) {
      if (stateTokens.has(w)) {
        hasNeg = true;
        break;
      }
    }

    // Alarm / domain boost
    const alarmTokens = new Set([
      "scam", "fraud", "wire", "urgent", "phishing", "attack", "critical", "breach",
      "refund", "stolen", "cancel", "ransomware", "hazard", "threat", "hacked", "emergency"
    ]);
    let alarmOverlap = 0;
    for (const w of alarmTokens) {
      if (stateTokens.has(w)) alarmOverlap++;
    }

    const queryLower = (noul.instructions || "").toLowerCase();
    const queryIsThreat = ["security", "threat", "hazard", "scam", "urgent", "refund"].some((w) =>
      queryLower.includes(w)
    );

    let keywordOverlap = 0;
    for (const w of queryTokens) {
      if (stateTokens.has(w)) keywordOverlap++;
    }

    let effectiveSim = sim + Math.min(0.30, keywordOverlap * 0.08);
    if (queryIsThreat && alarmOverlap > 0) {
      effectiveSim = Math.max(effectiveSim, 0.15 + alarmOverlap * 0.05);
    }

    // Sigmoid probability calibration
    const adjustedSim = effectiveSim - (hasNeg ? 0.15 : 0.0);
    const logit = (adjustedSim - 0.06) / this.temperature;
    const clampedLogit = Math.max(-20.0, Math.min(20.0, logit));
    const prob = 1.0 / (1.0 + Math.exp(-clampedLogit));

    return noul.resolve(prob);
  }

  _evaluateChoice(state, stateVec, choice) {
    if (!choice.options || choice.options.length === 0) {
      return choice.resolve("", {});
    }

    const rawSims = [];
    for (const opt of choice.options) {
      const desc = choice.criteria && choice.criteria[opt] ? choice.criteria[opt] : "";
      const optText = `${choice.instructions} ${opt} ${desc}`.trim();
      const optVec = this.encoder.encode(optText);
      rawSims.push(cosineSimilarity(stateVec, optVec));
    }

    // Softmax over option similarities
    const scaled = rawSims.map((s) => s / this.temperature);
    const maxS = Math.max(...scaled);
    const expS = scaled.map((s) => Math.exp(s - maxS));
    const sumExp = expS.reduce((a, b) => a + b, 0);

    const dist = {};
    let bestOpt = choice.options[0];
    let bestProb = -1;

    for (let i = 0; i < choice.options.length; i++) {
      const opt = choice.options[i];
      const p = Math.round((expS[i] / sumExp) * 10000) / 10000;
      dist[opt] = p;
      if (p > bestProb) {
        bestProb = p;
        bestOpt = opt;
      }
    }

    return choice.resolve(bestOpt, dist);
  }

  _evaluateScore(state, stateVec, score) {
    const queryVec = this.encoder.encode(score.instructions);
    const sim = Math.max(0.0, Math.min(1.0, cosineSimilarity(stateVec, queryVec)));
    const val = score.minVal + sim * (score.maxVal - score.minVal);
    const confidence = Math.round((0.5 + Math.abs(sim - 0.5)) * 1000) / 1000;
    return score.resolve(Math.round(val * 100) / 100, confidence);
  }
}
