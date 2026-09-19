/**
 * Reflex Client for Edge, Browser, and Node.js.
 * Universal System-1 AI Runtime & Dual-Brain Decision Gateway.
 */

import { Noul, Choice, Score, DecisionResult } from "./primitives.js";
import { PureSemanticEngine } from "./engine.js";
import { SemanticVectorEncoder, cosineSimilarity } from "./encoder.js";
import { InstinctCache } from "./cache.js";
import { GuardrailSuite } from "./guardrails.js";
import { CompiledInstinct } from "./compiler.js";

export class Reflex {
  /**
   * @param {Object} [options]
   * @param {string} [options.backend="semantic"] - "semantic" or remote URL.
   * @param {number} [options.temperature=0.25] - Softmax / sigmoid calibration temperature.
   * @param {boolean|InstinctCache} [options.cache=false] - Cache toggle or instance.
   * @param {boolean|GuardrailSuite} [options.guardrails=false] - Guardrail suite.
   * @param {string|null} [options.baseUrl=null] - Optional URL to Reflex REST gateway (e.g., http://localhost:8000).
   * @param {boolean} [options.learning=false] - Enable online active learning instinct head.
   * @param {CompiledInstinct|null} [options.compiledInstinct=null] - Pre-compiled .reflex decision model.
   */
  constructor({
    backend = "semantic",
    temperature = 0.25,
    cache = false,
    guardrails = false,
    baseUrl = null,
    learning = false,
    compiledInstinct = null,
  } = {}) {
    this.backendName = backend;
    this.temperature = temperature;
    this.baseUrl = baseUrl;
    this.compiledInstinct = compiledInstinct;
    this.engine = new PureSemanticEngine({ temperature });
    this.encoder = new SemanticVectorEncoder();

    // Cache initialization
    if (cache instanceof InstinctCache) {
      this.cache = cache;
    } else if (cache) {
      this.cache = new InstinctCache();
    } else {
      this.cache = null;
    }

    // Guardrail suite initialization
    if (guardrails instanceof GuardrailSuite) {
      this.guardrails = guardrails;
    } else if (guardrails) {
      this.guardrails = new GuardrailSuite();
    } else {
      this.guardrails = null;
    }

    // Online Active Learning weights: questionKey -> Float64Array(384)
    this.learning = learning;
    this.weights = {};
    this.biases = {};
  }

  /**
   * Evaluates machine-native questions against given state context.
   */
  async evaluate(state, questions) {
    // 1. Check InstinctCache
    if (this.cache) {
      const cached = this.cache.get(state, questions);
      if (cached) return cached;
    }

    // 2. Fast-path: Pre-compiled .reflex model evaluation (<20µs)
    if (this.compiledInstinct) {
      const compRes = this.compiledInstinct.predict(state);
      const decisions = {};
      for (const [k, q] of Object.entries(questions)) {
        if (q instanceof Choice) {
          const c = compRes.decisions.choice;
          decisions[k] = c ? q.resolve(c.selected, c.distribution) : q.resolve(this.compiledInstinct.options[0] || "", {});
        } else if (q instanceof Noul) {
          const n = compRes.decisions.noul;
          decisions[k] = n ? q.resolve(n.probability) : q.resolve(0.5);
        } else if (q instanceof Score) {
          const s = compRes.decisions.score;
          decisions[k] = s ? q.resolve(s.score, s.confidence) : q.resolve(5.0);
        } else {
          decisions[k] = q.resolve();
        }
      }
      const res = new DecisionResult({
        decisions,
        latencyMs: compRes.latencyMs,
        backend: `compiled:${this.compiledInstinct.name}`,
        inputTokens: compRes.inputTokens,
        outputTokens: 0,
        costUsd: 0.0,
      });
      if (this.cache) this.cache.set(state, questions, res);
      return res;
    }

    let result = null;

    // 2. Remote REST execution if baseUrl configured
    if (this.baseUrl && typeof fetch !== "undefined") {
      try {
        const payloadQuestions = {};
        for (const [k, q] of Object.entries(questions)) {
          payloadQuestions[k] = q.toJSON();
        }
        const resp = await fetch(`${this.baseUrl.replace(/\/+$/, "")}/v1/evaluate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ state, questions: payloadQuestions }),
        });
        if (resp.ok) {
          const data = await resp.json();
          const parsedDecisions = {};
          for (const [k, d] of Object.entries(data.decisions || {})) {
            if (d.type === "noul") {
              parsedDecisions[k] = new Noul(d);
            } else if (d.type === "choice") {
              parsedDecisions[k] = new Choice(d);
            } else if (d.type === "score") {
              parsedDecisions[k] = new Score({
                instructions: d.instructions,
                minVal: d.min_val,
                maxVal: d.max_val,
                score: d.score,
                confidence: d.confidence,
              });
            }
          }
          result = new DecisionResult({
            decisions: parsedDecisions,
            latencyMs: data.latency_ms || 0.0,
            backend: `remote:${this.baseUrl}`,
            inputTokens: data.input_tokens || 0,
            outputTokens: data.output_tokens || 0,
            costUsd: data.cost_usd || 0.0,
          });
        }
      } catch (err) {
        // Graceful fallback to local engine on network error
        result = null;
      }
    }

    // 3. Local PureSemanticEngine evaluation
    if (!result) {
      result = this.engine.evaluate(state, questions);
    }

    // 4. Online active learning weights override
    if (Object.keys(this.weights).length > 0) {
      const stateVec = this.encoder.encode(state);
      for (const [k, q] of Object.entries(questions)) {
        if (q instanceof Noul && this.weights[k]) {
          const prob = this._predictBinary(stateVec, k);
          result.decisions[k] = q.resolve(prob);
        } else if (q instanceof Choice) {
          const hasOptionWeights = q.options.some((opt) => this.weights[`${k}::${opt}`]);
          if (hasOptionWeights) {
            const { selected, distribution } = this._predictChoice(stateVec, q.options, k);
            result.decisions[k] = q.resolve(selected, distribution);
          }
        }
      }
    }

    // 5. Store in InstinctCache
    if (this.cache) {
      this.cache.set(state, questions, result);
    }

    return result;
  }

  /**
   * Fast Boolean instinct returning calibrated probability [0.0 - 1.0].
   */
  async noul(instructions, state, threshold = 0.85) {
    const res = await this.evaluate(state, {
      _q: new Noul({ instructions, threshold }),
    });
    return res.decisions._q.probability;
  }

  /**
   * Fast dynamic rubric selection returning the selected option string.
   */
  async choice(instructions, options, state, criteria = null) {
    const res = await this.evaluate(state, {
      _q: new Choice({ instructions, options, criteria }),
    });
    return res.decisions._q.selected;
  }

  /**
   * Fast continuous scoring returning value between minVal and maxVal.
   */
  async score(instructions, state, minVal = 0.0, maxVal = 1.0) {
    const res = await this.evaluate(state, {
      _q: new Score({ instructions, minVal, maxVal }),
    });
    return res.decisions._q.score;
  }

  /**
   * Instant sub-0.1ms security and prompt injection check.
   */
  guardrail(text) {
    const suite = this.guardrails || new GuardrailSuite();
    return suite.check(text);
  }

  /**
   * Online Active Learning: updates local instinct weights from System 2 ground truth in <0.05ms.
   *
   * @param {string} state - Input prompt or context.
   * @param {string} questionKey - Identifier of the decision task.
   * @param {boolean|string|number} groundTruth - Verified ground truth.
   * @param {Object} [options]
   * @param {number} [options.lr=0.05] - Learning rate.
   * @param {string[]} [options.candidateOptions] - Options list for Choice tuning.
   */
  teach(state, questionKey, groundTruth, { lr = 0.05, candidateOptions = null } = {}) {
    if (this.cache) {
      this.cache.clear();
    }
    const x = this.encoder.encode(state);

    if (typeof groundTruth === "boolean" || typeof groundTruth === "number") {
      const y = typeof groundTruth === "boolean" ? (groundTruth ? 1.0 : 0.0) : Number(groundTruth);
      return this._updateBinary(x, questionKey, y, lr);
    } else if (typeof groundTruth === "string") {
      const opts = candidateOptions || [groundTruth];
      return this._updateChoice(x, questionKey, groundTruth, opts, lr);
    }
    return 0.0;
  }

  _predictBinary(x, key) {
    const w = this.weights[key];
    const b = this.biases[key] || 0.0;
    let dot = b;
    for (let i = 0; i < 384; i++) dot += w[i] * x[i];
    const clamped = Math.max(-20.0, Math.min(20.0, dot));
    return 1.0 / (1.0 + Math.exp(-clamped));
  }

  _updateBinary(x, key, y, lr) {
    if (!this.weights[key]) {
      this.weights[key] = new Float64Array(384);
      this.biases[key] = 0.0;
    }
    const p = this._predictBinary(x, key);
    const grad = p - y;
    const w = this.weights[key];

    for (let i = 0; i < 384; i++) {
      w[i] -= lr * grad * x[i];
    }
    this.biases[key] -= lr * grad;

    const eps = 1e-9;
    const loss = -(y * Math.log(p + eps) + (1.0 - y) * Math.log(1.0 - p + eps));
    return Math.round(loss * 10000) / 10000;
  }

  _predictChoice(x, options, prefix) {
    const logits = [];
    for (const opt of options) {
      const key = `${prefix}::${opt}`;
      const w = this.weights[key];
      const b = this.biases[key] || 0.0;
      let dot = b;
      if (w) {
        for (let i = 0; i < 384; i++) dot += w[i] * x[i];
      }
      logits.push(dot);
    }
    const maxL = Math.max(...logits);
    const expL = logits.map((l) => Math.exp(l - maxL));
    const sumExp = expL.reduce((a, b) => a + b, 0);

    const distribution = {};
    let bestOpt = options[0];
    let bestP = -1;

    for (let i = 0; i < options.length; i++) {
      const opt = options[i];
      const p = Math.round((expL[i] / sumExp) * 10000) / 10000;
      distribution[opt] = p;
      if (p > bestP) {
        bestP = p;
        bestOpt = opt;
      }
    }
    return { selected: bestOpt, distribution };
  }

  _updateChoice(x, prefix, targetOpt, options, lr) {
    for (const opt of options) {
      const key = `${prefix}::${opt}`;
      if (!this.weights[key]) {
        this.weights[key] = new Float64Array(384);
        this.biases[key] = 0.0;
      }
    }

    const { distribution } = this._predictChoice(x, options, prefix);
    let targetP = distribution[targetOpt] || 1e-4;

    for (const opt of options) {
      const key = `${prefix}::${opt}`;
      const p = distribution[opt] || 0.0;
      const y = opt === targetOpt ? 1.0 : 0.0;
      const grad = p - y;
      const w = this.weights[key];

      for (let i = 0; i < 384; i++) {
        w[i] -= lr * grad * x[i];
      }
      this.biases[key] -= lr * grad;
    }

    const loss = -Math.log(Math.max(1e-9, targetP));
    return Math.round(loss * 10000) / 10000;
  }

  /**
   * Loads a .reflex binary model artifact into this client instance.
   * @param {Uint8Array|ArrayBuffer|string} bufferOrPath
   * @returns {Promise<CompiledInstinct>}
   */
  async loadCompiledModel(bufferOrPath) {
    if (typeof bufferOrPath === "string") {
      this.compiledInstinct = await CompiledInstinct.fromFile(bufferOrPath);
    } else {
      this.compiledInstinct = CompiledInstinct.fromBinary(bufferOrPath);
    }
    return this.compiledInstinct;
  }

  /**
   * Executes sub-20µs direct inference using the loaded compiled instinct head.
   * @param {string} state
   * @returns {DecisionResult}
   */
  predict(state) {
    if (!this.compiledInstinct) {
      throw new Error("No compiled .reflex model loaded. Pass compiledInstinct to constructor or call rx.loadCompiledModel().");
    }
    return this.compiledInstinct.predict(state);
  }
}
