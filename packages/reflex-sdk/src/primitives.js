/**
 * Reflex Primitives: Universal Machine-Native Decision Types.
 * Standardizing Noul, Choice, and Score across Edge, Browser, and Node.js.
 */

export class Noul {
  /**
   * Noul: Probabilistic Boolean primitive.
   * Represents calibrated epistemic probability in [0.0, 1.0].
   */
  constructor({
    instructions,
    probability = null,
    threshold = 0.85,
    uncertaintyLow = 0.35,
    uncertaintyHigh = 0.65,
  }) {
    this.instructions = instructions;
    this.probability = probability !== null ? Math.max(0.0, Math.min(1.0, Number(probability))) : null;
    this.threshold = threshold;
    this.uncertaintyLow = uncertaintyLow;
    this.uncertaintyHigh = uncertaintyHigh;
  }

  resolve(prob) {
    const clamped = Math.max(0.0, Math.min(1.0, Number(prob)));
    return new Noul({
      instructions: this.instructions,
      probability: clamped,
      threshold: this.threshold,
      uncertaintyLow: this.uncertaintyLow,
      uncertaintyHigh: this.uncertaintyHigh,
    });
  }

  get isTrue() {
    return this.probability !== null && this.probability >= this.threshold;
  }

  get isFalse() {
    return this.probability !== null && this.probability <= (1.0 - this.threshold);
  }

  get isUncertain() {
    if (this.probability === null) return true;
    return this.probability >= this.uncertaintyLow && this.probability <= this.uncertaintyHigh;
  }

  toJSON() {
    return {
      type: "noul",
      instructions: this.instructions,
      probability: this.probability !== null ? Number(this.probability.toFixed(4)) : null,
      is_true: this.isTrue,
      is_false: this.isFalse,
      is_uncertain: this.isUncertain,
    };
  }
}

export class Choice {
  /**
   * Choice: Dynamic Rubric & Multi-Class Selection primitive.
   * Returns the selected option and probability distribution.
   */
  constructor({
    instructions,
    options = [],
    criteria = null,
    selected = null,
    distribution = {},
  }) {
    this.instructions = instructions;
    this.options = [...options];
    this.criteria = criteria ? { ...criteria } : null;
    this.selected = selected;
    this.distribution = { ...distribution };
  }

  resolve(selected, distribution = null) {
    const dist = distribution || { [selected]: 1.0 };
    return new Choice({
      instructions: this.instructions,
      options: this.options,
      criteria: this.criteria,
      selected,
      distribution: dist,
    });
  }

  getProb(option) {
    return this.distribution[option] || 0.0;
  }

  toJSON() {
    return {
      type: "choice",
      instructions: this.instructions,
      options: this.options,
      selected: this.selected,
      distribution: this.distribution,
    };
  }
}

export class Score {
  /**
   * Score: Continuous scoring metric [minVal, maxVal].
   */
  constructor({
    instructions,
    minVal = 0.0,
    maxVal = 1.0,
    score = null,
    confidence = null,
  }) {
    this.instructions = instructions;
    this.minVal = minVal;
    this.maxVal = maxVal;
    this.score = score !== null ? Number(score) : null;
    this.confidence = confidence !== null ? Number(confidence) : null;
  }

  resolve(score, confidence = null) {
    const clamped = Math.max(this.minVal, Math.min(this.maxVal, Number(score)));
    return new Score({
      instructions: this.instructions,
      minVal: this.minVal,
      maxVal: this.maxVal,
      score: clamped,
      confidence: confidence !== null ? Math.max(0.0, Math.min(1.0, Number(confidence))) : null,
    });
  }

  toJSON() {
    return {
      type: "score",
      instructions: this.instructions,
      min_val: this.minVal,
      max_val: this.maxVal,
      score: this.score,
      confidence: this.confidence,
    };
  }
}

export class DecisionResult {
  constructor({
    decisions = {},
    latencyMs = 0.0,
    backend = "semantic-pure",
    inputTokens = 0,
    outputTokens = 0,
    costUsd = 0.0,
  } = {}) {
    this.decisions = decisions;
    this.latencyMs = latencyMs;
    this.backend = backend;
    this.inputTokens = inputTokens;
    this.outputTokens = outputTokens;
    this.costUsd = costUsd;
  }

  get isUncertain() {
    return Object.values(this.decisions).some((d) => {
      if (d instanceof Noul) return d.isUncertain;
      if (d && typeof d === "object" && "is_uncertain" in d) return Boolean(d.is_uncertain);
      return false;
    });
  }

  toJSON() {
    const decMap = {};
    for (const [k, v] of Object.entries(this.decisions)) {
      decMap[k] = v && typeof v.toJSON === "function" ? v.toJSON() : v;
    }
    return {
      decisions: decMap,
      latency_ms: this.latencyMs,
      backend: this.backend,
      input_tokens: this.inputTokens,
      output_tokens: this.outputTokens,
      cost_usd: this.costUsd,
      is_uncertain: this.isUncertain,
    };
  }
}
