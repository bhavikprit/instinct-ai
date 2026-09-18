/**
 * Instant Local Guardrails Suite for Reflex in JavaScript.
 * Zero-dependency, sub-0.1ms prompt injection, PII, and security protection.
 */

export class GuardrailResult {
  constructor({
    isSafe = true,
    blocked = false,
    riskScore = 0.0,
    category = null,
    reason = null,
    detectedEntities = [],
    latencyMs = 0.0,
  } = {}) {
    this.isSafe = isSafe;
    this.blocked = blocked;
    this.riskScore = Math.round(riskScore * 10000) / 10000;
    this.category = category;
    this.reason = reason;
    this.detectedEntities = [...detectedEntities];
    this.latencyMs = Math.round(latencyMs * 1000) / 1000;
  }

  toJSON() {
    return {
      is_safe: this.isSafe,
      blocked: this.blocked,
      risk_score: this.riskScore,
      category: this.category,
      reason: this.reason,
      detected_entities: this.detectedEntities,
      latency_ms: this.latencyMs,
    };
  }
}

export class PromptInjectionGuardrail {
  constructor({ threshold = 0.70 } = {}) {
    this.name = "prompt_injection";
    this.threshold = threshold;
    this.patterns = [
      /ignore\s+(all\s+)?(prior|previous|above)\s+(instructions?|rules?|directives?)/i,
      /disregard\s+(all\s+)?(prior|previous|above)\s+(instructions?|rules?)/i,
      /forget\s+(all\s+)?(prior|previous)\s+rules?/i,
      /\byou\s+are\s+now\s+(in\s+)?(dan|developer\s+mode|unrestricted|jailbroken)\b/i,
      /\bdo\s+anything\s+now\b/i,
      /print\s+(your\s+)?(secret\s+)?system\s+(prompt|instructions?)/i,
      /repeat\s+the\s+(text|words|instructions)\s+above\s+verbatim/i,
      /<\s*\|\s*im_start\s*\|\s*>\s*system/i,
      /```\s*system/i,
      /\[SYSTEM(?:\s+PROMPT)?\]/i,
    ];
  }

  check(text) {
    const startTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const matches = [];

    for (const pat of this.patterns) {
      const found = pat.exec(text);
      if (found) {
        matches.push(found[0]);
      }
    }

    const endTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const latencyMs = endTime - startTime;

    if (matches.length > 0) {
      return new GuardrailResult({
        isSafe: false,
        blocked: true,
        riskScore: 0.98,
        category: this.name,
        reason: `Detected prompt injection pattern: '${matches[0]}'`,
        detectedEntities: matches,
        latencyMs,
      });
    }

    return new GuardrailResult({
      isSafe: true,
      blocked: false,
      riskScore: 0.02,
      category: this.name,
      reason: null,
      latencyMs,
    });
  }
}

export class PIIGuardrail {
  constructor() {
    this.name = "pii_leakage";
    this.ssnPattern = /\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b/;
    this.apiKeyPatterns = [
      [/\bsk-[a-zA-Z0-9]{32,}\b/, "OpenAI API Key"],
      [/\bAKIA[0-9A-Z]{16}\b/, "AWS Access Key"],
      [/\bghp_[a-zA-Z0-9]{36}\b/, "GitHub Personal Access Token"],
    ];
    this.ccPattern = /\b(?:\d{4}[-\s]?){3}\d{4}\b|\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b|\b(?:\d[-\s]?){12,18}\d\b/g;
  }

  check(text) {
    const startTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const detected = [];

    // 1. API Keys
    for (const [pat, keyType] of this.apiKeyPatterns) {
      if (pat.test(text)) {
        detected.push(`Secret Credential (${keyType})`);
      }
    }

    // 2. SSN
    if (this.ssnPattern.test(text)) {
      detected.push("Social Security Number (SSN)");
    }

    // 3. Credit Card with Luhn validation
    const ccMatches = text.match(this.ccPattern);
    if (ccMatches) {
      for (const m of ccMatches) {
        const rawDigits = m.replace(/\D/g, "");
        if (this._luhnVerify(rawDigits)) {
          detected.push("Credit Card Number (Luhn Validated)");
          break;
        }
      }
    }

    const endTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const latencyMs = endTime - startTime;

    if (detected.length > 0) {
      return new GuardrailResult({
        isSafe: false,
        blocked: true,
        riskScore: 0.99,
        category: this.name,
        reason: `PII / Credential leak detected: ${detected.join(", ")}`,
        detectedEntities: detected,
        latencyMs,
      });
    }

    return new GuardrailResult({
      isSafe: true,
      blocked: false,
      riskScore: 0.01,
      category: this.name,
      reason: null,
      latencyMs,
    });
  }

  _luhnVerify(cardNumber) {
    if (cardNumber.length < 13 || cardNumber.length > 19) return false;
    let checksum = 0;
    const rev = cardNumber.split("").reverse();
    for (let i = 0; i < rev.length; i++) {
      const d = parseInt(rev[i], 10);
      if (i % 2 === 1) {
        const doubled = d * 2;
        checksum += doubled > 9 ? doubled - 9 : doubled;
      } else {
        checksum += d;
      }
    }
    return checksum % 10 === 0;
  }
}

export class GuardrailSuite {
  constructor({ guardrails = null, failFast = true } = {}) {
    this.guardrails = guardrails || [
      new PromptInjectionGuardrail(),
      new PIIGuardrail(),
    ];
    this.failFast = failFast;
  }

  check(text) {
    const startTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const violations = [];
    let highestRisk = 0.0;

    for (const g of this.guardrails) {
      const res = g.check(text);
      if (res.riskScore > highestRisk) {
        highestRisk = res.riskScore;
      }

      if (!res.isSafe) {
        violations.push(res);
        if (this.failFast) break;
      }
    }

    const endTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const totalLatencyMs = endTime - startTime;

    if (violations.length > 0) {
      const firstV = violations[0];
      const allEntities = violations.flatMap((v) => v.detectedEntities);
      return new GuardrailResult({
        isSafe: false,
        blocked: true,
        riskScore: highestRisk,
        category: firstV.category,
        reason: firstV.reason,
        detectedEntities: allEntities,
        latencyMs: totalLatencyMs,
      });
    }

    return new GuardrailResult({
      isSafe: true,
      blocked: false,
      riskScore: highestRisk,
      category: "all_clear",
      reason: null,
      latencyMs: totalLatencyMs,
    });
  }
}
