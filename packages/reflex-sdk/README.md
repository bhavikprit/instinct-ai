# sys1

> **Universal System-1 AI Runtime & Dual-Brain Gateway for Edge, Cloudflare Workers, Node.js, and Browsers.**  
> *OpenAI built o1 for System 2. We built sys1 for System 1.*  
> *Zero external dependencies. Sub-0.05ms execution. 100% mathematical vector parity with Python sys1.*

---

## ⚡ Why sys1 at the Edge?

Modern AI agents and web services suffer from high latency, massive cloud LLM bills, and lack of real-time security guardrails. 

`sys1` brings machine-native System-1 decision architecture directly into TypeScript and JavaScript environments:
- **Cloudflare Workers & Vercel Edge**: Filter prompt injections, PII, and routine queries at the edge in <0.05ms before calling OpenAI/Anthropic.
- **Node.js & Next.js Backends**: Sub-millisecond tool routing, dynamic schema selection, and semantic memory caching.
- **Client-Side Browsers**: 100% local, zero-network instinct evaluation with zero API key exposure.

---

## 📦 Installation

```bash
npm install sys1
# or pnpm add sys1
# or bun add sys1
```

*(Has **zero production dependencies** and runs on any standard ECMAScript 2022+ / Web runtime).*

---

## 🚀 Quickstart

### 1. Instant System-1 Instincts

```javascript
import { Sys1, Noul, Choice } from "sys1";

const rx = new Sys1({ cache: true, guardrails: true });

// Probabilistic Boolean decision (Noul)
const isUrgent = await rx.noul(
  "Is this an urgent production database incident?",
  "PostgreSQL primary node replica lag exceeded 60s"
);
console.log("Urgent probability:", isUrgent); // e.g. 0.94

// Dynamic rubric selection (Choice)
const tool = await rx.choice(
  "Select next autonomous agent action",
  ["search_docs", "restart_service", "page_oncall"],
  "Service timeout on port 5432 after memory leak"
);
console.log("Selected tool:", tool); // e.g. "restart_service"
```

---

### 2. Cloudflare Worker Edge Dual-Brain Gateway

```javascript
import { Sys1, Noul, Choice } from "sys1";

const rx = new Sys1({ cache: true, guardrails: true });

export default {
  async fetch(request) {
    const { prompt } = await request.json();

    // Step 1: Sub-0.05ms Instant Guardrail Check
    const security = rx.guardrail(prompt);
    if (security.blocked) {
      return new Response(JSON.stringify({ error: security.reason }), { status: 400 });
    }

    // Step 2: Edge Triage
    const triage = await rx.evaluate(prompt, {
      is_faq: new Noul({ instructions: "Is this a routine FAQ question?", threshold: 0.85 }),
      route: new Choice({ instructions: "Select route", options: ["resolve_at_edge", "escalate"] })
    });

    if (triage.decisions.is_faq.isTrue && triage.decisions.route.selected === "resolve_at_edge") {
      // 0 tokens used, 0 cloud cost!
      return new Response(JSON.stringify({ answer: "Routine answer cached at edge." }));
    }

    // Step 3: High epistemic uncertainty -> Escalate to System 2 (Claude 3.5 or GPT-4o)
    const completion = await callUpstreamLLM(prompt);
    return new Response(JSON.stringify({ answer: completion }));
  }
};
```

---

### 3. Online Active Learning (`rx.teach(...)`)

Reflex can self-improve online in microseconds when System 2 provides verified ground truth:

```javascript
const rx = new Reflex({ learning: true });

const query = "Authorize $50,000 international wire transfer";

// Reflex initial evaluation
const prob1 = await rx.noul("Is this a high risk financial action?", query);

// Teach Reflex from System 2 ground truth in <0.05ms
rx.teach(query, "_q", true, { lr: 0.15 });

// Subsequent calls adapt immediately without touching cloud models!
const prob2 = await rx.noul("Is this a high risk financial action?", query);
console.log(prob2 > prob1); // true
```

---

## 🏛️ Core Primitives

- `Noul`: Calibrated epistemic probability $[0.0, 1.0]$ with `isTrue`, `isFalse`, and `isUncertain` triggers.
- `Choice`: Dynamic multi-class rubric selection with Dirichlet-style softmax distribution.
- `Score`: Continuous bounded metric with confidence intervals.
- `InstinctCache`: Multi-tier L1 exact hash + L2 384-d semantic cosine vector memory.
- `GuardrailSuite`: Prompt injection, jailbreak, SSN, API key, and Luhn credit card detection in sub-0.1ms.

---

## 🧪 Testing

Run native Node.js tests without installing any test runners:

```bash
npm test
# runs: node --test test/test_reflex.mjs
```

---

## 📄 License

Apache 2.0. Open-source and machine-native.
