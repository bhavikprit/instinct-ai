# ⚡ Reflex

### Universal System-1 AI Runtime & Dual-Brain Gateway
*Make decisions, not strings. The open-source standard for machine-native AI decision models.*

---

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/bhavikprit/reflex-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/bhavikprit/reflex-ai/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)](pyproject.toml)
[![Speed](https://img.shields.io/badge/latency-%3C15ms%20local-cyan.svg)](#benchmarks)
[![Output Cost](https://img.shields.io/badge/output%20tokens-%240.00%20(FREE)-emerald.svg)](#why-reflex)

For three years, the AI industry has suffered from an architectural antipattern: **using 70B+ parameter autoregressive models to make boolean decisions and route software traffic.**

Software codebases natively speak `if`, `else if`, and `switch`. Forcing an autoregressive LLM to generate sequential text tokens just to output `{"is_spam": true}` burns 2.5 seconds, drains batteries, risks JSON schema hallucinations, and balloons cloud bills.

**Reflex** is the open-source **dual-brain runtime** that bridges instant local instincts (<15ms, $0 cost) with heavy cloud reasoning models, providing a single unified standard for AI decision-making.

---

## 🧠 The Dual-Brain Architecture

```
                             Your Application / Agent
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │          REFLEX UNIVERSAL RUNTIME           │
                 │   • Unified Protocol: Noul, Choice, Score   │
                 │   • Epistemic Gate & Auto-Escalation        │
                 └──────────────────────┬──────────────────────┘
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
[TIER 1: THE SPINAL REFLEX]                                     [TIER 2: THE CORTEX]
Local & Fast (<15ms • $0.00)                                    Heavy Reasoning (2s - 4s)
• Reflex Local Engine (CPU / Metal)                             • Claude 3.5 Sonnet
• TypeSafe Jev API (~typesafe/jev-latest)                       • GPT-4o / DeepSeek
• Fast Logit-Scorer (ModernBERT / Qwen)                         (Only awakened when confidence < 0.85)
```

---

## ⚡ Quickstart (60 Seconds)

### 1. Installation
```bash
pip install reflex-ai
```

### 2. Multi-Primitive Decision in a Single Pass
```python
from reflex import Reflex, Noul, Choice, Score

# Auto-detects local sub-15ms engine or cloud Jev API
rx = Reflex()

result = rx.evaluate(
    state="Customer: I was billed $499 twice on my Visa today. Reverse the duplicate charge immediately!",
    questions={
        "is_refund": Noul("Is the customer demanding a refund or chargeback?"),
        "target_queue": Choice(
            instructions="Select operations queue",
            options=["billing", "technical_support", "fraud_investigation", "spam"]
        ),
        "frustration": Score("Customer distress score 1-10", min_val=1.0, max_val=10.0)
    }
)

# Access typed primitives (Zero schema hallucinations)
print(f"Refund Requested? -> {result['is_refund'].probability:.3f}")
print(f"Target Queue      -> {result['target_queue'].selected}")
print(f"Latency           -> {result.latency_ms} ms (Cost: ${result.cost_usd})")
```

### 3. Inline Shortcuts
```python
# Returns float probability in [0.0 - 1.0]
is_scam = rx.noul("Is this a deceptive emergency scam?", sms_text)

# Returns winning string option directly
target_tool = rx.choice("Select next agent tool", ["search", "calc", "sql"], agent_context)
```

---

## 🔌 The Drop-in OpenAI Proxy (`reflex-proxy`)

Already have thousands of lines of existing OpenAI or Anthropic code? **Zero code refactoring required.**

### 1. Start the Reflex Proxy
```bash
python3 -m reflex.cli serve --port 8080
```

### 2. Point Your Client to Reflex
```python
from openai import OpenAI

# Simply change your baseURL to Reflex!
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-reflex"
)

# If the prompt is a classification or routing decision, 
# Reflex intercepts it and resolves it in <15ms for $0.00!
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Classify this ticket: my account is locked out"}],
    response_format={"type": "json_object"}
)

print(response.choices[0].message.content)
# Output tokens are billed as 0!
```

## 🔌 Model Context Protocol (MCP) Server

Connect Reflex directly to **Claude Desktop**, **Cursor**, or any MCP-compatible agent runtime:

### Claude Desktop Configuration
Add this to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "reflex": {
      "command": "python3",
      "args": ["-m", "reflex.cli", "mcp"]
    }
  }
}
```

### Tools Exposed to Your Agent:
* `reflex_noul`: Sub-15ms calibrated boolean evaluation (`is_true`, `is_uncertain`, probability).
* `reflex_choice`: Sub-15ms multi-enum tool and route selection.
* `reflex_guardrail`: Pre-flight prompt-injection and security gatekeeper.

---

## 🦜 LangChain & LangGraph Integration

Replace slow 3-second LLM routing steps with sub-15ms Reflex routing:

```python
from reflex.integrations.langchain import ReflexRouterNode, ReflexGuardrailNode

# 1. Pre-flight Guardrail
guardrail = ReflexGuardrailNode()
security_check = guardrail("User prompt to inspect")
# Returns {'safe': True, 'blocked': False, 'latency_ms': 0.08}

# 2. Drop-in LangGraph Routing Node
router = ReflexRouterNode(
    routes={
        "billing": "Invoice and payment inquiries",
        "tech_support": "System bugs and crashes",
        "sales": "Enterprise pricing"
    },
    input_key="messages",
    output_key="next_step"
)

state = router({"messages": "Need refund for double charge on invoice #9102"})
print(state["next_step"]) # -> "billing" (Evaluated in 0.08ms)
```

---

## 📊 DecisionBench Standardized Benchmark

Run the standardized benchmark testing calibration (ECE), latency, and injection stress:

```bash
python3 -m benchmarks.decision_bench
```

---

## 🗺️ Project Roadmap
 
 - [x] **Phase 1: Core SDK & Drop-in Proxy**
   - [x] Universal `Noul`, `Choice`, `Score` protocol
   - [x] Multi-backend routing (Local, TypeSafe Jev, OpenRouter, Fallback)
   - [x] Zero-dependency OpenAI-compatible reverse proxy
 - [x] **Phase 2: Framework Integrations & Agent Tools**
   - [x] Model Context Protocol (MCP) server for Claude Desktop & Cursor
   - [x] LangChain & LangGraph `ReflexRouterNode` and `ReflexGuardrailNode`
   - [x] DecisionBench standardized benchmark suite
 - [x] **Phase 3: Local Neural Engine (`reflex.backends.onnx_engine`)**
   - [x] ONNX Runtime INT8 quantized execution with sub-5ms latency
   - [x] Zero-dependency graceful fallback
   - [x] Softmax probability calibration and temperature scaling
 - [x] **Phase 4: OpenRLCD (Reinforcement Learning for Calibrated Decisions)**
   - [x] Synthetic calibration dataset generator (`reflex dataset-gen`)
   - [x] Standardized Brier Score & Expected Calibration Error (ECE) loss metrics
   - [x] Epistemic entropy uncertainty scoring
 - [ ] **Phase 5: Pretrained Model Weights**
   - [ ] Release fine-tuned `Reflex-0.5B` ONNX checkpoints on HuggingFace

---

## 🤝 Contributing

Reflex is an open-source project welcoming contributions from AI engineers, system architects, and researchers.

```bash
git clone https://github.com/bhavikprit/reflex-ai.git
cd reflex-ai
python3 -m unittest discover -s tests
```

---

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
