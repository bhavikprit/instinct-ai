# 🦀 Reflex Rust SDK (`reflex-rs`)

### High-Performance System-1 AI Decision Runtime & Dual-Brain Gateway
*Make decisions, not strings. Pure Rust, zero external dependencies.*

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-0%20crates-brightgreen.svg)](#)
[![Speed](https://img.shields.io/badge/throughput-79k%20ops%2Fs-orange.svg)](#)

---

## ⚡ Features

- **Zero External Dependencies**: Standard library only. No C compilers, OpenSSL, or heavy frameworks required.
- **Microsecond Latency**:
  - `Noul` boolean decision: **12.6 µs**
  - Dense vector encoding (384-d): **11.7 µs**
  - Instant guardrails: **0.2 µs**
  - Peak throughput: **79,000+ ops/sec**
- **100% Mathematical Parity**: Bit-for-bit identical vector embeddings and decision calibrations with Python `reflex-core` and JavaScript `@reflex-ai/sdk`.
- **WASM & Edge Compatible**: Compiles cleanly to `wasm32-unknown-unknown` and `wasm32-wasi`.

---

## 🚀 Quickstart

Add `reflex-rs` to your `Cargo.toml`:

```toml
[dependencies]
reflex-rs = "0.2.0"
```

### Usage

```rust
use reflex_rs::Reflex;

fn main() {
    let rx = Reflex::new();
    let state = "Customer: I was billed twice on my invoice today. Refund immediately!";

    // 1. Noul Boolean Decision (<15µs)
    let noul = rx.noul("Is the customer demanding a refund?", state);
    if noul.is_true {
        println!("Refund requested! (Confidence: {:.2})", noul.confidence);
    }

    // 2. Choice Rubric Selection (<15µs)
    let choice = rx.choice(
        "Route ticket to department",
        vec!["billing".into(), "support".into(), "sales".into()],
        state,
    );
    println!("Selected queue: {}", choice.selected);

    // 3. Instant Guardrails (<1µs)
    let guard = rx.guardrail(state);
    assert!(guard.is_safe);
}
```

---

## 🧪 Testing & Verification

```bash
cargo test
cargo run --example quickstart
cargo run --example bench --release 5000
```
