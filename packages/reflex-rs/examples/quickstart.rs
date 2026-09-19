//! Reflex Rust Quickstart (Phase 17).
//! Zero-dependency System-1 AI decision runtime for Rust.

use reflex_rs::Reflex;

fn main() {
    println!("==================================================================");
    println!("🦀 Reflex Rust SDK (reflex-rs) Quickstart");
    println!("==================================================================");

    let rx = Reflex::new();
    let state = "Customer: I was billed $299 twice on my Visa card today. Refund immediately!";

    // 1. Noul Boolean Primitive (<15µs)
    let noul = rx.noul("Is the customer demanding a refund or chargeback?", state);
    println!("\n1. Noul Boolean Decision:");
    println!("   • Probability : {:.4}", noul.probability);
    println!("   • Is True?    : {}", noul.is_true);
    println!("   • Confidence  : {:.4}", noul.confidence);
    println!("   • Uncertain?  : {}", noul.is_uncertain);

    // 2. Choice Multi-Class Rubric (<15µs)
    let options = vec![
        "billing".to_string(),
        "technical_support".to_string(),
        "sales".to_string(),
    ];
    let choice = rx.choice("Select operational department queue", options, state);
    println!("\n2. Choice Rubric Selection:");
    println!("   • Selected    : {}", choice.selected);
    println!("   • Confidence  : {:.4}", choice.confidence);
    for (opt, prob) in &choice.distribution {
        println!("     - {:<18} : {:.4}", opt, prob);
    }

    // 3. Instant Guardrails (<1µs)
    let guard = rx.guardrail(state);
    println!("\n3. Guardrail Security Audit:");
    println!("   • Is Safe?    : {}", guard.is_safe);
    println!("   • Blocked?    : {}", guard.blocked);
    println!("   • Risk Score  : {:.2}", guard.risk_score);

    println!("\n==================================================================");
    println!("✅ Rust System-1 execution complete. Zero external crate dependencies.");
    println!("==================================================================");
}
