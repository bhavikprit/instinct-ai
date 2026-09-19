use reflex_rs::{GuardrailSuite, Reflex, SemanticVectorEncoder};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let sample = "Urgent: Suspicious activity on your account. Click here to cancel wire #8129!";

    let rx = Reflex::new();
    let encoder = SemanticVectorEncoder::new();
    let suite = GuardrailSuite::new();

    // 1. Encode
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = encoder.encode(sample);
    }
    let t_enc = t0.elapsed().as_secs_f64();

    // 2. Noul
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = rx.noul("Is this a security threat or phishing scam?", sample);
    }
    let t_noul = t0.elapsed().as_secs_f64();

    // 3. Guardrail
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = suite.evaluate(sample);
    }
    let t_guard = t0.elapsed().as_secs_f64();

    let encode_us = (t_enc / iters as f64) * 1_000_000.0;
    let noul_us = (t_noul / iters as f64) * 1_000_000.0;
    let guard_us = (t_guard / iters as f64) * 1_000_000.0;
    let throughput_ops = iters as f64 / t_noul;

    println!(
        r#"{{"encode_us": {:.2}, "noul_us": {:.2}, "guard_us": {:.2}, "throughput_ops": {:.2}}}"#,
        encode_us, noul_us, guard_us, throughput_ops
    );
}
