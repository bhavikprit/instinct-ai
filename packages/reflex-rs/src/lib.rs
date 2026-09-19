//! Reflex: Universal System-1 AI Runtime & Dual-Brain Gateway.
//! High-performance, zero-dependency Rust crate with WebAssembly support.

pub mod client;
pub mod compiler;
pub mod encoder;
pub mod guardrails;
pub mod primitives;

pub use client::Reflex;
pub use compiler::{crc32, CompiledInstinct, CompiledResult};
pub use encoder::{cosine_similarity, Md5, SemanticVectorEncoder, VECTOR_DIM};
pub use guardrails::{GuardrailResult, GuardrailSuite};
pub use primitives::{Choice, ChoiceResult, Noul, NoulResult, Score, ScoreResult};
