//! High-level Reflex runtime client for Rust applications.

use crate::encoder::{cosine_similarity, SemanticVectorEncoder, VECTOR_DIM};
use crate::guardrails::{GuardrailResult, GuardrailSuite};
use crate::primitives::{Choice, ChoiceResult, Noul, NoulResult, Score, ScoreResult};

/// Universal System-1 AI Runtime client.
#[derive(Clone, Debug, Default)]
pub struct Reflex {
    encoder: SemanticVectorEncoder,
    guardrails: GuardrailSuite,
}

impl Reflex {
    pub fn new() -> Self {
        Self {
            encoder: SemanticVectorEncoder::new(),
            guardrails: GuardrailSuite::new(),
        }
    }

    /// Evaluates a Noul (probabilistic boolean) against a state string.
    pub fn noul(&self, instructions: impl Into<String>, state: &str) -> NoulResult {
        let noul = Noul::new(instructions);
        let state_vec = self.encoder.encode(state);
        noul.evaluate(state, &state_vec, &self.encoder)
    }

    /// Evaluates a Choice rubric selection across options.
    pub fn choice(&self, instructions: impl Into<String>, options: Vec<String>, state: &str) -> ChoiceResult {
        let choice = Choice::new(instructions, options);
        let state_vec = self.encoder.encode(state);
        choice.evaluate(state, &state_vec, &self.encoder)
    }

    /// Evaluates a continuous Score on [min_val, max_val].
    pub fn score(&self, instructions: impl Into<String>, min_val: f32, max_val: f32, state: &str) -> ScoreResult {
        let score = Score::new(instructions, min_val, max_val);
        let state_vec = self.encoder.encode(state);
        score.evaluate(state, &state_vec, &self.encoder)
    }

    /// Inspects input text for security threats, jailbreaks, and PII.
    pub fn guardrail(&self, text: &str) -> GuardrailResult {
        self.guardrails.evaluate(text)
    }

    /// Encodes input text into an L2-normalized 384-dimensional vector.
    pub fn encode(&self, text: &str) -> [f32; VECTOR_DIM] {
        self.encoder.encode(text)
    }

    /// Computes cosine similarity between two text strings.
    pub fn similarity(&self, text_a: &str, text_b: &str) -> f32 {
        let vec_a = self.encoder.encode(text_a);
        let vec_b = self.encoder.encode(text_b);
        cosine_similarity(&vec_a, &vec_b)
    }
}
