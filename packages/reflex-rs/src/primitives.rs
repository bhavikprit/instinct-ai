//! Machine-native System-1 Decision Primitives: Noul, Choice, Score.
//! 100% mathematical parity with Python reflex-core and JavaScript @reflex-ai/sdk.

use std::collections::HashSet;
use crate::encoder::{cosine_similarity, SemanticVectorEncoder, VECTOR_DIM};

/// Noul: Probabilistic Boolean primitive.
#[derive(Clone, Debug)]
pub struct Noul {
    pub instructions: String,
    pub threshold: f32,
    pub temperature: f32,
    pub uncertainty_low: f32,
    pub uncertainty_high: f32,
}

impl Noul {
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            threshold: 0.85,
            temperature: 0.25,
            uncertainty_low: 0.35,
            uncertainty_high: 0.65,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(0.01);
        self
    }

    pub fn evaluate(
        &self,
        state: &str,
        state_vec: &[f32; VECTOR_DIM],
        encoder: &SemanticVectorEncoder,
    ) -> NoulResult {
        let query_vec = encoder.encode(&self.instructions);
        let sim = cosine_similarity(state_vec, &query_vec);

        let state_tokens: HashSet<String> = extract_tokens(state);
        let query_tokens: HashSet<String> = extract_tokens(&self.instructions);

        // Contrast polarity adjustment
        let neg_words = ["not", "never", "safe", "normal", "routine", "false", "ignore"];
        let has_neg = neg_words.iter().any(|w| state_tokens.contains(*w));

        // Alarm / domain boost
        let alarm_tokens = [
            "scam", "fraud", "wire", "urgent", "phishing", "attack", "critical", "breach",
            "refund", "stolen", "cancel", "ransomware", "hazard", "threat", "hacked", "emergency",
        ];
        let alarm_overlap = alarm_tokens
            .iter()
            .filter(|w| state_tokens.contains(**w))
            .count();

        let query_lower = self.instructions.to_lowercase();
        let query_is_threat = ["security", "threat", "hazard", "scam", "urgent", "refund"]
            .iter()
            .any(|w| query_lower.contains(w));

        let keyword_overlap = query_tokens
            .iter()
            .filter(|w| state_tokens.contains(*w))
            .count();

        let mut effective_sim = sim + (0.30f32).min(keyword_overlap as f32 * 0.08);
        if query_is_threat && alarm_overlap > 0 {
            effective_sim = effective_sim.max(0.15 + alarm_overlap as f32 * 0.05);
        }

        // Calibrate similarity into probability via sigmoid
        let adjusted_sim = effective_sim - if has_neg { 0.15 } else { 0.0 };
        let logit = (adjusted_sim - 0.06) / self.temperature;
        let clamped_logit = logit.clamp(-20.0, 20.0);
        let prob = 1.0 / (1.0 + (-clamped_logit).exp());
        let clamped = prob.clamp(0.0, 1.0);

        let is_true = clamped >= self.threshold;
        let is_false = clamped <= (1.0 - self.threshold);
        let is_uncertain = clamped >= self.uncertainty_low && clamped <= self.uncertainty_high;
        let confidence = clamped.max(1.0 - clamped);

        NoulResult {
            probability: clamped,
            confidence,
            is_true,
            is_false,
            is_uncertain,
        }
    }
}

/// Result of evaluating a Noul primitive.
#[derive(Clone, Debug, PartialEq)]
pub struct NoulResult {
    pub probability: f32,
    pub confidence: f32,
    pub is_true: bool,
    pub is_false: bool,
    pub is_uncertain: bool,
}

/// Choice: Dynamic Rubric & Multi-Class Selection primitive.
#[derive(Clone, Debug)]
pub struct Choice {
    pub instructions: String,
    pub options: Vec<String>,
    pub temperature: f32,
}

impl Choice {
    pub fn new(instructions: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            instructions: instructions.into(),
            options,
            temperature: 0.25,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(0.01);
        self
    }

    pub fn evaluate(
        &self,
        _state: &str,
        state_vec: &[f32; VECTOR_DIM],
        encoder: &SemanticVectorEncoder,
    ) -> ChoiceResult {
        if self.options.is_empty() {
            return ChoiceResult {
                selected: String::new(),
                confidence: 0.0,
                distribution: Vec::new(),
            };
        }

        let mut raw_sims: Vec<f32> = Vec::with_capacity(self.options.len());
        for opt in &self.options {
            let opt_text = format!("{} {}", self.instructions, opt);
            let opt_vec = encoder.encode(&opt_text);
            let sim = cosine_similarity(state_vec, &opt_vec);
            raw_sims.push(sim / self.temperature);
        }

        // Softmax with numerical stability
        let max_score = raw_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw_sims.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();

        let mut distribution: Vec<(String, f32)> = Vec::with_capacity(self.options.len());
        let mut best_idx = 0;
        let mut best_prob = -1.0f32;

        for (i, opt) in self.options.iter().enumerate() {
            let p = if sum_exp > 0.0 { exps[i] / sum_exp } else { 0.0 };
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
            distribution.push((opt.clone(), p));
        }

        ChoiceResult {
            selected: self.options[best_idx].clone(),
            confidence: best_prob,
            distribution,
        }
    }
}

/// Result of evaluating a Choice primitive.
#[derive(Clone, Debug)]
pub struct ChoiceResult {
    pub selected: String,
    pub confidence: f32,
    pub distribution: Vec<(String, f32)>,
}

impl ChoiceResult {
    pub fn get_prob(&self, option: &str) -> f32 {
        self.distribution
            .iter()
            .find(|(k, _)| k == option)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }
}

/// Score: Continuous Scaling primitive.
#[derive(Clone, Debug)]
pub struct Score {
    pub instructions: String,
    pub min_val: f32,
    pub max_val: f32,
}

impl Score {
    pub fn new(instructions: impl Into<String>, min_val: f32, max_val: f32) -> Self {
        Self {
            instructions: instructions.into(),
            min_val,
            max_val,
        }
    }

    pub fn evaluate(
        &self,
        _state: &str,
        state_vec: &[f32; VECTOR_DIM],
        encoder: &SemanticVectorEncoder,
    ) -> ScoreResult {
        let instr_vec = encoder.encode(&self.instructions);
        let sim = cosine_similarity(state_vec, &instr_vec).clamp(0.0, 1.0);
        let score = self.min_val + sim * (self.max_val - self.min_val);
        let confidence = (0.50 + (sim - 0.50).abs()).clamp(0.0, 1.0);

        ScoreResult {
            score,
            confidence,
        }
    }
}

/// Result of evaluating a Score primitive.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreResult {
    pub score: f32,
    pub confidence: f32,
}

fn extract_tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}
