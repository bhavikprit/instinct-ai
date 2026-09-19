//! Instant Sub-Millisecond Guardrails for Prompt Injection, Jailbreaks, and PII.

/// Result of guardrail inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct GuardrailResult {
    pub is_safe: bool,
    pub blocked: bool,
    pub risk_score: f32,
    pub category: String,
    pub reason: String,
}

impl GuardrailResult {
    pub fn safe() -> Self {
        Self {
            is_safe: true,
            blocked: false,
            risk_score: 0.0,
            category: "benign".to_string(),
            reason: String::new(),
        }
    }

    pub fn violation(category: impl Into<String>, reason: impl Into<String>, risk_score: f32) -> Self {
        Self {
            is_safe: false,
            blocked: true,
            risk_score,
            category: category.into(),
            reason: reason.into(),
        }
    }
}

/// Zero-dependency guardrail suite.
#[derive(Clone, Debug, Default)]
pub struct GuardrailSuite;

impl GuardrailSuite {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates input text for prompt injection, jailbreaks, and PII violations.
    pub fn evaluate(&self, text: &str) -> GuardrailResult {
        let lower = text.to_lowercase();

        // 1. Prompt Injection & Jailbreak Patterns
        let injection_patterns = [
            "ignore all prior instructions",
            "ignore previous instructions",
            "disregard all previous prompts",
            "you are now in dan mode",
            "system prompt override",
            "jailbreak enabled",
            "developer mode activated",
            "reveal system prompt",
        ];

        for pattern in &injection_patterns {
            if lower.contains(pattern) {
                return GuardrailResult::violation(
                    "prompt_injection",
                    format!("Detected prompt injection phrase: '{}'", pattern),
                    0.98,
                );
            }
        }

        // 2. SSN Detection: ###-##-####
        if let Some(ssn) = find_ssn(text) {
            return GuardrailResult::violation(
                "pii_ssn",
                format!("Detected Social Security Number: {}", ssn),
                0.95,
            );
        }

        // 3. Credit Card Detection with Luhn algorithm
        if let Some(cc) = find_credit_card(text) {
            return GuardrailResult::violation(
                "pii_credit_card",
                format!("Detected valid payment card sequence: {}", cc),
                0.99,
            );
        }

        GuardrailResult::safe()
    }
}

fn find_ssn(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    if bytes.len() < 11 {
        return None;
    }
    for i in 0..=bytes.len() - 11 {
        let chunk = &bytes[i..i + 11];
        if chunk[0].is_ascii_digit()
            && chunk[1].is_ascii_digit()
            && chunk[2].is_ascii_digit()
            && chunk[3] == b'-'
            && chunk[4].is_ascii_digit()
            && chunk[5].is_ascii_digit()
            && chunk[6] == b'-'
            && chunk[7].is_ascii_digit()
            && chunk[8].is_ascii_digit()
            && chunk[9].is_ascii_digit()
            && chunk[10].is_ascii_digit()
        {
            return Some(String::from_utf8_lossy(chunk).to_string());
        }
    }
    None
}

fn find_credit_card(text: &str) -> Option<String> {
    let mut digits = String::with_capacity(20);
    for c in text.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else if c != '-' && c != ' ' {
            if digits.len() >= 13 && digits.len() <= 19 && luhn_check(&digits) {
                return Some(digits);
            }
            digits.clear();
        }
    }
    if digits.len() >= 13 && digits.len() <= 19 && luhn_check(&digits) {
        return Some(digits);
    }
    None
}

fn luhn_check(number: &str) -> bool {
    let mut sum = 0;
    let mut alternate = false;
    for c in number.chars().rev() {
        if let Some(mut digit) = c.to_digit(10) {
            if alternate {
                digit *= 2;
                if digit > 9 {
                    digit -= 9;
                }
            }
            sum += digit;
            alternate = !alternate;
        } else {
            return false;
        }
    }
    sum % 10 == 0
}
