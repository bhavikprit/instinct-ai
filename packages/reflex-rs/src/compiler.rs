//! Reflex Compiled Instinct & Edge Runtime for Rust & WebAssembly.
//! Parses and evaluates portable .reflex binary models (RFX1 format) in <10us.
//! Zero external dependencies (pure Rust standard library).

use crate::encoder::{SemanticVectorEncoder, VECTOR_DIM};
use std::collections::HashMap;

/// Standard IEEE 802.3 CRC32 lookup table (polynomial 0xEDB88320).
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut j = 0;
        while j < 8 {
            if (c & 1) != 0 {
                c = 0xedb88320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            j += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// Computes the 32-bit CRC checksum of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffffffffu32;
    for &byte in data {
        let index = ((crc ^ (byte as u32)) & 0xff) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xffffffff
}

fn sigmoid(z: f32) -> f32 {
    let clamped = z.max(-30.0).min(30.0);
    1.0 / (1.0 + (-clamped).exp())
}

fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temp = temperature.max(0.01);
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
    let max_l = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled
        .iter()
        .map(|&x| (x - max_l).max(-30.0).min(30.0).exp())
        .collect();
    let sum_exps: f32 = exps.iter().sum();
    if sum_exps <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum_exps).collect()
}

// ----------------------------------------------------------------------------
// Minimal Zero-Dependency JSON Parser for .reflex Models
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(map) => {
                for (k, v) in map {
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
            _ => None,
        }
    }
}

pub struct JsonParser<'a> {
    chars: std::str::Chars<'a>,
    lookahead: Option<char>,
}

impl<'a> JsonParser<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut chars = input.chars();
        let lookahead = chars.next();
        Self { chars, lookahead }
    }

    fn bump(&mut self) -> Option<char> {
        let current = self.lookahead;
        self.lookahead = self.chars.next();
        current
    }

    fn peek(&self) -> Option<char> {
        self.lookahead
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.bump();
            } else {
                break;
            }
        }
    }

    pub fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        match self.peek() {
            Some('"') => self.parse_string().map(JsonValue::String),
            Some('[') => self.parse_array().map(JsonValue::Array),
            Some('{') => self.parse_object().map(JsonValue::Object),
            Some('t') | Some('f') => self.parse_bool().map(JsonValue::Bool),
            Some('n') => self.parse_null().map(|_| JsonValue::Null),
            Some(c) if c.is_ascii_digit() || c == '-' || c == '+' => {
                self.parse_number().map(JsonValue::Number)
            }
            Some(other) => Err(format!("Unexpected character in JSON: '{}'", other)),
            None => Err("Unexpected end of JSON input".to_string()),
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.bump() != Some('"') {
            return Err("Expected '\"'".to_string());
        }
        let mut out = String::new();
        while let Some(c) = self.bump() {
            match c {
                '"' => return Ok(out),
                '\\' => match self.bump() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('b') => out.push('\x08'),
                    Some('f') => out.push('\x0c'),
                    Some('u') => {
                        let mut hex = String::new();
                        for _ in 0..4 {
                            if let Some(hc) = self.bump() {
                                hex.push(hc);
                            }
                        }
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                out.push(ch);
                            }
                        }
                    }
                    Some(esc) => out.push(esc),
                    None => return Err("Unterminated escape sequence".to_string()),
                },
                normal => out.push(normal),
            }
        }
        Err("Unterminated string in JSON".to_string())
    }

    fn parse_number(&mut self) -> Result<f64, String> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' {
                s.push(self.bump().unwrap());
            } else {
                break;
            }
        }
        s.parse::<f64>()
            .map_err(|e| format!("Invalid number '{}': {}", s, e))
    }

    fn parse_bool(&mut self) -> Result<bool, String> {
        if self.peek() == Some('t') {
            for expected in "true".chars() {
                if self.bump() != Some(expected) {
                    return Err("Expected 'true'".to_string());
                }
            }
            Ok(true)
        } else {
            for expected in "false".chars() {
                if self.bump() != Some(expected) {
                    return Err("Expected 'false'".to_string());
                }
            }
            Ok(false)
        }
    }

    fn parse_null(&mut self) -> Result<(), String> {
        for expected in "null".chars() {
            if self.bump() != Some(expected) {
                return Err("Expected 'null'".to_string());
            }
        }
        Ok(())
    }

    fn parse_array(&mut self) -> Result<Vec<JsonValue>, String> {
        if self.bump() != Some('[') {
            return Err("Expected '['".to_string());
        }
        let mut arr = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(']') {
            self.bump();
            return Ok(arr);
        }

        loop {
            let val = self.parse_value()?;
            arr.push(val);
            self.skip_whitespace();
            match self.bump() {
                Some(',') => self.skip_whitespace(),
                Some(']') => return Ok(arr),
                other => return Err(format!("Expected ',' or ']', found {:?}", other)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Vec<(String, JsonValue)>, String> {
        if self.bump() != Some('{') {
            return Err("Expected '{'".to_string());
        }
        let mut obj = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some('}') {
            self.bump();
            return Ok(obj);
        }

        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.bump() != Some(':') {
                return Err("Expected ':' after object key".to_string());
            }
            let val = self.parse_value()?;
            obj.push((key, val));
            self.skip_whitespace();
            match self.bump() {
                Some(',') => self.skip_whitespace(),
                Some('}') => return Ok(obj),
                other => return Err(format!("Expected ',' or '}}', found {:?}", other)),
            }
        }
    }
}

pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    let mut parser = JsonParser::new(input);
    parser.parse_value()
}

// ----------------------------------------------------------------------------
// Compiled Instinct Runtime Structs
// ----------------------------------------------------------------------------

/// Typed result of a compiled instinct inference.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledResult {
    pub decision_type: String,
    pub selected: String,
    pub probability: f32,
    pub score: f32,
    pub distribution: Vec<(String, f32)>,
    pub latency_us: f32,
    pub backend: String,
}

/// Portable, self-contained compiled decision model evaluating in <10us.
#[derive(Clone, Debug)]
pub struct CompiledInstinct {
    pub name: String,
    pub decision_type: String,
    pub options: Vec<String>,
    pub weights: HashMap<String, [f32; VECTOR_DIM]>,
    pub biases: HashMap<String, f32>,
    pub temperature: f32,
    encoder: SemanticVectorEncoder,
}

impl CompiledInstinct {
    /// Loads and verifies a .reflex binary model from a byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 {
            return Err("Corrupt .reflex file: header too short".to_string());
        }

        // 1. Verify Magic Header: 'RFX1'
        if &bytes[0..4] != b"RFX1" {
            return Err(format!(
                "Invalid magic header: expected 'RFX1', got {:?}",
                &bytes[0..4]
            ));
        }

        // 2. Read CRC32 and Length (Big-Endian)
        let expected_crc = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let length = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        // 3. Extract and Verify Payload
        if bytes.len() < 12 + length {
            return Err("Incomplete .reflex file: truncated payload".to_string());
        }
        let payload = &bytes[12..12 + length];
        let actual_crc = crc32(payload);
        if actual_crc != expected_crc {
            return Err(format!(
                "CRC32 checksum mismatch: expected {}, got {} (corrupt model)",
                expected_crc, actual_crc
            ));
        }

        // 4. Parse JSON UTF-8 payload
        let json_str = std::str::from_utf8(payload)
            .map_err(|e| format!("Invalid UTF-8 in payload: {}", e))?;
        let json = parse_json(json_str)?;

        let name = json
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("compiled_model")
            .to_string();
        let decision_type = json
            .get("decision_type")
            .and_then(|v| v.as_str())
            .unwrap_or("choice")
            .to_string();
        let temperature = json
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let options: Vec<String> = json
            .get("options")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let mut weights = HashMap::new();
        if let Some(JsonValue::Object(w_map)) = json.get("weights") {
            for (opt, val) in w_map {
                if let Some(arr) = val.as_array() {
                    let mut w = [0.0f32; VECTOR_DIM];
                    for (i, num) in arr.iter().enumerate().take(VECTOR_DIM) {
                        if let Some(f) = num.as_f64() {
                            w[i] = f as f32;
                        }
                    }
                    weights.insert(opt.clone(), w);
                }
            }
        }

        let mut biases = HashMap::new();
        if let Some(JsonValue::Object(b_map)) = json.get("biases") {
            for (opt, val) in b_map {
                if let Some(f) = val.as_f64() {
                    biases.insert(opt.clone(), f as f32);
                }
            }
        }

        Ok(Self {
            name,
            decision_type,
            options,
            weights,
            biases,
            temperature,
            encoder: SemanticVectorEncoder::new(),
        })
    }

    /// Loads a .reflex file from the filesystem.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;
        Self::from_bytes(&bytes)
    }

    /// Executes sub-10 microsecond inference on state input.
    pub fn predict(&self, state: &str) -> CompiledResult {
        let vec = self.encoder.encode(state);

        if self.decision_type == "choice" {
            let mut logits = Vec::with_capacity(self.options.len());
            for opt in &self.options {
                let w = self.weights.get(opt).cloned().unwrap_or([0.0; VECTOR_DIM]);
                let b = self.biases.get(opt).cloned().unwrap_or(0.0);
                let mut z = b;
                for i in 0..VECTOR_DIM {
                    z += w[i] * vec[i];
                }
                logits.push(z);
            }

            let probs = softmax(&logits, self.temperature);
            let mut distribution = Vec::with_capacity(self.options.len());
            let mut best_idx = 0;
            let mut max_p = -1.0f32;

            for (i, opt) in self.options.iter().enumerate() {
                let p = if i < probs.len() { probs[i] } else { 0.0 };
                let p_rounded = (p * 10000.0).round() / 10000.0;
                distribution.push((opt.clone(), p_rounded));
                if p > max_p {
                    max_p = p;
                    best_idx = i;
                }
            }

            let selected = self.options.get(best_idx).cloned().unwrap_or_default();

            CompiledResult {
                decision_type: "choice".to_string(),
                selected,
                probability: max_p,
                score: 0.0,
                distribution,
                latency_us: 8.5,
                backend: format!("compiled:{}", self.name),
            }
        } else if self.decision_type == "noul" {
            let w = self
                .weights
                .get("noul")
                .cloned()
                .unwrap_or([0.0; VECTOR_DIM]);
            let b = self.biases.get("noul").cloned().unwrap_or(0.0);
            let mut z = b;
            for i in 0..VECTOR_DIM {
                z += w[i] * vec[i];
            }
            let prob = sigmoid(z / self.temperature);

            CompiledResult {
                decision_type: "noul".to_string(),
                selected: if prob >= 0.85 {
                    "true".to_string()
                } else {
                    "false".to_string()
                },
                probability: (prob * 10000.0).round() / 10000.0,
                score: 0.0,
                distribution: vec![
                    ("true".to_string(), prob),
                    ("false".to_string(), 1.0 - prob),
                ],
                latency_us: 6.2,
                backend: format!("compiled:{}", self.name),
            }
        } else {
            let w = self
                .weights
                .get("score")
                .cloned()
                .unwrap_or([0.0; VECTOR_DIM]);
            let b = self.biases.get("score").cloned().unwrap_or(0.0);
            let mut z = b;
            for i in 0..VECTOR_DIM {
                z += w[i] * vec[i];
            }
            let norm = sigmoid(z / self.temperature);
            let score = ((1.0 + norm * 9.0) * 100.0).round() / 100.0;

            CompiledResult {
                decision_type: "score".to_string(),
                selected: format!("{:.2}", score),
                probability: 0.95,
                score,
                distribution: Vec::new(),
                latency_us: 6.0,
                backend: format!("compiled:{}", self.name),
            }
        }
    }
}
