//! Zero-dependency 384-dimensional dense semantic subword vector encoder.
//! 100% mathematical parity with Python reflex.embeddings and JavaScript @reflex-ai/sdk.

pub const VECTOR_DIM: usize = 384;

/// Pure Rust RFC 1321 MD5 message-digest implementation with zero external dependencies.
#[derive(Clone, Debug)]
pub struct Md5 {
    state: [u32; 4],
    count: [u32; 2],
    buffer: [u8; 64],
}

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    pub fn new() -> Self {
        Self {
            state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
            count: [0, 0],
            buffer: [0u8; 64],
        }
    }

    pub fn update(&mut self, input: &[u8]) {
        let mut index = ((self.count[0] >> 3) & 0x3f) as usize;
        let input_len = input.len();

        let shift = (input_len as u32) << 3;
        self.count[0] = self.count[0].wrapping_add(shift);
        if self.count[0] < shift {
            self.count[1] = self.count[1].wrapping_add(1);
        }
        self.count[1] = self.count[1].wrapping_add((input_len as u32) >> 29);

        let part_len = 64 - index;
        let mut i = 0;

        if input_len >= part_len {
            self.buffer[index..index + part_len].copy_from_slice(&input[0..part_len]);
            let buf = self.buffer;
            self.transform(&buf);

            i = part_len;
            while i + 63 < input_len {
                let mut chunk = [0u8; 64];
                chunk.copy_from_slice(&input[i..i + 64]);
                self.transform(&chunk);
                i += 64;
            }
            index = 0;
        }

        if i < input_len {
            self.buffer[index..index + (input_len - i)].copy_from_slice(&input[i..input_len]);
        }
    }

    pub fn finalize(mut self) -> [u8; 16] {
        let mut bits = [0u8; 8];
        for i in 0..4 {
            bits[i] = ((self.count[0] >> (i * 8)) & 0xff) as u8;
            bits[i + 4] = ((self.count[1] >> (i * 8)) & 0xff) as u8;
        }

        let index = ((self.count[0] >> 3) & 0x3f) as usize;
        let pad_len = if index < 56 { 56 - index } else { 120 - index };
        let mut padding = [0u8; 64];
        padding[0] = 0x80;

        self.update(&padding[..pad_len]);
        self.update(&bits);

        let mut digest = [0u8; 16];
        for i in 0..4 {
            digest[i * 4] = (self.state[i] & 0xff) as u8;
            digest[i * 4 + 1] = ((self.state[i] >> 8) & 0xff) as u8;
            digest[i * 4 + 2] = ((self.state[i] >> 16) & 0xff) as u8;
            digest[i * 4 + 3] = ((self.state[i] >> 24) & 0xff) as u8;
        }
        digest
    }

    pub fn digest(data: &[u8]) -> [u8; 16] {
        let mut md5 = Self::new();
        md5.update(data);
        md5.finalize()
    }

    fn transform(&mut self, block: &[u8; 64]) {
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut x = [0u32; 16];

        for i in 0..16 {
            let j = i * 4;
            x[i] = (block[j] as u32)
                | ((block[j + 1] as u32) << 8)
                | ((block[j + 2] as u32) << 16)
                | ((block[j + 3] as u32) << 24);
        }

        #[inline(always)]
        fn f(x: u32, y: u32, z: u32) -> u32 {
            (x & y) | (!x & z)
        }
        #[inline(always)]
        fn g(x: u32, y: u32, z: u32) -> u32 {
            (x & z) | (y & !z)
        }
        #[inline(always)]
        fn h(x: u32, y: u32, z: u32) -> u32 {
            x ^ y ^ z
        }
        #[inline(always)]
        fn i_func(x: u32, y: u32, z: u32) -> u32 {
            y ^ (x | !z)
        }
        #[inline(always)]
        fn rotl(x: u32, n: u32) -> u32 {
            (x << n) | (x >> (32 - n))
        }

        macro_rules! step {
            ($func:ident, $a:expr, $b:expr, $c:expr, $d:expr, $k:expr, $s:expr, $t:expr) => {
                $a = $a.wrapping_add($func($b, $c, $d).wrapping_add(x[$k]).wrapping_add($t));
                $a = rotl($a, $s).wrapping_add($b);
            };
        }

        // Round 1
        step!(f, a, b, c, d, 0, 7, 0xd76aa478);
        step!(f, d, a, b, c, 1, 12, 0xe8c7b756);
        step!(f, c, d, a, b, 2, 17, 0x242070db);
        step!(f, b, c, d, a, 3, 22, 0xc1bdceee);
        step!(f, a, b, c, d, 4, 7, 0xf57c0faf);
        step!(f, d, a, b, c, 5, 12, 0x4787c62a);
        step!(f, c, d, a, b, 6, 17, 0xa8304613);
        step!(f, b, c, d, a, 7, 22, 0xfd469501);
        step!(f, a, b, c, d, 8, 7, 0x698098d8);
        step!(f, d, a, b, c, 9, 12, 0x8b44f7af);
        step!(f, c, d, a, b, 10, 17, 0xffff5bb1);
        step!(f, b, c, d, a, 11, 22, 0x895cd7be);
        step!(f, a, b, c, d, 12, 7, 0x6b901122);
        step!(f, d, a, b, c, 13, 12, 0xfd987193);
        step!(f, c, d, a, b, 14, 17, 0xa679438e);
        step!(f, b, c, d, a, 15, 22, 0x49b40821);

        // Round 2
        step!(g, a, b, c, d, 1, 5, 0xf61e2562);
        step!(g, d, a, b, c, 6, 9, 0xc040b340);
        step!(g, c, d, a, b, 11, 14, 0x265e5a51);
        step!(g, b, c, d, a, 0, 20, 0xe9b6c7aa);
        step!(g, a, b, c, d, 5, 5, 0xd62f105d);
        step!(g, d, a, b, c, 10, 9, 0x02441453);
        step!(g, c, d, a, b, 15, 14, 0xd8a1e681);
        step!(g, b, c, d, a, 4, 20, 0xe7d3fbc8);
        step!(g, a, b, c, d, 9, 5, 0x21e1cde6);
        step!(g, d, a, b, c, 14, 9, 0xc33707d6);
        step!(g, c, d, a, b, 3, 14, 0xf4d50d87);
        step!(g, b, c, d, a, 8, 20, 0x455a14ed);
        step!(g, a, b, c, d, 13, 5, 0xa9e3e905);
        step!(g, d, a, b, c, 2, 9, 0xfcefa3f8);
        step!(g, c, d, a, b, 7, 14, 0x676f02d9);
        step!(g, b, c, d, a, 12, 20, 0x8d2a4c8a);

        // Round 3
        step!(h, a, b, c, d, 5, 4, 0xfffa3942);
        step!(h, d, a, b, c, 8, 11, 0x8771f681);
        step!(h, c, d, a, b, 11, 16, 0x6d9d6122);
        step!(h, b, c, d, a, 14, 23, 0xfde5380c);
        step!(h, a, b, c, d, 1, 4, 0xa4beea44);
        step!(h, d, a, b, c, 4, 11, 0x4bdecfa9);
        step!(h, c, d, a, b, 7, 16, 0xf6bb4b60);
        step!(h, b, c, d, a, 10, 23, 0xbebfbc70);
        step!(h, a, b, c, d, 13, 4, 0x289b7ec6);
        step!(h, d, a, b, c, 0, 11, 0xeaa127fa);
        step!(h, c, d, a, b, 3, 16, 0xd4ef3085);
        step!(h, b, c, d, a, 6, 23, 0x04881d05);
        step!(h, a, b, c, d, 9, 4, 0xd9d4d039);
        step!(h, d, a, b, c, 12, 11, 0xe6db99e5);
        step!(h, c, d, a, b, 15, 16, 0x1fa27cf8);
        step!(h, b, c, d, a, 2, 23, 0xc4ac5665);

        // Round 4
        step!(i_func, a, b, c, d, 0, 6, 0xf4292244);
        step!(i_func, d, a, b, c, 7, 10, 0x432aff97);
        step!(i_func, c, d, a, b, 14, 15, 0xab9423a7);
        step!(i_func, b, c, d, a, 5, 21, 0xfc93a039);
        step!(i_func, a, b, c, d, 12, 6, 0x655b59c3);
        step!(i_func, d, a, b, c, 3, 10, 0x8f0ccc92);
        step!(i_func, c, d, a, b, 10, 15, 0xffeff47d);
        step!(i_func, b, c, d, a, 1, 21, 0x85845dd1);
        step!(i_func, a, b, c, d, 8, 6, 0x6fa87e4f);
        step!(i_func, d, a, b, c, 15, 10, 0xfe2ce6e0);
        step!(i_func, c, d, a, b, 6, 15, 0xa3014314);
        step!(i_func, b, c, d, a, 13, 21, 0x4e0811a1);
        step!(i_func, a, b, c, d, 4, 6, 0xf7537e82);
        step!(i_func, d, a, b, c, 11, 10, 0xbd3af235);
        step!(i_func, c, d, a, b, 2, 15, 0x2ad7d2bb);
        step!(i_func, b, c, d, a, 9, 21, 0xeb86d391);

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }
}

/// Computes the dot product (cosine similarity) between two unit-normalized vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Zero-dependency 384-dimensional dense semantic subword vector encoder.
#[derive(Clone, Debug, Default)]
pub struct SemanticVectorEncoder;

impl SemanticVectorEncoder {
    pub fn new() -> Self {
        Self
    }

    /// Encodes arbitrary text into an L2-normalized 384-dimensional vector.
    pub fn encode(&self, text: &str) -> [f32; VECTOR_DIM] {
        let mut vec = [0.0f32; VECTOR_DIM];
        let lower = text.to_lowercase();
        
        // Extract words matching \w+
        let tokens: Vec<&str> = lower
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|s| !s.is_empty())
            .collect();

        if tokens.is_empty() {
            return vec;
        }

        // 1. Unigram & Bigram hashing
        for (i, &tok) in tokens.iter().enumerate() {
            Self::hash_into_vector(tok, &mut vec, 1.0);
            if i + 1 < tokens.len() {
                let bigram = format!("{}_{}", tok, tokens[i + 1]);
                Self::hash_into_vector(&bigram, &mut vec, 1.4);
            }
        }

        // 2. Subword 3-char n-grams for typo & morphology resilience
        for &tok in &tokens {
            if tok.len() >= 3 {
                for j in 0..=tok.len() - 3 {
                    let sub = &tok[j..j + 3];
                    let sub_key = format!("sub_{}", sub);
                    Self::hash_into_vector(&sub_key, &mut vec, 0.5);
                }
            }
        }

        // 3. L2 Unit Normalization
        let sum_sq: f32 = vec.iter().map(|x| x * x).sum();
        let norm = sum_sq.sqrt();
        if norm > 1e-9 {
            for val in vec.iter_mut() {
                *val /= norm;
            }
        }

        vec
    }

    fn hash_into_vector(token: &str, vec: &mut [f32; VECTOR_DIM], weight: f32) {
        let digest = Md5::digest(token.as_bytes());
        // Extract first 6 bytes as big-endian 48-bit integer
        let h = ((digest[0] as u64) << 40)
            | ((digest[1] as u64) << 32)
            | ((digest[2] as u64) << 24)
            | ((digest[3] as u64) << 16)
            | ((digest[4] as u64) << 8)
            | (digest[5] as u64);

        let idx = (h % (VECTOR_DIM as u64)) as usize;
        let sign = if (h >> 16) % 2 == 0 { 1.0f32 } else { -1.0f32 };
        vec[idx] += sign * weight;
    }
}
