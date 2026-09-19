/**
 * Reflex C ABI: Hardware-Accelerated Micro-Embedding SIMD Kernel (Phase 28).
 * Pure C99 with ARM NEON and x86_64 AVX2/FMA intrinsics + INT8 & 1-bit quantization.
 * Zero external dependencies.
 */

#ifndef REFLEX_SIMD_H
#define REFLEX_SIMD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/**
 * CPU hardware instruction set capabilities.
 */
typedef struct {
    int has_neon;
    int has_avx2;
    int has_avx512;
    int has_fma;
    int has_popcnt;
    char arch_name[32];
} reflex_simd_caps_t;

/* Capability Detection */
void reflex_detect_simd_capabilities(reflex_simd_caps_t* caps);

/* --- 1. FP32 SIMD Vector Math --- */

/**
 * High-performance vectorized dot product of two float arrays.
 * Uses ARM NEON (vfmaq_f32) or AVX2/FMA (_mm256_fmadd_ps) when available.
 */
float reflex_dot_product_f32_simd(const float* a, const float* b, int dim);

/**
 * Vectorized cosine similarity of two float vectors.
 */
float reflex_cosine_similarity_f32_simd(const float* a, const float* b, int dim);

/* --- 2. INT8 Symmetric Quantization & Vector Dot Product --- */

/**
 * Quantizes an FP32 vector to signed 8-bit integers [-127, 127] with scale factor s.
 * v_i approx s * q_i, where s = max(|v_i|) / 127.0.
 */
void reflex_quantize_i8(const float* in_vec, int8_t* out_vec, int dim, float* out_scale);

/**
 * Computes dot product between two int8 vectors using SIMD (vdotq_s32 / _mm256_maddubs_epi16).
 * Returns raw integer accumulation.
 */
int32_t reflex_dot_product_i8_simd(const int8_t* a, const int8_t* b, int dim);

/**
 * Reconstructed cosine similarity from two INT8 quantized vectors:
 * dot = (scale_a * scale_b) * dot_product_i8(a, b).
 */
float reflex_quantized_similarity_i8(
    const int8_t* a,
    float scale_a,
    const int8_t* b,
    float scale_b,
    int dim
);

/* --- 3. 1-Bit Binary Sign Quantization & Hamming Distance --- */

/**
 * Binarizes a 384-dimensional vector into 384 bits (6 x uint64_t words = 48 bytes).
 * Bit i is 1 if in_vec[i] >= 0, else 0.
 */
void reflex_binarize_384(const float* in_vec, uint64_t* out_bits);

/**
 * Computes Hamming distance between two 384-bit binary vectors using POPCOUNT.
 * Returns number of mismatched signs [0, 384].
 */
int reflex_hamming_distance_384(const uint64_t* a, const uint64_t* b);

/**
 * Approximates cosine similarity from Hamming distance:
 * sim approx cos(pi * hamming / 384.0).
 */
float reflex_binary_similarity_384(const uint64_t* a, const uint64_t* b);

/* --- 4. 4-Bit Nibble Quantization --- */

/**
 * Quantizes an FP32 vector into packed 4-bit signed integers [-8, 7].
 * Two values per byte: low nibble = vec[2i], high nibble = vec[2i+1].
 * 384 dimensions compress to 192 bytes.
 */
void reflex_quantize_i4(const float* in_vec, uint8_t* out_packed, int dim, float* out_scale);

/**
 * Computes dot product between packed 4-bit weights and an INT8 input vector.
 */
int32_t reflex_dot_product_i4(const uint8_t* packed_w, const int8_t* q_vec, int dim);

/**
 * Reconstructed similarity from 4-bit weights and INT8 vector.
 */
float reflex_quantized_similarity_i4(
    const uint8_t* packed_w,
    float scale_w,
    const int8_t* q_vec,
    float scale_q,
    int dim
);

#ifdef __cplusplus
}
#endif

#endif /* REFLEX_SIMD_H */
