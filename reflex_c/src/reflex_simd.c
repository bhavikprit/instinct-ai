/**
 * Reflex C ABI: Hardware-Accelerated Micro-Embedding SIMD Kernel (Phase 28).
 * Pure C99 implementation with ARM NEON and x86_64 AVX2/FMA vector intrinsics.
 * Zero external dependencies.
 */

#include "reflex_simd.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- Architecture & SIMD Header Inclusion --- */

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
    #include <arm_neon.h>
    #define REFLEX_HAVE_NEON 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__AVX2__)
        #include <immintrin.h>
        #define REFLEX_HAVE_AVX2 1
    #endif
#endif

/* --- Capability Detection --- */

void reflex_detect_simd_capabilities(reflex_simd_caps_t* caps) {
    if (!caps) return;
    memset(caps, 0, sizeof(reflex_simd_caps_t));

#if defined(__aarch64__) || defined(_M_ARM64)
    caps->has_neon = 1;
    caps->has_popcnt = 1;
    strncpy(caps->arch_name, "arm64_neon", sizeof(caps->arch_name) - 1);
#elif defined(REFLEX_HAVE_NEON)
    caps->has_neon = 1;
    caps->has_popcnt = 1;
    strncpy(caps->arch_name, "arm32_neon", sizeof(caps->arch_name) - 1);
#elif defined(REFLEX_HAVE_AVX2)
    caps->has_avx2 = 1;
    caps->has_fma = 1;
    caps->has_popcnt = 1;
    strncpy(caps->arch_name, "x86_64_avx2", sizeof(caps->arch_name) - 1);
#elif defined(__x86_64__) || defined(_M_X64)
    caps->has_popcnt = 1;
    strncpy(caps->arch_name, "x86_64_scalar", sizeof(caps->arch_name) - 1);
#else
    strncpy(caps->arch_name, "generic_scalar", sizeof(caps->arch_name) - 1);
#endif
}

/* --- 1. FP32 SIMD Vector Dot Product & Cosine Similarity --- */

float reflex_dot_product_f32_simd(const float* a, const float* b, int dim) {
    if (!a || !b || dim <= 0) return 0.0f;

#if defined(REFLEX_HAVE_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 15 < dim; i += 16) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }

    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

#if defined(__aarch64__)
    float total = vaddvq_f32(sum0);
#else
    float32x2_t r = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
    float total = vget_lane_f32(vpadd_f32(r, r), 0);
#endif

    for (; i < dim; i++) {
        total += a[i] * b[i];
    }
    return total;

#elif defined(REFLEX_HAVE_AVX2)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 15 < dim; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);

    __m128 vlow = _mm256_castps256_ps128(sum0);
    __m128 vhigh = _mm256_extractf128_ps(sum0, 1);
    __m128 v128 = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(v128);
    __m128 sums = _mm_add_ps(v128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float total = _mm_cvtss_f32(sums);

    for (; i < dim; i++) {
        total += a[i] * b[i];
    }
    return total;

#else
    /* Optimized 4-way unrolled scalar fallback */
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        s0 += (double)a[i] * (double)b[i];
        s1 += (double)a[i + 1] * (double)b[i + 1];
        s2 += (double)a[i + 2] * (double)b[i + 2];
        s3 += (double)a[i + 3] * (double)b[i + 3];
    }
    double total = s0 + s1 + s2 + s3;
    for (; i < dim; i++) {
        total += (double)a[i] * (double)b[i];
    }
    return (float)total;
#endif
}

float reflex_cosine_similarity_f32_simd(const float* a, const float* b, int dim) {
    return reflex_dot_product_f32_simd(a, b, dim);
}

/* --- 2. INT8 Symmetric Quantization & Vector Dot Product --- */

void reflex_quantize_i8(const float* in_vec, int8_t* out_vec, int dim, float* out_scale) {
    if (!in_vec || !out_vec || dim <= 0) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    float max_val = 0.0f;
    for (int i = 0; i < dim; i++) {
        float abs_v = fabsf(in_vec[i]);
        if (abs_v > max_val) {
            max_val = abs_v;
        }
    }

    if (max_val < 1e-9f) {
        memset(out_vec, 0, (size_t)dim);
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    float scale = max_val / 127.0f;
    float inv_scale = 127.0f / max_val;

    for (int i = 0; i < dim; i++) {
        float scaled = in_vec[i] * inv_scale;
        int val = (int)roundf(scaled);
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        out_vec[i] = (int8_t)val;
    }

    if (out_scale) {
        *out_scale = scale;
    }
}

int32_t reflex_dot_product_i8_simd(const int8_t* a, const int8_t* b, int dim) {
    if (!a || !b || dim <= 0) return 0;

#if defined(REFLEX_HAVE_NEON)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);

    int i = 0;
    for (; i + 15 < dim; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        int16x8_t prod_low = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t prod_high = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        acc0 = vpadalq_s16(acc0, prod_low);
        acc1 = vpadalq_s16(acc1, prod_high);
    }

    int32x4_t acc = vaddq_s32(acc0, acc1);

#if defined(__aarch64__)
    int32_t total = vaddvq_s32(acc);
#else
    int32_t total = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) +
                    vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
#endif

    for (; i < dim; i++) {
        total += (int32_t)a[i] * (int32_t)b[i];
    }
    return total;

#else
    /* Portable 4-way unrolled scalar fallback */
    int32_t total0 = 0, total1 = 0, total2 = 0, total3 = 0;
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        total0 += (int32_t)a[i] * (int32_t)b[i];
        total1 += (int32_t)a[i + 1] * (int32_t)b[i + 1];
        total2 += (int32_t)a[i + 2] * (int32_t)b[i + 2];
        total3 += (int32_t)a[i + 3] * (int32_t)b[i + 3];
    }
    int32_t total = total0 + total1 + total2 + total3;
    for (; i < dim; i++) {
        total += (int32_t)a[i] * (int32_t)b[i];
    }
    return total;
#endif
}

float reflex_quantized_similarity_i8(
    const int8_t* a,
    float scale_a,
    const int8_t* b,
    float scale_b,
    int dim
) {
    int32_t raw_dot = reflex_dot_product_i8_simd(a, b, dim);
    return (scale_a * scale_b) * (float)raw_dot;
}

/* --- 3. 1-Bit Binary Sign Quantization & Hamming Distance --- */

void reflex_binarize_384(const float* in_vec, uint64_t* out_bits) {
    if (!in_vec || !out_bits) return;

    for (int w = 0; w < 6; w++) {
        uint64_t word = 0;
        int base = w * 64;
        for (int b = 0; b < 64; b++) {
            if (in_vec[base + b] >= 0.0f) {
                word |= (1ULL << b);
            }
        }
        out_bits[w] = word;
    }
}

int reflex_hamming_distance_384(const uint64_t* a, const uint64_t* b) {
    if (!a || !b) return 384;

    int dist = 0;
    for (int w = 0; w < 6; w++) {
        uint64_t diff = a[w] ^ b[w];
#if defined(__GNUC__) || defined(__clang__)
        dist += __builtin_popcountll(diff);
#elif defined(_MSC_VER) && defined(_M_X64)
        dist += (int)__popcnt64(diff);
#else
        /* Portable Brian Kernighan bit count */
        while (diff) {
            diff &= (diff - 1);
            dist++;
        }
#endif
    }
    return dist;
}

float reflex_binary_similarity_384(const uint64_t* a, const uint64_t* b) {
    int dist = reflex_hamming_distance_384(a, b);
    double norm_dist = (double)dist / 384.0;
    return (float)cos(M_PI * norm_dist);
}

/* --- 4. 4-Bit Nibble Quantization --- */

void reflex_quantize_i4(const float* in_vec, uint8_t* out_packed, int dim, float* out_scale) {
    if (!in_vec || !out_packed || dim <= 0) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    float max_val = 0.0f;
    for (int i = 0; i < dim; i++) {
        float abs_v = fabsf(in_vec[i]);
        if (abs_v > max_val) max_val = abs_v;
    }

    if (max_val < 1e-9f) {
        memset(out_packed, 0, (size_t)(dim / 2));
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    float scale = max_val / 7.0f;
    float inv_scale = 7.0f / max_val;

    for (int i = 0; i < dim; i += 2) {
        int v0 = (int)roundf(in_vec[i] * inv_scale);
        if (v0 > 7) v0 = 7;
        if (v0 < -8) v0 = -8;

        int v1 = 0;
        if (i + 1 < dim) {
            v1 = (int)roundf(in_vec[i + 1] * inv_scale);
            if (v1 > 7) v1 = 7;
            if (v1 < -8) v1 = -8;
        }

        uint8_t nib0 = (uint8_t)(v0 & 0x0F);
        uint8_t nib1 = (uint8_t)(v1 & 0x0F);
        out_packed[i / 2] = (uint8_t)(nib0 | (nib1 << 4));
    }

    if (out_scale) *out_scale = scale;
}

int32_t reflex_dot_product_i4(const uint8_t* packed_w, const int8_t* q_vec, int dim) {
    if (!packed_w || !q_vec || dim <= 0) return 0;

    int32_t total = 0;
    for (int i = 0; i < dim; i += 2) {
        uint8_t byte = packed_w[i / 2];

        /* Arithmetic sign extension of 4-bit nibbles */
        int8_t w0 = (int8_t)((int8_t)(byte << 4) >> 4);
        int8_t w1 = (int8_t)((int8_t)byte >> 4);

        total += (int32_t)w0 * (int32_t)q_vec[i];
        if (i + 1 < dim) {
            total += (int32_t)w1 * (int32_t)q_vec[i + 1];
        }
    }
    return total;
}

float reflex_quantized_similarity_i4(
    const uint8_t* packed_w,
    float scale_w,
    const int8_t* q_vec,
    float scale_q,
    int dim
) {
    int32_t raw_dot = reflex_dot_product_i4(packed_w, q_vec, dim);
    return (scale_w * scale_q) * (float)raw_dot;
}
