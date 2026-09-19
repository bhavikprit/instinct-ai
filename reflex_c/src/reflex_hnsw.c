/**
 * Reflex C ABI: SIMD-Accelerated Batch Vector Operations for HNSW (Phase 30).
 * High-throughput batch distance kernels for Hierarchical Navigable Small World graphs.
 * Pure C99 with zero external dependencies.
 */

#include "reflex_hnsw.h"
#include <math.h>

void reflex_batch_dot_product_f32(
    const float* query,
    const float* vectors,
    int count,
    int dim,
    float* out_dots
) {
    if (!query || !vectors || !out_dots || count <= 0 || dim <= 0) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        const float* vec_i = vectors + (i * dim);
        out_dots[i] = reflex_dot_product_f32_simd(query, vec_i, dim);
    }
}

void reflex_batch_cosine_similarity_f32(
    const float* query,
    const float* vectors,
    int count,
    int dim,
    float* out_sims
) {
    if (!query || !vectors || !out_sims || count <= 0 || dim <= 0) {
        return;
    }

    // Compute norm of query once
    float norm_q = 0.0f;
    for (int d = 0; d < dim; ++d) {
        norm_q += query[d] * query[d];
    }
    norm_q = sqrtf(norm_q);

    // If query is unit vector (norm ~ 1.0), evaluate batch dot products directly
    if (fabsf(norm_q - 1.0f) < 1e-4f) {
        for (int i = 0; i < count; ++i) {
            const float* vec_i = vectors + (i * dim);
            out_sims[i] = reflex_dot_product_f32_simd(query, vec_i, dim);
        }
    } else {
        for (int i = 0; i < count; ++i) {
            const float* vec_i = vectors + (i * dim);
            float dot = reflex_dot_product_f32_simd(query, vec_i, dim);
            
            float norm_v = 0.0f;
            for (int d = 0; d < dim; ++d) {
                norm_v += vec_i[d] * vec_i[d];
            }
            norm_v = sqrtf(norm_v);
            float denom = norm_q * norm_v;
            out_sims[i] = (denom > 1e-9f) ? (dot / denom) : 0.0f;
        }
    }
}

void reflex_batch_similarity_i8(
    const int8_t* query,
    float query_scale,
    const int8_t* vectors,
    const float* scales,
    int count,
    int dim,
    float* out_sims
) {
    if (!query || !vectors || !scales || !out_sims || count <= 0 || dim <= 0) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        const int8_t* vec_i = vectors + (i * dim);
        int32_t raw_dot = reflex_dot_product_i8_simd(query, vec_i, dim);
        out_sims[i] = (float)raw_dot * (query_scale * scales[i]);
    }
}
