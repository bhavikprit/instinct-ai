/**
 * Reflex C ABI: SIMD-Accelerated Batch Vector Operations for HNSW (Phase 30).
 * High-throughput batch distance kernels for Hierarchical Navigable Small World graphs.
 * Pure C99 with ARM NEON and x86_64 AVX2/FMA intrinsics.
 * Zero external dependencies.
 */

#ifndef REFLEX_HNSW_H
#define REFLEX_HNSW_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include "reflex_simd.h"

/**
 * Computes dot products between a single query vector and a contiguous array of N vectors.
 * query: float array of length dim
 * vectors: contiguous float array of length (count * dim)
 * count: number of vectors in array
 * dim: vector dimension (e.g. 384)
 * out_dots: output float array of length count
 */
void reflex_batch_dot_product_f32(
    const float* query,
    const float* vectors,
    int count,
    int dim,
    float* out_dots
);

/**
 * Computes cosine similarities between a single query vector and a contiguous array of N vectors.
 * If vectors are pre-normalized, this directly evaluates batch dot products.
 */
void reflex_batch_cosine_similarity_f32(
    const float* query,
    const float* vectors,
    int count,
    int dim,
    float* out_sims
);

/**
 * Computes batch quantized similarities between INT8 query and contiguous array of N INT8 vectors.
 * query: int8_t array of length dim
 * query_scale: scale factor for query
 * vectors: contiguous int8_t array of length (count * dim)
 * scales: float array of length count containing scale for each vector
 * count: number of vectors
 * dim: vector dimension
 * out_sims: output float array of reconstructed similarities
 */
void reflex_batch_similarity_i8(
    const int8_t* query,
    float query_scale,
    const int8_t* vectors,
    const float* scales,
    int count,
    int dim,
    float* out_sims
);

#ifdef __cplusplus
}
#endif

#endif /* REFLEX_HNSW_H */
