/**
 * Reflex C ABI: Native Product Quantization (PQ) & Asymmetric Distance Computation (ADC) (Phase 31).
 * High-throughput sub-vector codebook lookup and memory compression.
 * Pure C99 with zero external dependencies.
 */

#ifndef REFLEX_PQ_H
#define REFLEX_PQ_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define REFLEX_PQ_METRIC_COSINE 0
#define REFLEX_PQ_METRIC_L2     1

/**
 * Computes the Asymmetric Distance Computation (ADC) Lookup Table (LUT)
 * between a query vector and codebook centroids.
 *
 * query: float array of length (M * d_sub)
 * centroids: contiguous float array of length (M * K * d_sub)
 * M: number of sub-vectors (e.g. 48)
 * d_sub: dimension per sub-vector (e.g. 8)
 * K: number of centroids per sub-quantizer (e.g. 256)
 * metric: 0 for cosine distance (1.0 - dot), 1 for squared Euclidean (L2)
 * out_lut: output float array of size (M * K)
 */
void reflex_compute_adc_lut_f32(
    const float* query,
    const float* centroids,
    int M,
    int d_sub,
    int K,
    int metric,
    float* out_lut
);

/**
 * Evaluates asymmetric distances for a contiguous batch of N quantized vectors
 * using the precomputed ADC Lookup Table (LUT).
 * Multiplier-free: consists purely of table lookups and vector accumulations.
 *
 * lut: float array of size (M * K)
 * codes: contiguous uint8_t array of size (count * M)
 * count: number of vectors in batch
 * M: number of sub-vectors (e.g. 48)
 * K: number of centroids per sub-quantizer (e.g. 256)
 * out_dists: output float array of length count
 */
void reflex_batch_adc_dist_u8(
    const float* lut,
    const uint8_t* codes,
    int count,
    int M,
    int K,
    float* out_dists
);

/**
 * Quantizes a single full-dimensional vector into M codebook indices.
 *
 * vector: float array of length (M * d_sub)
 * centroids: contiguous float array of length (M * K * d_sub)
 * M: number of sub-vectors (e.g. 48)
 * d_sub: dimension per sub-vector (e.g. 8)
 * K: number of centroids per sub-quantizer (e.g. 256)
 * metric: 0 for cosine distance, 1 for squared Euclidean
 * out_codes: output uint8_t array of length M
 */
void reflex_quantize_vector_pq(
    const float* vector,
    const float* centroids,
    int M,
    int d_sub,
    int K,
    int metric,
    uint8_t* out_codes
);

/**
 * Assigns each sub-vector of length d_sub to its closest centroid among K candidates.
 * Accelerates Lloyd's K-Means E-step by >80x.
 */
void reflex_assign_centroids_subvector(
    const float* sub_vectors,
    int count,
    const float* centroids,
    int K,
    int d_sub,
    int metric,
    int* out_assignments
);

/**
 * Computes residuals r = vector - centroid for a batch of vectors (Phase 32).
 * Accelerates IVF-PQ encoding.
 *
 * vectors: contiguous float array of shape (count * dim)
 * centroids: contiguous float array of shape (num_centroids * dim)
 * assignments: int array of shape count, where assignments[i] is the centroid index for vector i
 * count: number of vectors
 * dim: vector dimension (e.g. 384)
 * residuals_out: output float array of shape (count * dim)
 */
void reflex_compute_residuals(
    const float* vectors,
    const float* centroids,
    const int* assignments,
    int count,
    int dim,
    float* residuals_out
);

/**
 * Finds the nearest centroid for each vector in a batch using L2 or Cosine distance (Phase 32).
 * Accelerates coarse quantizer assignment in IVF-PQ.
 *
 * vectors: contiguous float array of shape (count * dim)
 * count: number of vectors
 * centroids: contiguous float array of shape (num_centroids * dim)
 * num_centroids: number of coarse centroids (e.g. 256)
 * dim: vector dimension (e.g. 384)
 * metric: 0 for cosine distance, 1 for squared L2 distance
 * out_assignments: output int array of length count
 */
void reflex_find_nearest_centroids(
    const float* vectors,
    int count,
    const float* centroids,
    int num_centroids,
    int dim,
    int metric,
    int* out_assignments
);

#ifdef __cplusplus
}
#endif

#endif /* REFLEX_PQ_H */
