/**
 * Reflex C ABI: Native Product Quantization (PQ) & Asymmetric Distance Computation (ADC) (Phase 31).
 * High-throughput sub-vector codebook lookup and memory compression.
 * Pure C99 with zero external dependencies.
 */

#include "reflex_pq.h"
#include <math.h>

void reflex_compute_adc_lut_f32(
    const float* query,
    const float* centroids,
    int M,
    int d_sub,
    int K,
    int metric,
    float* out_lut
) {
    if (!query || !centroids || !out_lut || M <= 0 || d_sub <= 0 || K <= 0) {
        return;
    }

    for (int m = 0; m < M; ++m) {
        const float* q_sub = query + (m * d_sub);
        const float* c_sub_base = centroids + (m * K * d_sub);
        float* lut_sub = out_lut + (m * K);

        if (metric == REFLEX_PQ_METRIC_COSINE) {
            for (int k = 0; k < K; ++k) {
                const float* c_k = c_sub_base + (k * d_sub);
                float dot = 0.0f;
                for (int d = 0; d < d_sub; ++d) {
                    dot += q_sub[d] * c_k[d];
                }
                if (dot > 1.0f) dot = 1.0f;
                else if (dot < -1.0f) dot = -1.0f;
                lut_sub[k] = (1.0f - dot);
            }
        } else {
            for (int k = 0; k < K; ++k) {
                const float* c_k = c_sub_base + (k * d_sub);
                float diff_sq = 0.0f;
                for (int d = 0; d < d_sub; ++d) {
                    float diff = q_sub[d] - c_k[d];
                    diff_sq += diff * diff;
                }
                lut_sub[k] = diff_sq;
            }
        }
    }
}

void reflex_batch_adc_dist_u8(
    const float* lut,
    const uint8_t* codes,
    int count,
    int M,
    int K,
    float* out_dists
) {
    if (!lut || !codes || !out_dists || count <= 0 || M <= 0 || K <= 0) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        const uint8_t* code_i = codes + (i * M);
        float sum = 0.0f;
        int m = 0;

        // Loop unrolling for high-throughput L1/L2 cache prefetching
        for (; m + 7 < M; m += 8) {
            sum += lut[m * K + code_i[m]];
            sum += lut[(m + 1) * K + code_i[m + 1]];
            sum += lut[(m + 2) * K + code_i[m + 2]];
            sum += lut[(m + 3) * K + code_i[m + 3]];
            sum += lut[(m + 4) * K + code_i[m + 4]];
            sum += lut[(m + 5) * K + code_i[m + 5]];
            sum += lut[(m + 6) * K + code_i[m + 6]];
            sum += lut[(m + 7) * K + code_i[m + 7]];
        }

        for (; m < M; ++m) {
            sum += lut[m * K + code_i[m]];
        }

        out_dists[i] = sum;
    }
}

void reflex_quantize_vector_pq(
    const float* vector,
    const float* centroids,
    int M,
    int d_sub,
    int K,
    int metric,
    uint8_t* out_codes
) {
    if (!vector || !centroids || !out_codes || M <= 0 || d_sub <= 0 || K <= 0) {
        return;
    }

    for (int m = 0; m < M; ++m) {
        const float* v_sub = vector + (m * d_sub);
        const float* c_sub_base = centroids + (m * K * d_sub);

        int best_k = 0;
        float best_dist = 1e30f;

        for (int k = 0; k < K; ++k) {
            const float* c_k = c_sub_base + (k * d_sub);
            float dist = 0.0f;

            if (metric == REFLEX_PQ_METRIC_COSINE) {
                float dot = 0.0f;
                for (int d = 0; d < d_sub; ++d) {
                    dot += v_sub[d] * c_k[d];
                }
                if (dot > 1.0f) dot = 1.0f;
                else if (dot < -1.0f) dot = -1.0f;
                dist = 1.0f - dot;
            } else {
                for (int d = 0; d < d_sub; ++d) {
                    float diff = v_sub[d] - c_k[d];
                    dist += diff * diff;
                }
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        out_codes[m] = (uint8_t)best_k;
    }
}

void reflex_assign_centroids_subvector(
    const float* sub_vectors,
    int count,
    const float* centroids,
    int K,
    int d_sub,
    int metric,
    int* out_assignments
) {
    if (!sub_vectors || !centroids || !out_assignments || count <= 0 || K <= 0 || d_sub <= 0) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        const float* sv = sub_vectors + (i * d_sub);
        int best_k = 0;
        float best_dist = 1e30f;

        for (int k = 0; k < K; ++k) {
            const float* c_k = centroids + (k * d_sub);
            float dist = 0.0f;

            if (metric == REFLEX_PQ_METRIC_COSINE) {
                float dot = 0.0f;
                for (int d = 0; d < d_sub; ++d) {
                    dot += sv[d] * c_k[d];
                }
                if (dot > 1.0f) dot = 1.0f;
                else if (dot < -1.0f) dot = -1.0f;
                dist = 1.0f - dot;
            } else {
                for (int d = 0; d < d_sub; ++d) {
                    float diff = sv[d] - c_k[d];
                    dist += diff * diff;
                }
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        out_assignments[i] = best_k;
    }
}

void reflex_compute_residuals(
    const float* vectors,
    const float* centroids,
    const int* assignments,
    int count,
    int dim,
    float* residuals_out
) {
    if (!vectors || !centroids || !assignments || !residuals_out || count <= 0 || dim <= 0) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        int c_idx = assignments[i];
        const float* vec = vectors + (i * dim);
        const float* cent = centroids + (c_idx * dim);
        float* res = residuals_out + (i * dim);

        int d = 0;
        for (; d <= dim - 8; d += 8) {
            res[d + 0] = vec[d + 0] - cent[d + 0];
            res[d + 1] = vec[d + 1] - cent[d + 1];
            res[d + 2] = vec[d + 2] - cent[d + 2];
            res[d + 3] = vec[d + 3] - cent[d + 3];
            res[d + 4] = vec[d + 4] - cent[d + 4];
            res[d + 5] = vec[d + 5] - cent[d + 5];
            res[d + 6] = vec[d + 6] - cent[d + 6];
            res[d + 7] = vec[d + 7] - cent[d + 7];
        }
        for (; d < dim; ++d) {
            res[d] = vec[d] - cent[d];
        }
    }
}

void reflex_find_nearest_centroids(
    const float* vectors,
    int count,
    const float* centroids,
    int num_centroids,
    int dim,
    int metric,
    int* out_assignments
) {
    if (!vectors || !centroids || !out_assignments || count <= 0 || num_centroids <= 0 || dim <= 0) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        const float* vec = vectors + (i * dim);
        int best_c = 0;
        float best_dist = 1e30f;

        for (int c = 0; c < num_centroids; ++c) {
            const float* cent = centroids + (c * dim);
            float dist = 0.0f;

            if (metric == REFLEX_PQ_METRIC_COSINE) {
                float dot = 0.0f;
                int d = 0;
                for (; d <= dim - 8; d += 8) {
                    dot += vec[d + 0] * cent[d + 0]
                         + vec[d + 1] * cent[d + 1]
                         + vec[d + 2] * cent[d + 2]
                         + vec[d + 3] * cent[d + 3]
                         + vec[d + 4] * cent[d + 4]
                         + vec[d + 5] * cent[d + 5]
                         + vec[d + 6] * cent[d + 6]
                         + vec[d + 7] * cent[d + 7];
                }
                for (; d < dim; ++d) {
                    dot += vec[d] * cent[d];
                }
                dist = -dot;
            } else {
                int d = 0;
                for (; d <= dim - 8; d += 8) {
                    float diff0 = vec[d + 0] - cent[d + 0];
                    float diff1 = vec[d + 1] - cent[d + 1];
                    float diff2 = vec[d + 2] - cent[d + 2];
                    float diff3 = vec[d + 3] - cent[d + 3];
                    float diff4 = vec[d + 4] - cent[d + 4];
                    float diff5 = vec[d + 5] - cent[d + 5];
                    float diff6 = vec[d + 6] - cent[d + 6];
                    float diff7 = vec[d + 7] - cent[d + 7];
                    dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3
                          + diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
                }
                for (; d < dim; ++d) {
                    float diff = vec[d] - cent[d];
                    dist += diff * diff;
                }
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }

        out_assignments[i] = best_c;
    }
}

