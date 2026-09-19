/**
 * Reflex C ABI: Universal Machine-Native System-1 AI Runtime.
 * Pure C99, zero-dependency header for embedded systems, Go, Rust, and Python FFI.
 */

#ifndef REFLEX_H
#define REFLEX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "reflex_simd.h"

#define REFLEX_VERSION "0.2.0"
#define REFLEX_VECTOR_DIM 384
#define REFLEX_MAX_OPTIONS 32
#define REFLEX_MAX_STR_LEN 128

/**
 * Result of a Noul (Probabilistic Boolean) evaluation.
 */
typedef struct {
    float probability;
    int is_true;
    int is_false;
    int is_uncertain;
} reflex_noul_result_t;

/**
 * Result of a Choice (Dynamic Rubric Selection) evaluation.
 */
typedef struct {
    char selected[REFLEX_MAX_STR_LEN];
    float confidence;
    int num_options;
    char option_names[REFLEX_MAX_OPTIONS][REFLEX_MAX_STR_LEN];
    float distribution[REFLEX_MAX_OPTIONS];
} reflex_choice_result_t;

/**
 * Result of a continuous Score evaluation.
 */
typedef struct {
    float score;
    float confidence;
} reflex_score_result_t;

/**
 * Result of instant Guardrail evaluation.
 */
typedef struct {
    int is_safe;
    int blocked;
    float risk_score;
    char category[REFLEX_MAX_STR_LEN];
    char reason[REFLEX_MAX_STR_LEN * 2];
    float latency_us;
} reflex_guardrail_result_t;

/* --- Core Mathematical Operations --- */

/**
 * Encodes arbitrary text into an L2-normalized 384-dimensional dense vector.
 * Exactly matches Python and JavaScript SemanticVectorEncoder outputs.
 * 
 * @param text UTF-8 input string.
 * @param out_vec Pointer to a float array of at least REFLEX_VECTOR_DIM (384) elements.
 */
void reflex_encode_384(const char* text, float* out_vec);

/**
 * Computes dot product (cosine similarity) between two unit-normalized vectors.
 */
float reflex_cosine_similarity(const float* vec_a, const float* vec_b, int dim);

/* --- System-1 Decision Primitives --- */

/**
 * Evaluates a Noul boolean primitive in sub-10 microseconds.
 */
void reflex_evaluate_noul(
    const char* state,
    const char* instructions,
    float threshold,
    float temperature,
    reflex_noul_result_t* out
);

/**
 * Evaluates a Choice rubric primitive across multiple options.
 */
void reflex_evaluate_choice(
    const char* state,
    const char* instructions,
    const char** options,
    int num_options,
    float temperature,
    reflex_choice_result_t* out
);

/**
 * Evaluates continuous Score primitive [min_val, max_val].
 */
void reflex_evaluate_score(
    const char* state,
    const char* instructions,
    float min_val,
    float max_val,
    reflex_score_result_t* out
);

/* --- Instant Security Guardrails --- */

/**
 * Evaluates prompt injection, jailbreaks, PII, and credit card Luhn check in <1us.
 */
void reflex_guardrail_check(
    const char* text,
    reflex_guardrail_result_t* out
);

#ifdef __cplusplus
}
#endif

#endif /* REFLEX_H */
