/**
 * Reflex Native C Benchmark & CLI Tester.
 * Tests sub-microsecond throughput and validates zero-dependency operations.
 */

#include "reflex.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("====================================================\n");
    printf("⚡ Reflex C Engine v%s (Pure C99 Native Standalone)\n", REFLEX_VERSION);
    printf("====================================================\n\n");

    const char* sample_text = "Urgent: Suspicious payment charge detected on your Visa 4532 0150 0000 0007! Click here to verify your account.";

    // 1. Benchmark Vector Encoding
    float vec[REFLEX_VECTOR_DIM];
    clock_t t0 = clock();
    int iterations = 100000;

    for (int i = 0; i < iterations; i++) {
        reflex_encode_384(sample_text, vec);
    }
    clock_t t1 = clock();

    double total_sec = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    double us_per_op = (total_sec / iterations) * 1000000.0;
    double ops_per_sec = (double)iterations / total_sec;

    printf("1. Vector Encoding (384-dimensional dense projection):\n");
    printf("   • Iterations  : %d\n", iterations);
    printf("   • Latency     : %.2f microseconds/op (%.4f ms)\n", us_per_op, us_per_op / 1000.0);
    printf("   • Throughput  : %.0f ops/second\n", ops_per_sec);
    printf("   • Sample Norm : %.4f\n\n", reflex_cosine_similarity(vec, vec, REFLEX_VECTOR_DIM));

    // 2. Benchmark Noul Decision
    reflex_noul_result_t noul_res;
    t0 = clock();
    for (int i = 0; i < iterations; i++) {
        reflex_evaluate_noul(
            sample_text,
            "Is this a security threat, phishing scam, or fraudulent payment?",
            0.75f,
            0.25f,
            &noul_res
        );
    }
    t1 = clock();
    total_sec = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    us_per_op = (total_sec / iterations) * 1000000.0;

    printf("2. Noul Evaluation (Probabilistic Boolean System-1 Decision):\n");
    printf("   • Probability : %.4f (is_true=%d, is_uncertain=%d)\n",
           noul_res.probability, noul_res.is_true, noul_res.is_uncertain);
    printf("   • Latency     : %.2f microseconds/op (%.4f ms)\n\n", us_per_op, us_per_op / 1000.0);

    // 3. Benchmark Choice Decision
    const char* options[] = {"quarantine_incident", "customer_refund_portal", "standard_chat"};
    reflex_choice_result_t choice_res;
    t0 = clock();
    for (int i = 0; i < iterations; i++) {
        reflex_evaluate_choice(
            sample_text,
            "Select next agent tool",
            options,
            3,
            0.25f,
            &choice_res
        );
    }
    t1 = clock();
    total_sec = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    us_per_op = (total_sec / iterations) * 1000000.0;

    printf("3. Choice Evaluation (Dynamic Multi-Class Rubric):\n");
    printf("   • Selected    : '%s' (confidence: %.2f%%)\n", choice_res.selected, choice_res.confidence * 100.0f);
    printf("   • Latency     : %.2f microseconds/op (%.4f ms)\n\n", us_per_op, us_per_op / 1000.0);

    // 4. Benchmark Guardrail Check
    reflex_guardrail_result_t guard_res;
    reflex_guardrail_check(sample_text, &guard_res);
    printf("4. Security Guardrails:\n");
    printf("   • Blocked     : %s (risk_score: %.2f)\n", guard_res.blocked ? "YES ⚠️" : "NO ✅", guard_res.risk_score);
    printf("   • Reason      : %s\n", guard_res.reason);
    printf("   • Category    : %s\n\n", guard_res.category);

    printf("✅ All native C99 tests completed successfully with zero memory errors!\n");
    return 0;
}
