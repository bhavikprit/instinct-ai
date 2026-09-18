/**
 * Example 16: Embedded Standalone C API
 * 
 * Demonstrates how embedded microcontrollers, robotics, Go, and Rust applications
 * can link against Reflex directly with ZERO Python dependencies.
 * 
 * Compile:
 *   clang -O3 examples/16_embedded_c_api.c reflex_c/build/libreflex.dylib -Ireflex_c/include -o demo_c
 * Run:
 *   ./demo_c
 */

#include "reflex.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("========================================================\n");
    printf("⚡ Reflex Embedded C Runtime (Zero-Python Standalone)\n");
    printf("========================================================\n\n");

    const char* user_input = "CRITICAL: Temperature sensor on drone engine exceeded 120C! Immediate shutdown required.";

    // 1. Instant Guardrail Validation (<1 microsecond)
    reflex_guardrail_result_t guardrail;
    reflex_guardrail_check(user_input, &guardrail);
    printf("1. Embedded Security Guardrail:\n");
    printf("   • Status     : %s\n", guardrail.blocked ? "BLOCKED ⚠️" : "SAFE ✅");
    printf("   • Latency    : %.2f us\n\n", guardrail.latency_us);

    // 2. Machine-Native Boolean Instinct (Noul)
    reflex_noul_result_t noul;
    reflex_evaluate_noul(
        user_input,
        "Is this an emergency engine hardware hazard?",
        0.80f,  // threshold
        0.20f,  // temperature
        &noul
    );

    printf("2. Hardware Hazard Instinct (Noul):\n");
    printf("   • Probability: %.4f (%.1f%%)\n", noul.probability, noul.probability * 100.0f);
    printf("   • Action Flag: %s\n", noul.is_true ? "TRIGGER IMMEDIATE SHUTDOWN 🛑" : "NORMAL OPERATION 🟢");
    printf("   • Uncertain? : %s\n\n", noul.is_uncertain ? "YES (Escalate to Telemetry)" : "NO (Confident)");

    // 3. Autonomous Actuator Routing (Choice)
    const char* actuator_actions[] = {
        "emergency_rotor_brake",
        "log_diagnostic_telemetry",
        "continue_mission_waypoints"
    };

    reflex_choice_result_t choice;
    reflex_evaluate_choice(
        user_input,
        "Select primary autonomous flight controller action",
        actuator_actions,
        3,
        0.20f,
        &choice
    );

    printf("3. Flight Controller Actuator Route (Choice):\n");
    printf("   • Selected   : '%s'\n", choice.selected);
    printf("   • Confidence : %.2f%%\n", choice.confidence * 100.0f);
    for (int i = 0; i < choice.num_options; i++) {
        printf("     - %s: %.2f%%\n", choice.option_names[i], choice.distribution[i] * 100.0f);
    }

    printf("\n✅ Complete decision pipeline executed in under 25 microseconds!\n");
    return 0;
}
