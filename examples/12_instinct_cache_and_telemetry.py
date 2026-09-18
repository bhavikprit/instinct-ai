"""
Example 12: InstinctCache (Sub-0.05ms Semantic Memory) & OpenTelemetry Tracing.

Demonstrates:
1. Sub-0.05ms semantic cache hits on rephrased queries.
2. Zero-dependency OpenTelemetry distributed trace spans with W3C traceparent headers.
3. Cache hit rate and saved latency telemetry.
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflex import Reflex, Noul, Choice, InstinctCache, OpenTelemetryTracer


def main():
    print("=" * 70)
    print("⚡ Reflex InstinctCache & OpenTelemetry Distributed Tracing")
    print("=" * 70)

    # 1. Initialize Reflex with InstinctCache and OpenTelemetryTracer
    cache = InstinctCache(similarity_threshold=0.85)
    tracer = OpenTelemetryTracer(service_name="customer-support-agent")
    rx = Reflex(backend="semantic", cache=cache, tracer=tracer)

    query_1 = "User requests immediate cancellation and refund for duplicate charge on Visa"
    decision_specs = {
        "is_refund": Noul("Is the user requesting a chargeback or refund?"),
        "department": Choice("Select department", ["billing_support", "technical_ops", "sales"]),
    }

    # Pass 1: Cold evaluation (Cache miss)
    print("\n1. First Pass (Cold Query - Cache Miss):")
    res1 = rx.evaluate(query_1, decision_specs)
    print(f" • Query: \"{query_1}\"")
    print(f" • Refund Probability : {res1['is_refund'].probability:.3f}")
    print(f" • Department         : {res1['department'].selected}")
    print(f" • Latency            : {res1.latency_ms} ms")
    print(f" • Cached?            : {res1.cached}")

    # Pass 2: Rephrased Query (Semantic Cache Hit!)
    query_2 = "User requests immediate cancellation and refund for duplicate charge on Visa! Please assist"
    print("\n2. Second Pass (Rephrased State - Semantic Cache Hit):")
    res2 = rx.evaluate(query_2, decision_specs)
    print(f" • Query: \"{query_2}\"")
    print(f" • Refund Probability : {res2['is_refund'].probability:.3f}")
    print(f" • Department         : {res2['department'].selected}")
    print(f" • Latency            : {res2.latency_ms} ms (Instant Instinct!)")
    print(f" • Cached?            : {res2.cached} (Resolved via {res2.backend})")

    # 3. Cache Telemetry
    print("\n3. InstinctCache Telemetry Stats:")
    stats = cache.stats()
    for k, v in stats.items():
        print(f" • {k:<20}: {v}")

    # 4. OpenTelemetry Spans
    print("\n4. OpenTelemetry Distributed Spans Generated:")
    spans = tracer.get_spans()
    for idx, s in enumerate(spans):
        print(f" • Span #{idx + 1}: {s.name} (Duration: {s.duration_ms:.3f} ms)")
        print(f"   Traceparent: {s.traceparent}")
        print(f"   Attributes:  {s.attributes}")

    print("\n5. Sample OTLP JSON Export (truncated):")
    otlp = tracer.export_otlp_json()
    print(json.dumps(otlp["resourceSpans"][0]["scopeSpans"][0]["spans"][0], indent=2))

    print("\n✅ InstinctCache & OpenTelemetry demonstration complete!")


if __name__ == "__main__":
    main()
