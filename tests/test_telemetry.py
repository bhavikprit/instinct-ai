"""
Unit tests for Reflex OpenTelemetry distributed tracer.
"""

import unittest
from reflex.telemetry import OpenTelemetryTracer
from reflex.client import Reflex
from reflex.primitives import Noul


class TestOpenTelemetryTracer(unittest.TestCase):

    def setUp(self):
        self.tracer = OpenTelemetryTracer(service_name="test-reflex-service")

    def test_span_lifecycle(self):
        with self.tracer.start_span("test.decision") as span:
            span.set_attribute("reflex.action", "classify")
            span.set_attribute("reflex.probability", 0.95)

        self.assertEqual(len(self.tracer.spans), 1)
        s = self.tracer.spans[0]
        self.assertEqual(s.name, "test.decision")
        self.assertEqual(s.status, "OK")
        self.assertGreaterEqual(s.duration_ms, 0.0)
        self.assertEqual(s.attributes["service.name"], "test-reflex-service")
        self.assertEqual(s.attributes["reflex.action"], "classify")
        self.assertAlmostEqual(s.attributes["reflex.probability"], 0.95)

    def test_w3c_traceparent_format(self):
        with self.tracer.start_span("test.traceparent") as span:
            tp = span.traceparent
            parts = tp.split("-")
            self.assertEqual(len(parts), 4)
            self.assertEqual(parts[0], "00")  # W3C version
            self.assertEqual(len(parts[1]), 32)  # 16-byte traceId hex
            self.assertEqual(len(parts[2]), 16)  # 8-byte spanId hex
            self.assertEqual(parts[3], "01")  # TraceFlags

    def test_otlp_json_export(self):
        with self.tracer.start_span("test.otlp") as span:
            span.set_attribute("cost_saved_usd", 0.03)

        otlp = self.tracer.export_otlp_json()
        self.assertIn("resourceSpans", otlp)
        resource_spans = otlp["resourceSpans"][0]
        self.assertEqual(
            resource_spans["resource"]["attributes"][0]["value"]["stringValue"],
            "test-reflex-service",
        )
        scope_spans = resource_spans["scopeSpans"][0]
        span_dict = scope_spans["spans"][0]
        self.assertEqual(span_dict["name"], "test.otlp")
        self.assertEqual(span_dict["status"]["code"], 1)  # OK code

    def test_reflex_client_tracer_integration(self):
        rx = Reflex(backend="local", tracer=self.tracer)
        res = rx.evaluate("Server down!", {"p0": Noul("Is outage?")})

        self.assertGreaterEqual(len(self.tracer.spans), 1)
        span = self.tracer.spans[-1]
        self.assertEqual(span.name, "reflex.evaluate")
        self.assertIn("reflex.latency_ms", span.attributes)
        self.assertIn("reflex.cost_usd", span.attributes)
        self.assertFalse(span.attributes["reflex.cached"])


if __name__ == "__main__":
    unittest.main()
