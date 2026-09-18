"""
Reflex OpenTelemetry Distributed Tracing & W3C TraceContext Emitter.
Provides zero-dependency tracing spans compatible with Datadog, Dynatrace, Honeycomb, and Langfuse.
"""

from __future__ import annotations
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


def _generate_trace_id() -> str:
    """Generates a 16-byte hex W3C trace ID."""
    return uuid.uuid4().hex


def _generate_span_id() -> str:
    """Generates an 8-byte hex W3C span ID."""
    return uuid.uuid4().hex[:16]


@dataclass
class Span:
    """OpenTelemetry-compliant Trace Span."""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time_nano: int = field(default_factory=lambda: time.time_ns())
    end_time_nano: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "UNSET"
    error_message: Optional[str] = None

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def end(self, status: str = "OK", error: Optional[str] = None):
        self.end_time_nano = time.time_ns()
        self.status = status
        if error:
            self.error_message = error

    @property
    def duration_ms(self) -> float:
        if self.end_time_nano is None:
            return 0.0
        return (self.end_time_nano - self.start_time_nano) / 1_000_000.0

    @property
    def traceparent(self) -> str:
        """Returns W3C traceparent header string."""
        return f"00-{self.trace_id}-{self.span_id}-01"

    def to_otlp_dict(self) -> Dict[str, Any]:
        """Converts to OpenTelemetry Protobuf-compatible JSON representation."""
        attrs = []
        for k, v in self.attributes.items():
            if isinstance(v, bool):
                val = {"boolValue": v}
            elif isinstance(v, int):
                val = {"intValue": v}
            elif isinstance(v, float):
                val = {"doubleValue": v}
            else:
                val = {"stringValue": str(v)}
            attrs.append({"key": k, "value": val})

        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "kind": 1,  # SPAN_KIND_INTERNAL
            "startTimeUnixNano": str(self.start_time_nano),
            "endTimeUnixNano": str(self.end_time_nano or self.start_time_nano),
            "attributes": attrs,
            "status": {
                "code": 1 if self.status == "OK" else (2 if self.status == "ERROR" else 0),
                "message": self.error_message or "",
            },
        }


class OpenTelemetryTracer:
    """
    Zero-dependency OpenTelemetry distributed tracer.
    
    Generates standardized W3C tracecontext and spans for tracking
    System 1 latency, dual-brain auto-escalation, and cloud cost savings.
    """

    def __init__(self, service_name: str = "reflex"):
        self.service_name = service_name
        self.spans: List[Span] = []

    @contextmanager
    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """Context manager creating, timing, and recording a trace span."""
        t_id = trace_id or (parent.trace_id if parent else _generate_trace_id())
        p_id = parent.span_id if parent else None
        s_id = _generate_span_id()

        span = Span(
            name=name,
            trace_id=t_id,
            span_id=s_id,
            parent_span_id=p_id,
            attributes=attributes or {},
        )
        span.set_attribute("service.name", self.service_name)

        try:
            yield span
            if span.end_time_nano is None:
                span.end(status="OK")
        except Exception as e:
            span.end(status="ERROR", error=str(e))
            raise
        finally:
            self.spans.append(span)

    def get_spans(self) -> List[Span]:
        return list(self.spans)

    def clear(self):
        self.spans.clear()

    def export_otlp_json(self) -> Dict[str, Any]:
        """Exports all recorded spans formatted as an OTLP JSON payload."""
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}},
                            {"key": "telemetry.sdk.language", "value": {"stringValue": "python"}},
                            {"key": "telemetry.sdk.name", "value": {"stringValue": "reflex-otel"}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "reflex.tracer", "version": "0.2.0"},
                            "spans": [span.to_otlp_dict() for span in self.spans],
                        }
                    ],
                }
            ]
        }
