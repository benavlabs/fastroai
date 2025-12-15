"""Tracing module for distributed tracing and observability.

Provides a protocol-based tracing interface that can be implemented
for various observability backends (Logfire, OpenTelemetry, etc.).
"""

from .tracer import NoOpTracer, SimpleTracer, Tracer

__all__ = [
    "Tracer",
    "SimpleTracer",
    "NoOpTracer",
]
