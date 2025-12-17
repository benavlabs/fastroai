"""Tracing module for distributed tracing and observability.

Provides a protocol-based tracing interface that can be implemented
for various observability backends, plus built-in implementations:

- SimpleTracer: Logging-based tracing for development
- LogfireTracer: Integration with Pydantic's Logfire platform
- NoOpTracer: Does nothing, for testing or disabled tracing
"""

from .tracer import LogfireTracer, NoOpTracer, SimpleTracer, Tracer

__all__ = [
    "Tracer",
    "SimpleTracer",
    "LogfireTracer",
    "NoOpTracer",
]
