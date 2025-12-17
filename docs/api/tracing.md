# Tracing

The tracing module provides a protocol-based interface for distributed tracing integration. Implement the `Tracer` protocol to connect FastroAI with your observability platform, or use one of the built-in tracers.

## Tracer

::: fastroai.tracing.Tracer
    options:
      show_root_heading: true
      show_source: false

## SimpleTracer

::: fastroai.tracing.SimpleTracer
    options:
      show_root_heading: true
      show_source: false

## LogfireTracer

::: fastroai.tracing.LogfireTracer
    options:
      show_root_heading: true
      show_source: false

## NoOpTracer

::: fastroai.tracing.NoOpTracer
    options:
      show_root_heading: true
      show_source: false

---

[← Usage](usage.md){ .md-button } [API Overview →](index.md){ .md-button .md-button--primary }
