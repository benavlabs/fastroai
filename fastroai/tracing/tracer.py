"""Tracer protocol and implementations for distributed tracing.

This module provides a protocol-based tracing interface that allows
FastroAI to integrate with any observability backend. Users can implement
the Tracer protocol for their preferred platform (Logfire, OpenTelemetry, etc.)
or use the provided tracers:

- SimpleTracer: Basic logging-based tracing for development
- LogfireTracer: Integration with Pydantic's Logfire platform
- NoOpTracer: Does nothing, for testing or disabled tracing
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import logfire as logfire_module
else:
    try:
        import logfire as logfire_module
    except ImportError:  # pragma: no cover
        logfire_module = None  # type: ignore[assignment]


@runtime_checkable
class Tracer(Protocol):
    """Protocol for distributed tracing implementations.

    Implement this protocol to integrate FastroAI with your preferred
    observability platform (OpenTelemetry, Datadog, etc.).

    FastroAI provides built-in implementations:
    - LogfireTracer: For Pydantic's Logfire platform
    - SimpleTracer: For logging-based tracing
    - NoOpTracer: For disabled tracing

    Examples:
        Using the built-in LogfireTracer:
        ```python
        from fastroai import FastroAgent, LogfireTracer

        tracer = LogfireTracer()
        agent = FastroAgent(model="openai:gpt-4o")
        response = await agent.run("Hello!", tracer=tracer)
        ```

        Custom implementation for OpenTelemetry:
        ```python
        from opentelemetry import trace as otel_trace

        class OTelTracer:
            def __init__(self):
                self.tracer = otel_trace.get_tracer("fastroai")

            @asynccontextmanager
            async def span(self, name: str, **attrs):
                trace_id = str(uuid.uuid4())
                with self.tracer.start_as_current_span(name) as span:
                    for key, value in attrs.items():
                        span.set_attribute(key, value)
                    yield trace_id

            def log_metric(self, trace_id: str, name: str, value):
                span = otel_trace.get_current_span()
                span.set_attribute(f"metric.{name}", value)

            def log_error(self, trace_id: str, error: Exception, context=None):
                span = otel_trace.get_current_span()
                span.record_exception(error)
        ```
    """

    def span(self, name: str, **attributes: Any) -> AbstractAsyncContextManager[str]:
        """Create a traced span for an operation.

        Args:
            name: Name of the operation being traced.
            **attributes: Additional context to attach to the span.

        Returns:
            Async context manager that yields a unique trace ID.
        """
        ...

    def log_metric(self, trace_id: str, name: str, value: Any) -> None:
        """Log a metric associated with a trace.

        Args:
            trace_id: Trace ID to associate the metric with.
            name: Metric name.
            value: Metric value.
        """
        ...

    def log_error(self, trace_id: str, error: Exception, context: dict[str, Any] | None = None) -> None:
        """Log an error associated with a trace.

        Args:
            trace_id: Trace ID to associate the error with.
            error: The exception that occurred.
            context: Additional error context.
        """
        ...


class SimpleTracer:
    """Basic tracer implementation using Python's logging module.

    Provides simple tracing functionality for development and debugging.
    For production use, consider implementing a Tracer for your
    observability platform.

    Examples:
        ```python
        tracer = SimpleTracer()

        async with tracer.span("my_operation", user_id="123") as trace_id:
            # Your operation here
            result = await do_something()
            tracer.log_metric(trace_id, "result_size", len(result))

        # Logs:
        # INFO [abc12345] Starting my_operation
        # INFO [abc12345] Metric result_size=42
        # INFO [abc12345] Completed my_operation in 0.123s
        ```
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize SimpleTracer.

        Args:
            logger: Logger to use. Defaults to 'fastroai.tracing'.
        """
        self.logger = logger or logging.getLogger("fastroai.tracing")

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[str]:
        """Create a traced span with timing.

        Args:
            name: Name of the operation.
            **attributes: Additional context logged with the span.

        Yields:
            Unique trace ID (first 8 chars shown in logs for readability).
        """
        trace_id = str(uuid.uuid4())
        short_id = trace_id[:8]
        start = time.perf_counter()

        self.logger.info(
            f"[{short_id}] Starting {name}",
            extra={"trace_id": trace_id, "span": name, **attributes},
        )

        try:
            yield trace_id
        except Exception as e:
            duration = time.perf_counter() - start
            self.logger.error(
                f"[{short_id}] FAILED {name} after {duration:.3f}s: {e}",
                exc_info=True,
                extra={"trace_id": trace_id, "span": name, "error": str(e)},
            )
            raise
        else:
            duration = time.perf_counter() - start
            self.logger.info(
                f"[{short_id}] Completed {name} in {duration:.3f}s",
                extra={"trace_id": trace_id, "span": name, "duration_seconds": duration},
            )

    def log_metric(self, trace_id: str, name: str, value: Any) -> None:
        """Log a metric with trace correlation.

        Args:
            trace_id: Trace ID for correlation.
            name: Metric name.
            value: Metric value.
        """
        short_id = trace_id[:8]
        self.logger.debug(
            f"[{short_id}] Metric {name}={value}",
            extra={"trace_id": trace_id, "metric_name": name, "metric_value": value},
        )

    def log_error(self, trace_id: str, error: Exception, context: dict[str, Any] | None = None) -> None:
        """Log an error with trace correlation.

        Args:
            trace_id: Trace ID for correlation.
            error: The exception that occurred.
            context: Additional error context.
        """
        short_id = trace_id[:8]
        self.logger.error(
            f"[{short_id}] Error: {error}",
            extra={"trace_id": trace_id, "error_type": type(error).__name__, **(context or {})},
        )


class NoOpTracer:
    """Tracer that does nothing. Use when tracing is disabled.

    This tracer satisfies the Tracer protocol but performs no operations,
    making it suitable for testing or when tracing overhead is undesirable.

    Examples:
        ```python
        tracer = NoOpTracer()

        async with tracer.span("operation") as trace_id:
            # trace_id is still generated for compatibility
            result = await do_something()
        ```
    """

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[str]:
        """Create a no-op span that just yields a trace ID.

        Args:
            name: Ignored.
            **attributes: Ignored.

        Yields:
            Unique trace ID (still generated for compatibility).
        """
        yield str(uuid.uuid4())

    def log_metric(self, trace_id: str, name: str, value: Any) -> None:
        """No-op metric logging.

        Args:
            trace_id: Ignored.
            name: Ignored.
            value: Ignored.
        """
        pass

    def log_error(self, trace_id: str, error: Exception, context: dict[str, Any] | None = None) -> None:
        """No-op error logging.

        Args:
            trace_id: Ignored.
            error: Ignored.
            context: Ignored.
        """
        pass


class LogfireTracer:
    """Tracer implementation for Pydantic's Logfire observability platform.

    Integrates FastroAI with Logfire for production-grade observability,
    including distributed tracing, metrics, and error tracking. Requires
    the `logfire` package to be installed.

    Note:
        Install logfire with: `pip install logfire`
        Configure logfire before use: `logfire.configure()`

    Examples:
        ```python
        import logfire
        from fastroai import FastroAgent, LogfireTracer

        # Configure logfire (typically done once at startup)
        logfire.configure()

        tracer = LogfireTracer()
        agent = FastroAgent(model="openai:gpt-4o")
        response = await agent.run("Hello!", tracer=tracer)

        # View traces in Logfire dashboard at https://logfire.pydantic.dev
        ```

        With pipelines:
        ```python
        from fastroai import Pipeline, LogfireTracer

        tracer = LogfireTracer()
        result = await pipeline.execute(
            {"document": doc},
            deps=my_deps,
            tracer=tracer,
        )
        ```
    """

    def __init__(self) -> None:
        """Initialize LogfireTracer.

        Raises:
            ImportError: If the logfire package is not installed.
        """
        if logfire_module is None:
            raise ImportError("logfire is required for LogfireTracer. Install it with: pip install logfire")
        self._logfire = logfire_module

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[str]:
        """Create a traced span using Logfire.

        Wraps Logfire's span context manager and generates a unique trace ID
        for correlation across FastroAI operations.

        Args:
            name: Name of the operation being traced.
            **attributes: Additional context to attach to the span.
                These appear as attributes in the Logfire dashboard.

        Yields:
            Unique trace ID for correlating metrics and errors.

        Examples:
            ```python
            async with tracer.span("my_operation", user_id="123") as trace_id:
                result = await do_something()
                tracer.log_metric(trace_id, "result_size", len(result))
            ```
        """
        trace_id = str(uuid.uuid4())
        with self._logfire.span(name, _tags=["fastroai"], trace_id=trace_id, **attributes):
            yield trace_id

    def log_metric(self, trace_id: str, name: str, value: Any) -> None:
        """Log a metric to Logfire with trace correlation.

        Metrics are logged as info-level spans with the metric name and value
        as attributes, allowing them to be queried and visualized in Logfire.

        Args:
            trace_id: Trace ID for correlation.
            name: Metric name (e.g., "input_tokens", "cost_microcents").
            value: Metric value.

        Examples:
            ```python
            tracer.log_metric(trace_id, "input_tokens", 150)
            tracer.log_metric(trace_id, "cost_microcents", 2500)
            ```
        """
        self._logfire.info(
            "metric.{name}",
            name=name,
            value=value,
            trace_id=trace_id,
            _tags=["fastroai", "metric"],
        )

    def log_error(self, trace_id: str, error: Exception, context: dict[str, Any] | None = None) -> None:
        """Log an error to Logfire with trace correlation.

        Records the error with full exception information for debugging
        in the Logfire dashboard.

        Args:
            trace_id: Trace ID for correlation.
            error: The exception that occurred.
            context: Additional error context (e.g., step_id, operation).

        Examples:
            ```python
            try:
                result = await risky_operation()
            except Exception as e:
                tracer.log_error(trace_id, e, {"step": "data_processing"})
                raise
            ```
        """
        self._logfire.error(
            "{error_type}: {error_message}",
            error_type=type(error).__name__,
            error_message=str(error),
            trace_id=trace_id,
            _exc_info=error,
            _tags=["fastroai", "error"],
            **(context or {}),
        )
