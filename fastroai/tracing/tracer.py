"""Tracer protocol and implementations for distributed tracing.

This module provides a protocol-based tracing interface that allows
FastroAI to integrate with any observability backend. Users can implement
the Tracer protocol for their preferred platform (Logfire, OpenTelemetry, etc.)
or use the provided SimpleTracer for basic logging-based tracing.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracer(Protocol):
    """Protocol for distributed tracing implementations.

    Implement this protocol to integrate FastroAI with your preferred
    observability platform (Logfire, OpenTelemetry, Datadog, etc.).

    Example implementation for Logfire:
        import logfire

        class LogfireTracer:
            @asynccontextmanager
            async def span(self, name: str, **attrs) -> AsyncIterator[str]:
                trace_id = str(uuid.uuid4())
                with logfire.span(name, trace_id=trace_id, **attrs):
                    yield trace_id

            def log_metric(self, trace_id: str, name: str, value: Any) -> None:
                logfire.metric(name, value, trace_id=trace_id)

            def log_error(self, trace_id: str, error: Exception, context: dict | None = None) -> None:
                logfire.error(str(error), trace_id=trace_id, **(context or {}))
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

    Example:
        tracer = SimpleTracer()

        async with tracer.span("my_operation", user_id="123") as trace_id:
            # Your operation here
            result = await do_something()
            tracer.log_metric(trace_id, "result_size", len(result))

        # Logs:
        # INFO [abc12345] Starting my_operation
        # INFO [abc12345] Metric result_size=42
        # INFO [abc12345] Completed my_operation in 0.123s
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

    Example:
        tracer = NoOpTracer()

        async with tracer.span("operation") as trace_id:
            # trace_id is still generated for compatibility
            result = await do_something()
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
