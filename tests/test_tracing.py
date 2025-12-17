"""Tests for the tracing module."""

import asyncio
import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from fastroai.tracing import LogfireTracer, NoOpTracer, SimpleTracer, Tracer


class TestTracerProtocol:
    """Tests for Tracer protocol compliance."""

    def test_simple_tracer_is_tracer(self) -> None:
        """SimpleTracer should satisfy Tracer protocol."""
        tracer = SimpleTracer()
        assert isinstance(tracer, Tracer)

    def test_noop_tracer_is_tracer(self) -> None:
        """NoOpTracer should satisfy Tracer protocol."""
        tracer = NoOpTracer()
        assert isinstance(tracer, Tracer)

    def test_logfire_tracer_is_tracer(self) -> None:
        """LogfireTracer should satisfy Tracer protocol."""
        tracer = LogfireTracer()
        assert isinstance(tracer, Tracer)


class TestSimpleTracer:
    """Tests for SimpleTracer."""

    @pytest.fixture
    def tracer(self) -> SimpleTracer:
        """Create a SimpleTracer with a test logger."""
        logger = logging.getLogger("test.tracing")
        logger.setLevel(logging.DEBUG)
        return SimpleTracer(logger=logger)

    async def test_span_returns_trace_id(self, tracer: SimpleTracer) -> None:
        """Span should yield a valid trace ID."""
        async with tracer.span("test_operation") as trace_id:
            assert trace_id is not None
            assert len(trace_id) == 36  # UUID format
            assert "-" in trace_id

    async def test_span_timing(self, tracer: SimpleTracer) -> None:
        """Span should measure execution time."""
        async with tracer.span("timed_operation") as trace_id:
            await asyncio.sleep(0.05)
        # If we got here without error, timing worked
        assert trace_id is not None

    async def test_span_with_attributes(self, tracer: SimpleTracer) -> None:
        """Span should accept arbitrary attributes."""
        async with tracer.span("attributed_operation", user_id="123", model="gpt-4") as trace_id:
            assert trace_id is not None

    async def test_span_propagates_exceptions(self, tracer: SimpleTracer) -> None:
        """Span should propagate exceptions after logging."""
        with pytest.raises(ValueError, match="test error"):
            async with tracer.span("failing_operation"):
                raise ValueError("test error")

    async def test_span_logs_error_on_exception(self, tracer: SimpleTracer, caplog: pytest.LogCaptureFixture) -> None:
        """Span should log errors when exceptions occur."""
        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
            async with tracer.span("error_operation"):
                raise RuntimeError("something broke")

        assert "FAILED error_operation" in caplog.text
        assert "something broke" in caplog.text

    def test_log_metric(self, tracer: SimpleTracer, caplog: pytest.LogCaptureFixture) -> None:
        """log_metric should log at debug level."""
        with caplog.at_level(logging.DEBUG):
            tracer.log_metric("abc-123", "token_count", 150)

        assert "token_count=150" in caplog.text

    def test_log_error(self, tracer: SimpleTracer, caplog: pytest.LogCaptureFixture) -> None:
        """log_error should log at error level."""
        with caplog.at_level(logging.ERROR):
            tracer.log_error("abc-123", ValueError("bad input"), {"step": "validation"})

        assert "bad input" in caplog.text

    def test_custom_logger(self) -> None:
        """Should use custom logger when provided."""
        custom_logger = logging.getLogger("custom.logger")
        tracer = SimpleTracer(logger=custom_logger)
        assert tracer.logger is custom_logger

    def test_default_logger(self) -> None:
        """Should use default logger when none provided."""
        tracer = SimpleTracer()
        assert tracer.logger.name == "fastroai.tracing"


class TestNoOpTracer:
    """Tests for NoOpTracer."""

    @pytest.fixture
    def tracer(self) -> NoOpTracer:
        """Create a NoOpTracer."""
        return NoOpTracer()

    async def test_span_returns_trace_id(self, tracer: NoOpTracer) -> None:
        """NoOp span should still return a trace ID."""
        async with tracer.span("operation") as trace_id:
            assert trace_id is not None
            assert len(trace_id) == 36

    async def test_span_doesnt_error(self, tracer: NoOpTracer) -> None:
        """NoOp span should work without errors."""
        async with tracer.span("operation", foo="bar", count=42) as trace_id:
            pass
        assert trace_id is not None

    def test_log_metric_doesnt_error(self, tracer: NoOpTracer) -> None:
        """NoOp log_metric should not raise."""
        tracer.log_metric("trace-123", "metric", 42)

    def test_log_error_doesnt_error(self, tracer: NoOpTracer) -> None:
        """NoOp log_error should not raise."""
        tracer.log_error("trace-123", ValueError("err"), {"context": "test"})

    async def test_noop_has_no_overhead(self, tracer: NoOpTracer) -> None:
        """NoOp tracer should have minimal overhead."""
        # This is more of a sanity check than a performance test
        for _ in range(100):
            async with tracer.span("operation"):
                pass
            tracer.log_metric("id", "m", 1)
            tracer.log_error("id", Exception("e"))


class TestLogfireTracer:
    """Tests for LogfireTracer."""

    @pytest.fixture
    def mock_logfire(self):
        """Create a mock logfire module."""
        mock = MagicMock()

        @contextmanager
        def mock_span(*args, **kwargs):
            yield

        mock.span = MagicMock(side_effect=mock_span)
        mock.info = MagicMock()
        mock.error = MagicMock()
        return mock

    @pytest.fixture
    def tracer(self, mock_logfire) -> LogfireTracer:
        """Create a LogfireTracer with mocked logfire."""
        tracer = LogfireTracer()
        tracer._logfire = mock_logfire
        return tracer

    async def test_span_returns_trace_id(self, tracer: LogfireTracer) -> None:
        """Span should yield a valid trace ID."""
        async with tracer.span("test_operation") as trace_id:
            assert trace_id is not None
            assert len(trace_id) == 36  # UUID format
            assert "-" in trace_id

    async def test_span_calls_logfire_span(self, tracer: LogfireTracer, mock_logfire) -> None:
        """Span should call logfire.span with correct arguments."""
        async with tracer.span("my_operation", user_id="123") as trace_id:
            pass

        mock_logfire.span.assert_called_once()
        call_args = mock_logfire.span.call_args
        assert call_args[0][0] == "my_operation"
        assert call_args[1]["user_id"] == "123"
        assert call_args[1]["trace_id"] == trace_id
        assert "fastroai" in call_args[1]["_tags"]

    async def test_span_with_multiple_attributes(self, tracer: LogfireTracer) -> None:
        """Span should accept multiple attributes."""
        async with tracer.span("operation", user_id="123", model="gpt-4", tokens=100) as trace_id:
            assert trace_id is not None

    async def test_span_propagates_exceptions(self, tracer: LogfireTracer) -> None:
        """Span should propagate exceptions."""
        with pytest.raises(ValueError, match="test error"):
            async with tracer.span("failing_operation"):
                raise ValueError("test error")

    def test_log_metric_calls_logfire_info(self, tracer: LogfireTracer, mock_logfire) -> None:
        """log_metric should call logfire.info with correct arguments."""
        tracer.log_metric("trace-123", "token_count", 150)

        mock_logfire.info.assert_called_once()
        call_args = mock_logfire.info.call_args
        assert "metric" in call_args[0][0]
        assert call_args[1]["name"] == "token_count"
        assert call_args[1]["value"] == 150
        assert call_args[1]["trace_id"] == "trace-123"
        assert "metric" in call_args[1]["_tags"]

    def test_log_metric_with_various_types(self, tracer: LogfireTracer, mock_logfire) -> None:
        """log_metric should handle various value types."""
        tracer.log_metric("trace-1", "int_metric", 42)
        tracer.log_metric("trace-2", "float_metric", 3.14)
        tracer.log_metric("trace-3", "string_metric", "value")

        assert mock_logfire.info.call_count == 3

    def test_log_error_calls_logfire_error(self, tracer: LogfireTracer, mock_logfire) -> None:
        """log_error should call logfire.error with correct arguments."""
        error = ValueError("bad input")
        tracer.log_error("trace-123", error, {"step": "validation"})

        mock_logfire.error.assert_called_once()
        call_args = mock_logfire.error.call_args
        # Template string is passed as first positional arg
        assert "{error_type}" in call_args[0][0]
        # Actual values are passed as kwargs
        assert call_args[1]["error_type"] == "ValueError"
        assert call_args[1]["error_message"] == "bad input"
        assert call_args[1]["trace_id"] == "trace-123"
        assert call_args[1]["step"] == "validation"
        assert call_args[1]["_exc_info"] is error
        assert "error" in call_args[1]["_tags"]

    def test_log_error_without_context(self, tracer: LogfireTracer, mock_logfire) -> None:
        """log_error should work without context."""
        tracer.log_error("trace-123", RuntimeError("failed"))

        mock_logfire.error.assert_called_once()

    def test_log_error_with_empty_context(self, tracer: LogfireTracer, mock_logfire) -> None:
        """log_error should work with empty context."""
        tracer.log_error("trace-123", RuntimeError("failed"), {})

        mock_logfire.error.assert_called_once()


class TestLogfireTracerImportError:
    """Tests for LogfireTracer when logfire is not installed."""

    def test_raises_import_error_when_logfire_not_available(self) -> None:
        """LogfireTracer should raise ImportError when logfire is not installed."""
        with (
            patch("fastroai.tracing.tracer.logfire_module", None),
            pytest.raises(ImportError, match="logfire is required"),
        ):
            LogfireTracer()
