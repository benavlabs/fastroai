"""Tests for the tools module."""

import asyncio

from fastroai.tools import (
    DEFAULT_TOOL_MAX_RETRIES,
    DEFAULT_TOOL_TIMEOUT,
    FunctionToolsetBase,
    SafeToolset,
    safe_tool,
)


class TestSafeToolDefaults:
    """Tests for safe_tool default values."""

    def test_default_timeout(self) -> None:
        """Default timeout should be 30 seconds."""
        assert DEFAULT_TOOL_TIMEOUT == 30

    def test_default_max_retries(self) -> None:
        """Default max retries should be 3."""
        assert DEFAULT_TOOL_MAX_RETRIES == 3


class TestSafeToolSuccess:
    """Tests for successful tool execution."""

    async def test_returns_result_on_success(self) -> None:
        """Should return function result on success."""

        @safe_tool()
        async def successful_tool(x: int) -> str:
            return f"Result: {x}"

        result = await successful_tool(42)
        assert result == "Result: 42"

    async def test_preserves_function_metadata(self) -> None:
        """Should preserve function name and docstring."""

        @safe_tool()
        async def documented_tool(x: int) -> str:
            """This is the docstring."""
            return str(x)

        assert documented_tool.__name__ == "documented_tool"
        assert documented_tool.__doc__ == "This is the docstring."

    async def test_passes_args_and_kwargs(self) -> None:
        """Should correctly pass positional and keyword arguments."""

        @safe_tool()
        async def multi_arg_tool(a: int, b: str, c: bool = False) -> str:
            return f"{a}-{b}-{c}"

        result = await multi_arg_tool(1, "two", c=True)
        assert result == "1-two-True"


class TestSafeToolTimeout:
    """Tests for timeout handling."""

    async def test_returns_message_on_timeout(self) -> None:
        """Should return error message when timing out."""

        @safe_tool(timeout=0.05, max_retries=1)
        async def slow_tool() -> str:
            await asyncio.sleep(1)
            return "never reached"

        result = await slow_tool()
        assert "timed out" in result.lower()

    async def test_custom_timeout_message(self) -> None:
        """Should use custom timeout message when provided."""

        @safe_tool(timeout=0.05, max_retries=1, on_timeout="Too slow!")
        async def slow_tool() -> str:
            await asyncio.sleep(1)
            return "never reached"

        result = await slow_tool()
        assert result == "Too slow!"

    async def test_retries_on_timeout(self) -> None:
        """Should retry on timeout before giving up."""
        attempts = []

        @safe_tool(timeout=0.05, max_retries=3)
        async def flaky_timeout_tool() -> str:
            attempts.append(1)
            await asyncio.sleep(1)
            return "never reached"

        await flaky_timeout_tool()
        assert len(attempts) == 3


class TestSafeToolErrors:
    """Tests for error handling."""

    async def test_returns_message_on_error(self) -> None:
        """Should return error message instead of raising."""

        @safe_tool(max_retries=1)
        async def failing_tool() -> str:
            raise ValueError("something broke")

        result = await failing_tool()
        assert "Tool failed:" in result
        assert "something broke" in result

    async def test_custom_error_message(self) -> None:
        """Should use custom error message when provided."""

        @safe_tool(max_retries=1, on_error="Oops: {error}")
        async def failing_tool() -> str:
            raise ValueError("bad input")

        result = await failing_tool()
        assert result == "Oops: bad input"

    async def test_retries_on_error(self) -> None:
        """Should retry on error before giving up."""
        attempts = []

        @safe_tool(max_retries=3)
        async def flaky_tool() -> str:
            attempts.append(1)
            raise RuntimeError("flaky failure")

        await flaky_tool()
        assert len(attempts) == 3

    async def test_succeeds_after_retry(self) -> None:
        """Should succeed if a retry works."""
        attempts = []

        @safe_tool(max_retries=3)
        async def eventually_works() -> str:
            attempts.append(1)
            if len(attempts) < 2:
                raise RuntimeError("not yet")
            return "success"

        result = await eventually_works()
        assert result == "success"
        assert len(attempts) == 2


class TestSafeToolBackoff:
    """Tests for exponential backoff."""

    async def test_exponential_backoff_timing(self) -> None:
        """Should use exponential backoff between retries."""
        times = []

        @safe_tool(timeout=0.01, max_retries=3)
        async def tracked_tool() -> str:
            times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(1)  # Will timeout
            return "never"

        await tracked_tool()

        # Should have 3 attempts
        assert len(times) == 3

        # Check backoff between attempts
        # First backoff: 0.1s, Second backoff: 0.2s
        # Allow some tolerance for test execution
        if len(times) >= 2:
            first_gap = times[1] - times[0]
            assert first_gap >= 0.05  # At least some backoff
        if len(times) >= 3:
            second_gap = times[2] - times[1]
            assert second_gap >= first_gap * 0.5  # Second should be longer


class TestFunctionToolsetBase:
    """Tests for FunctionToolsetBase."""

    def test_accepts_tools_list(self) -> None:
        """Should accept a list of tools."""

        async def tool1() -> str:
            return "1"

        async def tool2() -> str:
            return "2"

        toolset = FunctionToolsetBase(tools=[tool1, tool2], name="test")
        assert toolset.name == "test"

    def test_default_name_is_class_name(self) -> None:
        """Should use class name as default name."""

        async def tool() -> str:
            return "x"

        class MyToolset(FunctionToolsetBase):
            def __init__(self) -> None:
                super().__init__(tools=[tool])

        toolset = MyToolset()
        assert toolset.name == "MyToolset"

    def test_custom_name(self) -> None:
        """Should use custom name when provided."""

        async def tool() -> str:
            return "x"

        toolset = FunctionToolsetBase(tools=[tool], name="custom_name")
        assert toolset.name == "custom_name"


class TestSafeToolset:
    """Tests for SafeToolset."""

    def test_is_function_toolset_base(self) -> None:
        """SafeToolset should extend FunctionToolsetBase."""

        async def tool() -> str:
            return "x"

        toolset = SafeToolset(tools=[tool], name="safe")
        assert isinstance(toolset, FunctionToolsetBase)

    def test_can_create_subclass(self) -> None:
        """Should be able to subclass SafeToolset."""

        @safe_tool(timeout=5)
        async def calc(expr: str) -> str:
            return str(eval(expr, {"__builtins__": {}}, {}))

        class MathToolset(SafeToolset):
            def __init__(self) -> None:
                super().__init__(tools=[calc], name="math")

        toolset = MathToolset()
        assert toolset.name == "math"
        assert isinstance(toolset, SafeToolset)


class TestSafeToolIntegration:
    """Integration tests combining decorator with toolsets."""

    async def test_safe_tools_in_toolset(self) -> None:
        """Should work when safe_tool decorated functions are in a toolset."""

        @safe_tool(timeout=5)
        async def add(a: int, b: int) -> str:
            return str(a + b)

        @safe_tool(timeout=5)
        async def multiply(a: int, b: int) -> str:
            return str(a * b)

        class MathToolset(SafeToolset):
            def __init__(self) -> None:
                super().__init__(tools=[add, multiply], name="math")

        toolset = MathToolset()
        assert toolset.name == "math"

        # Verify tools still work
        result = await add(2, 3)
        assert result == "5"

        result = await multiply(4, 5)
        assert result == "20"
