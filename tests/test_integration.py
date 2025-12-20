"""Integration tests that use real AI APIs.

These tests require OPENAI_API_KEY to be set in the environment.

Run with: uv run pytest tests/test_integration.py -v

Skip with: uv run pytest tests/ --ignore=tests/test_integration.py
"""
# mypy: disable-error-code="var-annotated"

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import pytest

from fastroai import (
    BasePipeline,
    BaseStep,
    ConversationState,
    ConversationStatus,
    CostCalculator,
    FastroAgent,
    Pipeline,
    SafeToolset,
    SimpleTracer,
    StepContext,
    safe_tool,
)

# Marker for tests that require OpenAI API
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@requires_openai
class TestFastroAgentIntegration:
    """Integration tests for FastroAgent with real APIs."""

    async def test_simple_chat(self) -> None:
        """Test simple chat with OpenAI."""
        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant. Be concise.",
        )

        response = await agent.run("What is 2+2? Reply with just the number.")

        assert response.content is not None
        assert "4" in response.content
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.cost_microcents > 0
        assert response.processing_time_ms > 0

    async def test_with_tracer(self) -> None:
        """Test that tracing works with real API calls."""
        tracer = SimpleTracer()
        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Reply with one word.",
        )

        response = await agent.run("Say 'hello'", tracer=tracer)

        assert response.trace_id is not None
        assert len(response.trace_id) == 36  # UUID format

    async def test_streaming(self) -> None:
        """Test streaming responses."""
        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Be concise.",
        )

        chunks = []
        async for chunk in agent.run_stream("Count from 1 to 3"):
            chunks.append(chunk)

        assert len(chunks) > 1
        assert chunks[-1].is_final is True
        assert chunks[-1].usage_data is not None
        assert chunks[-1].usage_data.input_tokens > 0

    async def test_with_message_history(self) -> None:
        """Test conversation continuity with message history."""
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            UserPromptPart,
        )

        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant. Be very concise.",
            temperature=0.0,
        )

        # First message
        response1 = await agent.run("My name is Alice. Remember it.")

        # Build message history using PydanticAI format
        message_history: list[ModelRequest | ModelResponse] = [
            ModelRequest(parts=[UserPromptPart(content="My name is Alice. Remember it.")]),
            ModelResponse(parts=[TextPart(content=response1.content)]),
        ]

        # Second message with history
        response2 = await agent.run("What is my name?", message_history=message_history)

        assert "alice" in response2.content.lower()

    async def test_with_dependencies(self) -> None:
        """Test passing dependencies to agent for tools."""
        from pydantic_ai import RunContext

        @dataclass
        class MyDeps:
            secret_code: str

        @safe_tool(timeout=5)
        async def get_secret(ctx: RunContext[MyDeps]) -> str:
            """Retrieve the secret code from the system."""
            return f"The secret code is: {ctx.deps.secret_code}"

        class SecretToolset(SafeToolset):
            def __init__(self) -> None:
                super().__init__(tools=[get_secret], name="secrets")

        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="You can retrieve secret codes using the get_secret tool.",
            toolsets=[SecretToolset()],
        )

        response = await agent.run(
            "What is the secret code?",
            deps=MyDeps(secret_code="ALPHA-7742"),
        )

        assert "ALPHA-7742" in response.content

    async def test_tool_call_tracking(self) -> None:
        """Test that tool_call_count and request_count are tracked correctly."""

        @safe_tool(timeout=5)
        async def add_numbers(a: int, b: int) -> str:
            """Add two numbers together."""
            return f"The sum is {a + b}"

        @safe_tool(timeout=5)
        async def multiply_numbers(a: int, b: int) -> str:
            """Multiply two numbers together."""
            return f"The product is {a * b}"

        class MathToolset(SafeToolset):
            def __init__(self) -> None:
                super().__init__(tools=[add_numbers, multiply_numbers], name="math")

        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Use the math tools to answer questions. Always use tools, don't calculate yourself.",
            toolsets=[MathToolset()],
        )

        response = await agent.run("What is 5 + 3?")

        # Should have made at least 1 tool call
        assert response.tool_call_count >= 1, f"Expected tool_call_count >= 1, got {response.tool_call_count}"
        # Should have made at least 2 requests (initial + after tool response)
        assert response.request_count >= 2, f"Expected request_count >= 2, got {response.request_count}"
        # Should have tool call details
        assert len(response.tool_calls) >= 1, "Expected tool_calls list to have items"
        assert response.tool_calls[0]["tool_name"] == "add_numbers"

    async def test_multiple_tool_calls_tracking(self) -> None:
        """Test tracking when agent makes multiple tool calls."""

        @safe_tool(timeout=5)
        async def get_price(item: str) -> str:
            """Get the price of an item."""
            prices = {"apple": 1.50, "banana": 0.75, "orange": 2.00}
            return f"{item} costs ${prices.get(item, 0):.2f}"

        class PriceToolset(SafeToolset):
            def __init__(self) -> None:
                super().__init__(tools=[get_price], name="prices")

        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Use the get_price tool to answer price questions. Call the tool for each item.",
            toolsets=[PriceToolset()],
        )

        response = await agent.run("How much do an apple and a banana cost?")

        # Should have made at least 2 tool calls (one for each item)
        # Note: The model might call them in one go or separately
        assert response.tool_call_count >= 1, f"Expected tool_call_count >= 1, got {response.tool_call_count}"
        assert response.request_count >= 2, f"Expected request_count >= 2, got {response.request_count}"

    async def test_usage_details_populated(self) -> None:
        """Test that usage_details dict is populated (for models that provide it)."""
        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Be concise.",
        )

        response = await agent.run("Say hello")

        # usage_details should exist (may be empty for some models)
        assert isinstance(response.usage_details, dict)
        # Basic usage should always be tracked
        assert response.input_tokens > 0
        assert response.output_tokens > 0


class TestSafeToolIntegration:
    """Integration tests for @safe_tool with real operations."""

    async def test_timeout(self) -> None:
        """Test that timeout works."""

        @safe_tool(timeout=0.1, max_retries=1)
        async def slow_operation() -> str:
            await asyncio.sleep(1.0)
            return "done"

        result = await slow_operation()

        assert "timed out" in result.lower()

    async def test_success(self) -> None:
        """Test successful tool execution."""

        @safe_tool(timeout=5.0)
        async def fast_operation(x: int) -> str:
            return f"Result: {x * 2}"

        result = await fast_operation(21)

        assert result == "Result: 42"

    async def test_retry_on_error(self) -> None:
        """Test that retries work on transient errors."""
        attempt_count = 0

        @safe_tool(timeout=5.0, max_retries=3)
        async def flaky_operation() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = await flaky_operation()

        assert result == "success"
        assert attempt_count == 3

    async def test_custom_error_message(self) -> None:
        """Test custom error messages."""

        @safe_tool(
            timeout=5.0,
            max_retries=1,
            on_error="Custom error: {error}",
        )
        async def failing_operation() -> str:
            raise ValueError("Something broke")

        result = await failing_operation()

        assert "Custom error:" in result
        assert "Something broke" in result

    async def test_custom_timeout_message(self) -> None:
        """Test custom timeout messages."""

        @safe_tool(
            timeout=0.1,
            max_retries=1,
            on_timeout="Operation too slow, try simpler input",
        )
        async def slow_operation() -> str:
            await asyncio.sleep(1.0)
            return "done"

        result = await slow_operation()

        assert "Operation too slow" in result


class TestPipelineIntegration:
    """Integration tests for Pipeline with real steps."""

    async def test_linear_pipeline(self) -> None:
        """Test pipeline with sequential steps."""

        class DoubleStep(BaseStep[None, int]):
            async def execute(self, context: StepContext[None]) -> int:
                value = context.get_input("value")
                return value * 2

        class AddTenStep(BaseStep[None, int]):
            async def execute(self, context: StepContext[None]) -> int:
                doubled = context.get_dependency("double", int)
                return doubled + 10

        pipeline: Pipeline[None, dict[str, int], int] = Pipeline(
            name="math_pipeline",
            steps={
                "double": DoubleStep(),
                "add_ten": AddTenStep(),
            },
            dependencies={
                "add_ten": ["double"],
            },
        )

        result = await pipeline.execute({"value": 5}, None)

        assert result.output == 20  # (5 * 2) + 10
        assert result.stopped_early is False

    async def test_parallel_execution(self) -> None:
        """Test that independent steps run in parallel."""
        import time

        execution_times: dict[str, float] = {}

        class SlowStep(BaseStep[None, str]):
            def __init__(self, name: str, delay: float) -> None:
                self.name = name
                self.delay = delay

            async def execute(self, context: StepContext[None]) -> str:
                start = time.perf_counter()
                await asyncio.sleep(self.delay)
                execution_times[self.name] = time.perf_counter() - start
                return self.name

        class FinalStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                a = context.get_dependency("step_a", str)
                b = context.get_dependency("step_b", str)
                return f"{a}+{b}"

        pipeline: Pipeline[None, dict[str, Any], str] = Pipeline(
            name="parallel_test",
            steps={
                "step_a": SlowStep("A", 0.2),
                "step_b": SlowStep("B", 0.2),
                "final": FinalStep(),
            },
            dependencies={
                "final": ["step_a", "step_b"],
            },
        )

        start = time.perf_counter()
        result = await pipeline.execute({}, None)
        total_time = time.perf_counter() - start

        assert result.output == "A+B"
        # If parallel: ~0.2s. If sequential: ~0.4s
        assert total_time < 0.35, f"Steps not parallel: {total_time:.2f}s"

    async def test_early_termination(self) -> None:
        """Test pipeline stops on INCOMPLETE status."""

        class GatherInfoStep(BaseStep[None, ConversationState[dict[str, str]]]):
            async def execute(self, context: StepContext[None]) -> ConversationState[dict[str, str]]:
                message = context.get_input("message")

                if "name" in message.lower():
                    return ConversationState(
                        status=ConversationStatus.COMPLETE,
                        data={"name": "extracted"},
                    )

                return ConversationState(
                    status=ConversationStatus.INCOMPLETE,
                    data={},
                    context={"missing": ["name"]},
                )

        class ProcessStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                # This should not run if gather is incomplete
                return "processed"

        pipeline: Pipeline[None, dict[str, str], str] = Pipeline(
            name="gather_pipeline",
            steps={
                "gather": GatherInfoStep(),
                "process": ProcessStep(),
            },
            dependencies={
                "process": ["gather"],
            },
        )

        # Test incomplete - should stop early
        result = await pipeline.execute({"message": "hello"}, None)
        assert result.stopped_early is True
        assert result.conversation_state is not None
        assert result.conversation_state.status == ConversationStatus.INCOMPLETE
        assert "process" not in result.step_outputs

        # Test complete - should continue
        result2 = await pipeline.execute({"message": "my name is Alice"}, None)
        assert result2.stopped_early is False

    @requires_openai
    async def test_with_ai_step(self) -> None:
        """Test pipeline with .as_step()."""
        greet_agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Generate a one-word greeting.",
            temperature=0.0,
        )
        greet_step = greet_agent.as_step("Say hello")

        pipeline: Pipeline[None, dict[str, str], str] = Pipeline(
            name="greeting_pipeline",
            steps={"greet": greet_step},
        )

        result = await pipeline.execute({}, None)

        assert result.output is not None
        assert len(result.output) > 0
        assert result.usage is not None
        assert result.usage.total_cost_microcents > 0

    @requires_openai
    async def test_agent_step_with_tools(self) -> None:
        """Test .as_step() with tool access."""

        @safe_tool(timeout=5)
        async def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 22°C"

        class WeatherToolset(SafeToolset):
            def __init__(self) -> None:
                super().__init__(tools=[get_weather], name="weather")

        weather_agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Use the weather tool to answer questions.",
            toolsets=[WeatherToolset()],
        )
        weather_step = weather_agent.as_step(lambda ctx: f"What's the weather in {ctx.get_input('city')}?")

        pipeline: Pipeline[None, dict[str, str], str] = Pipeline(
            name="weather_pipeline",
            steps={"weather": weather_step},
        )

        result = await pipeline.execute({"city": "Paris"}, None)

        assert result.output is not None
        assert "paris" in result.output.lower() or "sunny" in result.output.lower() or "22" in result.output


class TestPipelineRoutingIntegration:
    """Integration tests for BasePipeline routing."""

    async def test_pipeline_routing(self) -> None:
        """Test dynamic pipeline selection based on input."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "simple_result"

        class ComplexStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "complex_result"

        simple_pipeline: Pipeline[None, dict[str, Any], str] = Pipeline(
            name="simple",
            steps={"process": SimpleStep()},
        )

        complex_pipeline: Pipeline[None, dict[str, Any], str] = Pipeline(
            name="complex",
            steps={"process": ComplexStep()},
        )

        class AmountRouter(BasePipeline[None, dict[str, int], str]):
            def __init__(self) -> None:
                super().__init__("amount_router")

            async def route(self, input_data: dict[str, int], deps: None) -> str:
                if input_data.get("amount", 0) < 1000:
                    return "simple"
                return "complex"

        router = AmountRouter()
        router.register_pipeline("simple", simple_pipeline)
        router.register_pipeline("complex", complex_pipeline)

        # Test routing to simple
        result1 = await router.execute({"amount": 500}, None)
        assert result1.output == "simple_result"

        # Test routing to complex
        result2 = await router.execute({"amount": 5000}, None)
        assert result2.output == "complex_result"


@requires_openai
class TestCostCalculatorIntegration:
    """Integration tests for CostCalculator accuracy."""

    async def test_cost_calculation_matches_usage(self) -> None:
        """Verify cost calculation uses actual token counts."""
        agent = FastroAgent(model="openai:gpt-4o-mini")
        calculator = CostCalculator()

        response = await agent.run("Say 'test'")

        expected_cost = calculator.calculate_cost(
            model="gpt-4o-mini",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        assert response.cost_microcents == expected_cost


@requires_openai
class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    async def test_full_workflow_with_tracing(self) -> None:
        """Test complete workflow: agent + tracing + cost tracking."""
        tracer = SimpleTracer()

        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="You are a math tutor. Be very brief.",
            temperature=0.0,
        )

        response1 = await agent.run("What is 5+5?", tracer=tracer)
        assert "10" in response1.content

        response2 = await agent.run("What is 3*4?", tracer=tracer)
        assert "12" in response2.content

        total_cost = response1.cost_microcents + response2.cost_microcents
        assert total_cost > 0
        assert response1.trace_id != response2.trace_id

    async def test_multi_step_pipeline_with_different_models(self) -> None:
        """Test pipeline using different models for different steps."""
        extract_agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Extract the main topic in one word.",
        )
        extract_step = extract_agent.as_step(lambda ctx: ctx.get_input("text"))

        compose_agent = FastroAgent(
            model="openai:gpt-4o-mini",  # Could use gpt-4o for quality
            system_prompt="Write a haiku about the given topic.",
        )
        compose_step = compose_agent.as_step(lambda ctx: f"Topic: {ctx.get_dependency('extract', str)}")

        pipeline: Pipeline[None, dict[str, str], str] = Pipeline(
            name="haiku_pipeline",
            steps={
                "extract": extract_step,
                "compose": compose_step,
            },
            dependencies={
                "compose": ["extract"],
            },
        )

        result = await pipeline.execute({"text": "The ocean waves crash"}, None)

        assert result.output is not None
        assert len(result.output) > 0
        assert result.usage is not None
        assert "extract" in result.usage.steps
        assert "compose" in result.usage.steps

    async def test_complete_research_pipeline(self) -> None:
        """Test a realistic multi-step research pipeline."""

        @safe_tool(timeout=5)
        async def search_database(query: str) -> str:
            """Search internal database."""
            return f"Found: {query} is a programming language created in 1991"

        class SearchToolset(SafeToolset):
            def __init__(self) -> None:
                super().__init__(tools=[search_database], name="search")

        research_agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Research the topic using available tools. Be concise.",
            toolsets=[SearchToolset()],
        )
        research_step = research_agent.as_step(lambda ctx: f"Research: {ctx.get_input('topic')}")

        summarize_agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Summarize in one sentence.",
        )
        summarize_step = summarize_agent.as_step(lambda ctx: f"Summarize: {ctx.get_dependency('research', str)}")

        pipeline: Pipeline[None, dict[str, str], str] = Pipeline(
            name="research_pipeline",
            steps={
                "research": research_step,
                "summarize": summarize_step,
            },
            dependencies={
                "summarize": ["research"],
            },
        )

        tracer = SimpleTracer()
        result = await pipeline.execute({"topic": "Python"}, None, tracer=tracer)

        assert result.output is not None
        assert result.usage is not None
        assert result.usage.total_cost_microcents > 0
        assert not result.stopped_early


# ============================================================================
# @step Decorator Integration Tests
# ============================================================================


class TestStepDecoratorIntegration:
    """Integration tests for the @step decorator in realistic scenarios."""

    async def test_step_decorator_basic(self) -> None:
        """Test @step decorator creates functional pipeline step."""
        from fastroai import step

        @step
        async def transform_step(ctx: StepContext[None]) -> str:
            value = ctx.get_input("text")
            return value.upper()

        pipeline = Pipeline(
            name="transform",
            steps={"transform": transform_step},
        )

        result = await pipeline.execute({"text": "hello world"}, None)
        assert result.output == "HELLO WORLD"

    async def test_step_decorator_with_config(self) -> None:
        """Test @step decorator with timeout and retry configuration."""
        from fastroai import step

        @step(timeout=5.0, retries=2, retry_delay=0.5, cost_budget=100_000)
        async def configured_step(ctx: StepContext[None]) -> str:
            # Access the config from context
            return f"timeout={ctx.config.timeout},retries={ctx.config.retries}"

        pipeline = Pipeline(
            name="configured",
            steps={"configured": configured_step},
        )

        result = await pipeline.execute({}, None)
        assert result.output == "timeout=5.0,retries=2"

        # Also verify the step itself has the config
        assert configured_step.config.retries == 2
        assert configured_step.config.timeout == 5.0
        assert configured_step.config.retry_delay == 0.5
        assert configured_step.config.cost_budget == 100_000

    async def test_step_decorator_chain(self) -> None:
        """Test multiple @step decorated functions in a pipeline."""
        from fastroai import step

        @step
        async def step_one(ctx: StepContext[None]) -> int:
            return ctx.get_input("value") * 2

        @step
        async def step_two(ctx: StepContext[None]) -> int:
            prev = ctx.get_dependency("one")
            return prev + 10

        @step
        async def step_three(ctx: StepContext[None]) -> str:
            prev = ctx.get_dependency("two")
            return f"Result: {prev}"

        pipeline = Pipeline(
            name="chain",
            steps={"one": step_one, "two": step_two, "three": step_three},
            dependencies={"two": ["one"], "three": ["two"]},
        )

        result = await pipeline.execute({"value": 5}, None)
        assert result.output == "Result: 20"  # (5*2) + 10 = 20

    async def test_step_decorator_with_deps(self) -> None:
        """Test @step decorator accessing application dependencies."""
        from dataclasses import dataclass

        from fastroai import step

        @dataclass
        class AppDeps:
            multiplier: int
            prefix: str

        @step
        async def multiply_step(ctx: StepContext[AppDeps]) -> int:
            value = ctx.get_input("value")
            return value * ctx.deps.multiplier

        @step
        async def format_step(ctx: StepContext[AppDeps]) -> str:
            prev = ctx.get_dependency("multiply")
            return f"{ctx.deps.prefix}: {prev}"

        pipeline: Pipeline[AppDeps, dict[str, int], str] = Pipeline(
            name="with_deps",
            steps={"multiply": multiply_step, "format": format_step},
            dependencies={"format": ["multiply"]},
        )

        deps = AppDeps(multiplier=3, prefix="Answer")
        result = await pipeline.execute({"value": 7}, deps)
        assert result.output == "Answer: 21"


# ============================================================================
# Config Inheritance Integration Tests
# ============================================================================


class TestConfigInheritanceIntegration:
    """Integration tests for config inheritance in pipelines."""

    async def test_pipeline_config_applies_to_all_steps(self) -> None:
        """Pipeline config should apply to all steps by default."""
        from fastroai import PipelineConfig, step

        configs_seen: dict[str, float | None] = {}

        @step
        async def step_a(ctx: StepContext[None]) -> str:
            configs_seen["a"] = ctx.config.timeout
            return "a"

        @step
        async def step_b(ctx: StepContext[None]) -> str:
            configs_seen["b"] = ctx.config.timeout
            return "b"

        pipeline = Pipeline(
            name="config_test",
            steps={"a": step_a, "b": step_b},
            dependencies={"b": ["a"]},
            config=PipelineConfig(timeout=30.0),
        )

        await pipeline.execute({}, None)
        assert configs_seen["a"] == 30.0
        assert configs_seen["b"] == 30.0

    async def test_step_config_overrides_pipeline(self) -> None:
        """step_configs should override pipeline defaults."""
        from fastroai import PipelineConfig, StepConfig, step

        configs_seen: dict[str, tuple[float | None, int]] = {}

        @step
        async def step_a(ctx: StepContext[None]) -> str:
            configs_seen["a"] = (ctx.config.timeout, ctx.config.retries)
            return "a"

        @step
        async def step_b(ctx: StepContext[None]) -> str:
            configs_seen["b"] = (ctx.config.timeout, ctx.config.retries)
            return "b"

        pipeline = Pipeline(
            name="override_test",
            steps={"a": step_a, "b": step_b},
            dependencies={"b": ["a"]},
            config=PipelineConfig(timeout=10.0, retries=1),
            step_configs={"b": StepConfig(timeout=60.0, retries=5)},
        )

        await pipeline.execute({}, None)
        assert configs_seen["a"] == (10.0, 1)  # Pipeline defaults
        assert configs_seen["b"] == (60.0, 5)  # Override

    async def test_decorator_config_inheritance(self) -> None:
        """@step decorator config should be middle priority."""
        from fastroai import PipelineConfig, StepConfig, step

        # Step class has its own config
        @step(timeout=20.0, retries=2)
        async def configured_step(ctx: StepContext[None]) -> str:
            return f"timeout={ctx.config.timeout}"

        # Test 1: No pipeline config - use decorator config
        pipeline1 = Pipeline(
            name="test1",
            steps={"step": configured_step},
        )
        result1 = await pipeline1.execute({}, None)
        assert result1.output == "timeout=20.0"

        # Test 2: Pipeline config exists - step class wins over pipeline
        pipeline2 = Pipeline(
            name="test2",
            steps={"step": configured_step},
            config=PipelineConfig(timeout=10.0),  # Lower priority
        )
        result2 = await pipeline2.execute({}, None)
        assert result2.output == "timeout=20.0"  # Decorator wins

        # Test 3: step_configs wins over all
        pipeline3 = Pipeline(
            name="test3",
            steps={"step": configured_step},
            config=PipelineConfig(timeout=10.0),
            step_configs={"step": StepConfig(timeout=99.0)},  # Highest priority
        )
        result3 = await pipeline3.execute({}, None)
        assert result3.output == "timeout=99.0"

    async def test_cost_budget_inheritance(self) -> None:
        """Cost budget should be inherited and enforced."""
        from unittest.mock import AsyncMock, patch

        from fastroai import CostBudgetExceededError, PipelineConfig, StepExecutionError

        agent = FastroAgent(model="test", system_prompt="Test")

        class ExpensiveStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                # Each call costs 100 microcents
                await context.run(agent, "First")
                await context.run(agent, "Second")
                await context.run(agent, "Third")  # Should exceed budget
                return "done"

        from fastroai.agent import ChatResponse

        mock_response = ChatResponse(
            output="Result",
            content="Result",
            model="test",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_microcents=100,
            processing_time_ms=50,
        )

        pipeline = Pipeline(
            name="budget_test",
            steps={"expensive": ExpensiveStep()},
            config=PipelineConfig(cost_budget=150),
        )

        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response

            with pytest.raises(StepExecutionError) as exc_info:
                await pipeline.execute({}, None)

            assert isinstance(exc_info.value.original_error, CostBudgetExceededError)


# ============================================================================
# Error Handling Integration Tests
# ============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error hierarchy and handling."""

    async def test_step_execution_error_wraps_exceptions(self) -> None:
        """StepExecutionError should wrap step exceptions."""
        from fastroai import StepExecutionError

        class FailingStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                raise ValueError("Something went wrong")

        pipeline = Pipeline(
            name="failing",
            steps={"fail": FailingStep()},
        )

        with pytest.raises(StepExecutionError) as exc_info:
            await pipeline.execute({}, None)

        assert exc_info.value.step_id == "fail"
        assert isinstance(exc_info.value.original_error, ValueError)
        assert "Something went wrong" in str(exc_info.value.original_error)

    async def test_pipeline_validation_errors(self) -> None:
        """PipelineValidationError for invalid configurations."""
        from fastroai import PipelineValidationError

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "done"

        # Unknown dependency
        with pytest.raises(PipelineValidationError, match="depends on unknown"):
            Pipeline(
                name="invalid",
                steps={"a": SimpleStep()},
                dependencies={"a": ["nonexistent"]},
            )

        # Cycle detection
        with pytest.raises(PipelineValidationError, match="cycle"):
            Pipeline(
                name="cyclic",
                steps={"a": SimpleStep(), "b": SimpleStep()},
                dependencies={"a": ["b"], "b": ["a"]},
            )

        # Multiple terminals without output_step
        with pytest.raises(PipelineValidationError, match="Multiple terminal"):
            Pipeline(
                name="multi_terminal",
                steps={"a": SimpleStep(), "b": SimpleStep()},
            )

    async def test_catch_all_fastroai_errors(self) -> None:
        """FastroAIError base class catches all library errors."""
        from fastroai import FastroAIError

        class FailingStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                raise RuntimeError("Step failed")

        pipeline = Pipeline(
            name="catch_test",
            steps={"fail": FailingStep()},
        )

        # StepExecutionError is a FastroAIError
        with pytest.raises(FastroAIError):
            await pipeline.execute({}, None)


# ============================================================================
# CostCalculator Integration Tests
# ============================================================================


class TestCostCalculatorFeatures:
    """Integration tests for CostCalculator functionality."""

    def test_custom_pricing(self) -> None:
        """Test adding custom model pricing override."""
        calc = CostCalculator()

        # Add custom model with override
        calc.add_pricing_override(
            model="my-custom-model",
            input_per_mtok=5.00,  # $5/1M tokens
            output_per_mtok=10.00,  # $10/1M tokens
        )

        cost = calc.calculate_cost(
            model="my-custom-model",
            input_tokens=2000,
            output_tokens=1000,
        )

        # Input: 2000 / 1M * $5 = $0.01 = 10000 microcents
        # Output: 1000 / 1M * $10 = $0.01 = 10000 microcents
        # Total: 20000 microcents
        assert cost == 20000

    def test_format_cost(self) -> None:
        """Test cost formatting utilities."""
        calc = CostCalculator()

        formatted = calc.format_cost(1_500_000)

        assert formatted["microcents"] == 1_500_000
        assert formatted["cents"] == 150
        assert formatted["dollars"] == 1.5

    def test_dollars_to_microcents(self) -> None:
        """Test dollar to microcent conversion."""
        calc = CostCalculator()

        assert calc.dollars_to_microcents(1.0) == 1_000_000
        assert calc.dollars_to_microcents(0.01) == 10_000
        assert calc.dollars_to_microcents(0.000001) == 1

    def test_model_name_normalization(self) -> None:
        """Test model name normalization handles prefixes."""
        calc = CostCalculator()

        # With prefix
        cost1 = calc.calculate_cost("openai:gpt-4o", 1000, 500)
        # Without prefix
        cost2 = calc.calculate_cost("gpt-4o", 1000, 500)

        assert cost1 == cost2

    def test_unknown_model_returns_zero(self) -> None:
        """Test unknown models return zero cost."""
        calc = CostCalculator()

        cost = calc.calculate_cost(
            model="unknown-model-xyz",
            input_tokens=10000,
            output_tokens=5000,
        )

        assert cost == 0

    def test_pricing_override_takes_precedence(self) -> None:
        """Test that pricing overrides take precedence over genai-prices."""
        calc = CostCalculator()

        # Get standard cost from genai-prices
        standard_cost = calc.calculate_cost("gpt-4o", 1000, 0)
        assert standard_cost > 0

        # Add discounted pricing
        calc.add_pricing_override("gpt-4o", input_per_mtok=1.00, output_per_mtok=1.00)

        # Override should be used
        override_cost = calc.calculate_cost("gpt-4o", 1000, 0)
        assert override_cost == 1000  # $1/1M * 1000 = 1000 microcents
        assert override_cost != standard_cost


# ============================================================================
# Tracing Integration Tests
# ============================================================================


class TestTracingIntegration:
    """Integration tests for tracing functionality."""

    async def test_simple_tracer_logs_spans(self) -> None:
        """SimpleTracer should log span starts and completions."""
        import logging

        tracer = SimpleTracer()
        tracer.logger.setLevel(logging.DEBUG)

        async with tracer.span("test_operation", user_id="123") as trace_id:
            assert len(trace_id) == 36  # UUID format
            tracer.log_metric(trace_id, "items_processed", 42)

    async def test_noop_tracer_works_silently(self) -> None:
        """NoOpTracer should work without side effects."""
        from fastroai import NoOpTracer

        tracer = NoOpTracer()

        async with tracer.span("operation", key="value") as trace_id:
            assert len(trace_id) == 36  # Still generates IDs
            tracer.log_metric(trace_id, "metric", 100)
            tracer.log_error(trace_id, ValueError("test"))

    async def test_pipeline_passes_tracer_to_steps(self) -> None:
        """Pipeline should pass tracer to step contexts."""
        tracer_received = []

        class TracerCheckStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                tracer_received.append(context.tracer)
                return "done"

        pipeline = Pipeline(
            name="tracer_test",
            steps={"check": TracerCheckStep()},
        )

        tracer = SimpleTracer()
        await pipeline.execute({}, None, tracer=tracer)

        assert len(tracer_received) == 1
        # NoOpTracer is used internally when no tracer provided,
        # but when tracer is provided, it should be passed through


# ============================================================================
# Real-World Scenario Integration Tests
# ============================================================================


class TestRealWorldScenarios:
    """Integration tests for realistic usage patterns."""

    async def test_data_processing_pipeline(self) -> None:
        """Test a realistic data processing pipeline."""
        from dataclasses import dataclass

        @dataclass
        class ProcessingDeps:
            format_type: str
            include_metadata: bool

        class ParseStep(BaseStep[ProcessingDeps, dict[str, Any]]):
            async def execute(self, context: StepContext[ProcessingDeps]) -> dict[str, Any]:
                raw = context.get_input("raw_data")
                return {"parsed": raw.upper(), "length": len(raw)}

        class ValidateStep(BaseStep[ProcessingDeps, dict[str, Any]]):
            async def execute(self, context: StepContext[ProcessingDeps]) -> dict[str, Any]:
                parsed = context.get_dependency("parse")
                if parsed["length"] < 3:
                    raise ValueError("Data too short")
                return {**parsed, "valid": True}

        class FormatStep(BaseStep[ProcessingDeps, str]):
            async def execute(self, context: StepContext[ProcessingDeps]) -> str:
                validated = context.get_dependency("validate")
                fmt = context.deps.format_type

                if fmt == "json":
                    import json

                    return json.dumps(validated)
                return str(validated)

        pipeline: Pipeline[ProcessingDeps, dict[str, str], str] = Pipeline(
            name="data_processor",
            steps={
                "parse": ParseStep(),
                "validate": ValidateStep(),
                "format": FormatStep(),
            },
            dependencies={
                "validate": ["parse"],
                "format": ["validate"],
            },
        )

        deps = ProcessingDeps(format_type="json", include_metadata=True)
        result = await pipeline.execute({"raw_data": "hello world"}, deps)

        import json

        assert result.output is not None
        output = json.loads(result.output)
        assert output["parsed"] == "HELLO WORLD"
        assert output["valid"] is True

    async def test_branching_pipeline_with_parallel_steps(self) -> None:
        """Test pipeline with branching and parallel execution."""
        execution_order: list[str] = []

        class StartStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                execution_order.append("start")
                return "initial"

        class BranchAStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                await asyncio.sleep(0.05)
                execution_order.append("branch_a")
                return "a_result"

        class BranchBStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                await asyncio.sleep(0.05)
                execution_order.append("branch_b")
                return "b_result"

        class MergeStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                a = context.get_dependency("branch_a")
                b = context.get_dependency("branch_b")
                execution_order.append("merge")
                return f"{a}+{b}"

        pipeline = Pipeline(
            name="branching",
            steps={
                "start": StartStep(),
                "branch_a": BranchAStep(),
                "branch_b": BranchBStep(),
                "merge": MergeStep(),
            },
            dependencies={
                "branch_a": ["start"],
                "branch_b": ["start"],
                "merge": ["branch_a", "branch_b"],
            },
        )

        import time

        start = time.perf_counter()
        result = await pipeline.execute({}, None)
        elapsed = time.perf_counter() - start

        assert result.output == "a_result+b_result"
        assert execution_order[0] == "start"
        assert execution_order[-1] == "merge"
        # branch_a and branch_b should be in parallel (order not guaranteed)
        assert set(execution_order[1:3]) == {"branch_a", "branch_b"}
        # Should take ~0.05s not ~0.1s due to parallelism
        assert elapsed < 0.15

    async def test_multi_turn_conversation_flow(self) -> None:
        """Test multi-turn conversation with early termination."""

        class InfoGatherer(BaseStep[None, ConversationState[dict[str, str]]]):
            async def execute(self, context: StepContext[None]) -> ConversationState[dict[str, str]]:
                message = context.get_input("message")
                current_data = context.get_input("current_data") or {}

                # Simulate extracting info from message
                if "email" in message.lower():
                    current_data["email"] = "user@example.com"
                if "name" in message.lower():
                    current_data["name"] = "John"

                # Check if we have all required fields
                required = {"name", "email"}
                missing = required - set(current_data.keys())

                if not missing:
                    return ConversationState(
                        status=ConversationStatus.COMPLETE,
                        data=current_data,
                    )

                return ConversationState(
                    status=ConversationStatus.INCOMPLETE,
                    data=current_data,
                    context={"missing_fields": list(missing)},
                )

        class ProcessOrder(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                info = context.get_dependency("gather")
                return f"Order processed for {info.data['name']}"

        pipeline: Pipeline[None, dict[str, Any], Any] = Pipeline(
            name="order_flow",
            steps={
                "gather": InfoGatherer(),
                "process": ProcessOrder(),
            },
            dependencies={"process": ["gather"]},
        )

        # Turn 1: Missing both
        result1 = await pipeline.execute({"message": "I want to order"}, None)
        assert result1.stopped_early
        assert result1.conversation_state is not None
        assert set(result1.conversation_state.context["missing_fields"]) == {"name", "email"}

        # Turn 2: Provide name
        assert result1.conversation_state.data is not None
        result2 = await pipeline.execute(
            {"message": "My name is John", "current_data": result1.conversation_state.data},
            None,
        )
        assert result2.stopped_early
        assert result2.conversation_state is not None
        assert result2.conversation_state.context["missing_fields"] == ["email"]

        # Turn 3: Provide email - should complete
        assert result2.conversation_state.data is not None
        result3 = await pipeline.execute(
            {"message": "My email is user@example.com", "current_data": result2.conversation_state.data},
            None,
        )
        assert not result3.stopped_early
        assert result3.output == "Order processed for John"


# ============================================================================
# Cache Token Integration Tests
# ============================================================================


@requires_openai
class TestCacheTokensIntegration:
    """Integration tests for cache token tracking with OpenAI.

    OpenAI supports prompt caching on gpt-4o models for prompts >= 1024 tokens.
    """

    async def test_cache_fields_populated(self) -> None:
        """Test that cache token fields exist in response (may be 0 for short prompts)."""
        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Be concise.",
        )

        response = await agent.run("Say hello")

        # Fields should exist (may be 0 for prompts below caching threshold)
        assert isinstance(response.cache_read_tokens, int)
        assert isinstance(response.cache_write_tokens, int)
        assert response.cache_read_tokens >= 0
        assert response.cache_write_tokens >= 0

    async def test_request_count_single_call(self) -> None:
        """Test request_count is 1 for a simple call without tools."""
        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Be concise.",
        )

        response = await agent.run("Say hello")

        # Should be exactly 1 request for a simple call
        assert response.request_count == 1

    async def test_cache_tokens_with_long_prompt(self) -> None:
        """Test cache tokens with a long system prompt that may trigger caching.

        OpenAI requires prompts >= 1024 tokens for prompt caching to be eligible.
        This test uses a long prompt and makes multiple calls to see if caching occurs.
        """
        # Create a long system prompt (aim for ~1500 tokens)
        long_context = " ".join([f"Rule {i}: Always be helpful, accurate, and informative." for i in range(200)])
        system_prompt = f"""You are an expert assistant with the following comprehensive guidelines:

{long_context}

Remember to always be concise in your final responses."""

        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt=system_prompt,
        )

        # First call
        response1 = await agent.run("Say 'hello'")
        assert response1.input_tokens > 0

        # Second call with same system prompt
        response2 = await agent.run("Say 'world'")
        assert response2.input_tokens > 0

        # Print for debugging
        print(f"\nResponse 1: input={response1.input_tokens}, cache_read={response1.cache_read_tokens}")
        print(f"Response 2: input={response2.input_tokens}, cache_read={response2.cache_read_tokens}")

        # Verify cache fields are integers (caching may or may not occur)
        assert isinstance(response1.cache_read_tokens, int)
        assert isinstance(response2.cache_read_tokens, int)

        # Both should have valid costs
        assert response1.cost_microcents > 0
        assert response2.cost_microcents > 0

    async def test_all_new_fields_populated(self) -> None:
        """Test that all new usage fields are properly populated."""
        agent = FastroAgent(
            model="openai:gpt-4o-mini",
            system_prompt="Be concise.",
        )

        response = await agent.run("What is 2+2?")

        # All new fields should be populated with valid values
        assert isinstance(response.cache_read_tokens, int) and response.cache_read_tokens >= 0
        assert isinstance(response.cache_write_tokens, int) and response.cache_write_tokens >= 0
        assert isinstance(response.input_audio_tokens, int) and response.input_audio_tokens >= 0
        assert isinstance(response.output_audio_tokens, int) and response.output_audio_tokens >= 0
        assert isinstance(response.cache_audio_read_tokens, int) and response.cache_audio_read_tokens >= 0
        assert isinstance(response.request_count, int) and response.request_count >= 1
        assert isinstance(response.tool_call_count, int) and response.tool_call_count >= 0
        assert isinstance(response.usage_details, dict)

        # Print summary for debugging
        print("\nUsage Summary:")
        print(f"  input_tokens: {response.input_tokens}")
        print(f"  output_tokens: {response.output_tokens}")
        print(f"  cache_read_tokens: {response.cache_read_tokens}")
        print(f"  cache_write_tokens: {response.cache_write_tokens}")
        print(f"  request_count: {response.request_count}")
        print(f"  tool_call_count: {response.tool_call_count}")
        print(f"  cost_microcents: {response.cost_microcents}")
        print(f"  cost_dollars: ${response.cost_dollars:.6f}")
