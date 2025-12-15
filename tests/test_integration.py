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
