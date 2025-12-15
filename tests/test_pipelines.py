"""Tests for the pipelines module."""
# mypy: disable-error-code="var-annotated,arg-type"

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from fastroai.agent import AgentStepWrapper, ChatResponse, FastroAgent
from fastroai.pipelines import (
    BasePipeline,
    BaseStep,
    ConversationState,
    ConversationStatus,
    Pipeline,
    PipelineUsage,
    StepContext,
    StepExecutionError,
    StepUsage,
)

# ============================================================================
# StepUsage Tests
# ============================================================================


class TestStepUsage:
    """Tests for StepUsage."""

    def test_create_step_usage(self) -> None:
        """Should create StepUsage with fields."""
        usage = StepUsage(
            input_tokens=100,
            output_tokens=50,
            cost_microcents=175,
            processing_time_ms=500,
            model="gpt-4o",
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cost_microcents == 175
        assert usage.processing_time_ms == 500
        assert usage.model == "gpt-4o"

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        usage = StepUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cost_microcents == 0
        assert usage.processing_time_ms == 0
        assert usage.model is None

    def test_from_chat_response(self) -> None:
        """Should create from ChatResponse."""
        response = ChatResponse(
            output="Test",
            content="Test",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_microcents=175,
            processing_time_ms=500,
        )
        usage = StepUsage.from_chat_response(response)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cost_microcents == 175
        assert usage.processing_time_ms == 500
        assert usage.model == "gpt-4o"

    def test_add_usages(self) -> None:
        """Should combine two usages."""
        usage1 = StepUsage(
            input_tokens=100,
            output_tokens=50,
            cost_microcents=175,
            processing_time_ms=500,
            model="gpt-4o",
        )
        usage2 = StepUsage(
            input_tokens=200,
            output_tokens=100,
            cost_microcents=350,
            processing_time_ms=300,
        )
        combined = usage1 + usage2
        assert combined.input_tokens == 300
        assert combined.output_tokens == 150
        assert combined.cost_microcents == 525
        assert combined.processing_time_ms == 800
        assert combined.model == "gpt-4o"  # First non-None model


# ============================================================================
# PipelineUsage Tests
# ============================================================================


class TestPipelineUsage:
    """Tests for PipelineUsage."""

    def test_from_step_usages(self) -> None:
        """Should aggregate from step usages."""
        step_usages = {
            "extract": StepUsage(input_tokens=100, output_tokens=50, cost_microcents=100),
            "classify": StepUsage(input_tokens=200, output_tokens=100, cost_microcents=200),
        }
        usage = PipelineUsage.from_step_usages(step_usages)
        assert usage.total_input_tokens == 300
        assert usage.total_output_tokens == 150
        assert usage.total_cost_microcents == 300
        assert len(usage.steps) == 2
        assert "extract" in usage.steps
        assert "classify" in usage.steps

    def test_total_cost_dollars(self) -> None:
        """Should calculate cost in dollars."""
        usage = PipelineUsage(total_cost_microcents=1_000_000)
        assert usage.total_cost_dollars == 1.0

        usage = PipelineUsage(total_cost_microcents=500)
        assert usage.total_cost_dollars == 0.0005


# ============================================================================
# ConversationState Tests
# ============================================================================


class TestConversationState:
    """Tests for ConversationState."""

    def test_complete_state(self) -> None:
        """Should create COMPLETE state."""
        state = ConversationState(
            status=ConversationStatus.COMPLETE,
            data={"result": "done"},
        )
        assert state.status == ConversationStatus.COMPLETE
        assert state.data == {"result": "done"}
        assert state.context == {}

    def test_incomplete_state(self) -> None:
        """Should create INCOMPLETE state with context."""
        state = ConversationState(
            status=ConversationStatus.INCOMPLETE,
            data={"partial": "data"},
            context={"missing": ["field1", "field2"]},
        )
        assert state.status == ConversationStatus.INCOMPLETE
        assert state.data == {"partial": "data"}
        assert state.context == {"missing": ["field1", "field2"]}


# ============================================================================
# StepContext Tests
# ============================================================================


class TestStepContext:
    """Tests for StepContext."""

    def test_create_context(self) -> None:
        """Should create context with all fields."""
        context = StepContext(
            step_id="test_step",
            inputs={"key": "value"},
            deps={"db": "session"},
            step_outputs={"prev_step": "output"},
        )
        assert context.step_id == "test_step"
        assert context.deps == {"db": "session"}
        assert context.tracer is None

    def test_get_input(self) -> None:
        """Should get input value."""
        context = StepContext(
            step_id="test",
            inputs={"document": "Hello World"},
            deps=None,
            step_outputs={},
        )
        assert context.get_input("document") == "Hello World"
        assert context.get_input("missing") is None
        assert context.get_input("missing", "default") == "default"

    def test_get_dependency(self) -> None:
        """Should get dependency output."""
        context = StepContext(
            step_id="classify",
            inputs={},
            deps=None,
            step_outputs={"extract": "extracted_text"},
        )
        assert context.get_dependency("extract") == "extracted_text"

    def test_get_dependency_missing_raises(self) -> None:
        """Should raise ValueError for missing dependency."""
        context = StepContext(
            step_id="classify",
            inputs={},
            deps=None,
            step_outputs={},
        )
        with pytest.raises(ValueError, match="not a dependency"):
            context.get_dependency("extract")

    def test_get_dependency_or_none(self) -> None:
        """Should return None for missing optional dependency."""
        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={"a": "value_a"},
        )
        assert context.get_dependency_or_none("a") == "value_a"
        assert context.get_dependency_or_none("b") is None


# ============================================================================
# BaseStep Tests
# ============================================================================


class TestBaseStep:
    """Tests for BaseStep."""

    def test_step_is_abstract(self) -> None:
        """BaseStep.execute should be abstract."""
        # Can't instantiate directly
        with pytest.raises(TypeError):
            BaseStep()  # type: ignore

    async def test_concrete_step_implementation(self) -> None:
        """Should be able to implement concrete step."""

        class UppercaseStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                text = context.get_input("text")
                return text.upper()

        step = UppercaseStep()
        context = StepContext(
            step_id="upper",
            inputs={"text": "hello"},
            deps=None,
            step_outputs={},
        )
        result = await step.execute(context)
        assert result == "HELLO"


# ============================================================================
# Pipeline Executor Tests
# ============================================================================


class TestPipelineExecutor:
    """Tests for pipeline executor (via Pipeline)."""

    async def test_validates_unknown_step_reference(self) -> None:
        """Should raise on unknown step in dependencies."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "result"

        with pytest.raises(ValueError, match="depends on unknown"):
            Pipeline(
                name="test",
                steps={"a": SimpleStep()},
                dependencies={"a": ["unknown_step"]},
            )

    async def test_validates_unknown_step_in_deps(self) -> None:
        """Should raise on dependency for unknown step."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "result"

        with pytest.raises(ValueError, match="Dependency for unknown step"):
            Pipeline(
                name="test",
                steps={"a": SimpleStep()},
                dependencies={"unknown": ["a"]},
            )

    async def test_validates_cycles(self) -> None:
        """Should detect cycles in dependencies."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "result"

        with pytest.raises(ValueError, match="cycle"):
            Pipeline(
                name="test",
                steps={"a": SimpleStep(), "b": SimpleStep()},
                dependencies={"a": ["b"], "b": ["a"]},
            )


# ============================================================================
# Pipeline Tests
# ============================================================================


class TestPipeline:
    """Tests for Pipeline."""

    async def test_simple_single_step_pipeline(self) -> None:
        """Should execute single step pipeline."""

        class UpperStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return context.get_input("text").upper()

        pipeline = Pipeline(
            name="simple",
            steps={"upper": UpperStep()},
        )

        result = await pipeline.execute({"text": "hello"}, None)
        assert result.output == "HELLO"
        assert result.stopped_early is False

    async def test_linear_pipeline(self) -> None:
        """Should execute linear pipeline in order."""
        execution_order: list[str] = []

        class StepA(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                execution_order.append("a")
                return "a_result"

        class StepB(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                execution_order.append("b")
                prev = context.get_dependency("a")
                return f"{prev}_b"

        pipeline = Pipeline(
            name="linear",
            steps={"a": StepA(), "b": StepB()},
            dependencies={"b": ["a"]},
        )

        result = await pipeline.execute({}, None)
        assert execution_order == ["a", "b"]
        assert result.output == "a_result_b"

    async def test_parallel_execution(self) -> None:
        """Should execute independent steps in parallel."""
        execution_times: dict[str, float] = {}

        class SlowStep(BaseStep[None, str]):
            def __init__(self, name: str, delay: float):
                self.name = name
                self.delay = delay

            async def execute(self, context: StepContext[None]) -> str:
                start = time.perf_counter()
                await asyncio.sleep(self.delay)
                execution_times[self.name] = time.perf_counter() - start
                return f"{self.name}_done"

        class FinalStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                b = context.get_dependency("b")
                c = context.get_dependency("c")
                return f"{b}_{c}"

        pipeline = Pipeline(
            name="parallel",
            steps={
                "a": SlowStep("a", 0.01),
                "b": SlowStep("b", 0.05),
                "c": SlowStep("c", 0.05),
                "d": FinalStep(),
            },
            dependencies={
                "b": ["a"],
                "c": ["a"],
                "d": ["b", "c"],
            },
        )

        start = time.perf_counter()
        result = await pipeline.execute({}, None)
        total_time = time.perf_counter() - start

        # If b and c ran in parallel, total time should be < sum of all steps
        # Sequential would be: 0.01 + 0.05 + 0.05 = 0.11
        # Parallel should be: 0.01 + max(0.05, 0.05) ≈ 0.06
        assert total_time < 0.10  # Allow some overhead
        assert result.output == "b_done_c_done"

    async def test_early_termination_on_incomplete(self) -> None:
        """Should stop pipeline on INCOMPLETE status."""
        executed_steps: list[str] = []

        class GatherStep(BaseStep[None, ConversationState[dict]]):
            async def execute(self, context: StepContext[None]) -> ConversationState[dict]:
                executed_steps.append("gather")
                return ConversationState(
                    status=ConversationStatus.INCOMPLETE,
                    data={"partial": True},
                    context={"missing": ["field1"]},
                )

        class CalculateStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                executed_steps.append("calculate")
                return "calculated"

        pipeline = Pipeline(
            name="multi_turn",
            steps={"gather": GatherStep(), "calculate": CalculateStep()},
            dependencies={"calculate": ["gather"]},
            output_step="calculate",
        )

        result = await pipeline.execute({}, None)
        assert executed_steps == ["gather"]  # calculate should not run
        assert result.stopped_early is True
        assert result.conversation_state is not None
        assert result.conversation_state.status == ConversationStatus.INCOMPLETE
        assert result.output is None

    async def test_complete_conversation_continues(self) -> None:
        """Should continue after COMPLETE status."""
        executed_steps: list[str] = []

        class GatherStep(BaseStep[None, ConversationState[dict]]):
            async def execute(self, context: StepContext[None]) -> ConversationState[dict]:
                executed_steps.append("gather")
                return ConversationState(
                    status=ConversationStatus.COMPLETE,
                    data={"all_info": True},
                )

        class CalculateStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                executed_steps.append("calculate")
                return "calculated"

        pipeline = Pipeline(
            name="multi_turn",
            steps={"gather": GatherStep(), "calculate": CalculateStep()},
            dependencies={"calculate": ["gather"]},
            output_step="calculate",
        )

        result = await pipeline.execute({}, None)
        assert executed_steps == ["gather", "calculate"]
        assert result.stopped_early is False
        assert result.output == "calculated"

    async def test_step_outputs_collected(self) -> None:
        """Should collect all step outputs."""

        class StepA(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "a_output"

        class StepB(BaseStep[None, int]):
            async def execute(self, context: StepContext[None]) -> int:
                return 42

        pipeline = Pipeline(
            name="test",
            steps={"a": StepA(), "b": StepB()},
            output_step="b",
        )

        result = await pipeline.execute({}, None)
        assert result.step_outputs == {"a": "a_output", "b": 42}
        assert result.output == 42

    async def test_explicit_output_step(self) -> None:
        """Should use explicit output_step."""

        class StepA(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "a_output"

        class StepB(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "b_output"

        # Both steps have no deps - multiple terminals
        pipeline = Pipeline(
            name="test",
            steps={"a": StepA(), "b": StepB()},
            output_step="a",
        )

        result = await pipeline.execute({}, None)
        assert result.output == "a_output"

    async def test_requires_output_step_for_multiple_terminals(self) -> None:
        """Should require output_step when multiple terminal steps."""

        class StepA(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "a"

        class StepB(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "b"

        with pytest.raises(ValueError, match="Multiple terminal steps"):
            Pipeline(
                name="test",
                steps={"a": StepA(), "b": StepB()},
                # No output_step specified
            )

    async def test_step_execution_error(self) -> None:
        """Should wrap step errors in StepExecutionError."""

        class FailingStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                raise RuntimeError("Step failed!")

        pipeline = Pipeline(
            name="test",
            steps={"fail": FailingStep()},
        )

        with pytest.raises(StepExecutionError) as exc_info:
            await pipeline.execute({}, None)

        assert exc_info.value.step_id == "fail"
        assert isinstance(exc_info.value.original_error, RuntimeError)


# ============================================================================
# FastroAgent.as_step() Tests
# ============================================================================


class TestAgentAsStep:
    """Tests for FastroAgent.as_step()."""

    def test_as_step_creates_wrapper(self) -> None:
        """Should create AgentStepWrapper from agent."""
        agent = FastroAgent(
            model="test",
            system_prompt="You are a test agent.",
            temperature=0.5,
        )

        step = agent.as_step("Hello")

        assert isinstance(step, AgentStepWrapper)
        assert step.agent is agent
        assert step.last_usage is None

    def test_as_step_with_static_prompt(self) -> None:
        """Should work with static string prompt."""
        agent = FastroAgent(model="test", system_prompt="Test")
        step = agent.as_step("Static prompt")

        assert isinstance(step, AgentStepWrapper)

    def test_as_step_with_dynamic_prompt(self) -> None:
        """Should work with function prompt."""
        agent = FastroAgent(model="test", system_prompt="Test")
        step = agent.as_step(lambda ctx: f"Input: {ctx.get_input('text')}")

        assert isinstance(step, AgentStepWrapper)

    async def test_as_step_tracks_usage(self) -> None:
        """Should track usage after execute()."""
        agent = FastroAgent(model="test", system_prompt="Test")
        step = agent.as_step(lambda ctx: "Hello")

        # Mock the agent.run method
        mock_response = ChatResponse(
            output="Hello!",
            content="Hello!",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_microcents=75,
            processing_time_ms=100,
        )

        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response

            context = StepContext(
                step_id="test",
                inputs={},
                deps=None,
                step_outputs={},
            )
            result = await step.execute(context)

        assert result == "Hello!"
        assert step.last_usage is not None
        assert step.last_usage.cost_microcents == 75
        assert step.last_usage.input_tokens == 10

    async def test_as_step_forwards_deps_and_tracer(self) -> None:
        """Should forward context.deps and context.tracer to agent."""
        agent = FastroAgent(model="test", system_prompt="Test")
        step = agent.as_step("Hello")

        mock_response = ChatResponse(
            output="Result",
            content="Result",
            model="test",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_microcents=75,
            processing_time_ms=100,
        )

        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response

            deps = {"key": "value"}
            context = StepContext(
                step_id="test",
                inputs={},
                deps=deps,
                step_outputs={},
                tracer=None,  # Would be a real tracer in production
            )
            await step.execute(context)

            # Verify deps were forwarded
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["deps"] == deps


# ============================================================================
# BasePipeline Router Tests
# ============================================================================


class TestBasePipeline:
    """Tests for BasePipeline router."""

    async def test_register_and_route(self) -> None:
        """Should register pipelines and route correctly."""

        class SimpleStep(BaseStep[None, str]):
            def __init__(self, value: str):
                self.value = value

            async def execute(self, context: StepContext[None]) -> str:
                return self.value

        simple_pipeline = Pipeline(
            name="simple",
            steps={"step": SimpleStep("simple_result")},
        )
        complex_pipeline = Pipeline(
            name="complex",
            steps={"step": SimpleStep("complex_result")},
        )

        class TestRouter(BasePipeline[None, dict, str]):
            async def route(self, input_data: dict, deps: None) -> str:
                if input_data.get("amount", 0) < 1000:
                    return "simple"
                return "complex"

        router = TestRouter("test_router")
        router.register_pipeline("simple", simple_pipeline)
        router.register_pipeline("complex", complex_pipeline)

        # Test simple route
        result = await router.execute({"amount": 500}, None)
        assert result.output == "simple_result"

        # Test complex route
        result = await router.execute({"amount": 5000}, None)
        assert result.output == "complex_result"

    async def test_unknown_pipeline_raises(self) -> None:
        """Should raise for unknown pipeline name."""

        class TestRouter(BasePipeline[None, dict, str]):
            async def route(self, input_data: dict, deps: None) -> str:
                return "nonexistent"

        router = TestRouter("test")

        with pytest.raises(ValueError, match="Unknown pipeline"):
            await router.execute({}, None)
