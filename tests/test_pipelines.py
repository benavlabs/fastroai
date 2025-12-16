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
    CostBudgetExceededError,
    FastroAIError,
    Pipeline,
    PipelineConfig,
    PipelineUsage,
    PipelineValidationError,
    StepConfig,
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
# Config Tests
# ============================================================================


class TestStepConfig:
    """Tests for StepConfig."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = StepConfig()
        assert config.timeout is None
        assert config.retries == 0
        assert config.retry_delay == 1.0
        assert config.cost_budget is None

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = StepConfig(
            timeout=30.0,
            retries=3,
            retry_delay=2.0,
            cost_budget=10_000_000,
        )
        assert config.timeout == 30.0
        assert config.retries == 3
        assert config.retry_delay == 2.0
        assert config.cost_budget == 10_000_000


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = PipelineConfig()
        # Inherited from StepConfig
        assert config.timeout is None
        assert config.retries == 0
        assert config.retry_delay == 1.0
        assert config.cost_budget is None
        # PipelineConfig specific
        assert config.trace is True
        assert config.on_error == "fail"

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = PipelineConfig(
            timeout=60.0,
            retries=2,
            trace=False,
            on_error="continue",
        )
        assert config.timeout == 60.0
        assert config.retries == 2
        assert config.trace is False
        assert config.on_error == "continue"

    def test_inherits_from_step_config(self) -> None:
        """PipelineConfig should be a StepConfig."""
        config = PipelineConfig()
        assert isinstance(config, StepConfig)


# ============================================================================
# Error Hierarchy Tests
# ============================================================================


class TestErrorHierarchy:
    """Tests for FastroAI error hierarchy."""

    def test_fastroai_error_is_base(self) -> None:
        """FastroAIError should be the base for all errors."""
        assert issubclass(PipelineValidationError, FastroAIError)
        assert issubclass(StepExecutionError, FastroAIError)
        assert issubclass(CostBudgetExceededError, FastroAIError)

    def test_step_execution_error(self) -> None:
        """StepExecutionError should have step_id and original_error."""
        original = ValueError("something failed")
        error = StepExecutionError("my_step", original)
        assert error.step_id == "my_step"
        assert error.original_error is original
        assert "my_step" in str(error)
        assert isinstance(error, FastroAIError)

    def test_cost_budget_exceeded_error(self) -> None:
        """CostBudgetExceededError should have budget and actual."""
        error = CostBudgetExceededError(budget=1000, actual=1500, step_id="expensive_step")
        assert error.budget_microcents == 1000
        assert error.actual_microcents == 1500
        assert error.step_id == "expensive_step"
        assert "1500" in str(error)
        assert "1000" in str(error)
        assert "expensive_step" in str(error)
        assert isinstance(error, FastroAIError)

    def test_cost_budget_exceeded_error_no_step(self) -> None:
        """CostBudgetExceededError without step_id."""
        error = CostBudgetExceededError(budget=1000, actual=1500)
        assert error.step_id is None
        assert "in step" not in str(error)


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

    def test_usage_starts_at_zero(self) -> None:
        """Context should start with zero usage."""
        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
        )
        assert context.usage.input_tokens == 0
        assert context.usage.output_tokens == 0
        assert context.usage.cost_microcents == 0

    async def test_ctx_run_accumulates_usage(self) -> None:
        """ctx.run() should accumulate usage across multiple calls."""
        agent = FastroAgent(model="test", system_prompt="Test")

        mock_response1 = ChatResponse(
            output="First",
            content="First",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_microcents=75,
            processing_time_ms=100,
        )
        mock_response2 = ChatResponse(
            output="Second",
            content="Second",
            model="gpt-4o",
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            cost_microcents=150,
            processing_time_ms=200,
        )

        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
        )

        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = [mock_response1, mock_response2]

            response1 = await context.run(agent, "First message")
            response2 = await context.run(agent, "Second message")

        assert response1.output == "First"
        assert response2.output == "Second"

        # Usage should be accumulated
        assert context.usage.input_tokens == 30  # 10 + 20
        assert context.usage.output_tokens == 15  # 5 + 10
        assert context.usage.cost_microcents == 225  # 75 + 150
        assert context.usage.processing_time_ms == 300  # 100 + 200

    async def test_ctx_run_forwards_deps_and_tracer(self) -> None:
        """ctx.run() should forward deps and tracer to agent."""
        agent = FastroAgent(model="test", system_prompt="Test")

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

        deps = {"db": "session", "user_id": 123}
        context = StepContext(
            step_id="test",
            inputs={},
            deps=deps,
            step_outputs={},
            tracer=None,
        )

        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response

            await context.run(agent, "Hello")

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["deps"] == deps
            assert call_kwargs["tracer"] is None

    def test_context_accepts_config(self) -> None:
        """Should accept config parameter."""
        config = StepConfig(timeout=30.0, retries=2, cost_budget=100000)
        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
            config=config,
        )
        assert context.config.timeout == 30.0
        assert context.config.retries == 2
        assert context.config.cost_budget == 100000

    def test_context_default_config(self) -> None:
        """Should have default config if not provided."""
        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
        )
        assert context.config.timeout is None
        assert context.config.retries == 0
        assert context.config.cost_budget is None

    async def test_ctx_run_budget_exceeded_raises(self) -> None:
        """Should raise CostBudgetExceededError when budget is exceeded."""
        agent = FastroAgent(model="test", system_prompt="Test")

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

        # Set budget to 150 microcents
        config = StepConfig(cost_budget=150)
        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
            config=config,
        )

        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response

            # First call succeeds (usage 0 < 150)
            await context.run(agent, "First")
            assert context.usage.cost_microcents == 100

            # Second call succeeds (usage 100 < 150)
            await context.run(agent, "Second")
            assert context.usage.cost_microcents == 200

            # Third call should raise (usage 200 >= 150)
            with pytest.raises(CostBudgetExceededError) as exc_info:
                await context.run(agent, "Third")

            assert exc_info.value.budget_microcents == 150
            assert exc_info.value.actual_microcents == 200
            assert exc_info.value.step_id == "test"

    async def test_ctx_run_timeout(self) -> None:
        """Should raise TimeoutError when timeout exceeded."""
        agent = FastroAgent(model="test", system_prompt="Test")

        async def slow_agent_run(*args, **kwargs):
            await asyncio.sleep(1.0)  # Slow response

        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
            config=StepConfig(timeout=0.05),  # 50ms timeout
        )

        with patch.object(agent, "run", side_effect=slow_agent_run), pytest.raises(asyncio.TimeoutError):
            await context.run(agent, "Hello")

    async def test_ctx_run_retries_on_failure(self) -> None:
        """Should retry on failure with exponential backoff."""
        agent = FastroAgent(model="test", system_prompt="Test")

        mock_response = ChatResponse(
            output="Success",
            content="Success",
            model="test",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_microcents=75,
            processing_time_ms=100,
        )

        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient failure")
            return mock_response

        # Config with 2 retries and short delay for testing
        config = StepConfig(retries=2, retry_delay=0.01)
        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
            config=config,
        )

        with patch.object(agent, "run", side_effect=failing_then_success):
            result = await context.run(agent, "Hello")

        assert result.output == "Success"
        assert call_count == 3  # 1 initial + 2 retries

    async def test_ctx_run_retries_exhausted_raises(self) -> None:
        """Should raise after all retries exhausted."""
        agent = FastroAgent(model="test", system_prompt="Test")

        async def always_fails(*args, **kwargs):
            raise RuntimeError("Persistent failure")

        config = StepConfig(retries=2, retry_delay=0.01)
        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
            config=config,
        )

        with (
            patch.object(agent, "run", side_effect=always_fails),
            pytest.raises(RuntimeError, match="Persistent failure"),
        ):
            await context.run(agent, "Hello")

    async def test_ctx_run_per_call_overrides(self) -> None:
        """Should allow per-call timeout/retries overrides."""
        agent = FastroAgent(model="test", system_prompt="Test")

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

        call_count = 0

        async def failing_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First failure")
            return mock_response

        # Config has no retries
        config = StepConfig(retries=0)
        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
            config=config,
        )

        with patch.object(agent, "run", side_effect=failing_once):
            # Per-call override: retries=1
            result = await context.run(agent, "Hello", retries=1)

        assert result.output == "Result"
        assert call_count == 2  # 1 initial + 1 retry


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

        with pytest.raises(PipelineValidationError, match="depends on unknown"):
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

        with pytest.raises(PipelineValidationError, match="Dependency for unknown step"):
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

        with pytest.raises(PipelineValidationError, match="cycle"):
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

        with pytest.raises(PipelineValidationError, match="Multiple terminal steps"):
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

    async def test_as_step_tracks_usage_in_context(self) -> None:
        """Should track usage in context.usage after execute()."""
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
        # Usage is now tracked in context.usage, not step.last_usage
        assert context.usage.cost_microcents == 75
        assert context.usage.input_tokens == 10

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


# ============================================================================
# @step Decorator Tests
# ============================================================================


class TestStepDecorator:
    """Tests for the @step decorator."""

    def test_decorator_without_args(self) -> None:
        """Should work as @step without parentheses."""
        from fastroai.pipelines import step

        @step
        async def my_step(ctx: StepContext[None]) -> str:
            return "result"

        # Should be a BaseStep
        assert isinstance(my_step, BaseStep)
        # Should have default config
        assert my_step.config.timeout is None
        assert my_step.config.retries == 0

    def test_decorator_with_args(self) -> None:
        """Should work as @step(timeout=30) with args."""
        from fastroai.pipelines import step

        @step(timeout=30.0, retries=2, retry_delay=2.0, cost_budget=100000)
        async def my_step(ctx: StepContext[None]) -> str:
            return "result"

        # Should have config from decorator
        assert my_step.config.timeout == 30.0
        assert my_step.config.retries == 2
        assert my_step.config.retry_delay == 2.0
        assert my_step.config.cost_budget == 100000

    async def test_async_function_step(self) -> None:
        """Should execute async functions."""
        from fastroai.pipelines import step

        @step
        async def async_step(ctx: StepContext[None]) -> str:
            await asyncio.sleep(0.001)
            return "async_result"

        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
        )

        result = await async_step.execute(context)
        assert result == "async_result"

    async def test_sync_function_step(self) -> None:
        """Should execute sync functions."""
        from fastroai.pipelines import step

        @step
        def sync_step(ctx: StepContext[None]) -> str:
            return "sync_result"

        context = StepContext(
            step_id="test",
            inputs={},
            deps=None,
            step_outputs={},
        )

        result = await sync_step.execute(context)
        assert result == "sync_result"

    async def test_step_in_pipeline(self) -> None:
        """Should work alongside class-based steps in Pipeline."""
        from fastroai.pipelines import step

        @step
        async def step_a(ctx: StepContext[None]) -> str:
            return "a_done"

        @step
        async def step_b(ctx: StepContext[None]) -> str:
            a_result = ctx.get_dependency("a")
            return f"{a_result}_b_done"

        pipeline = Pipeline(
            name="test",
            steps={"a": step_a, "b": step_b},
            dependencies={"b": ["a"]},
        )

        result = await pipeline.execute({}, None)
        assert result.output == "a_done_b_done"

    async def test_step_with_ctx_run(self) -> None:
        """Should work with ctx.run() for agent calls."""
        from fastroai.pipelines import step

        agent = FastroAgent(model="test", system_prompt="Test")

        @step
        async def agent_step(ctx: StepContext[None]) -> str:
            response = await ctx.run(agent, "Hello")
            return response.output

        mock_response = ChatResponse(
            output="Hello!",
            content="Hello!",
            model="test",
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
            result = await agent_step.execute(context)

        assert result == "Hello!"
        assert context.usage.cost_microcents == 75

    async def test_step_can_access_inputs(self) -> None:
        """Should be able to access inputs via context."""
        from fastroai.pipelines import step

        @step
        async def input_step(ctx: StepContext[None]) -> str:
            name = ctx.get_input("name")
            return f"Hello, {name}!"

        context = StepContext(
            step_id="test",
            inputs={"name": "World"},
            deps=None,
            step_outputs={},
        )

        result = await input_step.execute(context)
        assert result == "Hello, World!"


# ============================================================================
# Pipeline Config Inheritance Tests
# ============================================================================


class TestPipelineConfigInheritance:
    """Tests for Pipeline config inheritance."""

    async def test_pipeline_accepts_config(self) -> None:
        """Should accept config parameter."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                # Config should be accessible via context
                assert context.config.timeout == 30.0
                assert context.config.retries == 2
                return "done"

        pipeline = Pipeline(
            name="test",
            steps={"simple": SimpleStep()},
            config=PipelineConfig(timeout=30.0, retries=2),
        )

        result = await pipeline.execute({}, None)
        assert result.output == "done"

    async def test_step_configs_override_pipeline(self) -> None:
        """step_configs should override pipeline config."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return f"timeout={context.config.timeout}"

        pipeline = Pipeline(
            name="test",
            steps={"a": SimpleStep(), "b": SimpleStep()},
            dependencies={"b": ["a"]},  # b depends on a
            config=PipelineConfig(timeout=10.0),
            step_configs={"b": StepConfig(timeout=60.0)},  # Override for b only
        )

        result = await pipeline.execute({}, None)

        # Step a uses pipeline default, step b uses override
        assert result.step_outputs["a"] == "timeout=10.0"
        assert result.step_outputs["b"] == "timeout=60.0"

    async def test_step_class_config_used(self) -> None:
        """Step class config should be used when no pipeline config."""
        from fastroai.pipelines import step

        @step(timeout=45.0, retries=3)
        async def configured_step(ctx: StepContext[None]) -> str:
            return f"timeout={ctx.config.timeout},retries={ctx.config.retries}"

        pipeline = Pipeline(
            name="test",
            steps={"configured": configured_step},
        )

        result = await pipeline.execute({}, None)
        assert result.output == "timeout=45.0,retries=3"

    async def test_config_inheritance_order(self) -> None:
        """Config inheritance: pipeline < step class < step_configs."""
        from fastroai.pipelines import step

        @step(timeout=20.0)  # Step class config
        async def my_step(ctx: StepContext[None]) -> str:
            return f"timeout={ctx.config.timeout}"

        # Pipeline config (lowest priority)
        # Step class has timeout=20.0 (overrides pipeline)
        # step_configs has timeout=30.0 (highest priority)
        pipeline = Pipeline(
            name="test",
            steps={"my": my_step},
            config=PipelineConfig(timeout=10.0),  # Lowest priority
            step_configs={"my": StepConfig(timeout=30.0)},  # Highest priority
        )

        result = await pipeline.execute({}, None)
        # step_configs should win
        assert result.output == "timeout=30.0"

    async def test_pipeline_config_cost_budget(self) -> None:
        """Pipeline config cost_budget should apply to steps."""
        agent = FastroAgent(model="test", system_prompt="Test")

        class BudgetStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                # First call: usage=0 < 150, proceeds. After: usage=100
                await context.run(agent, "First")
                # Second call: usage=100 < 150, proceeds. After: usage=200
                await context.run(agent, "Second")
                # Third call: usage=200 >= 150, raises CostBudgetExceededError
                await context.run(agent, "Third")
                return "done"

        mock_response = ChatResponse(
            output="Result",
            content="Result",
            model="test",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_microcents=100,  # Each call costs 100
            processing_time_ms=50,
        )

        pipeline = Pipeline(
            name="test",
            steps={"budget": BudgetStep()},
            config=PipelineConfig(cost_budget=150),  # Budget of 150
        )

        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response

            # Should raise StepExecutionError wrapping CostBudgetExceededError
            with pytest.raises(StepExecutionError) as exc_info:
                await pipeline.execute({}, None)

            assert isinstance(exc_info.value.original_error, CostBudgetExceededError)
            assert exc_info.value.original_error.budget_microcents == 150
            assert exc_info.value.original_error.actual_microcents == 200

    async def test_no_config_uses_defaults(self) -> None:
        """No config should use default values."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                assert context.config.timeout is None
                assert context.config.retries == 0
                assert context.config.cost_budget is None
                return "done"

        pipeline = Pipeline(
            name="test",
            steps={"simple": SimpleStep()},
            # No config provided
        )

        result = await pipeline.execute({}, None)
        assert result.output == "done"


# ============================================================================
# Coverage Gap Tests
# ============================================================================


class TestCoverageGaps:
    """Tests to cover remaining edge cases for 100% coverage."""

    def test_invalid_output_step_raises(self) -> None:
        """Should raise PipelineValidationError for unknown output_step."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "done"

        with pytest.raises(PipelineValidationError, match="output_step 'nonexistent' not in steps"):
            Pipeline(
                name="test",
                steps={"a": SimpleStep()},
                output_step="nonexistent",
            )

    async def test_step_execution_error_reraise(self) -> None:
        """Should re-raise StepExecutionError directly when step raises it."""

        class RaisingStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                # Step raises StepExecutionError directly (unusual but possible)
                raise StepExecutionError("inner_step", RuntimeError("Inner error"))

        pipeline = Pipeline(
            name="test",
            steps={"raising": RaisingStep()},
        )

        with pytest.raises(StepExecutionError) as exc_info:
            await pipeline.execute({}, None)

        # Should be the same error, not wrapped again
        assert exc_info.value.step_id == "inner_step"

    async def test_executor_without_tracer(self) -> None:
        """Should execute step without tracer (line 207)."""

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "no_tracer"

        pipeline = Pipeline(
            name="test",
            steps={"simple": SimpleStep()},
        )

        # Execute without tracer explicitly
        result = await pipeline.execute({}, None, tracer=None)
        assert result.output == "no_tracer"

    async def test_usage_extraction_with_nonzero_usage(self) -> None:
        """Should extract usage when cost or tokens are non-zero (lines 185, 216)."""
        agent = FastroAgent(model="test", system_prompt="Test")

        class UsageStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                await context.run(agent, "Hello")
                return "done"

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

        pipeline = Pipeline(
            name="test",
            steps={"usage": UsageStep()},
        )

        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response
            result = await pipeline.execute({}, None)

        assert result.output == "done"
        assert result.usage is not None
        assert result.usage.total_cost_microcents == 75

    async def test_executor_without_tracer_directly(self) -> None:
        """Test executor directly without tracer (covers line 207)."""
        from fastroai.pipelines.executor import PipelineExecutor

        class SimpleStep(BaseStep[None, str]):
            async def execute(self, context: StepContext[None]) -> str:
                return "executed"

        executor = PipelineExecutor(
            steps={"simple": SimpleStep()},
            dependencies={},
        )

        # Call executor directly with tracer=None
        result = await executor.execute({}, None, tracer=None)
        assert result.outputs["simple"] == "executed"
