"""Pipeline - Declarative DAG-based workflow orchestration.

Provides the main Pipeline class for multi-step AI workflows
with automatic parallelism and usage tracking.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ..errors import PipelineValidationError
from ..tracing import NoOpTracer, Tracer
from .base import BaseStep, ConversationState
from .config import PipelineConfig, StepConfig
from .executor import PipelineExecutor
from .schemas import PipelineUsage

DepsT = TypeVar("DepsT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class PipelineResult(BaseModel, Generic[OutputT]):
    """Result from pipeline execution.

    Attributes:
        output: Final step's output, or None if stopped early.
        step_outputs: All step outputs by ID.
        conversation_state: ConversationState if a step returned one.
        usage: Aggregated usage metrics.
        stopped_early: True if stopped due to INCOMPLETE status.

    Examples:
        ```python
        result = await pipeline.execute(data, deps)

        if result.stopped_early:
            missing = result.conversation_state.context["missing"]
            return {"status": "incomplete", "missing": missing}

        print(f"Cost: ${result.usage.total_cost_dollars:.6f}")
        return {"status": "complete", "output": result.output}
        ```
    """

    output: OutputT | None = None
    step_outputs: dict[str, Any] = {}
    conversation_state: ConversationState[Any] | None = None
    usage: PipelineUsage | None = None
    stopped_early: bool = False

    model_config = {"arbitrary_types_allowed": True}


class Pipeline(Generic[DepsT, InputT, OutputT]):
    """Declarative DAG pipeline for multi-step AI workflows.

    Features:
    - Automatic parallelism from dependencies
    - Type-safe dependency access
    - Early termination on INCOMPLETE status
    - Aggregated usage tracking
    - Distributed tracing

    Examples:
        ```python
        pipeline = Pipeline(
            name="document_processor",
            steps={
                "extract": ExtractStep(),
                "classify": ClassifyStep(),
                "summarize": SummarizeStep(),
            },
            dependencies={
                "classify": ["extract"],
                "summarize": ["classify"],
            },
        )

        result = await pipeline.execute({"document": doc}, deps, tracer)
        summary = result.output
        ```

        Parallelism example - steps at the same level run in parallel:

        ```python
        dependencies = {
            "classify": ["extract"],
            "fetch_market": ["classify"],
            "fetch_user": ["classify"],  # Same dep as above
            "calculate": ["fetch_market", "fetch_user"],
        }
        # Execution:
        # Level 0: extract
        # Level 1: classify
        # Level 2: fetch_market, fetch_user (PARALLEL)
        # Level 3: calculate
        ```
    """

    def __init__(
        self,
        name: str,
        steps: dict[str, BaseStep[DepsT, Any]],
        dependencies: dict[str, list[str]] | None = None,
        output_step: str | None = None,
        config: PipelineConfig | None = None,
        step_configs: dict[str, StepConfig] | None = None,
    ) -> None:
        """Initialize Pipeline.

        Args:
            name: Pipeline name (for tracing).
            steps: Dict of step_id -> step instance.
            dependencies: Dict of step_id -> [dependency_ids].
            output_step: Which step's output is the pipeline output.
                        Defaults to last step in topological order.
            config: Default configuration for all steps (timeout, retries, budget).
            step_configs: Per-step configuration overrides.

        Config Resolution (most specific wins):
            1. Pipeline default config
            2. Step class config (if step has .config attribute)
            3. step_configs[step_id] override
            4. Per-call overrides via ctx.run(timeout=..., retries=...)

        Raises:
            PipelineValidationError: Invalid deps, unknown output_step, or cycles.

        Examples:
            ```python
            pipeline = Pipeline(
                name="processor",
                steps={"extract": ExtractStep(), "classify": ClassifyStep()},
                config=PipelineConfig(timeout=30.0, retries=1),
                step_configs={"classify": StepConfig(timeout=60.0)},  # Override
            )
            ```
        """
        self.name = name
        self.steps = steps
        self.dependencies = dependencies or {}
        self.config = config
        self.step_configs = step_configs or {}

        if output_step is not None and output_step not in steps:
            raise PipelineValidationError(f"output_step '{output_step}' not in steps")

        self._executor = PipelineExecutor(steps, self.dependencies, self.config, self.step_configs)

        if output_step:
            self.output_step = output_step
        else:
            last_level = self._executor._execution_levels[-1]
            if len(last_level) > 1:
                raise PipelineValidationError(f"Multiple terminal steps: {last_level}. Specify output_step explicitly.")
            self.output_step = next(iter(last_level))

    async def execute(
        self,
        input_data: InputT,
        deps: DepsT,
        tracer: Tracer | None = None,
    ) -> PipelineResult[OutputT]:
        """Execute the pipeline.

        Args:
            input_data: Input accessible via context.get_input().
            deps: Your deps accessible via context.deps.
            tracer: For distributed tracing.

        Returns:
            PipelineResult with output and usage.

        Raises:
            StepExecutionError: If any step fails.

        Examples:
            ```python
            # Basic execution
            result = await pipeline.execute(
                {"document": "Hello world"},
                deps=MyDeps(api_key="..."),
            )
            print(result.output)

            # With tracing
            from fastroai import SimpleTracer

            tracer = SimpleTracer()
            result = await pipeline.execute(
                {"document": doc},
                deps=deps,
                tracer=tracer,
            )

            # Handle early termination (multi-turn)
            if result.stopped_early:
                missing = result.conversation_state.context["missing"]
                return {"status": "incomplete", "missing": missing}

            # Access usage metrics
            if result.usage:
                print(f"Cost: ${result.usage.total_cost_dollars:.6f}")
                print(f"Tokens: {result.usage.total_input_tokens}")
            ```
        """
        effective_tracer = tracer or NoOpTracer()

        inputs = input_data if isinstance(input_data, dict) else {"data": input_data}

        async with effective_tracer.span(f"pipeline.{self.name}"):
            exec_result = await self._executor.execute(inputs, deps, effective_tracer)

        output = exec_result.outputs.get(self.output_step)
        usage = PipelineUsage.from_step_usages(exec_result.usages) if exec_result.usages else None

        return PipelineResult(
            output=output,
            step_outputs=exec_result.outputs,
            conversation_state=exec_result.conversation_state,
            usage=usage,
            stopped_early=exec_result.stopped_early,
        )
