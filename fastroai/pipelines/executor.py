"""Pipeline executor with DAG scheduling and parallelism.

Internal implementation - use Pipeline for the public API.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar

from ..errors import FastroAIError, PipelineValidationError
from .base import BaseStep, ConversationState, ConversationStatus, StepContext
from .config import PipelineConfig, StepConfig
from .schemas import StepUsage

if TYPE_CHECKING:
    from ..tracing import Tracer

DepsT = TypeVar("DepsT")


class StepExecutionError(FastroAIError):
    """Raised when a pipeline step fails during execution.

    Attributes:
        step_id: The ID of the step that failed.
        original_error: The underlying exception that caused the failure.

    Examples:
        ```python
        try:
            result = await pipeline.execute(inputs, deps)
        except StepExecutionError as e:
            print(f"Step '{e.step_id}' failed: {e.original_error}")
        ```
    """

    def __init__(self, step_id: str, original_error: Exception) -> None:
        self.step_id = step_id
        self.original_error = original_error
        super().__init__(f"Step '{step_id}' failed: {original_error}")


class ExecutionResult:
    """Internal execution result before conversion to PipelineResult."""

    def __init__(self) -> None:
        self.outputs: dict[str, Any] = {}
        self.usages: dict[str, StepUsage] = {}
        self.conversation_state: ConversationState[Any] | None = None
        self.stopped_early: bool = False


class PipelineExecutor:
    """DAG executor with parallelism and early termination.

    Internal class - use Pipeline for the public API.

    Features:
    - Topological sort for execution order
    - Parallel execution of independent steps
    - Early termination on INCOMPLETE status
    - Usage extraction from ctx.usage (accumulated via ctx.run())
    - Config inheritance: pipeline -> step class -> step_configs
    """

    def __init__(
        self,
        steps: dict[str, BaseStep[Any, Any]],
        dependencies: dict[str, list[str]],
        pipeline_config: PipelineConfig | None = None,
        step_configs: dict[str, StepConfig] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            steps: Dict of step_id -> step instance.
            dependencies: Dict of step_id -> [dependency_ids].
            pipeline_config: Default config for all steps.
            step_configs: Per-step config overrides.

        Raises:
            PipelineValidationError: Invalid dependencies or cycles.
        """
        self.steps = steps
        self.dependencies = dependencies
        self.pipeline_config = pipeline_config
        self.step_configs = step_configs or {}
        self._validate()
        self._execution_levels = self._topological_sort()

    def _validate(self) -> None:
        """Validate step graph."""
        for step_id, deps in self.dependencies.items():
            if step_id not in self.steps:
                raise PipelineValidationError(f"Dependency for unknown step: '{step_id}'")
            for dep in deps:
                if dep not in self.steps:
                    raise PipelineValidationError(f"Step '{step_id}' depends on unknown: '{dep}'")

        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)

            for dep in self.dependencies.get(step_id, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(step_id)
            return False

        for step_id in self.steps:
            if step_id not in visited and has_cycle(step_id):
                raise PipelineValidationError("Dependency graph has a cycle")

    def _topological_sort(self) -> list[set[str]]:
        """Sort into execution levels for parallelism.

        Steps in the same level have no deps on each other and can run
        in parallel.

        Returns:
            List of sets, each set is one execution level.

        Examples:
            ```python
            dependencies = {
                "b": ["a"],
                "c": ["a"],
                "d": ["b", "c"],
            }
            # Returns: [{"a"}, {"b", "c"}, {"d"}]
            # a runs first, then b and c in parallel, then d
            ```
        """
        levels: list[set[str]] = []
        remaining = set(self.steps.keys())

        while remaining:
            level = {
                step_id
                for step_id in remaining
                if all(dep not in remaining for dep in self.dependencies.get(step_id, []))
            }

            if not level:  # pragma: no cover
                raise PipelineValidationError("Cycle detected")

            levels.append(level)
            remaining -= level

        return levels

    async def execute(
        self,
        inputs: dict[str, Any],
        deps: DepsT,
        tracer: Tracer | None = None,
    ) -> ExecutionResult:
        """Execute pipeline.

        Args:
            inputs: Pipeline inputs accessible via context.get_input().
            deps: Dependencies accessible via context.deps.
            tracer: Optional tracer for distributed tracing.

        Returns:
            ExecutionResult with outputs, usages, and conversation state.

        Raises:
            StepExecutionError: If any step fails.
        """
        result = ExecutionResult()

        for level in self._execution_levels:
            tasks = []
            step_ids = []
            contexts = []

            for step_id in level:
                step = self.steps[step_id]
                step_config = self._resolve_config(step_id, step)
                context = StepContext(
                    step_id=step_id,
                    inputs=inputs,
                    deps=deps,
                    step_outputs=result.outputs.copy(),
                    tracer=tracer,
                    config=step_config,
                )
                tasks.append(self._execute_step(step_id, step, context, tracer))
                step_ids.append(step_id)
                contexts.append(context)

            outputs = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, (step_id, output) in enumerate(zip(step_ids, outputs, strict=True)):
                if isinstance(output, Exception):
                    if isinstance(output, StepExecutionError):
                        raise output
                    raise StepExecutionError(step_id, output)

                result.outputs[step_id] = output

                context = contexts[idx]
                usage = self._extract_usage(context)
                if usage:
                    result.usages[step_id] = usage

                if isinstance(output, ConversationState):
                    if output.status == ConversationStatus.INCOMPLETE:
                        result.conversation_state = output
                        result.stopped_early = True
                        return result
                    result.conversation_state = output

        return result

    async def _execute_step(
        self,
        step_id: str,
        step: BaseStep[Any, Any],
        context: StepContext[Any],
        tracer: Tracer | None,
    ) -> Any:
        """Execute single step with tracing."""
        if tracer:
            async with tracer.span(f"step.{step_id}"):
                return await step.execute(context)
        return await step.execute(context)

    def _extract_usage(self, context: StepContext[Any]) -> StepUsage | None:
        """Extract usage from step context.

        All steps should use ctx.run() which accumulates usage in ctx.usage.
        """
        usage = context.usage
        if usage.cost_microcents > 0 or usage.input_tokens > 0:
            return usage
        return None

    def _resolve_config(self, step_id: str, step: BaseStep[Any, Any]) -> StepConfig:
        """Resolve config for a step using inheritance.

        Config resolution order (most specific wins):
        1. Pipeline default config (base)
        2. Step class config (if step has .config attribute)
        3. step_configs[step_id] override

        Per-call overrides happen in ctx.run().
        """
        if self.pipeline_config:
            config = StepConfig(
                timeout=self.pipeline_config.timeout,
                retries=self.pipeline_config.retries,
                retry_delay=self.pipeline_config.retry_delay,
                cost_budget=self.pipeline_config.cost_budget,
            )
        else:
            config = StepConfig()

        step_class_config = getattr(step, "config", None)
        if isinstance(step_class_config, StepConfig):
            config = self._merge_configs(config, step_class_config)

        if step_id in self.step_configs:
            config = self._merge_configs(config, self.step_configs[step_id])

        return config

    def _merge_configs(self, base: StepConfig, override: StepConfig) -> StepConfig:
        """Merge two configs, with override taking precedence for non-None values."""
        return replace(
            base,
            timeout=override.timeout if override.timeout is not None else base.timeout,
            retries=override.retries if override.retries != 0 else base.retries,
            retry_delay=override.retry_delay if override.retry_delay != 1.0 else base.retry_delay,
            cost_budget=override.cost_budget if override.cost_budget is not None else base.cost_budget,
        )
