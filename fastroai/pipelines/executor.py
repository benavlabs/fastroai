"""Pipeline executor with DAG scheduling and parallelism.

Internal implementation - use Pipeline for the public API.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar

from ..agent.schemas import ChatResponse
from .base import BaseStep, ConversationState, ConversationStatus, StepContext
from .schemas import StepUsage

if TYPE_CHECKING:
    from ..tracing import Tracer

DepsT = TypeVar("DepsT")


class StepExecutionError(Exception):
    """Raised when a pipeline step fails."""

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
    - Usage extraction from steps with last_usage property
    """

    def __init__(
        self,
        steps: dict[str, BaseStep[Any, Any]],
        dependencies: dict[str, list[str]],
    ) -> None:
        self.steps = steps
        self.dependencies = dependencies
        self._validate()
        self._execution_levels = self._topological_sort()

    def _validate(self) -> None:
        """Validate step graph."""
        for step_id, deps in self.dependencies.items():
            if step_id not in self.steps:
                raise ValueError(f"Dependency for unknown step: '{step_id}'")
            for dep in deps:
                if dep not in self.steps:
                    raise ValueError(f"Step '{step_id}' depends on unknown: '{dep}'")

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
                raise ValueError("Dependency graph has a cycle")

    def _topological_sort(self) -> list[set[str]]:
        """Sort into execution levels for parallelism.

        Steps in the same level have no deps on each other.

        Example:
            dependencies = {
                "b": ["a"],
                "c": ["a"],
                "d": ["b", "c"],
            }
            # Returns: [{"a"}, {"b", "c"}, {"d"}]
            # a runs first, then b and c in parallel, then d
        """
        levels: list[set[str]] = []
        remaining = set(self.steps.keys())

        while remaining:
            level = {
                step_id
                for step_id in remaining
                if all(dep not in remaining for dep in self.dependencies.get(step_id, []))
            }

            if not level:
                raise ValueError("Cycle detected")

            levels.append(level)
            remaining -= level

        return levels

    async def execute(
        self,
        inputs: dict[str, Any],
        deps: DepsT,
        tracer: Tracer | None = None,
    ) -> ExecutionResult:
        """Execute pipeline."""
        result = ExecutionResult()

        for level in self._execution_levels:
            tasks = []
            step_ids = []

            for step_id in level:
                step = self.steps[step_id]
                context = StepContext(
                    step_id=step_id,
                    inputs=inputs,
                    deps=deps,
                    step_outputs=result.outputs.copy(),
                    tracer=tracer,
                )
                tasks.append(self._execute_step(step_id, step, context, tracer))
                step_ids.append(step_id)

            outputs = await asyncio.gather(*tasks, return_exceptions=True)

            for step_id, output in zip(step_ids, outputs, strict=True):
                if isinstance(output, Exception):
                    if isinstance(output, StepExecutionError):
                        raise output
                    raise StepExecutionError(step_id, output)

                result.outputs[step_id] = output

                step = self.steps[step_id]
                usage = self._extract_usage(step, output)
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

    def _extract_usage(self, step: BaseStep[Any, Any], output: Any) -> StepUsage | None:
        """Extract usage from step or output."""
        if hasattr(step, "last_usage"):
            usage = getattr(step, "last_usage", None)
            if isinstance(usage, StepUsage):
                return usage

        if hasattr(output, "usage") and isinstance(output.usage, StepUsage):
            return output.usage

        if isinstance(output, ChatResponse):
            return StepUsage.from_chat_response(output)

        return None
