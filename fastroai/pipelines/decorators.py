"""Step decorator for concise pipeline step definitions.

Provides @step decorator as an alternative to class-based steps.

Example:
    from fastroai import step, Pipeline

    @step
    async def extract_text(ctx: StepContext[MyDeps]) -> str:
        doc = ctx.get_input("document")
        response = await ctx.run(extractor_agent, f"Extract: {doc}")
        return response.output

    @step(timeout=30.0, retries=2)
    async def classify(ctx: StepContext[MyDeps]) -> str:
        text = ctx.get_dependency("extract_text")
        response = await ctx.run(classifier_agent, f"Classify: {text}")
        return response.output

    pipeline = Pipeline(
        name="processor",
        steps={"extract_text": extract_text, "classify": classify},
        dependencies={"classify": ["extract_text"]},
    )
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, overload

from .base import BaseStep
from .config import StepConfig

if TYPE_CHECKING:
    from .base import StepContext

OutputT = TypeVar("OutputT")


class _FunctionStep(BaseStep[Any, Any]):
    """Internal step class wrapping a function.

    Created by the @step decorator. Supports both sync and async functions.
    """

    def __init__(self, func: Callable[..., Any], config: StepConfig) -> None:
        """Initialize function step.

        Args:
            func: The function to wrap (sync or async).
            config: Step configuration (timeout, retries, budget).
        """
        self._func = func
        self.config = config

    async def execute(self, context: StepContext[Any]) -> Any:
        """Execute the wrapped function.

        Args:
            context: Step execution context.

        Returns:
            The function's return value.
        """
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(context)
        return self._func(context)


@overload
def step(func: Callable[..., OutputT]) -> _FunctionStep: ...


@overload
def step(
    func: None = None,
    *,
    timeout: float | None = None,
    retries: int = 0,
    retry_delay: float = 1.0,
    cost_budget: int | None = None,
) -> Callable[[Callable[..., OutputT]], _FunctionStep]: ...


def step(
    func: Callable[..., Any] | None = None,
    *,
    timeout: float | None = None,
    retries: int = 0,
    retry_delay: float = 1.0,
    cost_budget: int | None = None,
) -> _FunctionStep | Callable[[Callable[..., Any]], _FunctionStep]:
    """Decorator to create a pipeline step from a function.

    Can be used with or without arguments:

        @step
        async def my_step(ctx): ...

        @step(timeout=30.0, retries=2)
        async def my_step(ctx): ...

    Args:
        func: The function to wrap (when used without parentheses).
        timeout: Maximum execution time in seconds.
        retries: Number of retry attempts on failure.
        retry_delay: Base delay between retries (exponential backoff).
        cost_budget: Maximum cost in microcents for this step.

    Returns:
        A BaseStep instance that can be used in a Pipeline.
    """
    config = StepConfig(
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        cost_budget=cost_budget,
    )

    def decorator(fn: Callable[..., Any]) -> _FunctionStep:
        return _FunctionStep(fn, config)

    if func is not None:
        return decorator(func)

    return decorator
