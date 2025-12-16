"""Base abstractions for pipeline steps.

Provides:
- ConversationStatus/ConversationState: Multi-turn conversation signaling
- StepContext: Execution context with inputs, deps, and step outputs
- BaseStep: Abstract base class for pipeline steps
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import BaseModel

from ..errors import CostBudgetExceededError
from .config import StepConfig
from .schemas import StepUsage

if TYPE_CHECKING:
    from ..agent import ChatResponse, FastroAgent
    from ..tracing import Tracer

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")
T = TypeVar("T")


class ConversationStatus(str, Enum):
    """Status of multi-turn conversation gathering.

    Attributes:
        COMPLETE: All required information has been gathered.
            The pipeline proceeds to subsequent steps.
        INCOMPLETE: More information is needed from the user.
            The pipeline pauses and returns partial state.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class ConversationState(BaseModel, Generic[T]):
    """Signal for multi-turn conversation steps.

    When a step returns ConversationState with INCOMPLETE status,
    the pipeline stops early. Partial data and context are preserved.

    Examples:
        ```python
        class GatherInfoStep(BaseStep[MyDeps, ConversationState[UserInfo]]):
            async def execute(self, context) -> ConversationState[UserInfo]:
                info = await self._extract(context.get_input("message"))

                if info.is_complete():
                    return ConversationState(
                        status=ConversationStatus.COMPLETE,
                        data=info,
                    )

                return ConversationState(
                    status=ConversationStatus.INCOMPLETE,
                    data=info,  # Partial data
                    context={"missing": info.missing_fields()},
                )
        ```
    """

    status: ConversationStatus
    data: T | None = None
    context: dict[str, Any] = {}


class StepContext(Generic[DepsT]):
    """Execution context provided to pipeline steps.

    Provides access to:
    - Pipeline inputs (the data passed to execute())
    - Outputs from dependency steps
    - Application dependencies (your db session, user, etc.)
    - Tracer for custom spans

    Examples:
        ```python
        class ProcessStep(BaseStep[MyDeps, Result]):
            async def execute(self, context: StepContext[MyDeps]) -> Result:
                # Get pipeline input
                document = context.get_input("document")

                # Get output from dependency step
                classification = context.get_dependency("classify", Classification)

                # Access your deps
                db = context.deps.session
                user_id = context.deps.user_id

                # Custom tracing
                if context.tracer:
                    async with context.tracer.span("custom_operation"):
                        result = await process(document)

                return result
        ```
    """

    def __init__(
        self,
        step_id: str,
        inputs: dict[str, Any],
        deps: DepsT,
        step_outputs: dict[str, Any],
        tracer: Tracer | None = None,
        config: StepConfig | None = None,
    ) -> None:
        """Initialize step context.

        Args:
            step_id: Unique identifier for this step.
            inputs: Pipeline inputs passed to execute().
            deps: Application dependencies (db session, user, etc.).
            step_outputs: Outputs from completed dependency steps.
            tracer: Optional tracer for distributed tracing.
            config: Step configuration (timeout, retries, budget).
        """
        self._step_id = step_id
        self._inputs = inputs
        self._deps = deps
        self._outputs = step_outputs
        self._tracer = tracer
        self._config = config or StepConfig()
        self._usage = StepUsage()

    @property
    def step_id(self) -> str:
        """Current step's ID.

        Returns:
            The step identifier string.
        """
        return self._step_id

    @property
    def deps(self) -> DepsT:
        """Application dependencies (your session, user, etc.).

        Returns:
            The dependencies object passed to pipeline.execute().
        """
        return self._deps

    @property
    def tracer(self) -> Tracer | None:
        """Tracer for custom spans.

        Returns:
            The tracer instance, or None if no tracing.
        """
        return self._tracer

    @property
    def usage(self) -> StepUsage:
        """Accumulated usage from all ctx.run() calls in this step.

        Returns:
            StepUsage with aggregated tokens and cost.
        """
        return self._usage

    @property
    def config(self) -> StepConfig:
        """Configuration for this step (timeout, retries, budget).

        Returns:
            The resolved StepConfig for this step.
        """
        return self._config

    def get_input(self, key: str, default: Any = None) -> Any:
        """Get value from pipeline inputs.

        Args:
            key: The input key to retrieve.
            default: Value to return if key not found.

        Returns:
            The input value, or default if not present.

        Examples:
            ```python
            class ProcessStep(BaseStep[MyDeps, str]):
                async def execute(self, ctx: StepContext[MyDeps]) -> str:
                    # Get required input
                    document = ctx.get_input("document")

                    # Get optional input with default
                    format_type = ctx.get_input("format", "json")

                    return f"Processing {document} as {format_type}"
            ```
        """
        return self._inputs.get(key, default)

    def get_dependency(
        self,
        step_id: str,
        output_type: type[T] | None = None,
    ) -> T:
        """Get output from a dependency step.

        Args:
            step_id: ID of the dependency step.
            output_type: Expected type (for IDE/type checker, not enforced).

        Returns:
            The output from the dependency step.

        Raises:
            ValueError: If step_id not in dependencies or hasn't run.

        Examples:
            ```python
            # With type hint (IDE knows extraction is ExtractionResult)
            extraction = context.get_dependency("extract", ExtractionResult)
            extraction.entities  # Autocomplete works!
            ```
        """
        if step_id not in self._outputs:
            raise ValueError(
                f"Step '{step_id}' not a dependency of '{self._step_id}' "
                f"or hasn't completed. Available: {list(self._outputs.keys())}"
            )
        return cast(T, self._outputs[step_id])

    def get_dependency_or_none(
        self,
        step_id: str,
        output_type: type[T] | None = None,
    ) -> T | None:
        """Get output from a dependency step, or None if not available.

        Use this for optional dependencies that may not have run.

        Args:
            step_id: ID of the dependency step.
            output_type: Expected type (for IDE/type checker, not enforced).

        Returns:
            The output from the dependency step, or None if not available.

        Examples:
            ```python
            class EnhanceStep(BaseStep[MyDeps, str]):
                async def execute(self, ctx: StepContext[MyDeps]) -> str:
                    # Required dependency
                    base_content = ctx.get_dependency("extract")

                    # Optional dependency - might not exist
                    metadata = ctx.get_dependency_or_none("fetch_metadata", dict)

                    if metadata:
                        return f"{base_content} (with metadata)"
                    return base_content
            ```
        """
        return self._outputs.get(step_id)

    async def run(
        self,
        agent: FastroAgent[OutputT],
        message: str,
        *,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> ChatResponse[OutputT]:
        """Run an agent with automatic tracer, usage tracking, and config.

        This is THE way to call agents from within a step. It:
        - Passes deps and tracer automatically
        - Accumulates usage in ctx.usage
        - Enforces cost budget (raises CostBudgetExceededError if exceeded)
        - Supports timeout and retries (from config or per-call override)

        Args:
            agent: The FastroAgent to run.
            message: The message/prompt to send.
            timeout: Per-call timeout override (seconds). Uses config if None.
            retries: Per-call retries override. Uses config if None.

        Returns:
            ChatResponse with output, content, usage data, etc.

        Raises:
            CostBudgetExceededError: If cost_budget is set and exceeded.
            asyncio.TimeoutError: If timeout exceeded after all retries.

        Examples:
            ```python
            class MyStep(BaseStep[MyDeps, str]):
                classifier = FastroAgent(model="gpt-4o-mini", output_type=Category)
                writer = FastroAgent(model="gpt-4o")

                async def execute(self, ctx: StepContext[MyDeps]) -> str:
                    # Both calls tracked in ctx.usage
                    category = await ctx.run(self.classifier, "Classify this")
                    result = await ctx.run(self.writer, f"Write about {category.output}")
                    return result.content

            # With per-call overrides:
            response = await ctx.run(agent, "msg", timeout=30.0, retries=2)
            ```
        """
        if self._config.cost_budget is not None and self._usage.cost_microcents >= self._config.cost_budget:
            raise CostBudgetExceededError(
                budget=self._config.cost_budget,
                actual=self._usage.cost_microcents,
                step_id=self._step_id,
            )

        effective_timeout = timeout if timeout is not None else self._config.timeout
        effective_retries = retries if retries is not None else self._config.retries
        response = await self._execute_with_config(agent, message, effective_timeout, effective_retries)

        self._usage = self._usage + StepUsage.from_chat_response(response)
        return response

    async def _execute_with_config(
        self,
        agent: FastroAgent[OutputT],
        message: str,
        timeout: float | None,
        retries: int,
    ) -> ChatResponse[OutputT]:
        """Execute agent call with timeout and retry logic."""
        last_error: Exception | None = None
        retry_delay = self._config.retry_delay

        for attempt in range(max(1, retries + 1)):
            try:
                coro = agent.run(message=message, deps=self._deps, tracer=self._tracer)
                if timeout:
                    return await asyncio.wait_for(coro, timeout=timeout)
                return await coro
            except TimeoutError:
                last_error = TimeoutError(
                    f"Agent call timed out after {timeout}s (attempt {attempt + 1}/{retries + 1})"
                )
            except Exception as e:
                last_error = e

            if attempt < retries:
                await asyncio.sleep(retry_delay * (2**attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected error in _execute_with_config")  # pragma: no cover


class BaseStep(ABC, Generic[DepsT, OutputT]):
    """Abstract base class for pipeline steps.

    A step is one unit of work. It:
    - Receives context with inputs and dependencies
    - Does something (AI call, computation, API call)
    - Returns typed output

    Steps should be stateless. Any state goes in deps or inputs.

    Examples:
        ```python
        class ExtractStep(BaseStep[MyDeps, ExtractionResult]):
            '''Extract entities from document.'''

            def __init__(self):
                self.agent = FastroAgent(system_prompt="Extract entities.")

            async def execute(self, context: StepContext[MyDeps]) -> ExtractionResult:
                document = context.get_input("document")
                response = await self.agent.run(f"Extract: {document}")
                return ExtractionResult.model_validate_json(response.content)
        ```
    """

    @abstractmethod
    async def execute(self, context: StepContext[DepsT]) -> OutputT:
        """Execute step logic.

        Override this method to implement your step's behavior.

        Args:
            context: Execution context with inputs, deps, and step outputs.

        Returns:
            The step's typed output.
        """
        ...
