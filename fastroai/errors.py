"""FastroAI exception hierarchy and error classification.

Provides structured exceptions for clear error handling:
- FastroAIError: Base exception for all FastroAI errors
- PipelineValidationError: Invalid pipeline configuration
- CostBudgetExceeded: Cost budget was exceeded
- DispatchSkippedError: Short-circuit signal raised from on_before_dispatch hooks

Plus the ErrorCategory enum, used by on_after_dispatch callbacks to classify
exceptions for downstream consumers (e.g. circuit breakers).
"""

from __future__ import annotations

from enum import StrEnum


class FastroAIError(Exception):
    """Base exception for all FastroAI errors.

    All FastroAI-specific exceptions inherit from this class,
    allowing you to catch all library errors with a single except clause.

    Examples:
        ```python
        try:
            result = await pipeline.execute(inputs, deps)
        except FastroAIError as e:
            logger.error(f"FastroAI error: {e}")
        ```
    """

    pass


class PipelineValidationError(FastroAIError):
    """Invalid pipeline configuration.

    Raised at pipeline construction time for:
    - Unknown step in dependencies
    - Circular dependencies
    - Missing output_step when needed

    Examples:
        ```python
        try:
            pipeline = Pipeline(
                name="test",
                steps={"a": step_a},
                dependencies={"a": ["unknown"]},  # Error!
            )
        except PipelineValidationError as e:
            print(f"Invalid config: {e}")
        ```
    """

    pass


class CostBudgetExceededError(FastroAIError):
    """Cost budget was exceeded.

    Raised when a step or pipeline exceeds its configured cost_budget.
    The current operation completes, but subsequent ctx.run() calls
    will raise this exception.

    Attributes:
        budget_microcents: The configured budget limit.
        actual_microcents: The actual cost incurred.
        step_id: The step where budget was exceeded (if in pipeline).

    Examples:
        ```python
        try:
            response = await ctx.run(agent, "message")
        except CostBudgetExceededError as e:
            print(f"Over budget: {e.actual_microcents} > {e.budget_microcents}")
        ```
    """

    def __init__(
        self,
        budget: int,
        actual: int,
        step_id: str | None = None,
    ) -> None:
        """Initialize CostBudgetExceededError.

        Args:
            budget: The configured budget limit in microcents.
            actual: The actual cost incurred in microcents.
            step_id: The step where budget was exceeded (optional).
        """
        self.budget_microcents = budget
        self.actual_microcents = actual
        self.step_id = step_id
        location = f" in step '{step_id}'" if step_id else ""
        super().__init__(f"Cost budget exceeded{location}: {actual} microcents > {budget} microcents budget")


class DispatchSkippedError(FastroAIError):
    """Raise from `on_before_dispatch` to short-circuit the dispatch.

    When raised, `agent.run()` is not called and the retry loop in
    `StepContext._execute_with_config` does not retry — the exception
    propagates immediately to the caller of `ctx.run()`. Use this for
    guards that should fail fast: circuit breakers being the canonical
    example. Subclass this for application-specific skip reasons.

    The retry loop distinguishes `DispatchSkippedError` from other exceptions:
    the former propagates without retry; the latter triggers exponential
    backoff. This avoids burning ``retries × retry_delay`` seconds on a
    sick provider that the guard already classified as "don't even try."

    Examples:
        ```python
        class BreakerOpen(DispatchSkippedError):
            def __init__(self, provider: str):
                super().__init__(f"{provider} circuit breaker open")
                self.provider = provider

        async def before_dispatch() -> None:
            if breaker.state == "open":
                raise BreakerOpen("deepseek")

        agent = FastroAgent(
            ...,
            on_before_dispatch=before_dispatch,
        )
        ```
    """

    pass


class ErrorCategory(StrEnum):
    """Bare-minimum vocabulary for error classification at the dispatch hooks layer.

    Concrete categorization (which exceptions are TRANSIENT for which
    provider) is the application's responsibility — fastroai stays
    provider-agnostic. Used by `on_after_dispatch` callbacks to feed
    downstream consumers (circuit breakers, retry budgets, alerting).

    Categories:
        TRANSIENT: Should retry / let breaker count toward opening.
            Network errors, rate limits, 5xx responses, request timeouts.
        PERMANENT: Should not retry / no breaker signal.
            4xx responses (except 408/429), schema validation errors,
            malformed input.
        RESOURCE_EXHAUSTION: Out of memory / disk / quota.
            Distinct from TRANSIENT because recovery typically needs
            operational action (scale up, free disk) rather than a
            retry-after delay.
        UNKNOWN: Unclassified — caller decides how to treat.
    """

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"
