"""FastroAI exception hierarchy.

Provides structured exceptions for clear error handling:
- FastroAIError: Base exception for all FastroAI errors
- PipelineValidationError: Invalid pipeline configuration
- CostBudgetExceeded: Cost budget was exceeded
"""

from __future__ import annotations


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
