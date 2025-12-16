"""Configuration classes for pipelines and steps.

Provides:
- StepConfig: Configuration for individual pipeline steps
- PipelineConfig: Configuration for pipelines with additional options
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class StepConfig:
    """Configuration for a pipeline step.

    Attributes:
        timeout: Maximum time in seconds for step execution. None = no timeout.
        retries: Number of retry attempts on failure. 0 = no retries.
        retry_delay: Delay in seconds between retry attempts.
        cost_budget: Maximum cost in microcents. None = no budget limit.

    Examples:
        ```python
        # Step with 30s timeout and 2 retries
        config = StepConfig(timeout=30.0, retries=2)

        # Step with cost budget of $0.10 (10 cents = 100_000 microcents)
        config = StepConfig(cost_budget=100_000)
        ```
    """

    timeout: float | None = None
    retries: int = 0
    retry_delay: float = 1.0
    cost_budget: int | None = None


@dataclass
class PipelineConfig(StepConfig):
    """Configuration for a pipeline with additional options.

    Inherits all StepConfig fields plus pipeline-specific options.

    Attributes:
        trace: Whether to enable tracing for this pipeline.
        on_error: Error handling strategy:
            - "fail": Stop pipeline on first error (default)
            - "continue": Continue executing other steps on error

    Examples:
        ```python
        # Pipeline with tracing and 60s timeout
        config = PipelineConfig(trace=True, timeout=60.0)

        # Pipeline that continues on errors
        config = PipelineConfig(on_error="continue")
        ```
    """

    trace: bool = True
    on_error: Literal["fail", "continue"] = "fail"
