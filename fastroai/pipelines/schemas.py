"""Usage tracking schemas for pipelines.

Provides StepUsage for individual step metrics and PipelineUsage
for aggregated pipeline-wide metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..agent import ChatResponse


class StepUsage(BaseModel):
    """Usage metrics for a single pipeline step.

    Automatically extracted from ChatResponse when using AgentStep.

    Example:
        # From ChatResponse
        usage = StepUsage.from_chat_response(response)

        # Manual creation
        usage = StepUsage(
            input_tokens=100,
            output_tokens=50,
            cost_microcents=175,
            processing_time_ms=500,
            model="gpt-4o",
        )

        # Combine usages
        total = usage1 + usage2
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_microcents: int = 0
    processing_time_ms: int = 0
    model: str | None = None

    @classmethod
    def from_chat_response(cls, response: ChatResponse[Any]) -> StepUsage:
        """Create from ChatResponse."""
        return cls(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_microcents=response.cost_microcents,
            processing_time_ms=response.processing_time_ms,
            model=response.model,
        )

    def __add__(self, other: StepUsage) -> StepUsage:
        """Combine two StepUsage instances."""
        return StepUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_microcents=self.cost_microcents + other.cost_microcents,
            processing_time_ms=self.processing_time_ms + other.processing_time_ms,
            model=self.model or other.model,
        )


class PipelineUsage(BaseModel):
    """Aggregated usage across all pipeline steps.

    Example:
        # From step usages
        usage = PipelineUsage.from_step_usages({
            "extract": StepUsage(cost_microcents=100, ...),
            "classify": StepUsage(cost_microcents=200, ...),
        })

        print(f"Total cost: ${usage.total_cost_dollars:.6f}")
        print(f"Steps: {list(usage.steps.keys())}")
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_microcents: int = 0
    total_processing_time_ms: int = 0
    steps: dict[str, StepUsage] = {}

    @classmethod
    def from_step_usages(cls, step_usages: dict[str, StepUsage]) -> PipelineUsage:
        """Aggregate from individual step usages."""
        return cls(
            total_input_tokens=sum(u.input_tokens for u in step_usages.values()),
            total_output_tokens=sum(u.output_tokens for u in step_usages.values()),
            total_cost_microcents=sum(u.cost_microcents for u in step_usages.values()),
            total_processing_time_ms=sum(u.processing_time_ms for u in step_usages.values()),
            steps=step_usages,
        )

    @property
    def total_cost_dollars(self) -> float:
        """Total cost in dollars for display purposes."""
        return self.total_cost_microcents / 1_000_000
