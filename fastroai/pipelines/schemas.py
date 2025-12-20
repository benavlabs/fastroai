"""Usage tracking schemas for pipelines.

Provides StepUsage for individual step metrics and PipelineUsage
for aggregated pipeline-wide metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..agent import ChatResponse


class StepUsage(BaseModel):
    """Usage metrics for a single pipeline step.

    Automatically extracted from ChatResponse when using AgentStep.

    Examples:
        ```python
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
        ```
    """

    input_tokens: int = Field(default=0, description="Tokens consumed by input/prompt.")
    output_tokens: int = Field(default=0, description="Tokens in response/completion.")

    cache_read_tokens: int = Field(default=0, description="Tokens read from prompt cache (typically 90% cheaper).")
    cache_write_tokens: int = Field(default=0, description="Tokens written to prompt cache (typically 25% premium).")

    input_audio_tokens: int = Field(default=0, description="Audio input tokens for multimodal models.")
    output_audio_tokens: int = Field(default=0, description="Audio output tokens for multimodal models.")
    cache_audio_read_tokens: int = Field(default=0, description="Audio tokens read from cache.")

    request_count: int = Field(default=0, description="Number of API requests made.")
    tool_call_count: int = Field(default=0, description="Number of tool invocations executed.")

    cost_microcents: int = Field(default=0, description="Cost in microcents (1/1,000,000 dollar).")
    processing_time_ms: int = Field(default=0, description="Wall-clock processing time in milliseconds.")

    model: str | None = Field(default=None, description="Model that generated the response.")

    @classmethod
    def from_chat_response(cls, response: ChatResponse[Any]) -> StepUsage:
        """Create StepUsage from a ChatResponse.

        Args:
            response: ChatResponse from an agent run.

        Returns:
            StepUsage with metrics extracted from the response.

        Examples:
            ```python
            response = await agent.run("Hello")
            usage = StepUsage.from_chat_response(response)

            print(f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out")
            print(f"Cost: {usage.cost_microcents} microcents")
            ```
        """
        return cls(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
            input_audio_tokens=response.input_audio_tokens,
            output_audio_tokens=response.output_audio_tokens,
            cache_audio_read_tokens=response.cache_audio_read_tokens,
            request_count=response.request_count,
            tool_call_count=response.tool_call_count,
            cost_microcents=response.cost_microcents,
            processing_time_ms=response.processing_time_ms,
            model=response.model,
        )

    def __add__(self, other: StepUsage) -> StepUsage:
        """Combine two StepUsage instances.

        Args:
            other: Another StepUsage to add.

        Returns:
            New StepUsage with summed metrics.

        Examples:
            ```python
            usage1 = StepUsage(input_tokens=100, cost_microcents=50)
            usage2 = StepUsage(input_tokens=200, cost_microcents=100)

            total = usage1 + usage2
            print(total.input_tokens)  # 300
            print(total.cost_microcents)  # 150
            ```
        """
        return StepUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            input_audio_tokens=self.input_audio_tokens + other.input_audio_tokens,
            output_audio_tokens=self.output_audio_tokens + other.output_audio_tokens,
            cache_audio_read_tokens=self.cache_audio_read_tokens + other.cache_audio_read_tokens,
            request_count=self.request_count + other.request_count,
            tool_call_count=self.tool_call_count + other.tool_call_count,
            cost_microcents=self.cost_microcents + other.cost_microcents,
            processing_time_ms=self.processing_time_ms + other.processing_time_ms,
            model=self.model or other.model,
        )


class PipelineUsage(BaseModel):
    """Aggregated usage across all pipeline steps.

    Examples:
        ```python
        # From step usages
        usage = PipelineUsage.from_step_usages({
            "extract": StepUsage(cost_microcents=100, ...),
            "classify": StepUsage(cost_microcents=200, ...),
        })

        print(f"Total cost: ${usage.total_cost_dollars:.6f}")
        print(f"Steps: {list(usage.steps.keys())}")
        ```
    """

    total_input_tokens: int = Field(default=0, description="Total tokens consumed by input/prompt.")
    total_output_tokens: int = Field(default=0, description="Total tokens in response/completion.")

    total_cache_read_tokens: int = Field(default=0, description="Total tokens read from prompt cache.")
    total_cache_write_tokens: int = Field(default=0, description="Total tokens written to prompt cache.")

    total_input_audio_tokens: int = Field(default=0, description="Total audio input tokens.")
    total_output_audio_tokens: int = Field(default=0, description="Total audio output tokens.")
    total_cache_audio_read_tokens: int = Field(default=0, description="Total audio tokens read from cache.")

    total_request_count: int = Field(default=0, description="Total number of API requests made.")
    total_tool_call_count: int = Field(default=0, description="Total number of tool invocations executed.")

    total_cost_microcents: int = Field(default=0, description="Total cost in microcents (1/1,000,000 dollar).")
    total_processing_time_ms: int = Field(default=0, description="Total wall-clock processing time in milliseconds.")

    steps: dict[str, StepUsage] = Field(default_factory=dict, description="Usage breakdown by step ID.")

    @classmethod
    def from_step_usages(cls, step_usages: dict[str, StepUsage]) -> PipelineUsage:
        """Aggregate metrics from individual step usages.

        Args:
            step_usages: Dict mapping step IDs to their usage metrics.

        Returns:
            PipelineUsage with summed totals and per-step breakdown.

        Examples:
            ```python
            step_usages = {
                "extract": StepUsage(input_tokens=100, cost_microcents=50),
                "classify": StepUsage(input_tokens=200, cost_microcents=100),
            }

            usage = PipelineUsage.from_step_usages(step_usages)
            print(f"Total tokens: {usage.total_input_tokens}")  # 300
            print(f"Total cost: ${usage.total_cost_dollars:.6f}")

            # Access per-step breakdown
            for step_id, step_usage in usage.steps.items():
                print(f"  {step_id}: {step_usage.cost_microcents} microcents")
            ```
        """
        return cls(
            total_input_tokens=sum(u.input_tokens for u in step_usages.values()),
            total_output_tokens=sum(u.output_tokens for u in step_usages.values()),
            total_cache_read_tokens=sum(u.cache_read_tokens for u in step_usages.values()),
            total_cache_write_tokens=sum(u.cache_write_tokens for u in step_usages.values()),
            total_input_audio_tokens=sum(u.input_audio_tokens for u in step_usages.values()),
            total_output_audio_tokens=sum(u.output_audio_tokens for u in step_usages.values()),
            total_cache_audio_read_tokens=sum(u.cache_audio_read_tokens for u in step_usages.values()),
            total_request_count=sum(u.request_count for u in step_usages.values()),
            total_tool_call_count=sum(u.tool_call_count for u in step_usages.values()),
            total_cost_microcents=sum(u.cost_microcents for u in step_usages.values()),
            total_processing_time_ms=sum(u.processing_time_ms for u in step_usages.values()),
            steps=step_usages,
        )

    @property
    def total_cost_dollars(self) -> float:
        """Total cost in dollars for display purposes.

        Returns:
            Total cost as a float in dollars.

        Note:
            Use total_cost_microcents for calculations to avoid floating-point errors.
        """
        return self.total_cost_microcents / 1_000_000
