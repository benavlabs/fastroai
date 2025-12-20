"""Schemas for the agent module.

Defines configuration and response models for FastroAgent.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

OutputT = TypeVar("OutputT")

DEFAULT_MODEL = "openai:gpt-4o"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


class AgentConfig(BaseModel):
    """Configuration for FastroAgent instances.

    All parameters have sensible defaults. Override as needed.

    Examples:
        ```python
        # Minimal - uses all defaults
        config = AgentConfig()

        # Custom configuration
        config = AgentConfig(
            model="anthropic:claude-3-5-sonnet",
            system_prompt="You are a financial advisor.",
            temperature=0.3,
        )

        # Use with agent
        agent = FastroAgent(config=config)

        # Or pass kwargs directly to FastroAgent
        agent = FastroAgent(model="openai:gpt-4o-mini", temperature=0.5)
        ```
    """

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model identifier (e.g., 'openai:gpt-4o', 'anthropic:claude-3-5-sonnet').",
    )
    system_prompt: str | None = Field(default=None, description="System prompt. If None, uses DEFAULT_SYSTEM_PROMPT.")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, description="Maximum tokens in response.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = creative).",
    )
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0, description="Request timeout in seconds.")
    max_retries: int = Field(default=DEFAULT_MAX_RETRIES, ge=0, description="Maximum retry attempts on failure.")

    def get_effective_system_prompt(self) -> str:
        """Get system prompt, using default if not set.

        Returns:
            The configured system prompt or DEFAULT_SYSTEM_PROMPT.
        """
        return self.system_prompt if self.system_prompt is not None else DEFAULT_SYSTEM_PROMPT


class ChatResponse(BaseModel, Generic[OutputT]):
    """Response from an AI agent interaction.

    Contains the response content plus comprehensive usage metrics
    for billing, analytics, and debugging.

    Examples:
        ```python
        response = await agent.run("What is 2+2?")

        print(f"Answer: {response.content}")
        print(f"Cost: ${response.cost_dollars:.6f}")
        print(f"Tokens: {response.total_tokens}")

        # Check cache effectiveness
        if response.cache_read_tokens > 0:
            cache_ratio = response.cache_read_tokens / response.input_tokens
            print(f"Cache hit ratio: {cache_ratio:.1%}")

        if response.tool_calls:
            for call in response.tool_calls:
                print(f"Used tool: {call['tool_name']}")

        # With structured output
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: int
            explanation: str

        agent = FastroAgent(output_type=Answer)
        response = await agent.run("What is 2+2?")
        print(response.output.value)  # 4
        print(response.output.explanation)  # "2 plus 2 equals 4"
        ```

    Note:
        Why microcents? Floating-point math has precision errors (0.1 + 0.2 = 0.30000000000000004).
        With integers, precision is exact. For billing systems, this matters.
    """

    output: OutputT = Field(description="The typed output from the agent.")
    content: str = Field(description="String representation of the output.")
    model: str = Field(description="Model that generated the response.")

    input_tokens: int = Field(description="Tokens consumed by input/prompt.")
    output_tokens: int = Field(description="Tokens in response/completion.")
    total_tokens: int = Field(description="Total tokens (input + output).")

    cache_read_tokens: int = Field(
        default=0,
        description="Tokens read from prompt cache (typically 90% cheaper).",
    )
    cache_write_tokens: int = Field(
        default=0,
        description="Tokens written to prompt cache (typically 25% premium).",
    )

    input_audio_tokens: int = Field(default=0, description="Audio input tokens for multimodal models.")
    output_audio_tokens: int = Field(default=0, description="Audio output tokens for multimodal models.")
    cache_audio_read_tokens: int = Field(default=0, description="Audio tokens read from cache.")

    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tools invoked during generation.")
    tool_call_count: int = Field(default=0, description="Number of tool invocations executed.")
    request_count: int = Field(default=1, description="Number of API requests made during this interaction.")

    cost_microcents: int = Field(description="Cost in microcents (1/1,000,000 dollar).")
    processing_time_ms: int = Field(description="Wall-clock processing time in milliseconds.")

    trace_id: str | None = Field(default=None, description="Distributed tracing correlation ID.")
    usage_details: dict[str, int] = Field(
        default_factory=dict,
        description="Provider-specific usage details (e.g., reasoning_tokens for o1).",
    )

    @property
    def cost_dollars(self) -> float:
        """Cost in dollars for display purposes.

        Returns:
            Cost as a float in dollars.

        Note:
            Use cost_microcents for calculations to avoid floating-point errors.
        """
        return self.cost_microcents / 1_000_000


class StreamChunk(BaseModel, Generic[OutputT]):
    """A chunk in a streaming response.

    Most chunks have content with is_final=False.
    The last chunk has is_final=True with complete usage data.

    Examples:
        ```python
        async for chunk in agent.run_stream("Tell me a story"):
            if chunk.is_final:
                print(f"\\nTotal cost: ${chunk.usage_data.cost_dollars:.6f}")
            else:
                print(chunk.content, end="", flush=True)
        ```
    """

    content: str = Field(default="", description="Text content of this chunk.")
    is_final: bool = Field(default=False, description="True if this is the final chunk with usage data.")
    usage_data: ChatResponse[OutputT] | None = Field(
        default=None, description="Complete usage data (only on final chunk)."
    )
