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

    model: str = DEFAULT_MODEL
    """Model identifier (e.g., 'openai:gpt-4o', 'anthropic:claude-3-5-sonnet')."""

    system_prompt: str | None = None
    """System prompt. If None, uses DEFAULT_SYSTEM_PROMPT."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    """Maximum tokens in response."""

    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    """Sampling temperature (0.0 = deterministic, 2.0 = creative)."""

    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    """Request timeout in seconds."""

    max_retries: int = Field(default=DEFAULT_MAX_RETRIES, ge=0)
    """Maximum retry attempts on failure."""

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

    Attributes:
        output: The typed output from the agent. For string agents, same as content.
        content: String representation of the output.
        model: Model that generated the response.
        input_tokens: Tokens consumed by input/prompt.
        output_tokens: Tokens in response/completion.
        total_tokens: input_tokens + output_tokens.
        tool_calls: Tools invoked during generation.
        cost_microcents: Cost in 1/10,000ths of a cent (integer).
        processing_time_ms: Wall-clock time in milliseconds.
        trace_id: Distributed tracing correlation ID.

    Examples:
        ```python
        response = await agent.run("What is 2+2?")

        print(f"Answer: {response.content}")
        print(f"Cost: ${response.cost_dollars:.6f}")
        print(f"Tokens: {response.total_tokens}")

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
        Why microcents?
        Floating-point math has precision errors:
        >>> 0.1 + 0.2
        0.30000000000000004

        With microcents (integers), precision is exact:
        >>> 100 + 200
        300

        For billing systems, this matters.
    """

    output: OutputT
    """The typed output from the agent."""

    content: str
    """String representation of the output."""

    model: str
    """Model that generated the response."""

    input_tokens: int
    """Tokens consumed by input/prompt."""

    output_tokens: int
    """Tokens in response/completion."""

    total_tokens: int
    """Total tokens (input + output)."""

    tool_calls: list[dict[str, Any]] = []
    """Tools invoked during generation."""

    cost_microcents: int
    """Cost in microcents (1/1,000,000 dollar)."""

    processing_time_ms: int
    """Wall-clock processing time in milliseconds."""

    trace_id: str | None = None
    """Distributed tracing correlation ID."""

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

    content: str = ""
    """Text content of this chunk."""

    is_final: bool = False
    """True if this is the final chunk with usage data."""

    usage_data: ChatResponse[OutputT] | None = None
    """Complete usage data (only on final chunk)."""
