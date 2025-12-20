"""Tests for the agent module."""
# mypy: disable-error-code="var-annotated"

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from fastroai.agent import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    AgentConfig,
    ChatResponse,
    FastroAgent,
    StreamChunk,
)
from fastroai.tracing import SimpleTracer
from fastroai.usage import CostCalculator


class TestAgentConfigDefaults:
    """Tests for AgentConfig default values."""

    def test_default_model(self) -> None:
        """Default model should be gpt-4o."""
        assert DEFAULT_MODEL == "openai:gpt-4o"

    def test_default_max_tokens(self) -> None:
        """Default max tokens should be 4096."""
        assert DEFAULT_MAX_TOKENS == 4096

    def test_default_temperature(self) -> None:
        """Default temperature should be 0.7."""
        assert DEFAULT_TEMPERATURE == 0.7

    def test_default_timeout(self) -> None:
        """Default timeout should be 120 seconds."""
        assert DEFAULT_TIMEOUT_SECONDS == 120

    def test_default_max_retries(self) -> None:
        """Default max retries should be 3."""
        assert DEFAULT_MAX_RETRIES == 3

    def test_default_system_prompt(self) -> None:
        """Default system prompt should be set."""
        assert DEFAULT_SYSTEM_PROMPT == "You are a helpful AI assistant."


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_values(self) -> None:
        """Should use defaults when no values provided."""
        config = AgentConfig()
        assert config.model == DEFAULT_MODEL
        assert config.max_tokens == DEFAULT_MAX_TOKENS
        assert config.temperature == DEFAULT_TEMPERATURE
        assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
        assert config.max_retries == DEFAULT_MAX_RETRIES
        assert config.system_prompt is None

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = AgentConfig(
            model="anthropic:claude-3-5-sonnet",
            system_prompt="Be brief.",
            max_tokens=1000,
            temperature=0.3,
            timeout_seconds=60,
            max_retries=5,
        )
        assert config.model == "anthropic:claude-3-5-sonnet"
        assert config.system_prompt == "Be brief."
        assert config.max_tokens == 1000
        assert config.temperature == 0.3
        assert config.timeout_seconds == 60
        assert config.max_retries == 5

    def test_get_effective_system_prompt_with_custom(self) -> None:
        """Should return custom system prompt when set."""
        config = AgentConfig(system_prompt="Custom prompt")
        assert config.get_effective_system_prompt() == "Custom prompt"

    def test_get_effective_system_prompt_default(self) -> None:
        """Should return default when system_prompt is None."""
        config = AgentConfig()
        assert config.get_effective_system_prompt() == DEFAULT_SYSTEM_PROMPT

    def test_temperature_validation(self) -> None:
        """Temperature should be between 0 and 2."""
        config = AgentConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = AgentConfig(temperature=2.0)
        assert config.temperature == 2.0

        with pytest.raises(ValueError):
            AgentConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            AgentConfig(temperature=2.1)

    def test_timeout_validation(self) -> None:
        """Timeout should be positive."""
        with pytest.raises(ValueError):
            AgentConfig(timeout_seconds=0)

        with pytest.raises(ValueError):
            AgentConfig(timeout_seconds=-1)


class TestChatResponse:
    """Tests for ChatResponse."""

    def test_create_response(self) -> None:
        """Should create response with all fields."""
        response = ChatResponse(
            output="Hello!",
            content="Hello!",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            tool_calls=[],
            cost_microcents=175,
            processing_time_ms=500,
            trace_id="abc-123",
        )
        assert response.output == "Hello!"
        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.total_tokens == 15
        assert response.cost_microcents == 175
        assert response.processing_time_ms == 500
        assert response.trace_id == "abc-123"

    def test_cost_dollars_property(self) -> None:
        """Should calculate cost in dollars correctly."""
        response = ChatResponse(
            output="Test",
            content="Test",
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_microcents=1_000_000,
            processing_time_ms=0,
        )
        assert response.cost_dollars == 1.0

        response = ChatResponse(
            output="Test",
            content="Test",
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_microcents=750,
            processing_time_ms=0,
        )
        assert response.cost_dollars == pytest.approx(0.00075)

    def test_tool_calls_default_empty(self) -> None:
        """Tool calls should default to empty list."""
        response = ChatResponse(
            output="Test",
            content="Test",
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_microcents=0,
            processing_time_ms=0,
        )
        assert response.tool_calls == []

    def test_trace_id_optional(self) -> None:
        """Trace ID should be optional."""
        response = ChatResponse(
            output="Test",
            content="Test",
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_microcents=0,
            processing_time_ms=0,
        )
        assert response.trace_id is None


class TestStreamChunk:
    """Tests for StreamChunk."""

    def test_content_chunk(self) -> None:
        """Should create content chunk."""
        chunk = StreamChunk(content="Hello", is_final=False)
        assert chunk.content == "Hello"
        assert chunk.is_final is False
        assert chunk.usage_data is None

    def test_final_chunk(self) -> None:
        """Should create final chunk with usage data."""
        usage = ChatResponse(
            output="Done",
            content="Done",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_microcents=175,
            processing_time_ms=500,
        )
        chunk = StreamChunk(content="", is_final=True, usage_data=usage)
        assert chunk.is_final is True
        assert chunk.usage_data is not None
        assert chunk.usage_data.cost_microcents == 175

    def test_defaults(self) -> None:
        """Should have sensible defaults."""
        chunk = StreamChunk()
        assert chunk.content == ""
        assert chunk.is_final is False
        assert chunk.usage_data is None


class TestFastroAgentInit:
    """Tests for FastroAgent initialization."""

    def test_default_init(self) -> None:
        """Should initialize with defaults."""
        # Use 'test' model to avoid needing API keys
        agent = FastroAgent(model="test")
        assert agent.config.model == "test"
        assert agent.toolsets == []
        assert isinstance(agent.cost_calculator, CostCalculator)

    def test_init_with_config(self) -> None:
        """Should use provided config."""
        config = AgentConfig(model="test", temperature=0.3)
        agent = FastroAgent(config=config)
        assert agent.config.model == "test"
        assert agent.config.temperature == 0.3

    def test_init_with_kwargs(self) -> None:
        """Should create config from kwargs."""
        agent = FastroAgent(model="test", temperature=0.5)
        assert agent.config.model == "test"
        assert agent.config.temperature == 0.5

    def test_init_with_custom_calculator(self) -> None:
        """Should use provided cost calculator."""
        custom_pricing = {"test-model": {"input_per_mtok": 1.00, "output_per_mtok": 2.00}}
        calc = CostCalculator(pricing_overrides=custom_pricing)
        agent = FastroAgent(model="test", cost_calculator=calc)
        assert agent.cost_calculator is calc

    def test_agent_property(self) -> None:
        """Should expose underlying PydanticAI agent."""
        agent = FastroAgent(model="test")
        assert agent.agent is not None


class TestFastroAgentRun:
    """Tests for FastroAgent.run() method."""

    @pytest.fixture
    def mock_result(self) -> MagicMock:
        """Create a mock PydanticAI result."""
        result = MagicMock()
        result.output = "Test response"

        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.total_tokens = 150
        usage.model = "gpt-4o"
        # Cache tokens
        usage.cache_read_tokens = 0
        usage.cache_write_tokens = 0
        # Audio tokens
        usage.input_audio_tokens = 0
        usage.output_audio_tokens = 0
        usage.cache_audio_read_tokens = 0
        # Request/tool metrics
        usage.requests = 1
        usage.tool_calls = 0
        usage.details = {}
        result.usage.return_value = usage

        result.new_messages.return_value = []
        return result

    async def test_run_returns_chat_response(self, mock_result: MagicMock) -> None:
        """Should return ChatResponse on success."""
        agent = FastroAgent(model="test")

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            response = await agent.run("Hello")

        assert isinstance(response, ChatResponse)
        assert response.content == "Test response"
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    async def test_run_calculates_cost(self, mock_result: MagicMock) -> None:
        """Should calculate cost correctly."""
        agent = FastroAgent(model="test")

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            response = await agent.run("Hello")

        # gpt-4o: $2.50/1M input, $10/1M output (from genai-prices)
        # 100 / 1M * $2.50 = $0.00025 = 250 microcents
        # 50 / 1M * $10 = $0.0005 = 500 microcents
        # Total: 750 microcents
        assert response.cost_microcents == 750

    async def test_run_includes_trace_id(self, mock_result: MagicMock) -> None:
        """Should include trace ID in response."""
        agent = FastroAgent(model="test")

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            response = await agent.run("Hello")

        assert response.trace_id is not None
        assert len(response.trace_id) == 36  # UUID format

    async def test_run_with_tracer(self, mock_result: MagicMock) -> None:
        """Should use provided tracer."""
        agent = FastroAgent(model="test")
        tracer = SimpleTracer()

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            response = await agent.run("Hello", tracer=tracer)

        assert response.trace_id is not None

    async def test_run_passes_message_history(self, mock_result: MagicMock) -> None:
        """Should pass message history to PydanticAI."""
        agent = FastroAgent(model="test")
        history: list = []  # type: ignore

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            await agent.run("Continue", message_history=history)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["message_history"] == history

    async def test_run_passes_deps(self, mock_result: MagicMock) -> None:
        """Should pass deps to PydanticAI."""
        agent = FastroAgent(model="test")
        deps = {"key": "value"}

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            await agent.run("Hello", deps=deps)

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["deps"] == deps

    async def test_run_measures_processing_time(self, mock_result: MagicMock) -> None:
        """Should measure processing time."""
        agent = FastroAgent(model="test")

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            response = await agent.run("Hello")

        assert response.processing_time_ms >= 0


class TestFastroAgentToolExtraction:
    """Tests for tool call extraction."""

    async def test_extracts_tool_calls(self) -> None:
        """Should extract tool calls from result."""
        agent = FastroAgent(model="test")

        # Create mock result with tool calls
        tool_part = MagicMock()
        tool_part.tool_name = "calculator"
        tool_part.args = {"expression": "2+2"}
        tool_part.tool_call_id = "call_123"

        message = MagicMock()
        message.parts = [tool_part]

        result = MagicMock()
        result.output = "The answer is 4"
        result.new_messages.return_value = [message]

        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        usage.total_tokens = 15
        usage.model = "gpt-4o"
        # Cache tokens
        usage.cache_read_tokens = 0
        usage.cache_write_tokens = 0
        # Audio tokens
        usage.input_audio_tokens = 0
        usage.output_audio_tokens = 0
        usage.cache_audio_read_tokens = 0
        # Request/tool metrics
        usage.requests = 1
        usage.tool_calls = 0
        usage.details = {}
        result.usage.return_value = usage

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = result
            response = await agent.run("What is 2+2?")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["tool_name"] == "calculator"
        assert response.tool_calls[0]["args"] == {"expression": "2+2"}
        assert response.tool_calls[0]["tool_call_id"] == "call_123"

    async def test_handles_no_tool_calls(self) -> None:
        """Should handle responses without tool calls."""
        agent = FastroAgent(model="test")

        result = MagicMock()
        result.output = "Hello"
        result.new_messages.return_value = []

        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        usage.total_tokens = 15
        usage.model = "gpt-4o"
        # Cache tokens
        usage.cache_read_tokens = 0
        usage.cache_write_tokens = 0
        # Audio tokens
        usage.input_audio_tokens = 0
        usage.output_audio_tokens = 0
        usage.cache_audio_read_tokens = 0
        # Request/tool metrics
        usage.requests = 1
        usage.tool_calls = 0
        usage.details = {}
        result.usage.return_value = usage

        with patch.object(agent._agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = result
            response = await agent.run("Hello")

        assert response.tool_calls == []


class TestFastroAgentEscapeHatch:
    """Tests for FastroAgent with custom PydanticAI agent."""

    def test_init_with_custom_agent(self) -> None:
        """Should accept a pre-configured PydanticAI agent."""
        # Use TestModel to avoid needing API keys
        custom_agent: Agent[None, str] = Agent(model=TestModel())
        fastro_agent = FastroAgent(agent=custom_agent, model="test")

        assert fastro_agent._agent is custom_agent
        # Config is still created for cost calculation
        assert fastro_agent.config is not None


class MockAsyncIterator:
    """Mock async iterator for testing streaming."""

    def __init__(self, items: list[str]) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class TestFastroAgentStreaming:
    """Tests for FastroAgent streaming functionality."""

    async def test_run_stream_yields_chunks(self) -> None:
        """run_stream should yield StreamChunk objects."""
        agent = FastroAgent(model="test")

        # Create mock streaming context
        mock_stream = MagicMock()
        mock_stream.stream_text.return_value = MockAsyncIterator(["Hello", " ", "world"])

        # Mock usage and output for final chunk
        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        usage.total_tokens = 15
        usage.model = "gpt-4o"
        # Cache tokens
        usage.cache_read_tokens = 0
        usage.cache_write_tokens = 0
        # Audio tokens
        usage.input_audio_tokens = 0
        usage.output_audio_tokens = 0
        usage.cache_audio_read_tokens = 0
        # Request/tool metrics
        usage.requests = 1
        usage.tool_calls = 0
        usage.details = {}
        mock_stream.usage.return_value = usage
        mock_stream.get_output.return_value = "Hello world"
        mock_stream.new_messages.return_value = []

        # Create async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_stream
        mock_context.__aexit__.return_value = None

        with patch.object(agent._agent, "run_stream", return_value=mock_context):
            chunks = []
            async for chunk in agent.run_stream("Test"):
                chunks.append(chunk)

        # Should have content chunks + final chunk
        assert len(chunks) == 4  # "Hello", " ", "world", final
        assert chunks[0].content == "Hello"
        assert chunks[0].is_final is False
        assert chunks[-1].is_final is True
        assert chunks[-1].usage_data is not None
        assert chunks[-1].usage_data.input_tokens == 10

    async def test_run_stream_final_chunk_has_usage(self) -> None:
        """Final chunk should contain complete usage data."""
        agent = FastroAgent(model="test")

        mock_stream = MagicMock()
        mock_stream.stream_text.return_value = MockAsyncIterator(["Response"])

        usage = MagicMock()
        usage.input_tokens = 50
        usage.output_tokens = 25
        usage.total_tokens = 75
        usage.model = "gpt-4o"
        # Cache tokens
        usage.cache_read_tokens = 0
        usage.cache_write_tokens = 0
        # Audio tokens
        usage.input_audio_tokens = 0
        usage.output_audio_tokens = 0
        usage.cache_audio_read_tokens = 0
        # Request/tool metrics
        usage.requests = 1
        usage.tool_calls = 0
        usage.details = {}
        mock_stream.usage.return_value = usage
        mock_stream.get_output.return_value = "Response"
        mock_stream.new_messages.return_value = []

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_stream
        mock_context.__aexit__.return_value = None

        with patch.object(agent._agent, "run_stream", return_value=mock_context):
            chunks = [chunk async for chunk in agent.run_stream("Test")]

        final = chunks[-1]
        assert final.is_final is True
        assert final.usage_data is not None
        assert final.usage_data.input_tokens == 50
        assert final.usage_data.output_tokens == 25


class TestToolCallExtraction:
    """Tests for tool call extraction edge cases."""

    def test_extract_tool_calls_handles_exception(self) -> None:
        """_extract_tool_calls should return empty list on exception."""
        agent = FastroAgent(model="test", system_prompt="Test")

        # Create a mock result that raises on new_messages()
        mock_result = MagicMock()
        mock_result.new_messages.side_effect = RuntimeError("Failed to get messages")

        tool_calls = agent._extract_tool_calls(mock_result)
        assert tool_calls == []

    def test_extract_tool_calls_from_messages_handles_exception(self) -> None:
        """_extract_tool_calls_from_messages should return empty list on exception."""
        agent = FastroAgent(model="test", system_prompt="Test")

        # Create messages that cause iteration to fail
        class BadMessages:
            def __iter__(self):
                raise RuntimeError("Iteration failed")

        tool_calls = agent._extract_tool_calls_from_messages(BadMessages())
        assert tool_calls == []
