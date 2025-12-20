"""FastroAgent - PydanticAI wrapper with usage tracking and tracing.

This module provides FastroAgent, a convenience wrapper around PydanticAI's
Agent that adds automatic cost calculation, distributed tracing, and a
consistent response format.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any, Generic, TypeVar, cast

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset

from ..pipelines.base import BaseStep, StepContext
from ..tracing import NoOpTracer, Tracer
from ..usage import CostCalculator
from .schemas import AgentConfig, ChatResponse, StreamChunk

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


class FastroAgent(Generic[OutputT]):
    """AI agent with usage tracking, cost calculation, and tracing.

    Wraps PydanticAI's Agent to provide:
    - Automatic cost calculation in microcents
    - Optional distributed tracing
    - Streaming and non-streaming modes
    - Consistent ChatResponse format
    - Structured output support via output_type

    The agent is STATELESS regarding conversation history.
    Callers load history from their storage and pass it to run().

    Examples:
        ```python
        # Basic usage (returns string)
        agent = FastroAgent(
            model="openai:gpt-4o",
            system_prompt="You are helpful.",
        )
        response = await agent.run("Hello!")
        print(response.content)
        print(f"Cost: ${response.cost_dollars:.6f}")

        # With structured output
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: int
            explanation: str

        agent = FastroAgent(
            model="openai:gpt-4o",
            output_type=Answer,
        )
        response = await agent.run("What is 2+2?")
        print(response.output.value)  # 4

        # With conversation history (you load it)
        history = await my_memory_service.load(user_id)
        response = await agent.run("Continue", message_history=history)
        await my_memory_service.save(user_id, "Continue", response.content)

        # With tracing
        from fastroai import SimpleTracer
        tracer = SimpleTracer()
        response = await agent.run("Hello", tracer=tracer)

        # With custom deps for tools
        response = await agent.run("Search for news", deps=MyDeps(api_key="..."))

        # Streaming
        async for chunk in agent.run_stream("Tell me a story"):
            if chunk.is_final:
                print(f"\\nCost: ${chunk.usage_data.cost_dollars:.6f}")
            else:
                print(chunk.content, end="", flush=True)
        ```
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        agent: Agent[Any, OutputT] | None = None,
        output_type: type[OutputT] | None = None,
        toolsets: list[AbstractToolset] | None = None,
        cost_calculator: CostCalculator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FastroAgent.

        Args:
            config: Agent configuration. If None, creates from kwargs.
            agent: Pre-configured PydanticAI Agent (escape hatch).
                  If provided, config is only used for cost calculation.
            output_type: Pydantic model for structured output. Defaults to str.
            toolsets: Tool sets available to the agent.
            cost_calculator: Cost calculator. Default uses standard pricing.
            **kwargs: Passed to AgentConfig if config is None.
                     Common: model, system_prompt, temperature, max_tokens.

        Examples:
            ```python
            # Using config object
            config = AgentConfig(model="gpt-4o", temperature=0.3)
            agent = FastroAgent(config=config)

            # Using kwargs (simpler)
            agent = FastroAgent(model="gpt-4o", temperature=0.5)

            # With structured output
            agent = FastroAgent(model="gpt-4o", output_type=MyResponseModel)

            # Custom pricing override (e.g., volume discount)
            calc = CostCalculator(pricing_overrides={
                "gpt-4o": {"input_per_mtok": 2.00, "output_per_mtok": 8.00}
            })
            agent = FastroAgent(cost_calculator=calc)

            # Escape hatch: your own PydanticAI agent
            from pydantic_ai import Agent
            pydantic_agent = Agent(model="gpt-4o", output_type=MyType)
            agent = FastroAgent(agent=pydantic_agent)
            ```
        """
        self.config = config or AgentConfig(**kwargs)
        self.toolsets = toolsets or []
        self.cost_calculator = cost_calculator or CostCalculator()
        self._output_type = output_type

        if agent is not None:
            self._agent = agent
        else:
            self._agent = Agent(
                model=self.config.model,
                system_prompt=self.config.get_effective_system_prompt(),
                toolsets=self.toolsets if self.toolsets else None,
                output_type=cast(type[OutputT], output_type or str),
            )

    @property
    def agent(self) -> Agent[Any, OutputT]:
        """Access the underlying PydanticAI agent.

        Returns:
            The wrapped PydanticAI Agent instance.
        """
        return self._agent

    async def run(
        self,
        message: str,
        deps: DepsT | None = None,
        message_history: list[ModelMessage] | None = None,
        model_settings: ModelSettings | None = None,
        tracer: Tracer | None = None,
        **kwargs: Any,
    ) -> ChatResponse[OutputT]:
        """Execute a single agent interaction.

        Args:
            message: User message to send.
            deps: Dependencies passed to tools. Can be any type.
            message_history: Previous messages (you load these from your storage).
            model_settings: Runtime model config overrides.
            tracer: Tracer for distributed tracing.
            **kwargs: Passed to PydanticAI Agent.run().

        Returns:
            ChatResponse with content, usage, cost, and trace_id.

        Examples:
            ```python
            # Simple usage
            response = await agent.run("Hello!")
            print(response.content)
            print(f"Cost: ${response.cost_dollars:.6f}")

            # With conversation history
            history = await memory.load(user_id)
            response = await agent.run("Continue", message_history=history)
            await memory.save(user_id, "Continue", response.content)

            # With tracing
            tracer = SimpleTracer()
            response = await agent.run("Hello", tracer=tracer)
            print(f"Trace ID: {response.trace_id}")
            ```
        """
        effective_tracer = tracer or NoOpTracer()

        async with effective_tracer.span(
            "fastroai.agent.run",
            model=self.config.model,
            has_history=message_history is not None,
            history_length=len(message_history) if message_history else 0,
        ) as trace_id:
            return await self._execute(
                message=message,
                deps=deps,
                message_history=message_history,
                model_settings=model_settings,
                trace_id=trace_id,
                tracer=effective_tracer,
                **kwargs,
            )

    async def run_stream(
        self,
        message: str,
        deps: DepsT | None = None,
        message_history: list[ModelMessage] | None = None,
        model_settings: ModelSettings | None = None,
        tracer: Tracer | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk[OutputT], None]:
        """Execute a streaming agent interaction.

        Yields StreamChunk objects as the response is generated.
        The final chunk has is_final=True and includes complete usage data.

        Args:
            message: User message to send.
            deps: Dependencies passed to tools.
            message_history: Previous messages.
            model_settings: Runtime model config overrides.
            tracer: Tracer for distributed tracing.
            **kwargs: Passed to PydanticAI Agent.run_stream().

        Yields:
            StreamChunk objects. Final chunk has usage_data.

        Examples:
            ```python
            async for chunk in agent.run_stream("Tell me a story"):
                if chunk.is_final:
                    print(f"\\nCost: ${chunk.usage_data.cost_dollars:.6f}")
                else:
                    print(chunk.content, end="", flush=True)
            ```
        """
        effective_tracer = tracer or NoOpTracer()

        async with effective_tracer.span(
            "fastroai.agent.run_stream",
            model=self.config.model,
        ) as trace_id:
            async for chunk in self._execute_stream(
                message=message,
                deps=deps,
                message_history=message_history,
                model_settings=model_settings,
                trace_id=trace_id,
                tracer=effective_tracer,
                **kwargs,
            ):
                yield chunk

    async def _execute(
        self,
        message: str,
        deps: Any,
        message_history: list[ModelMessage] | None,
        model_settings: ModelSettings | None,
        trace_id: str,
        tracer: Tracer,
        **kwargs: Any,
    ) -> ChatResponse[OutputT]:
        """Internal execution logic."""
        effective_settings = model_settings or ModelSettings(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        start_time = time.perf_counter()

        result = await self._agent.run(
            user_prompt=message,
            deps=deps,
            message_history=message_history,
            model_settings=effective_settings,
            **kwargs,
        )

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        response = self._create_response(result, trace_id, processing_time_ms)

        tracer.log_metric(trace_id, "input_tokens", response.input_tokens)
        tracer.log_metric(trace_id, "output_tokens", response.output_tokens)
        tracer.log_metric(trace_id, "cost_microcents", response.cost_microcents)

        return response

    async def _execute_stream(
        self,
        message: str,
        deps: Any,
        message_history: list[ModelMessage] | None,
        model_settings: ModelSettings | None,
        trace_id: str,
        tracer: Tracer,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk[OutputT], None]:
        """Internal streaming logic."""
        effective_settings = model_settings or ModelSettings(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        start_time = time.perf_counter()

        async with self._agent.run_stream(
            user_prompt=message,
            deps=deps,
            message_history=message_history,
            model_settings=effective_settings,
            **kwargs,
        ) as stream:
            async for text in stream.stream_text():
                yield StreamChunk(content=text, is_final=False)

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            final_response = await self._create_streaming_response(stream, trace_id, processing_time_ms)

            tracer.log_metric(trace_id, "input_tokens", final_response.input_tokens)
            tracer.log_metric(trace_id, "output_tokens", final_response.output_tokens)
            tracer.log_metric(trace_id, "cost_microcents", final_response.cost_microcents)

            yield StreamChunk(content="", is_final=True, usage_data=final_response)

    def _create_response(
        self,
        result: Any,
        trace_id: str,
        processing_time_ms: int,
    ) -> ChatResponse[OutputT]:
        """Create ChatResponse from PydanticAI result."""
        usage = result.usage()

        model = self.config.model
        if hasattr(usage, "model") and usage.model:
            model = usage.model

        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        total_tokens = usage.total_tokens or (input_tokens + output_tokens)

        cache_read_tokens = getattr(usage, "cache_read_tokens", 0) or 0
        cache_write_tokens = getattr(usage, "cache_write_tokens", 0) or 0

        input_audio_tokens = getattr(usage, "input_audio_tokens", 0) or 0
        output_audio_tokens = getattr(usage, "output_audio_tokens", 0) or 0
        cache_audio_read_tokens = getattr(usage, "cache_audio_read_tokens", 0) or 0

        request_count = getattr(usage, "requests", 1) or 1
        tool_call_count = getattr(usage, "tool_calls", 0) or 0

        usage_details: dict[str, int] = {}
        if hasattr(usage, "details") and usage.details:  # pragma: no cover
            usage_details = {k: v for k, v in usage.details.items() if isinstance(v, int)}

        cost_microcents = self.cost_calculator.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            input_audio_tokens=input_audio_tokens,
            output_audio_tokens=output_audio_tokens,
            cache_audio_read_tokens=cache_audio_read_tokens,
        )

        tool_calls = self._extract_tool_calls(result)
        output: OutputT = result.output
        content = output if isinstance(output, str) else str(output)

        return ChatResponse[OutputT](
            output=output,
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            input_audio_tokens=input_audio_tokens,
            output_audio_tokens=output_audio_tokens,
            cache_audio_read_tokens=cache_audio_read_tokens,
            tool_calls=tool_calls,
            tool_call_count=tool_call_count,
            request_count=request_count,
            cost_microcents=cost_microcents,
            processing_time_ms=processing_time_ms,
            trace_id=trace_id,
            usage_details=usage_details,
        )

    async def _create_streaming_response(
        self,
        stream: Any,
        trace_id: str,
        processing_time_ms: int,
    ) -> ChatResponse[OutputT]:
        """Create ChatResponse from StreamedRunResult.

        StreamedRunResult has a different interface than AgentRunResult:
        - .get_output() for the output (async)
        - .usage() for usage data
        - .new_messages() for messages
        """
        usage = stream.usage()

        model = self.config.model
        if hasattr(usage, "model") and usage.model:
            model = usage.model

        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        total_tokens = usage.total_tokens or (input_tokens + output_tokens)

        cache_read_tokens = getattr(usage, "cache_read_tokens", 0) or 0
        cache_write_tokens = getattr(usage, "cache_write_tokens", 0) or 0

        input_audio_tokens = getattr(usage, "input_audio_tokens", 0) or 0
        output_audio_tokens = getattr(usage, "output_audio_tokens", 0) or 0
        cache_audio_read_tokens = getattr(usage, "cache_audio_read_tokens", 0) or 0

        request_count = getattr(usage, "requests", 1) or 1
        tool_call_count = getattr(usage, "tool_calls", 0) or 0

        usage_details: dict[str, int] = {}
        if hasattr(usage, "details") and usage.details:  # pragma: no cover
            usage_details = {k: v for k, v in usage.details.items() if isinstance(v, int)}

        cost_microcents = self.cost_calculator.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            input_audio_tokens=input_audio_tokens,
            output_audio_tokens=output_audio_tokens,
            cache_audio_read_tokens=cache_audio_read_tokens,
        )

        tool_calls = self._extract_tool_calls_from_messages(stream.new_messages())
        raw_output = stream.get_output()
        if inspect.isawaitable(raw_output):  # pragma: no cover
            raw_output = await raw_output
        output: OutputT = raw_output
        content = output if isinstance(output, str) else str(output)

        return ChatResponse[OutputT](
            output=output,
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            input_audio_tokens=input_audio_tokens,
            output_audio_tokens=output_audio_tokens,
            cache_audio_read_tokens=cache_audio_read_tokens,
            tool_calls=tool_calls,
            tool_call_count=tool_call_count,
            request_count=request_count,
            cost_microcents=cost_microcents,
            processing_time_ms=processing_time_ms,
            trace_id=trace_id,
            usage_details=usage_details,
        )

    def _extract_tool_calls(self, result: Any) -> list[dict[str, Any]]:
        """Extract tool call information from result."""
        try:
            messages = result.new_messages()
            return self._extract_tool_calls_from_messages(messages)
        except Exception:
            return []

    def _extract_tool_calls_from_messages(self, messages: Any) -> list[dict[str, Any]]:
        """Extract tool call information from message list."""
        tool_calls: list[dict[str, Any]] = []

        try:
            for message in messages:
                if hasattr(message, "parts"):
                    for part in message.parts:
                        if hasattr(part, "tool_name") and hasattr(part, "args"):
                            tool_calls.append(
                                {
                                    "tool_name": part.tool_name,
                                    "args": part.args if isinstance(part.args, dict) else {},
                                    "tool_call_id": getattr(part, "tool_call_id", None),
                                }
                            )
        except Exception:
            pass

        return tool_calls

    def as_step(
        self,
        prompt: Callable[[StepContext[DepsT]], str] | str,
    ) -> AgentStepWrapper[DepsT, OutputT]:
        """Turn this agent into a pipeline step.

        Creates a BaseStep that runs this agent with the given prompt
        and returns the agent's output directly.

        Args:
            prompt: Either a static string or a function that builds
                   the prompt from the step context.

        Returns:
            A BaseStep that can be used in a Pipeline.

        Examples:
            ```python
            # Static prompt
            agent = FastroAgent(model="gpt-4o", system_prompt="Summarize text.")
            step = agent.as_step("Summarize the document.")

            # Dynamic prompt from context
            agent = FastroAgent(model="gpt-4o", system_prompt="Summarize text.")
            step = agent.as_step(lambda ctx: f"Summarize: {ctx.get_input('doc')}")

            # With structured output
            agent = FastroAgent(model="gpt-4o", output_type=Summary)
            step = agent.as_step(lambda ctx: f"Summarize: {ctx.get_input('doc')}")
            # step returns Summary directly

            # Use in pipeline
            pipeline = Pipeline(
                name="summarizer",
                steps={"summarize": step},
            )
            ```
        """
        return AgentStepWrapper(self, prompt)


class AgentStepWrapper(BaseStep[DepsT, OutputT]):
    """Pipeline step wrapper for FastroAgent.

    Created via FastroAgent.as_step(). Wraps an agent as a pipeline step.

    The wrapper uses ctx.run() for automatic tracer/deps forwarding and usage
    tracking, and returns the agent's typed output directly.

    Note:
        Use FastroAgent.as_step() to create instances rather than
        instantiating directly.
    """

    def __init__(
        self,
        agent: FastroAgent[OutputT],
        prompt: Callable[[StepContext[DepsT]], str] | str,
    ) -> None:
        """Initialize the step wrapper.

        Args:
            agent: The FastroAgent to wrap.
            prompt: Static string or function that builds the prompt from context.
        """
        self._agent = agent
        self._prompt = prompt

    @property
    def agent(self) -> FastroAgent[OutputT]:
        """Access the underlying FastroAgent.

        Returns:
            The wrapped FastroAgent instance.
        """
        return self._agent

    async def execute(self, context: StepContext[DepsT]) -> OutputT:
        """Execute the agent with the configured prompt.

        Args:
            context: Step execution context with inputs, deps, and config.

        Returns:
            The agent's typed output.
        """
        message = self._prompt if isinstance(self._prompt, str) else self._prompt(context)
        response = await context.run(self._agent, message)
        return response.output
