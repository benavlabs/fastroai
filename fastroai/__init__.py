"""FastroAI - Lightweight AI orchestration built on PydanticAI.

FastroAI provides production-ready primitives for building AI applications:
- FastroAgent: PydanticAI wrapper with usage tracking and tracing
- Pipeline: DAG-based workflow orchestration with automatic parallelism
- @safe_tool: Production-safe tool decorator with timeout and retry
- CostCalculator: Precise cost tracking with microcents accuracy

Example:
    from fastroai import FastroAgent, SimpleTracer

    agent = FastroAgent(model="openai:gpt-4o")
    response = await agent.run("Hello!")

    print(response.content)
    print(f"Cost: ${response.cost_dollars:.6f}")

Pipeline Example:
    from fastroai import Pipeline, BaseStep, StepContext

    class ExtractStep(BaseStep[None, str]):
        async def execute(self, context: StepContext[None]) -> str:
            return context.get_input("text").upper()

    pipeline = Pipeline(
        name="processor",
        steps={"extract": ExtractStep()},
    )

    result = await pipeline.execute({"text": "hello"}, None)
    print(result.output)  # "HELLO"
"""

__version__ = "0.1.0"

# Agent
from .agent import AgentConfig, AgentStepWrapper, ChatResponse, FastroAgent, StreamChunk

# Pipelines
from .pipelines import (
    BasePipeline,
    BaseStep,
    ConversationState,
    ConversationStatus,
    Pipeline,
    PipelineResult,
    PipelineUsage,
    StepContext,
    StepExecutionError,
    StepUsage,
)

# Tools
from .tools import FunctionToolsetBase, SafeToolset, safe_tool

# Tracing
from .tracing import NoOpTracer, SimpleTracer, Tracer

# Usage
from .usage import DEFAULT_PRICING, CostCalculator

__all__ = [
    "__version__",
    # Agent
    "FastroAgent",
    "AgentStepWrapper",
    "AgentConfig",
    "ChatResponse",
    "StreamChunk",
    # Pipelines
    "Pipeline",
    "PipelineResult",
    "BaseStep",
    "StepContext",
    "ConversationState",
    "ConversationStatus",
    "BasePipeline",
    "StepUsage",
    "PipelineUsage",
    "StepExecutionError",
    # Tools
    "safe_tool",
    "FunctionToolsetBase",
    "SafeToolset",
    # Tracing
    "Tracer",
    "SimpleTracer",
    "NoOpTracer",
    # Usage
    "CostCalculator",
    "DEFAULT_PRICING",
]
