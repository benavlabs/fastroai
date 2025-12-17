"""FastroAI - Lightweight AI orchestration built on PydanticAI.

FastroAI provides production-ready primitives for building AI applications:
- FastroAgent: PydanticAI wrapper with usage tracking and tracing
- Pipeline: DAG-based workflow orchestration with automatic parallelism
- @step: Decorator for concise pipeline step definitions
- @safe_tool: Production-safe tool decorator with timeout and retry
- CostCalculator: Precise cost tracking with microcents accuracy

Example:
    from fastroai import FastroAgent, SimpleTracer

    agent = FastroAgent(model="openai:gpt-4o")
    response = await agent.run("Hello!")

    print(response.content)
    print(f"Cost: ${response.cost_dollars:.6f}")

Pipeline Example:
    from fastroai import Pipeline, step, StepContext, PipelineConfig

    @step(timeout=30.0, retries=2)
    async def classify(ctx: StepContext[None]) -> str:
        response = await ctx.run(classifier, ctx.get_input("text"))
        return response.output

    pipeline = Pipeline(
        name="processor",
        steps={"classify": classify},
        config=PipelineConfig(timeout=60.0),
    )

    result = await pipeline.execute({"text": "hello"}, None)
    print(result.output)
"""

__version__ = "0.1.0"

from .agent import AgentConfig, AgentStepWrapper, ChatResponse, FastroAgent, StreamChunk
from .errors import CostBudgetExceededError, FastroAIError, PipelineValidationError
from .pipelines import (
    BasePipeline,
    BaseStep,
    ConversationState,
    ConversationStatus,
    Pipeline,
    PipelineConfig,
    PipelineResult,
    PipelineUsage,
    StepConfig,
    StepContext,
    StepExecutionError,
    StepUsage,
    step,
)
from .tools import FunctionToolsetBase, SafeToolset, safe_tool
from .tracing import LogfireTracer, NoOpTracer, SimpleTracer, Tracer
from .usage import CostCalculator

__all__ = [
    "__version__",
    # Agent
    "FastroAgent",
    "AgentStepWrapper",
    "AgentConfig",
    "ChatResponse",
    "StreamChunk",
    # Errors
    "FastroAIError",
    "PipelineValidationError",
    "CostBudgetExceededError",
    # Pipelines
    "Pipeline",
    "PipelineResult",
    "PipelineConfig",
    "BaseStep",
    "StepContext",
    "StepConfig",
    "step",
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
    "LogfireTracer",
    "NoOpTracer",
    # Usage
    "CostCalculator",
]
