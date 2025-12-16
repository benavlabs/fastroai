"""Pipeline system for DAG-based workflow orchestration.

Provides declarative multi-step workflows with automatic parallelism,
type-safe dependency passing, and usage tracking.

Example:
    from fastroai import Pipeline, BaseStep, StepContext, FastroAgent

    # Simple step using .as_step()
    summarizer = FastroAgent(model="gpt-4o", system_prompt="Summarize text.")
    summarize_step = summarizer.as_step(lambda ctx: ctx.get_input("document"))

    # Complex step using BaseStep
    class ClassifyStep(BaseStep[MyDeps, str]):
        async def execute(self, context: StepContext[MyDeps]) -> str:
            text = context.get_dependency("summarize")
            return "POSITIVE" if "good" in text.lower() else "NEGATIVE"

    pipeline = Pipeline(
        name="classifier",
        steps={"summarize": summarize_step, "classify": ClassifyStep()},
        dependencies={"classify": ["summarize"]},
    )

    result = await pipeline.execute({"document": "Good news!"}, deps)
    print(result.output)  # "POSITIVE"
"""

from ..errors import CostBudgetExceededError, FastroAIError, PipelineValidationError
from .base import BaseStep, ConversationState, ConversationStatus, StepContext
from .config import PipelineConfig, StepConfig
from .decorators import step
from .executor import StepExecutionError
from .pipeline import Pipeline, PipelineResult
from .router import BasePipeline
from .schemas import PipelineUsage, StepUsage

__all__ = [
    # Base
    "BaseStep",
    "StepContext",
    "ConversationStatus",
    "ConversationState",
    # Config
    "StepConfig",
    "PipelineConfig",
    # Decorators
    "step",
    # Errors
    "FastroAIError",
    "PipelineValidationError",
    "StepExecutionError",
    "CostBudgetExceededError",
    # Pipeline
    "Pipeline",
    "PipelineResult",
    # Router
    "BasePipeline",
    # Schemas
    "StepUsage",
    "PipelineUsage",
]
