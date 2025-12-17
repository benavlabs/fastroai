# API Reference

**Complete reference for FastroAI's public API.**

All classes, functions, and protocols documented here are considered stable and follow semantic versioning.

!!! tip "Looking for explanations?"
    This is a reference, not a tutorial. For explanations and examples, see the [Guides](../guides/index.md).

## Core Components

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **[Agent](agent.md)**

    ---

    FastroAgent, AgentConfig, ChatResponse, StreamChunk

-   :material-pipe:{ .lg .middle } **[Pipelines](pipelines.md)**

    ---

    Pipeline, BaseStep, StepContext, configurations

-   :material-tools:{ .lg .middle } **[Tools](tools.md)**

    ---

    @safe_tool decorator, SafeToolset, FunctionToolsetBase

-   :material-cash:{ .lg .middle } **[Usage](usage.md)**

    ---

    CostCalculator with microcents precision

-   :material-chart-line:{ .lg .middle } **[Tracing](tracing.md)**

    ---

    Tracer protocol, SimpleTracer, NoOpTracer

</div>

---

## Quick Import Reference

```python
from fastroai import (
    # Agent
    FastroAgent,
    AgentConfig,
    ChatResponse,
    StreamChunk,

    # Pipelines
    Pipeline,
    PipelineResult,
    PipelineConfig,
    BaseStep,
    StepContext,
    StepConfig,
    step,
    ConversationState,
    ConversationStatus,

    # Tools
    safe_tool,
    SafeToolset,
    FunctionToolsetBase,

    # Tracing
    Tracer,
    SimpleTracer,
    NoOpTracer,

    # Usage
    CostCalculator,

    # Errors
    FastroAIError,
    PipelineValidationError,
    CostBudgetExceededError,
)
```

## Error Hierarchy

All FastroAI exceptions inherit from `FastroAIError`, so you can catch all library errors with a single except clause:

```
FastroAIError                    # Base for all FastroAI errors
├── PipelineValidationError      # Invalid pipeline configuration
├── StepExecutionError           # Step failed during execution
└── CostBudgetExceededError      # Cost budget exceeded
```

```python
try:
    result = await pipeline.execute(inputs, deps)
except FastroAIError as e:
    logger.error(f"FastroAI error: {e}")
```

---

[← Recipes](../recipes/index.md){ .md-button } [Agent →](agent.md){ .md-button .md-button--primary }
