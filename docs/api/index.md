# API Reference

**Complete reference for FastroAI's public API.**

All classes, functions, and protocols documented here are considered stable and follow semantic versioning.

!!! tip "Looking for explanations?"
    This is a reference, not a tutorial. For explanations and examples, see the [Guides](../guides/index.md).

## Core Components

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **Agent**

    ---

    FastroAgent, AgentConfig, ChatResponse, StreamChunk

    [:octicons-arrow-right-24: Learn more](agent.md)

-   :material-pipe:{ .lg .middle } **Pipelines**

    ---

    Pipeline, BaseStep, StepContext, configurations

    [:octicons-arrow-right-24: Learn more](pipelines.md)

-   :material-tools:{ .lg .middle } **Tools**

    ---

    @safe_tool decorator, SafeToolset, FunctionToolsetBase

    [:octicons-arrow-right-24: Learn more](tools.md)

-   :material-cash:{ .lg .middle } **Usage**

    ---

    CostCalculator with microcents precision

    [:octicons-arrow-right-24: Learn more](usage.md)

-   :material-chart-line:{ .lg .middle } **Tracing**

    ---

    Tracer protocol, SimpleTracer, LogfireTracer, NoOpTracer

    [:octicons-arrow-right-24: Learn more](tracing.md)

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
    LogfireTracer,
    NoOpTracer,

    # Usage
    CostCalculator,

    # Errors
    FastroAIError,
    PipelineValidationError,
    CostBudgetExceededError,
    DispatchSkippedError,
    ErrorCategory,
)
```

## Error Hierarchy

All FastroAI exceptions inherit from `FastroAIError`, so you can catch all library errors with a single except clause:

```
FastroAIError                    # Base for all FastroAI errors
├── PipelineValidationError      # Invalid pipeline configuration
├── StepExecutionError           # Step failed during execution
├── CostBudgetExceededError      # Cost budget exceeded
└── DispatchSkippedError         # Raised from on_before_dispatch to short-circuit (no retry)
```

`ErrorCategory` (StrEnum: `TRANSIENT`, `PERMANENT`, `RESOURCE_EXHAUSTION`, `UNKNOWN`) is provided for callers that want to categorize exceptions inside `on_after_dispatch` (e.g., a circuit breaker that only counts transient failures toward its open threshold). The library doesn't auto-categorize.

```python
try:
    result = await pipeline.execute(inputs, deps)
except FastroAIError as e:
    logger.error(f"FastroAI error: {e}")
```

---

[← Recipes](../recipes/index.md){ .md-button } [Agent →](agent.md){ .md-button .md-button--primary }
