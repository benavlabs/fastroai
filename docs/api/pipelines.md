# Pipelines

The pipelines module provides DAG-based workflow orchestration with automatic parallelism, multi-turn conversation support, and usage tracking.

## Pipeline

::: fastroai.pipelines.Pipeline
    options:
      show_root_heading: true
      show_source: false

## PipelineResult

::: fastroai.pipelines.PipelineResult
    options:
      show_root_heading: true
      show_source: false

## BaseStep

::: fastroai.pipelines.BaseStep
    options:
      show_root_heading: true
      show_source: false

## StepContext

::: fastroai.pipelines.StepContext
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - step_id
        - deps
        - tracer
        - usage
        - config
        - get_input
        - get_dependency
        - get_dependency_or_none
        - run

## step

::: fastroai.pipelines.step
    options:
      show_root_heading: true
      show_source: false

## Configuration

### StepConfig

::: fastroai.pipelines.StepConfig
    options:
      show_root_heading: true
      show_source: false

### PipelineConfig

::: fastroai.pipelines.PipelineConfig
    options:
      show_root_heading: true
      show_source: false

## Conversation State

### ConversationStatus

::: fastroai.pipelines.ConversationStatus
    options:
      show_root_heading: true
      show_source: false

### ConversationState

::: fastroai.pipelines.ConversationState
    options:
      show_root_heading: true
      show_source: false

## Usage Tracking

### StepUsage

::: fastroai.pipelines.StepUsage
    options:
      show_root_heading: true
      show_source: false

### PipelineUsage

::: fastroai.pipelines.PipelineUsage
    options:
      show_root_heading: true
      show_source: false

## Errors

### StepExecutionError

::: fastroai.pipelines.StepExecutionError
    options:
      show_root_heading: true
      show_source: false

---

[← Agent](agent.md){ .md-button } [Tools →](tools.md){ .md-button .md-button--primary }
