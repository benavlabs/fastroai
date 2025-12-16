# FastroAI

**Lightweight AI orchestration built on PydanticAI.**

> **Warning**: FastroAI is highly experimental. The API may change between versions without notice. Use in production at your own risk.

---

## What is FastroAI?

FastroAI is a thin layer on top of [PydanticAI](https://ai.pydantic.dev/) that adds production essentials: cost tracking, multi-step pipelines, and tools that handle failures gracefully. We built it for ourselves but you're free to use.

PydanticAI is excellent for building AI agents. But when you start building real applications, you run into the same problems repeatedly. How much did that request cost? PydanticAI gives you token counts, but you need to look up pricing and calculate costs yourself. Need to run multiple AI steps, some in parallel (but don't want to define graphs for everything)? You end up writing your own orchestration logic.

FastroAI adds a small set of focused primitives to make these tasks a lot easier. It doesn't replace PydanticAI, it wraps it and adds quality of life features.

## Installation

```bash
pip install fastroai
```

Or with uv:

```bash
uv add fastroai
```

## Quick Start

```python
from fastroai import FastroAgent

agent = FastroAgent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
)

response = await agent.run("What is the capital of France?")

print(response.content)
print(f"Cost: ${response.cost_dollars:.6f}")
```

You get the response, token counts, and cost. No manual tracking required.

## Core Concepts

### FastroAgent

FastroAgent wraps PydanticAI's `Agent` class. You get the same functionality, plus automatic cost calculation and optional tracing.

The response includes everything PydanticAI provides, plus `cost_microcents` (exact cost in 1/1,000,000 of a dollar, using integer math to avoid floating-point errors), `cost_dollars` (same value as a float for display), `processing_time_ms`, and `trace_id` for distributed tracing.

Need the underlying PydanticAI agent? Access it directly:

```python
pydantic_agent = agent.agent
```

Already have a configured PydanticAI agent with custom output types or tools? Pass it in:

```python
from pydantic_ai import Agent

my_agent = Agent(model="openai:gpt-4o", output_type=MyCustomType)
fastro_agent = FastroAgent(agent=my_agent)
```

### Cost Tracking

Every response includes cost calculated from token usage. FastroAI uses integer microcents internally (1 microcent = $0.000001) so costs don't accumulate floating-point errors over thousands of requests.

```python
response = await agent.run("Explain quantum computing")

print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"Cost: ${response.cost_dollars:.6f}")
```

Pricing is included for OpenAI, Anthropic, Google, and Groq models. For other providers, add your own:

```python
from fastroai import CostCalculator

calc = CostCalculator()
calc.add_model_pricing(
    "my-custom-model",
    input_cost_per_1k_tokens=100,
    output_cost_per_1k_tokens=200,
)
agent = FastroAgent(model="my-custom-model", cost_calculator=calc)
```

### Conversation History

FastroAgent is stateless - it doesn't store conversation history. You load history from your storage, pass it in, and save the new messages yourself.

```python
history = await my_storage.load(user_id)

response = await agent.run(
    "Continue our conversation",
    message_history=history,
)

await my_storage.save(user_id, response.new_messages())
```

This keeps the agent simple and lets you use whatever storage fits your application.

### Streaming

Stream responses with the same cost tracking:

```python
async for chunk in agent.run_stream("Tell me a story"):
    if chunk.is_final:
        print(f"\nTotal cost: ${chunk.usage_data.cost_dollars:.6f}")
    else:
        print(chunk.content, end="", flush=True)
```

The final chunk includes complete usage data, so you don't lose cost tracking when streaming.

### Tracing

Pass a tracer to correlate AI calls with the rest of your application:

```python
from fastroai import SimpleTracer

tracer = SimpleTracer()
response = await agent.run("Hello", tracer=tracer)
print(response.trace_id)
```

`SimpleTracer` logs to Python's logging module. For production, implement the `Tracer` protocol to integrate with Logfire, OpenTelemetry, Datadog, or your preferred observability platform.

## Pipelines

For multi-step workflows, Pipeline orchestrates execution and parallelizes where possible.

### Three Ways to Define Steps

FastroAI offers three approaches, from simplest to most flexible:

#### 1. `agent.as_step()` - Single-Agent Steps

The simplest way to create a pipeline step:

```python
from fastroai import FastroAgent, Pipeline

summarizer = FastroAgent(
    model="openai:gpt-4o-mini",
    system_prompt="Summarize text concisely.",
)

pipeline = Pipeline(
    name="summarizer",
    steps={"summarize": summarizer.as_step(lambda ctx: ctx.get_input("text"))},
)

result = await pipeline.execute({"text": "Long article..."}, deps=None)
print(result.output)
```

#### 2. `@step` Decorator - Function-Based Steps

For steps that need logic beyond a single agent call:

```python
from fastroai import step, Pipeline, StepContext

@step
async def transform(ctx: StepContext[None]) -> str:
    text = ctx.get_input("text")
    return text.upper()

@step(timeout=30.0, retries=2)
async def classify(ctx: StepContext[None]) -> str:
    text = ctx.get_dependency("transform")
    response = await ctx.run(classifier_agent, f"Classify: {text}")
    return response.output

pipeline = Pipeline(
    name="processor",
    steps={"transform": transform, "classify": classify},
    dependencies={"classify": ["transform"]},
)
```

The `@step` decorator accepts configuration options:
- `timeout`: Maximum execution time in seconds for `ctx.run()` calls
- `retries`: Number of retry attempts on failure
- `retry_delay`: Base delay between retries (exponential backoff)
- `cost_budget`: Maximum cost in microcents for this step

#### 3. `BaseStep` Class - Complex Multi-Agent Steps

For steps that need multiple agents, conditional logic, or complex orchestration:

```python
from fastroai import BaseStep, StepContext, FastroAgent

class ResearchStep(BaseStep[MyDeps, dict]):
    def __init__(self):
        self.classifier = FastroAgent(model="gpt-4o-mini", system_prompt="Classify.")
        self.writer = FastroAgent(model="gpt-4o", system_prompt="Write reports.")

    async def execute(self, context: StepContext[MyDeps]) -> dict:
        topic = context.get_input("topic")

        # Use ctx.run() for automatic deps/tracer forwarding and usage tracking
        category = await context.run(self.classifier, f"Classify: {topic}")

        if "technical" in category.output.lower():
            report = await context.run(self.writer, f"Technical report on: {topic}")
        else:
            report = await context.run(self.writer, f"General summary of: {topic}")

        return {"category": category.output, "report": report.content}
```

### `ctx.run()` - The Key Integration Point

When calling agents from within a step, always use `ctx.run()`:

```python
response = await ctx.run(agent, "Your message")
```

This automatically:
- Passes your application dependencies (`deps`) to the agent
- Forwards the tracer for distributed tracing
- Accumulates usage in `ctx.usage` for pipeline-wide tracking
- Enforces timeout, retries, and cost budget from config

You can override config per-call:

```python
response = await ctx.run(agent, "message", timeout=60.0, retries=3)
```

### Multi-Step Pipelines with Parallelism

Chain steps by declaring dependencies. FastroAI runs independent steps in parallel:

```python
extract_agent = FastroAgent(
    model="openai:gpt-4o-mini",
    system_prompt="Extract named entities from text.",
)
classify_agent = FastroAgent(
    model="openai:gpt-4o-mini",
    system_prompt="Classify documents based on entities.",
)

pipeline = Pipeline(
    name="document_processor",
    steps={
        "extract": extract_agent.as_step(
            lambda ctx: f"Extract entities: {ctx.get_input('document')}"
        ),
        "classify": classify_agent.as_step(
            lambda ctx: f"Classify: {ctx.get_dependency('extract', str)}"
        ),
    },
    dependencies={"classify": ["extract"]},
)

result = await pipeline.execute({"document": "Apple announced..."}, deps=None)
print(f"Total cost: ${result.usage.total_cost_dollars:.6f}")
```

The prompt can be a static string or a function receiving the step context. Use `get_input()` for pipeline inputs and `get_dependency()` for outputs from previous steps.

### Pipeline Configuration

Configure defaults for all steps in a pipeline:

```python
from fastroai import Pipeline, PipelineConfig, StepConfig

pipeline = Pipeline(
    name="processor",
    steps={"extract": extract_step, "classify": classify_step},
    dependencies={"classify": ["extract"]},
    config=PipelineConfig(
        timeout=30.0,      # Default timeout for all steps
        retries=2,         # Default retry count
        cost_budget=100_000,  # Budget in microcents ($0.10)
    ),
    step_configs={
        "classify": StepConfig(timeout=60.0),  # Override for specific step
    },
)
```

**Config inheritance** (most specific wins):
1. Pipeline default config (`PipelineConfig`)
2. Step class config (`.config` attribute on `@step` decorated functions)
3. Per-step override (`step_configs[step_id]`)
4. Per-call override via `ctx.run(timeout=..., retries=...)`

### Multi-Turn Conversations

Steps can signal that more information is needed using `ConversationState`:

```python
from fastroai import BaseStep, ConversationState, ConversationStatus

class GatherInfoStep(BaseStep[None, ConversationState[dict]]):
    async def execute(self, context) -> ConversationState[dict]:
        message = context.get_input("message")
        current_data = context.get_input("current_data") or {}

        # Extract info from message...
        if "email" in message.lower():
            current_data["email"] = extract_email(message)

        # Check if complete
        required = {"name", "email"}
        missing = required - set(current_data.keys())

        if not missing:
            return ConversationState(
                status=ConversationStatus.COMPLETE,
                data=current_data,
            )

        return ConversationState(
            status=ConversationStatus.INCOMPLETE,
            data=current_data,
            context={"missing": list(missing)},
        )
```

When a step returns `INCOMPLETE`, the pipeline stops early. Use `result.stopped_early` and `result.conversation_state` to handle partial completion.

### Pipeline Routing

Route between pipelines based on input:

```python
from fastroai import BasePipeline

class InvestmentRouter(BasePipeline[MyDeps, dict, Plan]):
    def __init__(self):
        super().__init__("investment_router")
        self.register_pipeline("simple", simple_pipeline)
        self.register_pipeline("complex", complex_pipeline)

    async def route(self, input_data: dict, deps: MyDeps) -> str:
        if input_data.get("amount", 0) < 10000:
            return "simple"
        return "complex"

router = InvestmentRouter()
result = await router.execute({"amount": 50000}, deps)
```

## Safe Tools

The `@safe_tool` decorator wraps tools with timeout, retry, and graceful error handling. When something goes wrong, the AI receives an error message instead of the request failing entirely.

```python
from fastroai import safe_tool, SafeToolset

@safe_tool(timeout=10, max_retries=2)
async def fetch_weather(location: str) -> str:
    """Get current weather for a location."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.weather.com/{location}")
        return resp.text
```

If the API times out after two retries, the AI receives "Tool timed out after 2 attempts" and can respond appropriately or try a different approach.

Custom error messages:

```python
@safe_tool(
    timeout=30,
    on_timeout="Search is taking too long. Try a simpler query.",
    on_error="Search unavailable: {error}",
)
async def search(query: str) -> str:
    ...
```

Group tools into toolsets:

```python
class WeatherToolset(SafeToolset):
    def __init__(self):
        super().__init__(tools=[fetch_weather], name="weather")

agent = FastroAgent(
    model="openai:gpt-4o",
    system_prompt="You can check the weather.",
    toolsets=[WeatherToolset()],
)
```

## Error Handling

FastroAI provides a structured error hierarchy:

```python
from fastroai import FastroAIError, PipelineValidationError, StepExecutionError, CostBudgetExceededError

try:
    result = await pipeline.execute(inputs, deps)
except CostBudgetExceededError as e:
    print(f"Over budget: {e.actual_microcents} > {e.budget_microcents}")
except StepExecutionError as e:
    print(f"Step '{e.step_id}' failed: {e.original_error}")
except PipelineValidationError as e:
    print(f"Invalid pipeline config: {e}")
except FastroAIError as e:
    print(f"FastroAI error: {e}")
```

- `FastroAIError`: Base class for all FastroAI errors
- `PipelineValidationError`: Invalid pipeline configuration (cycles, unknown steps)
- `StepExecutionError`: Step failed during execution (wraps original exception)
- `CostBudgetExceededError`: Cost budget was exceeded

## Development

```bash
uv sync --all-extras # Install dependencies
uv run pytest        # Run tests
uv run mypy fastroai # Type checking
uv run ruff check .  # Linting
uv run ruff format . # Formatting
```

## Support

For questions and discussion, join our [Discord server](https://discord.com/invite/TEmPs22gqB).

For bugs and feature requests, open an issue on [GitHub](https://github.com/benavlabs/fastroai/issues).

## License

MIT
