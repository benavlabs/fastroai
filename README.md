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

### Pipelines

For multi-step workflows, Pipeline orchestrates execution and parallelizes where possible.

The simplest way to create a pipeline step is with `.as_step()`:

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

#### Multi-step Pipelines

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

#### Custom Steps with `BaseStep`

For steps that need conditional logic, multiple agent calls, or custom transformations, subclass `BaseStep`:

```python
from fastroai import BaseStep, StepContext, FastroAgent

class ResearchStep(BaseStep[None, dict]):
    def __init__(self):
        self.summarizer = FastroAgent(model="gpt-4o-mini", system_prompt="Summarize.")
        self.fact_checker = FastroAgent(model="gpt-4o", system_prompt="Verify facts.")

    async def execute(self, context: StepContext[None]) -> dict:
        topic = context.get_input("topic")
        summary = await self.summarizer.run(f"Summarize: {topic}")

        if "unverified" in summary.content.lower():
            verified = await self.fact_checker.run(f"Verify: {summary.content}")
            return {"summary": summary.content, "verified": verified.content}

        return {"summary": summary.content, "verified": None}
```

### Safe Tools

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
