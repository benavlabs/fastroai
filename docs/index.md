<style>
    .md-typeset h1,
    .md-content__button {
        display: none;
    }
</style>

<p align="center">
  <a href="https://github.com/benavlabs/fastroai">
    <img src="assets/logo.png" alt="FastroAI Logo" width="25%">
  </a>
</p>
<p align="center" markdown=1>
  <i>Lightweight AI orchestration built on PydanticAI.</i>
</p>
<p align="center" markdown=1>
<a href="https://pypi.org/project/fastroai/">
  <img src="https://img.shields.io/pypi/v/fastroai?color=%23E91E63&label=pypi%20package" alt="PyPi Version"/>
</a>
<a href="https://pypi.org/project/fastroai/">
  <img src="https://img.shields.io/pypi/pyversions/fastroai.svg?color=%23E91E63" alt="Supported Python Versions"/>
</a>
<a href="https://github.com/benavlabs/fastroai/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/>
</a>
</p>
<hr>
<p align="justify">
<b>FastroAI</b> wraps <a href="https://ai.pydantic.dev/">PydanticAI</a> with production essentials: <b>cost tracking in microcents</b>, <b>multi-step pipelines</b>, and <b>tools that handle failures gracefully</b>. You get everything PydanticAI offers, plus the infrastructure you'd build yourself anyway.
</p>
<hr>

> **Note**: FastroAI is experimental, it was extracted into a package from code that we had in production in different contexts. We built it for ourselves but you're free to use and contribute. The API may change between versions and you'll probably find bugs, we're here to fix them. Use in production at your own risk (we do).

## Features

- **Cost Tracking**: Automatic cost calculation in microcents. Integer math, no floating-point drift.
- **Pipelines**: DAG-based workflows with automatic parallelization and dependency resolution.
- **Safe Tools**: Timeout, retry, and graceful error handling for AI tools.
- **Tracing**: Protocol-based integration with any observability platform.
- **Structured Output**: Type-safe responses with Pydantic models.
- **Streaming**: Real-time responses with cost tracking on the final chunk.

## Requirements

- **Python 3.10+**: Modern async/await and type hints.
- **AI API Key**: OpenAI, Anthropic, Google, or other provider.

## Quick Start

### 1. Install FastroAI

=== "pip"

    ```bash
    pip install fastroai
    ```

=== "uv"

    ```bash
    uv add fastroai
    ```

=== "poetry"

    ```bash
    poetry add fastroai
    ```

### 2. Set Your API Key

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sk-your-key-here"
    ```

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY="sk-ant-your-key-here"
    ```

=== "Google"

    ```bash
    export GOOGLE_API_KEY="your-key-here"
    ```

### 3. Run Your First Agent

```python
import asyncio
from fastroai import FastroAgent

agent = FastroAgent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
)

async def main():
    response = await agent.run("What is 2 + 2?")

    print(response.output)
    print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
    print(f"Cost: ${response.cost_dollars:.6f}")

asyncio.run(main())
```

Output:

```
2 + 2 equals 4.
Tokens: 24 in, 8 out
Cost: $0.000120
```

That's it. You have an AI agent with automatic cost tracking.

## Usage

### Single Agent Calls

`FastroAgent` is a thin wrapper around [PydanticAI's Agent](https://ai.pydantic.dev/). It adds automatic cost tracking and a consistent response format, but otherwise stays out of your way. All PydanticAI features work exactly as documented:

```python
from fastroai import FastroAgent

agent = FastroAgent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
)

response = await agent.run("What is the capital of France?")
print(response.output)
print(f"Cost: ${response.cost_dollars:.6f}")
```

### Structured Output

Get Pydantic models back instead of strings:

```python
from pydantic import BaseModel
from fastroai import FastroAgent

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

agent = FastroAgent(
    model="openai:gpt-4o",
    output_type=MovieReview,
)

response = await agent.run("Review the movie Inception")
print(response.output.title)   # "Inception"
print(response.output.rating)  # 9
```

### Multi-Step Pipelines

Real applications often need multiple AI calls: extract entities, then classify them, then generate a response. You could chain these manually with `await`, but then you're writing boilerplate for dependency ordering, parallel execution, and cost aggregation.

Pipelines handle this. Declare your steps and dependencies, and FastroAI runs them in the right order, parallelizes independent steps, and tracks costs across the whole workflow. All FastroAI features flow through: you get cost tracking per step and per pipeline, plus distributed tracing across the entire flow.

For simple DAG workflows, this is less verbose than [pydantic-graph](https://ai.pydantic.dev/graph/) and far simpler than durable execution frameworks like Temporal. It's enough for most AI orchestration needs:

```python
from fastroai import Pipeline, step, StepContext, FastroAgent

extractor = FastroAgent(model="openai:gpt-4o-mini", system_prompt="Extract entities.")
classifier = FastroAgent(model="openai:gpt-4o-mini", system_prompt="Classify documents.")

@step
async def extract(ctx: StepContext[None]) -> str:
    document = ctx.get_input("document")
    response = await ctx.run(extractor, f"Extract entities: {document}")
    return response.output

@step(timeout=30.0, retries=2)
async def classify(ctx: StepContext[None]) -> str:
    entities = ctx.get_dependency("extract")
    response = await ctx.run(classifier, f"Classify based on: {entities}")
    return response.output

pipeline = Pipeline(
    name="document_processor",
    steps={"extract": extract, "classify": classify},
    dependencies={"classify": ["extract"]},
)

result = await pipeline.execute({"document": "Apple announced..."}, deps=None)
print(f"Total cost: ${result.usage.total_cost_dollars:.6f}")
```

### Safe Tools

When you give an AI agent tools that call external APIs, those APIs will eventually fail. They'll time out, return errors, or hang indefinitely. With regular tools, this crashes your entire request and the user sees an error page.

`@safe_tool` wraps tools with timeout, retry, and graceful error handling. When something fails, instead of raising an exception, the AI receives an error message it can work with:

```python
from fastroai import safe_tool

@safe_tool(timeout=10, max_retries=2)
async def fetch_weather(location: str) -> str:
    """Get weather for a location."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.weather.com/{location}")
        return resp.text
```

If the API times out, the AI sees "Tool timed out after 2 attempts" and can respond: "I'm having trouble checking the weather right now. Would you like me to try again?" Your request doesn't crash, you don't lose the prompt tokens, and the user gets a response.

## Response Fields

Every `ChatResponse` includes:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The response text |
| `output` | `OutputT` | Typed output (same as content for string agents) |
| `input_tokens` | `int` | Prompt tokens consumed |
| `output_tokens` | `int` | Completion tokens generated |
| `cost_microcents` | `int` | Cost in 1/1,000,000 of a dollar |
| `cost_dollars` | `float` | Cost in dollars (for display) |
| `processing_time_ms` | `int` | Wall-clock time |
| `trace_id` | `str` | Tracing correlation ID |

Use `cost_microcents` when aggregating costs across many calls. Use `cost_dollars` for display.

## License

[MIT](https://github.com/benavlabs/fastroai/blob/main/LICENSE)

---

## FastroAI Template

Looking for a complete AI SaaS starter? [FastroAI Template](https://fastro.ai) includes authentication, payments, background tasks, and more built on top of this library.

---

<div style="text-align: center; margin-top: 50px;">
    <a href="learn/" class="md-button md-button--primary">
        Start Learning
    </a>
    <a href="guides/" class="md-button">
        Browse Guides
    </a>
</div>
