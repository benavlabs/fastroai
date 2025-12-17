<div align="center">

# FastroAI

### **Lightweight AI orchestration built on PydanticAI.**

<p align="center">
  <a href="https://benavlabs.github.io/fastroai">
    <img src="https://github.com/benavlabs/fastroai/blob/main/docs/assets/logo.png?raw=true" alt="FastroAI Logo" width="25%">
  </a>
</p>

[Documentation](https://benavlabs.github.io/fastroai) • [Discord](https://discord.com/invite/TEmPs22gqB) • [GitHub](https://github.com/benavlabs/fastroai)

<br/>

[![PyPI](https://img.shields.io/pypi/v/fastroai?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/fastroai/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-E92063?style=for-the-badge&logoColor=white)](https://ai.pydantic.dev)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>

---

FastroAI wraps [PydanticAI](https://ai.pydantic.dev/) with production essentials: **cost tracking in microcents**, **multi-step pipelines**, and **tools that handle failures gracefully**.

> **Note**: FastroAI is experimental, it was extracted into a package from code that we had in production in different contexts. We built it for ourselves but you're free to use and contribute. The API may change between versions and you'll probably find bugs, we're here to fix them. Use in production at your own risk (we do).

## Features

- **Cost Tracking**: Automatic cost calculation in microcents. No floating-point drift.
- **Pipelines**: DAG-based workflows with automatic parallelization.
- **Safe Tools**: Timeout, retry, and graceful error handling for AI tools.
- **Tracing**: Built-in Logfire integration, or bring your own observability platform.

## Installation

```bash
pip install fastroai
```

With Logfire tracing:

```bash
pip install fastroai[logfire]
```

Or with uv:

```bash
uv add fastroai
uv add "fastroai[logfire]"  # With Logfire tracing
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

Every response includes token counts and cost. No manual tracking required.

## Pipelines

Chain multiple AI steps with automatic parallelization:

```python
from fastroai import FastroAgent, Pipeline

extract = FastroAgent(model="openai:gpt-4o-mini", system_prompt="Extract entities.")
classify = FastroAgent(model="openai:gpt-4o-mini", system_prompt="Classify documents.")

pipeline = Pipeline(
    name="processor",
    steps={
        "extract": extract.as_step(lambda ctx: ctx.get_input("text")),
        "classify": classify.as_step(lambda ctx: ctx.get_dependency("extract")),
    },
    dependencies={"classify": ["extract"]},
)

result = await pipeline.execute({"text": "Apple announced..."}, deps=None)
print(f"Total cost: ${result.usage.total_cost_dollars:.6f}")
```

## Safe Tools

Tools that don't crash when external services fail:

```python
from fastroai import safe_tool

@safe_tool(timeout=10, max_retries=2)
async def fetch_weather(location: str) -> str:
    """Get weather for a location."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.weather.com/{location}")
        return resp.text
```

If the API times out, the AI receives an error message and can respond gracefully.

## Documentation

- **[Quick Start](https://benavlabs.github.io/fastroai/#quick-start)**: Install and run your first agent in 2 minutes.
- **[Guides](https://benavlabs.github.io/fastroai/guides/)**: Deep dives into agents, pipelines, tools, and tracing.
- **[API Reference](https://benavlabs.github.io/fastroai/api/)**: Complete reference for all classes and functions.

## FastroAI Template

Looking for a complete AI SaaS starter? Check out [FastroAI Template](https://fastro.ai): authentication, payments, background tasks, and more built on top of this library.

## Support

- **Questions & Discussion**: [Discord](https://discord.com/invite/TEmPs22gqB)
- **Bugs & Features**: [GitHub Issues](https://github.com/benavlabs/fastroai/issues)

## License

MIT

---

<p align="center"><i>Built by <a href="https://benav.io">Benav Labs</a></i></p>
