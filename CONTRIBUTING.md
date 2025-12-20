# Contributing to FastroAI

Thank you for your interest in contributing to FastroAI! This guide will get you started quickly.

## Quick Setup

```sh
# Clone and setup
git clone https://github.com/benavlabs/fastroai.git
cd fastroai
uv sync

# Verify installation
uv run pytest
uv run mypy fastroai
uv run ruff check
```

## Architecture Overview

FastroAI is a lightweight AI orchestration library built on PydanticAI. It provides production-ready primitives for building AI applications.

### Core Components

```
fastroai/
├── agent/          # FastroAgent - PydanticAI wrapper with cost tracking
├── pipelines/      # DAG-based workflow orchestration
├── tools/          # @safe_tool decorator and toolsets
├── tracing/        # Tracer protocol and implementations
└── usage/          # CostCalculator with microcents precision
```

**Key Design Principles:**

- Stateless agents (conversation history is caller-managed)
- Integer microcents for billing (avoids floating-point errors)
- Protocol-based tracing (implement `Tracer` for any backend)
- Type-safe contexts throughout pipelines

## Before You Code

1. **Read the CLAUDE.md** - contains detailed architecture and patterns
2. **Use existing patterns** - look at similar code for consistency
3. **Write tests first** - especially for new functionality
4. **Understand the component you're modifying**

## Code Standards

### Import Organization

```python
# Standard library
from typing import Any, Optional

# Third-party
from pydantic import BaseModel
from pydantic_ai import Agent

# Local
from fastroai.agent import FastroAgent
from fastroai.usage import CostCalculator
```

### Where to Put New Code

| Type of Change | Location |
|---------------|----------|
| Agent functionality | `agent/agent.py` |
| Response/request schemas | `agent/schemas.py` |
| Pipeline steps | `pipelines/base.py` |
| Pipeline execution | `pipelines/executor.py` |
| Tool decorators | `tools/decorators.py` |
| Tracer implementations | `tracing/tracer.py` |
| Cost calculation | `usage/calculator.py` |

## Testing

```sh
# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check --fix .

# Run tests
uv run pytest

# With coverage
uv run pytest --cov=fastroai

# Type checking
uv run mypy fastroai

# All checks (run before submitting PR)
uv run ruff format . && uv run ruff check --fix . && uv run mypy fastroai && uv run pytest
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"`. Mock PydanticAI agents using `model="test"` or `TestModel()`.

## Pull Request Process

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes** following the architecture principles
3. **Add tests** for new functionality
4. **Run all checks** (format, lint, mypy, pytest)
5. **Submit PR** with clear description

## Common Scenarios

### Adding a New Agent Method

1. Add method to `FastroAgent` class (`agent/agent.py`)
2. Update `ChatResponse` schema if needed (`agent/schemas.py`)
3. Add tests (`tests/test_agent.py`)
4. Update documentation

### Adding a Pipeline Feature

1. Update `BaseStep` or `Pipeline` class (`pipelines/`)
2. Update `StepUsage`/`PipelineUsage` if tracking new metrics
3. Add tests (`tests/test_pipeline.py`)
4. Update documentation

### Adding a Tracer Implementation

1. Create new class implementing `Tracer` protocol (`tracing/tracer.py`)
2. Add tests (`tests/test_tracing.py`)
3. Export from `__init__.py` if public
4. Add documentation

## Performance Guidelines

- **Use integer arithmetic** for cost calculations (microcents)
- **Keep functions pure** when possible (easier to test)
- **Use TYPE_CHECKING imports** for type hints that would create circular deps
- **Stream large responses** instead of loading everything into memory

## Need Help?

- **Documentation**: Check the [docs](docs/) folder
- **Issues**: Create a GitHub issue for bugs/features
- **Discussions**: Use GitHub Discussions for questions

## Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming environment.

---

**Quick Reference:**

- Stateless agents, integer microcents, protocol-based tracing
- Test everything, respect existing patterns
- Run `uv run ruff format . && uv run ruff check --fix . && uv run mypy fastroai && uv run pytest` before submitting

Thank you for contributing to FastroAI!
