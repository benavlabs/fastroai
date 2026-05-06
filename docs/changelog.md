# FastroAI Changelog

## Introduction

The Changelog documents all notable changes made to FastroAI. This includes new features, bug fixes, and improvements. It's organized by version and date, providing a clear history of the library's development.

---

## [0.5.0] - May 6, 2026

#### Fixed
- **`FastroAgent` timeout is now actually enforced** by [@igorbenav](https://github.com/igorbenav)
  - The `timeout` parameter on `FastroAgent` was previously stored on `AgentConfig` but never read by `_execute()`. The configured value silently never reached the underlying model client, so a single LLM call could run for the full underlying httpx read timeout (~600s) × `max_retries+1` attempts before any error surfaced.
  - New `_build_default_model_settings()` helper forwards `self.config.timeout` into `ModelSettings.timeout`, where pydantic-ai passes it through to the model client (e.g. OpenAI SDK's `chat.completions.create(timeout=N)`).
  - Contributed to a 50-minute production hang in a downstream worker on 2026-05-06 — a hung pydantic-ai agent call that should have hit a 300s timeout per the configured constant ran for ~30 min until the worker was manually restarted.

#### Changed (Breaking)
- **`AgentConfig.timeout_seconds` renamed to `AgentConfig.timeout`** to match pydantic-ai's `ModelSettings.timeout` and to align with what `FastroAgent(timeout=N)` callers were already passing. Code using the old name will fail with a Pydantic validation error.
- **Default is now `None`** (opt-in), previously `120`. With `None`, no per-request timeout is forwarded and the model client's own default applies (typically 600s read on OpenAI). Code that wants the previous 120s behavior — though no such code worked correctly because the default was never enforced — should pass `timeout=120` explicitly.
- **`DEFAULT_TIMEOUT_SECONDS` removed** from `fastroai.agent` exports. No replacement constant; choose a value at the call site.

#### Migration

```python
# Before:
from fastroai.agent import DEFAULT_TIMEOUT_SECONDS
agent = FastroAgent(model="openai:gpt-4o", timeout_seconds=60)
config = AgentConfig(timeout_seconds=60)

# After:
agent = FastroAgent(model="openai:gpt-4o", timeout=60)
config = AgentConfig(timeout=60)
```

If you have a `*_TIMEOUT_SECONDS` constant that you've been passing into `FastroAgent` and trusting to bound LLM call wall time, that bound was not actually applying. Re-validate your value — you may have been over-budgeting because timeouts were getting absorbed by retries that no longer happen as silently.

#### Documentation
- Updated RELEASE.md with 0.5.0 release notes including breaking-change migration guide
- Updated `docs/guides/fastro-agent.md` configuration reference table

**Full Changelog**: https://github.com/benavlabs/fastroai/compare/v0.4.1...v0.5.0

---

## [0.4.1] - Jan 25, 2026

#### Fixed
- **FallbackModel Tracking** by [@igorbenav](https://github.com/igorbenav)
  - FastroAgent now correctly tracks the actual model that processed the request when using PydanticAI's `FallbackModel`
  - Model name is extracted from `ModelResponse.model_name` in the message history instead of non-existent `usage.model`
  - Fixes incorrect cost calculations when using model wrappers (e.g., DeepSeek primary with GPT-4o fallback)

- **No False Model Assumptions** by [@igorbenav](https://github.com/igorbenav)
  - When using escape hatch (`agent=`) without explicit `model=`, FastroAgent no longer assumes `gpt-4o`
  - If model can't be detected from response, returns `model=None` and `cost_microcents=0` with a warning
  - Tokens are still tracked accurately even when model is unknown

#### Changed
- `ChatResponse.model` is now `str | None` (was `str`) to represent unknown models
- `CostCalculator.calculate_cost()` accepts `model: str | None` and returns `0` for `None`
- Added logging when model detection fails and no fallback is configured

#### Documentation
- Updated RELEASE.md with 0.4.1 release notes

**Full Changelog**: https://github.com/benavlabs/fastroai/compare/v0.4.0...v0.4.1

---

## [0.4.0] - Dec 20, 2025

#### Added
- **Enhanced Cost Tracking** by [@igorbenav](https://github.com/igorbenav)
  - Cache token tracking (`cache_read_tokens`, `cache_write_tokens`) for accurate cost calculation with prompt caching
  - Audio token tracking for multimodal models
  - Request count tracking (`request_count`) for API call monitoring
  - Tool call count tracking (`tool_call_count`) for agentic behavior metrics
  - Provider-specific usage details (`usage_details` dict) for reasoning tokens, etc.

- **Accurate Prompt Caching Costs**
  - Cached tokens are now priced at 90% discount (Anthropic) automatically
  - `CostCalculator.calculate_cost()` accepts optional `cache_read_tokens`, `cache_write_tokens` parameters
  - Pricing overrides support cache token rates via `add_pricing_override(cache_read_per_mtok=..., cache_write_per_mtok=...)`
  - Fixes cost overreporting when prompt caching is enabled (~18% more accurate)

#### Changed
- All Pydantic schemas now use `Field(description=...)` for better API documentation
- `ChatResponse`, `StepUsage`, and `PipelineUsage` include new usage fields with sensible defaults
- Backward compatible - existing code continues to work without changes

#### Documentation
- Updated FastroAgent guide with new response fields
- Added Prompt Caching section to Cost Calculator guide
- Updated API reference with enhanced schemas

**Full Changelog**: https://github.com/benavlabs/fastroai/compare/v0.3.0...v0.4.0

---

## [0.3.0] - Dec 17, 2025

#### Added
- **LogfireTracer** by [@igorbenav](https://github.com/igorbenav)
  - Built-in tracer implementation for Pydantic's Logfire observability platform
  - Implements the `Tracer` protocol - drop-in replacement for `SimpleTracer`
  - Automatic span creation with `_tags=["fastroai"]` for easy filtering in Logfire dashboard
  - Metric logging via `logfire.info()` with trace correlation
  - Error logging with full exception info via `logfire.error()`
  - Clear `ImportError` when logfire package is not installed

- **Optional Dependency** by [@igorbenav](https://github.com/igorbenav)
  - Install with `pip install fastroai[logfire]` to enable Logfire support
  - Logfire remains optional - core functionality works without it

#### Documentation
- Added LogfireTracer to built-in tracers section in tracing guide
- Added LogfireTracer to API reference
- Updated README with Logfire installation instructions

#### What's Changed
- LogfireTracer implementation by [@igorbenav](https://github.com/igorbenav)

**Full Changelog**: https://github.com/benavlabs/fastroai/compare/v0.2.0...v0.3.0

---

## [0.2.0] - Dec 16, 2025

#### Added
- **FastroAgent** - PydanticAI wrapper with automatic cost calculation and tracing
- **Pipeline** - DAG-based workflow orchestration with automatic parallelization
- **@step decorator** - Concise function-based pipeline step definitions
- **@safe_tool** - Production-safe tool decorator with timeout and retry
- **CostCalculator** - Precise cost tracking using microcents (integer arithmetic)
- **Tracer Protocol** - Protocol-based tracing interface for observability integration
- **SimpleTracer** - Logging-based tracer for development
- **NoOpTracer** - No-op tracer for testing or disabled tracing

#### Documentation
- Complete documentation site with guides, API reference, and recipes
- MkDocs Material theme with dark/light mode support

**Full Changelog**: https://github.com/benavlabs/fastroai/compare/v0.1.0...v0.2.0

---

## [0.1.0] - Dec 15, 2025

Initial release.

- Core FastroAgent functionality
- Basic pipeline support
- Cost calculation primitives
