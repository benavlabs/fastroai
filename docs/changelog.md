# FastroAI Changelog

## Introduction

The Changelog documents all notable changes made to FastroAI. This includes new features, bug fixes, and improvements. It's organized by version and date, providing a clear history of the library's development.

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

## [0.2.0] - Dec 15, 2025

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

## [0.1.0] - Dec 10, 2025

Initial release.

- Core FastroAgent functionality
- Basic pipeline support
- Cost calculation primitives
