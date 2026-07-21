# FastroAI 0.6.1 Release Notes

FastroAI 0.6.1 raises the `genai-prices` floor to `>=0.0.71`, so `CostCalculator` prices the current
generation of models out of the box - notably DeepSeek's V4 family (`deepseek-v4-flash`,
`deepseek-v4-pro`), which the older pinned data did not know and therefore priced at zero. A cost of
zero for an unrecognized model is silent, so a stale price floor reads as "free" rather than "unknown"
- raising the floor closes that gap for every consumer on their next lock.

**Breaking Changes:** None. Dependency floor bump only; all APIs are unchanged.

---

# FastroAI 0.6.0 Release Notes

FastroAI 0.6.0 adds **dispatch hooks** to `FastroAgent` — async callbacks that fire before and after each `agent.run()` attempt inside the retry loop. These are the integration points downstream code needs to plug in circuit breakers, rate limiters, and kill switches without monkey-patching the dispatch path. A new `DispatchSkippedError` exception lets `on_before_dispatch` short-circuit a dispatch entirely (no retry, no after-hook), and an `ErrorCategory` enum is provided for callers that want a common vocabulary when classifying outcomes inside `on_after_dispatch`.

## Summary

**New:**
- **`FastroAgent(on_before_dispatch=..., on_after_dispatch=...)`.** Per-attempt async hooks wired into `StepContext._execute_with_config`. Fire on every retry, not just the final outcome. Use them for breakers, rate limiters, and any guard that needs to see each individual try.
- **`DispatchSkippedError`** — subclass of `FastroAIError`. Raise from `on_before_dispatch` to short-circuit. The retry loop propagates without retrying, and `on_after_dispatch` is **not** called. Use for breakers-open / kill-switches / "we already know this will fail."
- **`ErrorCategory`** StrEnum (`TRANSIENT`, `PERMANENT`, `RESOURCE_EXHAUSTION`, `UNKNOWN`) — provided for callers that want to categorize exceptions inside `on_after_dispatch`. The library doesn't auto-categorize; you map exceptions yourself.
- **`AgentConfig.timeout_name`** — optional human-readable label for the configured timeout, surfaced in span attributes and `TimeoutError` messages emitted by the retry loop. Observability tooling (e.g. Logfire) uses it to distinguish which configured timeout fired without parsing call-site context.

**Breaking Changes:** None. All additions are opt-in; existing code continues to work unchanged.

## Why This Exists

A 50-minute production hang on 2026-05-06 in a downstream worker exposed the gap between "retries are bounded by `max_retries`" and "downstream code can react to per-attempt outcomes." The 0.5.0 timeout fix bounded *individual* calls, but circuit-breaker integrations needed something else: visibility into each attempt as it happens, plus a way to fail fast when a breaker has already tripped. Patching that into application code on top of `agent.run()` worked for one consumer but didn't compose — every caller would have wrapped the same retry loop differently. So the hooks live in the library, fire per attempt, and have well-defined semantics around skipping (`DispatchSkippedError`) and pre-flight rejection (`CostBudgetExceededError`).

## How It Works

Hooks fire inside `StepContext._execute_with_config`, the retry loop that wraps `agent.run()`. Per attempt:

1. `on_before_dispatch()` if set. Any exception (including `DispatchSkippedError`) propagates immediately — no retry, no after-hook.
2. `agent.run(...)` (with timeout if configured).
3. `on_after_dispatch(exc_or_none)` if set. After-hook errors propagate without retry.
4. On agent success: return. On agent failure: continue retry loop unless the failure was `DispatchSkippedError`.

`CostBudgetExceededError` is a pre-flight rejection — it raises in `ctx.run()` before the dispatch loop is entered, so neither hook fires for it.

## Example: Circuit Breaker Integration

```python
from fastroai import FastroAgent, DispatchSkippedError, ErrorCategory

class BreakerOpenError(DispatchSkippedError):
    pass

async def before():
    if breaker.is_open():
        raise BreakerOpenError("downstream breaker open")

async def after(exc: Exception | None):
    if exc is None:
        breaker.record_success()
        return
    category = categorize(exc)  # your app's classifier, returning ErrorCategory
    if category == ErrorCategory.TRANSIENT:
        breaker.record_failure()
    # PERMANENT failures don't count toward the breaker

agent = FastroAgent(
    model="openai:gpt-4o",
    timeout=30,
    timeout_name="summarize_step.dispatch",
    on_before_dispatch=before,
    on_after_dispatch=after,
)
```

When the breaker is open, `before()` raises `BreakerOpenError` (a `DispatchSkippedError` subclass) — the agent never dispatches, no retry happens, and the application sees the breaker-specific exception type bubble up. When the breaker is closed but a request fails, `after()` decides whether to count the failure based on the application's categorization. The `timeout_name` shows up in span attributes so you can tell which configured timeout fired without scanning call sites.

## Upgrade Guide

No code changes required. `FastroAgent(...)` calls without the new params work exactly as before. To opt in:

```python
# Before (still works):
agent = FastroAgent(model="openai:gpt-4o", timeout=30)

# After (with hooks):
agent = FastroAgent(
    model="openai:gpt-4o",
    timeout=30,
    timeout_name="my_step.dispatch",  # optional, observability only
    on_before_dispatch=my_before_hook,
    on_after_dispatch=my_after_hook,
)
```

Hooks live on the `FastroAgent` instance, not in `AgentConfig` — they're `Callable`s and don't serialize. `timeout_name` is a string so it lives on `AgentConfig` like the rest of the configuration.

---

**Full Changelog**: https://github.com/benavlabs/fastroai/compare/v0.5.0...v0.6.0
