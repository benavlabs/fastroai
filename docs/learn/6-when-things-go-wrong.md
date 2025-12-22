# When Things Go Wrong

**AI applications fail in ways traditional software doesn't.**

A database query either returns rows or throws an exception. You can test every code path. But an LLM might work perfectly 99 times and fail on the 100th because the input triggered some edge case in its training data. The response might be valid JSON but contain hallucinated data. The API might rate-limit you mid-conversation.

Traditional error handling assumes deterministic failure modes. AI applications need a different approach.

## The Failure Modes

Some failures throw exceptions - network timeouts, rate limits, API outages. These are the infrastructure failures you're used to. Traditional error handling catches them.

Model failures are different. The model generates invalid JSON when you asked for structured output. It hallucinates a function that doesn't exist. It ignores your constraints. You get a 200 response with valid-looking content that's wrong. No exception to catch.

Then there are application failures - your code can't handle what came back. The model returned a category you didn't expect, a date in the wrong format, a confidence score outside your valid range. These are mismatches between what you assumed and what happened.

## Failing at the Right Level

When a tool makes an HTTP request that times out, where should that failure surface?

```python
async def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.stocks.com/{symbol}")
        return response.json()["price"]
```

If this throws, the exception bubbles up through the agent, terminating the entire request. The user sees an error page. You've paid for the tokens up to that point. The conversation is dead.

But the agent could have recovered. Given a chance, it might say "I couldn't check the stock price right now - would you like me to try something else?" That's a better user experience, and it doesn't waste the context you've built.

Section 3 introduced `@safe_tool` for exactly this:

```python
@safe_tool(timeout=10, max_retries=2)
async def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    ...
```

The tool catches its own failures and returns an error message instead of throwing. The agent sees the message and can respond appropriately. The failure is contained at the tool level, where recovery is possible.

A tool failure shouldn't necessarily crash the agent. An agent failure shouldn't crash the pipeline. Let failures surface where they can be handled.

## Model Failures

Infrastructure failures throw exceptions. Model failures don't - they look like success.

You asked for a JSON response:

```python
class StockAnalysis(BaseModel):
    symbol: str
    recommendation: Literal["buy", "hold", "sell"]
    confidence: float
```

The model returns:

```json
{"symbol": "AAPL", "recommendation": "strong buy", "confidence": 0.95}
```

Pydantic validation fails because "strong buy" isn't in your Literal. That's actually the good case - at least you caught it. The worse case is when the response is valid but wrong. The model confidently returns `"confidence": 0.99` for a recommendation it made up.

Structured output (Section 4) constrains format, but models can still hallucinate within those constraints. You need defenses:

| Problem | Example | Defense |
|---------|---------|---------|
| Invalid format | `"strong buy"` instead of `"buy"` | `Literal` types, Pydantic validation |
| Hallucinated data | Stock symbol that doesn't exist | Validate against external source |
| Inconsistent reasoning | High confidence for weak argument | Check recommendation matches reasoning |
| Repetition loops | Model repeats same phrase | Detect repeated content, abort |
| Excessive hedging | "It might possibly be..." | Check for hedge phrases, re-prompt |
| Contradictions | "Buy" after explaining why to sell | Compare conclusion to supporting text |

For critical outputs, temperature 0 gives more consistent results - not guaranteed correct, but less random.

How do you know your defenses work? You can't unit test an LLM the way you test deterministic code. Section 9 covers evals - systematic ways to test model behavior across many inputs.

## FastroAI Errors

FastroAI uses a structured exception hierarchy:

```python
FastroAIError                    # Base for all FastroAI errors
├── PipelineValidationError      # Invalid pipeline config
├── StepExecutionError           # Step failed during execution
└── CostBudgetExceededError      # Budget limit hit
```

Catch `FastroAIError` to handle all library-specific errors:

```python
from fastroai import FastroAIError

try:
    result = await pipeline.execute(inputs, deps)
except FastroAIError as e:
    logger.error(f"Pipeline error: {e}")
```

Or catch specific exceptions when you need different handling:

```python
from fastroai import CostBudgetExceededError, StepExecutionError

try:
    result = await pipeline.execute(inputs, deps)
except CostBudgetExceededError as e:
    # Graceful degradation - return partial results
    return partial_result
except StepExecutionError as e:
    # Log and retry the specific step
    logger.error(f"Step {e.step_id} failed: {e.original_error}")
```

`CostBudgetExceededError` includes details about what happened:

```python
except CostBudgetExceededError as e:
    print(f"Budget: {e.budget_microcents}")
    print(f"Actual: {e.actual_microcents}")
    print(f"Step: {e.step_id}")
```

## Designing for Failure

The goal isn't preventing all failures - it's ensuring failures don't cascade.

Configuration errors should fail fast. `PipelineValidationError` raises at construction time, not runtime - invalid dependencies, missing steps, circular references are caught before you waste tokens.

Tool failures can be contained or propagated depending on what you need. `@safe_tool` lets the agent recover gracefully - useful when the tool is optional or the agent can try alternatives. Let exceptions propagate when you need to handle them at a higher level, retry the whole operation, or fail fast.

Cost budgets prevent runaway loops. Pipelines accept a `cost_budget` that raises `CostBudgetExceededError` when exceeded - better to fail explicitly than discover a $500 bill.

And when something does fail, you need to know what led to it. The prompt, the model's reasoning, the tool calls, the responses. Section 8 covers tracing, but the point here is that debugging AI failures without context is guesswork.

## What's Next

Single-agent applications can only go so far. Real tasks often require multiple steps: classify the input, research the topic, draft a response, verify the facts, format the output. Each step might use a different model, different tools, different prompts.

Section 7 introduces pipelines - multi-step workflows that handle dependencies, run independent steps in parallel, and aggregate costs across the whole process.

[Multi-Step Workflows →](7-multi-step-workflows.md){ .md-button .md-button--primary }
