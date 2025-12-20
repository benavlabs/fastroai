# Cost Calculator

LLM APIs charge per token. When you're processing thousands of requests, small precision errors in cost tracking add up. Floating-point math is the classic culprit:

```python
>>> 0.1 + 0.2
0.30000000000000004
```

Your billing system shows $0.30000000000000004 and someone opens a support ticket. Or worse, you round incorrectly and underbill by a few cents per request, which becomes real money at scale.

FastroAI tracks costs in microcents - integer math that doesn't drift. One microcent equals 1/1,000,000 of a dollar, so $0.01 is 10,000 microcents. All calculations use integers. When you display the cost to users, convert to dollars.

## How It Works

When you use `FastroAgent`, cost calculation happens automatically:

```python
from fastroai import FastroAgent

agent = FastroAgent(model="openai:gpt-4o")
response = await agent.run("Hello!")

print(response.cost_microcents)  # 2500 (exact integer)
print(response.cost_dollars)     # 0.0025 (for display)
```

The calculator looks up the model's per-token pricing, multiplies by your token counts, and returns an integer. No floating-point operations in the calculation path.

The price data comes from [genai-prices](https://github.com/pydantic/genai-prices), which tracks pricing for OpenAI, Anthropic, Google, Groq, and other providers. The package is updated regularly as providers change their prices.

## Aggregating Costs

For a single request, precision doesn't matter much. But when you're aggregating across a conversation, a user's session, or your entire platform:

```python
# Track a conversation
total_cost = 0

for message in user_messages:
    response = await agent.run(message)
    total_cost += response.cost_microcents  # Integer addition, no drift

# Convert for display at the end
print(f"Session cost: ${calc.microcents_to_dollars(total_cost):.4f}")
```

After 10,000 additions, you have an exact count. No cumulative rounding errors.

## Prompt Caching

Anthropic and OpenAI support prompt caching, where repeated system prompts are stored and reused at a significant discount - typically 90% cheaper for cached tokens. FastroAI automatically tracks cache tokens and factors them into cost calculations.

When you use FastroAgent, cache tokens are tracked automatically:

```python
response = await agent.run("What's the weather?")

print(f"Input tokens: {response.input_tokens}")
print(f"Cached tokens: {response.cache_read_tokens}")  # Tokens from cache (90% discount)
print(f"Cost: ${response.cost_dollars:.6f}")  # Accurate cost with cache discount
```

If you're using a long system prompt that gets cached, subsequent requests to the same agent will show cache hits. The cost calculation automatically applies the discounted rate for cached tokens.

**Requirements for prompt caching:**

- **Anthropic:** Prompts >= 1024 tokens with cache_control markers
- **OpenAI:** Prompts >= 1024 tokens on gpt-4o models

## Direct Calculator Usage

Sometimes you need cost calculation without running an agent - estimating costs upfront, building dashboards, or custom tracking:

```python
from fastroai import CostCalculator

calc = CostCalculator()

cost = calc.calculate_cost(
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500,
)

print(f"Cost: {cost} microcents")  # 7500 microcents
print(f"Cost: ${calc.microcents_to_dollars(cost):.6f}")  # $0.007500
```

For accurate costs with prompt caching, pass cache tokens:

```python
cost = calc.calculate_cost(
    model="claude-3-5-sonnet",
    input_tokens=1000,
    output_tokens=500,
    cache_read_tokens=800,  # 800 of 1000 input tokens were cached
)
# Cost is lower because 800 tokens are priced at 90% discount
```

Model names get normalized automatically. Both `"gpt-4o"` and `"openai:gpt-4o"` work.

### Conversion Methods

```python
calc = CostCalculator()

# Microcents to dollars (for display)
dollars = calc.microcents_to_dollars(7500)  # 0.0075

# Dollars to microcents (for storage/budgets)
microcents = calc.dollars_to_microcents(0.10)  # 100000

# Formatted output for debugging
formatted = calc.format_cost(7500)
# {"microcents": 7500, "cents": 0, "dollars": 0.0075}
```

## Custom Pricing

genai-prices covers most models, but you might need custom pricing. Maybe you've negotiated volume discounts with your provider. Or you're running self-hosted or fine-tuned models that aren't in any public pricing list. Or a new model came out and genai-prices hasn't updated yet.

Override pricing at initialization:

```python
from fastroai import CostCalculator

calc = CostCalculator(pricing_overrides={
    "gpt-4o": {
        "input_per_mtok": 2.00,   # $2.00 per million input tokens
        "output_per_mtok": 8.00,  # $8.00 per million output tokens
    },
})
```

Or add overrides later:

```python
calc.add_pricing_override(
    model="my-local-model",
    input_per_mtok=0.10,
    output_per_mtok=0.20,
)
```

You can also specify cache token rates for models that support prompt caching:

```python
calc.add_pricing_override(
    model="my-cached-model",
    input_per_mtok=3.00,
    output_per_mtok=15.00,
    cache_read_per_mtok=0.30,   # 90% discount for cached tokens
    cache_write_per_mtok=3.75,  # 25% premium for cache writes
)
```

If you don't specify cache rates, default discounts apply (90% for reads, 25% premium for writes) when cache tokens are passed to `calculate_cost()`.

Overrides take precedence over genai-prices. Prices are in dollars per million tokens (the standard unit providers use).

### Using Custom Pricing with Agents

Pass your configured calculator to the agent:

```python
from fastroai import FastroAgent, CostCalculator

calc = CostCalculator()
calc.add_pricing_override("gpt-4o", input_per_mtok=2.00, output_per_mtok=8.00)

agent = FastroAgent(
    model="openai:gpt-4o",
    cost_calculator=calc,
)

# Now response.cost_microcents uses your pricing
response = await agent.run("Hello!")
```

## Unknown Models

If you use a model that isn't in genai-prices and doesn't have a custom override, the calculator returns 0 cost and logs a debug message. This way your code doesn't crash - you just get missing cost data.

Check your logs for "Unknown model" warnings. Either add a pricing override or open a PR on genai-prices.

## Cost Budgets

Track cumulative costs and stop when you hit a limit:

```python
from fastroai import CostCalculator

calc = CostCalculator()
total_cost = 0
budget = calc.dollars_to_microcents(1.00)  # $1.00 budget

for query in queries:
    response = await agent.run(query)
    total_cost += response.cost_microcents

    if total_cost >= budget:
        print("Budget exhausted")
        break

print(f"Total spent: ${calc.microcents_to_dollars(total_cost):.4f}")
```

This works fine for simple cases. For multi-step workflows where you want automatic budget enforcement, pipelines have built-in cost budgets that raise `CostBudgetExceededError` when exceeded. See the [Pipelines](pipelines.md) guide.

## Pricing Reference

Common model pricing as of January 2025:

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| claude-3-haiku | $0.25 | $1.25 |
| gemini-1.5-pro | $1.25 | $5.00 |
| gemini-1.5-flash | $0.075 | $0.30 |

Prices change. Check your provider's current pricing, and consider using pricing overrides if you're on a negotiated rate.

## Key Files

| Component | Location |
|-----------|----------|
| CostCalculator | `fastroai/usage/calculator.py` |

---

[← FastroAgent](fastro-agent.md){ .md-button } [Pipelines →](pipelines.md){ .md-button .md-button--primary }
