# Tracking What You Spend

**Understanding AI costs before they surprise you.**

Section 0 explained that LLMs work by predicting tokens. Every token flows through billions of parameters, running matrix multiplications on expensive GPUs. Providers charge per token because that's what actually costs them money.

## How Pricing Works

Providers charge per million tokens, separately for input and output. Output tokens cost more - the model has to generate them one at a time, while input tokens can be processed in parallel.

| Model | Input ($/1M) | Output ($/1M) |
|-------|--------------|---------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| claude-3-haiku | $0.25 | $1.25 |

A typical request might use 500 input tokens and 200 output tokens. On GPT-4o that's:

- Input: 500 × ($2.50 / 1,000,000) = $0.00125
- Output: 200 × ($10.00 / 1,000,000) = $0.002
- Total: $0.00325

Fractions of a cent. But if your application handles 100,000 requests per day, that's $325/day or roughly $10,000/month. And that's a modest workload.

The model you choose matters enormously. The same workload on claude-3-haiku instead of claude-3-5-sonnet costs 12x less. For many tasks - classification, extraction, simple Q&A - the cheaper model works fine.

## Prompt Caching

Remember from Section 0 that your prompt gets tokenized and embedded before the model processes it. If you're using the same long system prompt across many requests, that's redundant work.

Prompt caching stores the processed form of your prompt. When the same tokens appear again, the provider skips the processing and charges you less - typically 90% less for cached tokens.

This is more important when you have substantial system prompts. A 2,000-token system prompt processed 10,000 times costs $50 on Claude. With caching, subsequent requests cost $5 for that portion.

FastroAI tracks cache hits automatically:

```python
response = await agent.run("What's the weather?")
print(response.cache_read_tokens)  # Tokens served from cache
```

You don't need to do anything special - if the provider supports caching and your prompts qualify, FastroAI reports what happened.

## The Precision Problem

When you're aggregating costs across thousands of requests, floating-point math becomes a problem:

```python
>>> 0.1 + 0.2
0.30000000000000004
```

This isn't a Python bug - it's how IEEE 754 floating-point works. Most decimal fractions can't be represented exactly in binary. Each operation introduces tiny errors, and those errors accumulate.

After 10,000 additions, your billing shows $127.30000000000004. After a million, the drift is worse. You're either overcharging customers or losing money, and you can't tell which.

Financial systems solve this with integer arithmetic. Instead of tracking $0.0025, track 2,500 microcents (millionths of a dollar). Addition is exact:

```python
>>> 100 + 200
300
```

FastroAI uses microcents throughout:

```python
response = await agent.run("Hello!")
print(response.cost_microcents)  # 2500 (exact)
print(response.cost_dollars)     # 0.0025 (for display)
```

Use `cost_microcents` for everything except showing users a dollar amount.

## Estimating Before You Run

Before committing to an architecture, estimate what it'll cost. The `CostCalculator` lets you do the math without making API calls:

```python
from fastroai import CostCalculator

calc = CostCalculator()

# What would 1000 requests cost?
per_request = calc.calculate_cost("gpt-4o", input_tokens=500, output_tokens=200)
monthly = per_request * 1000 * 30

print(f"Monthly estimate: ${calc.microcents_to_dollars(monthly):.2f}")
```

This uses [genai-prices](https://github.com/pydantic/genai-prices) for current pricing. If you've negotiated volume discounts or you're using a model not in the database, override the pricing:

```python
calc = CostCalculator(pricing_overrides={
    "gpt-4o": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
})
```

## What's Next

Your agents can do things, return structured data, and you know what they cost. But that weather API from Section 3 will eventually time out. The database will be unreachable. The model will return something your code doesn't expect.

Section 6 is about what happens when things break - and how to keep your application running anyway.

[When Things Go Wrong →](6-when-things-go-wrong.md){ .md-button .md-button--primary }
