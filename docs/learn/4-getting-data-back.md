# Getting Data Back, Not Just Text

**From parsing strings to working with real data.**

Section 3 gave your agent capabilities: tools that calculate, fetch data, and interact with external systems. But when you ask "Is my $800 food spending reasonable on $5000/month?", the agent returns a string like "Your spending of 16% on food is a bit high but manageable."

Your code now has to figure out what to do with that. The answer is buried in natural language (is it reasonable? What was the percentage?).

## The Problem With Parsing Text

Let's say you're building an invoice processor. The agent analyzes invoices and you need to route them based on amount and category:

```python
response = await agent.run("Analyze this invoice: $2,340.50 for office supplies")
```

The agent might return:

> "This invoice is for $2,340.50 in the office supplies category. The amount exceeds the typical threshold for automatic approval."

Now what? You could regex out the dollar amount, look for keywords like "office supplies", check for "exceeds threshold". But then the agent changes its phrasing - "totaling $2,340.50" instead of "for $2,340.50", or "classified under office equipment" instead of "office supplies category." Your parsing breaks.

You're fighting the model. LLMs are optimized to generate natural, varied language - not to maintain consistent output formats. Every regex you write is betting against that.

## Structured Output

PydanticAI (which FastroAI builds on) solves this by constraining the model to return valid instances of Pydantic models. Instead of free-form text, you get typed data:

```python
from pydantic import BaseModel

class InvoiceAnalysis(BaseModel):
    amount: float
    category: str
    requires_approval: bool
    reason: str
```

The model's output must conform to this schema. Not "try to match it" - the output is validated and typed.

```python
from fastroai import FastroAgent

agent = FastroAgent(
    model="openai:gpt-4o",
    system_prompt="You analyze invoices and determine if they need approval.",
    output_type=InvoiceAnalysis,
)

response = await agent.run("Analyze: $2,340.50 for office supplies")
```

Now `response.output` is an `InvoiceAnalysis` instance:

```python
print(response.output.amount)           # 2340.5
print(response.output.category)         # "office supplies"
print(response.output.requires_approval) # True
print(response.output.reason)           # "Amount exceeds $1000 threshold"
```

No parsing, regex or hoping the model used the right words. The model's natural language ability is still there - it figured out the category, determined whether approval is needed, and explained why. But the output is structured data your code can work with.

## How It Works

When you specify `output_type`, PydanticAI sends the model a JSON schema derived from your Pydantic model. The model generates JSON that conforms to that schema, and PydanticAI validates and instantiates it.

This connects back to Section 0: the model is still predicting tokens, but now it's predicting tokens that form valid JSON matching your schema. Models are trained on vast amounts of structured data - JSON, code, formatted documents - so generating structured output is something they're already good at.

The `ChatResponse` gives you both forms:

```python
response.output   # InvoiceAnalysis instance - use this
response.content  # JSON string representation - rarely needed
```

For text agents (no `output_type`), `.output` and `.content` are both the string response. With structured output, `.output` is your typed model instance.

## Defining Good Schemas

Your schema teaches the model what you want. Field names and types matter:

```python
class SpendingAnalysis(BaseModel):
    percentage_of_income: float
    category: str
    assessment: str  # What does this mean? Vague.
```

The model has to guess what "assessment" should contain. Is it a grade? A description? A recommendation?

Better:

```python
from typing import Literal

class SpendingAnalysis(BaseModel):
    percentage_of_income: float
    category: str
    rating: Literal["reasonable", "high", "excessive"]
    recommendation: str
```

Now the model knows exactly what `rating` can be. It can't return "pretty good" or "could be better" - it must pick from the specified values. Your code can switch on `rating` without parsing.

Field descriptions and examples make it even clearer:

```python
from pydantic import Field

class SpendingAnalysis(BaseModel):
    """Analysis of spending in a budget category."""

    percentage_of_income: float = Field(
        description="Spending as percentage of monthly income (0-100)",
        examples=[16.0, 32.5],
    )
    category: str = Field(
        description="Budget category",
        examples=["food", "housing", "transportation"],
    )
    rating: Literal["reasonable", "high", "excessive"] = Field(
        description="Assessment based on standard financial guidelines",
    )
    recommendation: str = Field(
        description="One actionable suggestion for this category",
        examples=["Consider meal planning to reduce costs"],
    )
```

PydanticAI includes these in the JSON schema sent to the model. Clearer descriptions and concrete examples lead to better output.

## A Complete Example

Let's combine structured output with the spending analysis from Section 3. The tool does the calculation, and structured output ensures we get data we can use:

```python
from pydantic import BaseModel
from typing import Literal

class SpendingReport(BaseModel):
    """Report on spending analysis."""

    category: str
    amount: float
    percentage: float
    rating: Literal["reasonable", "high", "excessive"]
    suggestion: str
```

The agent and tool work together:

```python
from pydantic_ai.toolsets import FunctionToolset
from fastroai import FastroAgent

async def analyze_spending(
    monthly_income: float,
    amount: float,
    category: str
) -> str:
    """Analyze spending in a category relative to income."""
    percentage = (amount / monthly_income) * 100

    guidelines = {
        "food": {"reasonable": 15, "high": 20},
        "housing": {"reasonable": 30, "high": 35},
        "transportation": {"reasonable": 15, "high": 20},
    }

    limits = guidelines.get(category.lower(), {"reasonable": 10, "high": 15})

    if percentage <= limits["reasonable"]:
        assessment = "reasonable"
    elif percentage <= limits["high"]:
        assessment = "high but manageable"
    else:
        assessment = "above recommended guidelines"

    return f"{percentage:.1f}% of income on {category} - {assessment}"

toolset = FunctionToolset(tools=[analyze_spending])

agent = FastroAgent(
    model="openai:gpt-4o",
    system_prompt="You analyze personal finances and provide actionable advice.",
    output_type=SpendingReport,
    toolsets=[toolset],
)
```

When we run the agent:

```python
response = await agent.run(
    "I make $5000/month and spent $800 on food. How am I doing?"
)
```

The agent calls the tool, gets the calculation result, and returns a structured report:

```python
print(response.output.category)     # "food"
print(response.output.percentage)   # 16.0
print(response.output.rating)       # "high"
print(response.output.suggestion)   # "Consider meal planning to reduce food costs"
```

The tool did the math reliably (Section 1: use code for computation). The model interpreted the result and generated advice (Section 1: use LLMs for language). Structured output made the response programmatically useful.

## When to Use Structured Output

Use it when your code needs to act on the response:

- Routing decisions (send this to approval, file it here)
- Data extraction (pull fields from documents)
- Multi-step workflows (pass data to the next step)
- API responses (return JSON to clients)
- Validation (check that required fields exist)

Skip it when you just need text:

- Conversational responses to users
- Creative writing
- Explanations meant for humans

The overhead is minimal, but if you're just showing the response to a user, plain text is simpler.

## What's Next

You now have agents that can do things (tools) and return structured data (output types). But every token costs money - Section 0 explained why. When you're running thousands of requests, those costs add up fast.

Section 5 covers cost tracking: measuring what you spend, understanding where the money goes, and setting budgets before you get surprised.

[Tracking What You Spend →](5-tracking-what-you-spend.md){ .md-button .md-button--primary }
