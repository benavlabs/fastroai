# Learn FastroAI

Build AI applications by understanding how they actually work.

This isn't a feature tour. We start from how language models work, build up to agents and tools, and end with production patterns. Each section builds on the previous, and by the end you'll understand not just how to use FastroAI, but why it's designed the way it is.

!!! tip "Choose Your Starting Point"
    **New to LLMs?** Start from the beginning. We cover tokenization, transformers, and what's actually happening when you call an API.

    **Already understand LLMs?** Jump to Section 2 (Your First Agent) to start building.

    **Know PydanticAI?** Skip to Section 3 (Letting Agents Do Things) for tools and what FastroAI adds.

    **Just want code?** Check the [Quick Start](../index.md#quick-start) for the 2-minute setup.

## What You'll Learn

=== "Foundations"

    **Understand what you're building on**

    - How language models actually work (not just API calls)
    - Tokenization and why it matters for costs
    - What LLMs do well and where they fail
    - The mechanics behind "AI agents"

=== "Building Applications"

    **Create real AI features**

    - Agents with system prompts
    - Tools that let agents interact with the world
    - Structured output for type-safe responses
    - Cost tracking and error handling

=== "Production Patterns"

    **Ship with confidence**

    - Multi-step workflows with pipelines
    - Tracing and observability
    - RAG for when the model doesn't know enough

## The Learning Path

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **[0. How Language Models Work](0-how-language-models-work.md)**

    *Understanding what you're actually calling*

    ---

    From early NLP to modern transformers. Tokenization, attention, and why "predict the next token" leads to surprisingly capable systems. You'll understand what happens between your API call and the response.

    [:octicons-arrow-right-24: Start here](0-how-language-models-work.md)

-   :material-check-circle-outline:{ .lg .middle } **[1. What LLMs Can and Can't Do](1-what-llms-can-and-cant-do.md)**

    *Capabilities and limitations*

    ---

    LLMs are good at specific things and terrible at others. Understanding this gap is what separates working applications from impressive demos that break in production.

    [:octicons-arrow-right-24: Continue](1-what-llms-can-and-cant-do.md)

-   :material-robot:{ .lg .middle } **[2. Your First Agent](2-your-first-agent.md)**

    *From API calls to agents*

    ---

    Creating a FastroAgent, writing system prompts that work, running queries, and understanding what comes back. The foundation for everything else.

    [:octicons-arrow-right-24: Continue](2-your-first-agent.md)

-   :material-tools:{ .lg .middle } **[3. Letting Agents Do Things](3-letting-agents-do-things.md)**

    *Tools and function calling*

    ---

    An agent that can only talk isn't very useful. Tools let agents call APIs, query databases, and interact with the world. `@safe_tool` makes them production-safe.

    [:octicons-arrow-right-24: Continue](3-letting-agents-do-things.md)

-   :material-code-braces:{ .lg .middle } **[4. Getting Data Back, Not Just Text](4-getting-data-back.md)**

    *Structured output*

    ---

    Your agent returns text, but your application needs data. Using Pydantic models to get type-safe responses you can actually work with.

    [:octicons-arrow-right-24: Continue](4-getting-data-back.md)

-   :material-cash:{ .lg .middle } **5. Tracking What You Spend**

    *Tokens cost money*

    ---

    Remember tokens from Section 0? Each one costs money. Why microcents matter for billing, how to track usage across calls, and setting budgets before you get surprised.

    *(Coming soon)*

-   :material-alert-circle:{ .lg .middle } **6. When Things Go Wrong**

    *Error handling*

    ---

    APIs time out. External services fail. The model returns something unexpected. Building AI applications that handle failures gracefully instead of crashing.

    *(Coming soon)*

-   :material-pipe:{ .lg .middle } **7. Multi-Step Workflows**

    *Pipelines*

    ---

    Real tasks need multiple steps: classify, then research, then write. Pipelines handle dependencies, run independent steps in parallel, and aggregate costs across the workflow.

    *(Coming soon)*

-   :material-chart-line:{ .lg .middle } **8. Finding Problems in Production**

    *Tracing and observability*

    ---

    Something's slow. Something's expensive. But what? Tracing lets you see inside your AI calls and correlate them with the rest of your application.

    *(Coming soon)*

-   :material-database-search:{ .lg .middle } **9. Retrieval Augmented Generation**

    *When the model doesn't know enough*

    ---

    LLMs have knowledge cutoffs and don't know your data. RAG combines retrieval (finding relevant documents) with generation (answering based on them). Embeddings, vector search, and grounding responses in real data.

    *(Coming soon)*

</div>

---

## Alternative Learning Paths

=== "By Time"

    - **1 hour**: Sections 0-2 → Understand LLMs and build your first agent
    - **Half day**: Sections 0-6 → Build a complete AI feature with tools
    - **Full day**: All sections → Production-ready with pipelines, tracing, and RAG

=== "By Background"

    - **New to AI**: Start from Section 0, don't skip the foundations
    - **Know ML, new to LLMs**: Skim Section 0, focus on 1-2
    - **Know LLMs, new to PydanticAI**: Start at Section 2
    - **Know PydanticAI**: Jump to Section 3 for tools and `@safe_tool`

=== "By Goal"

    - **"I want to understand how LLMs work"** → Sections 0-1
    - **"I want to build an AI feature"** → Sections 2-6
    - **"I want to build complex workflows"** → Sections 2-5, then 7
    - **"I want to add RAG to my app"** → Sections 0-2, then 9

## Prerequisites

You should be comfortable with:

- Python async/await syntax
- Basic Pydantic models
- Environment variables and API keys

You don't need prior experience with:

- Machine learning or NLP (we start from the beginning)
- PydanticAI (we cover what you need)
- Production infrastructure (we build up to it)

---

[Start Learning →](0-how-language-models-work.md){ .md-button .md-button--primary } [Browse Guides →](../guides/index.md){ .md-button }
