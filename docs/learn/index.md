# Learn FastroAI

**Build AI applications step by step.**

This isn't a reference manual. Each section builds on the previous, and by the end you'll know how to use FastroAI's cost tracking, pipelines, and tools in production.

!!! tip "Choose Your Starting Point"
    **New to AI development?** Start from the beginning - we cover everything from basic LLM concepts.

    **Already know PydanticAI?** Jump to Cost Tracking (section 3) to see what FastroAI adds.

    **Just want code?** Check the [Quick Start](../index.md#quick-start) for the 2-minute setup.

## What You'll Build

Instead of toy examples, you'll work through progressively complex scenarios that mirror real production applications:

=== "Fundamentals"

    **Understand the building blocks**

    - What LLMs are and how to call them
    - Creating agents with system prompts
    - Understanding tokens and why costs matter
    - Structured output with Pydantic models

=== "Production Features"

    **Add real-world capabilities**

    - Tools that call external services safely
    - Conversation history and stateless patterns
    - Streaming responses for better UX
    - Tracing for debugging and observability

=== "Orchestration"

    **Build complex workflows**

    - Multi-step pipelines with dependencies
    - Parallel execution for performance
    - Cost budgets and early termination
    - Error handling that doesn't crash requests

## The Learning Path

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **1. LLM Basics**

    *"What are these things anyway?"*

    ---

    What language models are, how to call them, understanding tokens and why costs matter for production applications.

    *(Coming soon)*

-   :material-robot:{ .lg .middle } **2. Your First Agent**

    *"Hello, AI world"*

    ---

    Creating a FastroAgent, crafting system prompts, running queries, and understanding the response format.

    *(Coming soon)*

-   :material-cash:{ .lg .middle } **3. Cost Tracking**

    *"How much did that cost?"*

    ---

    Why microcents matter for billing, tracking usage across calls, and setting cost budgets.

    *(Coming soon)*

-   :material-code-json:{ .lg .middle } **4. Structured Output**

    *"Give me data, not strings"*

    ---

    Getting Pydantic models back instead of raw text. Type-safe responses for real applications.

    *(Coming soon)*

-   :material-tools:{ .lg .middle } **5. Tools**

    *"Let the agent do things"*

    ---

    Giving agents capabilities with `@safe_tool`. Timeout, retry, and graceful error handling.

    *(Coming soon)*

-   :material-message-text:{ .lg .middle } **6. Conversations**

    *"Remember what we talked about"*

    ---

    Message history patterns, stateless design, and when to use each approach.

    *(Coming soon)*

-   :material-play-speed:{ .lg .middle } **7. Streaming**

    *"Show me as you think"*

    ---

    Real-time responses, handling chunks, and cost tracking with streaming.

    *(Coming soon)*

-   :material-pipe:{ .lg .middle } **8. Pipelines**

    *"Chain steps together"*

    ---

    Multi-step workflows with automatic dependency resolution and cost aggregation.

    *(Coming soon)*

-   :material-run-fast:{ .lg .middle } **9. Parallel Execution**

    *"Run independent steps concurrently"*

    ---

    DAG-based execution, automatic parallelization, and performance optimization.

    *(Coming soon)*

-   :material-shield-check:{ .lg .middle } **10. Production**

    *"Ship it with confidence"*

    ---

    Tracing integration, error handling patterns, and observability for production systems.

    *(Coming soon)*

</div>

---

## Alternative Learning Paths

=== "By Time Available"

    - **30 minutes**: Sections 1-2 → Understand agents and basic usage
    - **2 hours**: Sections 1-5 → Build a functional AI application
    - **Half day**: Sections 1-8 → Production-ready with pipelines
    - **Full day**: All sections → Complete mastery

=== "By Experience Level"

    - **New to AI**: Start from section 1, don't skip fundamentals
    - **Know LLMs, new to PydanticAI**: Start at section 2
    - **Know PydanticAI**: Jump to section 3 (Cost Tracking) for FastroAI specifics
    - **Production developer**: Focus on sections 5, 8-10

=== "By Goal"

    - **"I want to understand AI agents"** → Sections 1-4
    - **"I want to build a chatbot"** → Sections 1-6
    - **"I want to build a data pipeline"** → Sections 1-4, then 8-9
    - **"I want to go to production"** → All sections, focus on 5 and 10

## Prerequisites

You should be comfortable with:

- Python async/await syntax
- Basic Pydantic models
- Environment variables and API keys

You don't need prior experience with:

- PydanticAI (we'll cover what you need)
- AI/LLM concepts (we start from basics)
- Production infrastructure (we'll build up to it)

---

**Ready to learn?**

The learning path is coming soon. In the meantime, check the [Quick Start](../index.md#quick-start) or explore the [Guides](../guides/index.md) for deep dives into specific features.

[Quick Start →](../index.md#quick-start){ .md-button .md-button--primary } [Browse Guides →](../guides/index.md){ .md-button }
