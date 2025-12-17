# Guides

**Deep dives into FastroAI's core components.**

Each guide explains what a component does, when you'd use it, and how to get it working. Start with whatever problem you're trying to solve.

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **FastroAgent**

    ---

    Wrap PydanticAI agents with automatic cost tracking. Stateless, production-ready, with consistent response formats.

    [FastroAgent →](fastro-agent.md)

-   :material-cash:{ .lg .middle } **Cost Calculator**

    ---

    Track token costs in microcents for exact billing. Override pricing for volume discounts or custom models.

    [Cost Calculator →](cost-calculator.md)

-   :material-pipe:{ .lg .middle } **Pipelines**

    ---

    Chain multiple AI steps with automatic parallelization. Track costs across entire workflows.

    [Pipelines →](pipelines.md)

-   :material-tools:{ .lg .middle } **Safe Tools**

    ---

    Timeout, retry, and graceful error handling for AI tools. Keep requests alive when external services fail.

    [Safe Tools →](safe-tools.md)

-   :material-chart-line:{ .lg .middle } **Tracing**

    ---

    Correlate AI calls with the rest of your request flow. Integrate with any observability platform.

    [Tracing →](tracing.md)

</div>

---

## Where to Start

!!! tip "Not sure which guide to read?"

    **Building an AI feature?** Start with [FastroAgent](fastro-agent.md). It gives you cost tracking on every AI call.

    **Multiple AI steps?** Read [Pipelines](pipelines.md) for execution order, parallelization, and aggregated costs.

    **Tools calling external APIs?** Check [Safe Tools](safe-tools.md) for graceful degradation when services fail.

    **Going to production?** Set up [Tracing](tracing.md) to debug slow or expensive requests.

Most projects start with FastroAgent - it gives you cost tracking on every AI call, which you'll want in production. As your application grows, add pipelines for multi-step workflows and safe tools for external service calls.

---

[← Home](../index.md){ .md-button } [FastroAgent →](fastro-agent.md){ .md-button .md-button--primary }
