"""The driver for the FastroAI 0.6 API."""

from collections.abc import Mapping
from typing import Any

from pydantic_ai import Agent

from fastroai import FastroAgent, Pipeline, StepContext, step
from tests.contract.drivers.base import AgentStep, FunctionStep, Outcome, Usage
from tests.contract.pricebook import PRICES, calculator
from tests.contract.scripted import Script

MISSING = object()


class View:
    def __init__(self, ctx: StepContext[Any]) -> None:
        self._ctx = ctx

    def input(self, key: str) -> Any:
        value = self._ctx.get_input(key, MISSING)
        if value is MISSING:
            raise KeyError(f"no pipeline input {key!r}")
        return value

    def dependency(self, step_id: str) -> Any:
        return self._ctx.get_dependency(step_id)


def function_step(spec: FunctionStep) -> Any:
    @step
    async def run(ctx: StepContext[Any]) -> Any:
        return await spec.body(View(ctx))

    return run


def agent_step(spec: AgentStep) -> Any:
    agent = spec.agent
    assert isinstance(agent, FastroAgent)

    @step
    async def run(ctx: StepContext[Any]) -> Any:
        response = await ctx.run(agent, spec.prompt(View(ctx)))
        return response.output

    return run


class V0Driver:
    def agent(self, script: Script) -> FastroAgent[Any]:
        if script.model_name not in PRICES:
            raise ValueError(f"scripted model {script.model_name!r} has no pricebook price")
        return FastroAgent(agent=Agent(script.model), cost_calculator=calculator())

    def pipeline(
        self,
        steps: Mapping[str, FunctionStep | AgentStep],
        dependencies: Mapping[str, list[str]],
        *,
        output_step: str | None = None,
    ) -> Pipeline[None, dict[str, Any], Any]:
        return Pipeline(
            name="contract",
            steps={
                step_id: function_step(spec) if isinstance(spec, FunctionStep) else agent_step(spec)
                for step_id, spec in steps.items()
            },
            dependencies=dict(dependencies),
            output_step=output_step,
        )

    async def run(self, pipeline: object, inputs: dict[str, Any]) -> Outcome:
        assert isinstance(pipeline, Pipeline)
        result = await pipeline.execute(inputs, None)
        steps = result.usage.steps if result.usage else {}
        return Outcome(
            output=result.output,
            step_outputs=dict(result.step_outputs),
            step_usage={
                step_id: Usage(
                    model=usage.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    cost_microcents=usage.cost_microcents,
                )
                for step_id, usage in steps.items()
            },
            total_cost_microcents=result.usage.total_cost_microcents if result.usage else 0,
        )
