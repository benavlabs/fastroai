"""The protocol every contract scenario is written against, and the neutral types it speaks."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from tests.contract.scripted import Script


class FeatureMissing(Exception):  # noqa: N818
    """The driver under test lacks a capability the scenario asks for."""


class StepView(Protocol):
    def input(self, key: str) -> Any: ...

    def dependency(self, step_id: str) -> Any: ...


@dataclass(frozen=True)
class FunctionStep:
    body: Callable[[StepView], Awaitable[Any]]


@dataclass(frozen=True)
class AgentStep:
    agent: object
    prompt: Callable[[StepView], str]


@dataclass(frozen=True)
class Usage:
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_microcents: int


@dataclass(frozen=True)
class Outcome:
    output: Any
    step_outputs: dict[str, Any]
    step_usage: dict[str, Usage]
    total_cost_microcents: int


class Driver(Protocol):
    def agent(self, script: Script) -> object: ...

    def pipeline(
        self,
        steps: Mapping[str, FunctionStep | AgentStep],
        dependencies: Mapping[str, list[str]],
        *,
        output_step: str | None = None,
    ) -> object: ...

    async def run(self, pipeline: object, inputs: dict[str, Any]) -> Outcome: ...
