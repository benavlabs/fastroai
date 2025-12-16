"""BasePipeline - Router between multiple pipelines.

Implements Strategy pattern for pipeline selection based on input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..tracing import Tracer
from .pipeline import Pipeline, PipelineResult

DepsT = TypeVar("DepsT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BasePipeline(ABC, Generic[DepsT, InputT, OutputT]):
    """Router between multiple pipelines.

    Implements Strategy pattern for pipeline selection.

    Use cases:
    - Simple vs complex paths
    - A/B testing
    - Fallback pipelines

    Examples:
        ```python
        class InvestmentRouter(BasePipeline[MyDeps, dict, Plan]):
            def __init__(self):
                super().__init__("investment_router")
                self.register_pipeline("simple", simple_pipeline)
                self.register_pipeline("complex", complex_pipeline)

            async def route(self, input_data: dict, deps: MyDeps) -> str:
                if input_data.get("amount", 0) < 10000:
                    return "simple"
                return "complex"

        router = InvestmentRouter()
        result = await router.execute({"amount": 50000}, deps)
        ```
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._pipelines: dict[str, Pipeline[DepsT, InputT, OutputT]] = {}

    def register_pipeline(
        self,
        name: str,
        pipeline: Pipeline[DepsT, InputT, OutputT],
    ) -> None:
        """Register a pipeline."""
        self._pipelines[name] = pipeline

    @abstractmethod
    async def route(self, input_data: InputT, deps: DepsT) -> str:
        """Determine which pipeline to execute. Return registered name."""
        ...

    async def execute(
        self,
        input_data: InputT,
        deps: DepsT,
        tracer: Tracer | None = None,
    ) -> PipelineResult[OutputT]:
        """Route and execute."""
        pipeline_name = await self.route(input_data, deps)

        if pipeline_name not in self._pipelines:
            raise ValueError(f"Unknown pipeline: '{pipeline_name}'. Registered: {list(self._pipelines.keys())}")

        return await self._pipelines[pipeline_name].execute(input_data, deps, tracer)
