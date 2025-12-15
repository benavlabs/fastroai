"""Base abstractions for pipeline steps.

Provides:
- ConversationStatus/ConversationState: Multi-turn conversation signaling
- StepContext: Execution context with inputs, deps, and step outputs
- BaseStep: Abstract base class for pipeline steps
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..tracing import Tracer

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")
T = TypeVar("T")


class ConversationStatus(str, Enum):
    """Status of multi-turn conversation gathering.

    Attributes:
        COMPLETE: All required information has been gathered.
            The pipeline proceeds to subsequent steps.
        INCOMPLETE: More information is needed from the user.
            The pipeline pauses and returns partial state.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class ConversationState(BaseModel, Generic[T]):
    """Signal for multi-turn conversation steps.

    When a step returns ConversationState with INCOMPLETE status,
    the pipeline stops early. Partial data and context are preserved.

    Example:
        class GatherInfoStep(BaseStep[MyDeps, ConversationState[UserInfo]]):
            async def execute(self, context) -> ConversationState[UserInfo]:
                info = await self._extract(context.get_input("message"))

                if info.is_complete():
                    return ConversationState(
                        status=ConversationStatus.COMPLETE,
                        data=info,
                    )

                return ConversationState(
                    status=ConversationStatus.INCOMPLETE,
                    data=info,  # Partial data
                    context={"missing": info.missing_fields()},
                )
    """

    status: ConversationStatus
    data: T | None = None
    context: dict[str, Any] = {}


class StepContext(Generic[DepsT]):
    """Execution context provided to pipeline steps.

    Provides access to:
    - Pipeline inputs (the data passed to execute())
    - Outputs from dependency steps
    - Application dependencies (your db session, user, etc.)
    - Tracer for custom spans

    Example:
        class ProcessStep(BaseStep[MyDeps, Result]):
            async def execute(self, context: StepContext[MyDeps]) -> Result:
                # Get pipeline input
                document = context.get_input("document")

                # Get output from dependency step
                classification = context.get_dependency("classify", Classification)

                # Access your deps
                db = context.deps.session
                user_id = context.deps.user_id

                # Custom tracing
                if context.tracer:
                    async with context.tracer.span("custom_operation"):
                        result = await process(document)

                return result
    """

    def __init__(
        self,
        step_id: str,
        inputs: dict[str, Any],
        deps: DepsT,
        step_outputs: dict[str, Any],
        tracer: Tracer | None = None,
    ) -> None:
        self._step_id = step_id
        self._inputs = inputs
        self._deps = deps
        self._outputs = step_outputs
        self._tracer = tracer

    @property
    def step_id(self) -> str:
        """Current step's ID."""
        return self._step_id

    @property
    def deps(self) -> DepsT:
        """Application dependencies (your session, user, etc.)."""
        return self._deps

    @property
    def tracer(self) -> Tracer | None:
        """Tracer for custom spans."""
        return self._tracer

    def get_input(self, key: str, default: Any = None) -> Any:
        """Get value from pipeline inputs."""
        return self._inputs.get(key, default)

    def get_dependency(
        self,
        step_id: str,
        output_type: type[T] | None = None,
    ) -> T:
        """Get output from a dependency step.

        Args:
            step_id: ID of the dependency step.
            output_type: Expected type (for IDE/type checker).

        Raises:
            ValueError: If step_id not in dependencies or hasn't run.

        Example:
            # With type hint (IDE knows extraction is ExtractionResult)
            extraction = context.get_dependency("extract", ExtractionResult)
            extraction.entities  # Autocomplete works!
        """
        if step_id not in self._outputs:
            raise ValueError(
                f"Step '{step_id}' not a dependency of '{self._step_id}' "
                f"or hasn't completed. Available: {list(self._outputs.keys())}"
            )
        return cast(T, self._outputs[step_id])

    def get_dependency_or_none(
        self,
        step_id: str,
        output_type: type[T] | None = None,
    ) -> T | None:
        """Get output or None if not available. For optional deps."""
        return self._outputs.get(step_id)


class BaseStep(ABC, Generic[DepsT, OutputT]):
    """Abstract base class for pipeline steps.

    A step is one unit of work. It:
    - Receives context with inputs and dependencies
    - Does something (AI call, computation, API call)
    - Returns typed output

    Steps should be stateless. Any state goes in deps or inputs.

    Example:
        class ExtractStep(BaseStep[MyDeps, ExtractionResult]):
            '''Extract entities from document.'''

            def __init__(self):
                self.agent = FastroAgent(system_prompt="Extract entities.")

            async def execute(self, context: StepContext[MyDeps]) -> ExtractionResult:
                document = context.get_input("document")
                response = await self.agent.run(f"Extract: {document}")
                return ExtractionResult.model_validate_json(response.content)
    """

    @abstractmethod
    async def execute(self, context: StepContext[DepsT]) -> OutputT:
        """Execute step logic. Return typed output."""
        ...
