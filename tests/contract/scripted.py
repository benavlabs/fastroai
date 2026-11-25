"""FunctionModels that answer from a fixed script and record every request they get."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponsePart, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage


class ScriptError(Exception):
    """A scripted model got a request its script can't answer."""


@dataclass(frozen=True)
class Reply:
    kind: Literal["text", "tool_calls", "structured"]
    content: str | tuple[tuple[str, dict[str, Any]], ...] | dict[str, Any]
    usage: RequestUsage

    def __post_init__(self) -> None:
        if not self.usage.has_values():
            raise ValueError("a scripted reply needs non-zero usage, or FunctionModel replaces it with an estimate")
        object.__setattr__(self, "usage", dataclasses.replace(self.usage))


def text(content: str, *, usage: RequestUsage) -> Reply:
    return Reply("text", content, usage)


def tool_calls(*calls: tuple[str, dict[str, Any]], usage: RequestUsage) -> Reply:
    if not calls:
        raise ValueError("a tool_calls reply needs at least one tool call")
    return Reply("tool_calls", calls, usage)


def structured(output: dict[str, Any], *, usage: RequestUsage) -> Reply:
    return Reply("structured", output, usage)


@dataclass(frozen=True)
class HTTPError:
    status: int

    def __post_init__(self) -> None:
        if self.status not in (503, 400):
            raise ValueError(f"a scripted HTTP error is a 503 or 400, got {self.status}")


def http_error(status: int) -> HTTPError:
    return HTTPError(status)


@dataclass(frozen=True)
class Request:
    model_name: str
    settings: ModelSettings | None
    messages: list[ModelMessage]


class Script:
    def __init__(self, model_name: str, replies: Sequence[Reply | HTTPError]) -> None:
        if not model_name or model_name != model_name.lower() or ":" in model_name or "/" in model_name:
            raise ValueError(f"a scripted model name must be lowercase with no ':' or '/', got {model_name!r}")
        self.model_name = model_name
        self.replies = list(replies)
        self.requests: list[Request] = []
        self.errors: list[str] = []
        self.model = FunctionModel(self._respond, model_name=model_name)

    async def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        settings = None if info.model_settings is None else ModelSettings(**info.model_settings)
        self.requests.append(Request(self.model_name, settings, messages))
        number = len(self.requests)
        if number > len(self.replies):
            self._fail(f"request {number} ran past the {len(self.replies)} scripted replies")
        reply = self.replies[number - 1]
        if isinstance(reply, HTTPError):
            raise ModelHTTPError(
                reply.status, self.model_name, {"error": f"scripted {reply.status} on request {number}"}
            )
        return ModelResponse(parts=self._parts(reply, info, number), usage=reply.usage)

    def _parts(self, reply: Reply, info: AgentInfo, number: int) -> list[ModelResponsePart]:
        if reply.kind == "text":
            assert isinstance(reply.content, str)
            return [TextPart(reply.content)]
        if reply.kind == "tool_calls":
            assert isinstance(reply.content, tuple)
            return [
                ToolCallPart(name, args, tool_call_id=f"call-{number}-{position}")
                for position, (name, args) in enumerate(reply.content, start=1)
            ]
        assert isinstance(reply.content, dict)
        if len(info.output_tools) != 1:
            self._fail(
                f"request {number} has a structured reply, which needs 1 output tool, not {len(info.output_tools)}"
            )
        return [ToolCallPart(info.output_tools[0].name, reply.content, tool_call_id=f"call-{number}-1")]

    def _fail(self, message: str) -> NoReturn:
        self.errors.append(message)
        raise ScriptError(message)

    def assert_consumed(self) -> None:
        assert not self.errors, self.errors
        assert len(self.requests) == len(self.replies), (
            f"{len(self.requests)} requests for {len(self.replies)} scripted replies"
        )
