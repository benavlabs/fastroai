"""Self-tests for the contract suite's helpers."""

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage

from fastroai import CostCalculator, FastroAgent
from tests.contract.pricebook import PRICES, Price, calculator, expected_microcents
from tests.contract.scripted import Script, ScriptError, http_error, streamed, structured, text, tool_calls
from tests.contract.test_no_mock import mock_uses


@pytest.mark.parametrize(
    "source",
    [
        "import unittest.mock",
        "import unittest.mock as um",
        "import os, unittest.mock",
        "from unittest import mock",
        "from unittest import mock as m",
        "from unittest import TestCase, mock",
        "from unittest.mock import AsyncMock",
        "from unittest.mock import patch as p",
        "import unittest\npatcher = unittest.mock.patch",
        "import unittest as ut\npatcher = ut.mock.patch",
        "def helper():\n    from unittest.mock import MagicMock\n    return MagicMock()",
    ],
)
def test_mock_uses_finds_every_way_to_reach_unittest_mock(source: str) -> None:
    assert mock_uses(source)


@pytest.mark.parametrize(
    "source",
    [
        "import unittest",
        "from unittest import TestCase",
        "import unittest\ncase = unittest.TestCase",
        "mock = object()",
        "settings = object()\nvalue = settings.mock",
        "import os as ut\nvalue = ut.mock",
        "class Fake:\n    mock = 1\nvalue = Fake().mock",
        "text = 'from unittest import mock'",
        "# import unittest.mock",
        "from tests.contract import mock",
        "import mock_helpers",
    ],
)
def test_mock_uses_ignores_code_that_does_not_reach_unittest_mock(source: str) -> None:
    assert mock_uses(source) == []


def test_mock_uses_reports_the_line_of_each_use_in_order() -> None:
    source = "def helper():\n    import unittest.mock\n\nfrom unittest import mock\nimport unittest\nunittest.mock.patch('os.getcwd')\n"
    assert mock_uses(source) == [2, 4, 6]


TEXT_USAGE = RequestUsage(input_tokens=120, output_tokens=30)
TOOL_USAGE = RequestUsage(input_tokens=200, output_tokens=40, cache_read_tokens=50)
FINAL_USAGE = RequestUsage(input_tokens=260, output_tokens=15)


class Report(BaseModel):
    title: str
    score: int


def responses(result: AgentRunResult[Any]) -> list[ModelResponse]:
    return [message for message in result.all_messages() if isinstance(message, ModelResponse)]


def parts_of(message: ModelMessage, kind: type) -> list[Any]:
    return [part for part in message.parts if isinstance(part, kind)]


async def test_a_text_reply_reaches_the_agent_with_its_usage_and_the_request_is_recorded() -> None:
    script = Script("scripted-text", [text("hello there", usage=TEXT_USAGE)])

    result = await Agent(script.model).run("say hello")

    assert result.output == "hello there"
    (response,) = responses(result)
    assert response.usage == TEXT_USAGE
    assert response.model_name == "scripted-text"
    (request,) = script.requests
    assert request.model_name == "scripted-text"
    assert [part.content for part in parts_of(request.messages[-1], UserPromptPart)] == ["say hello"]
    script.assert_consumed()


async def test_tool_call_replies_run_the_tools_and_their_results_reach_the_next_request() -> None:
    script = Script(
        "scripted-tools",
        [
            tool_calls(("lookup", {"code": "A1"}), ("lookup", {"code": "B2"}), usage=TOOL_USAGE),
            text("both found", usage=FINAL_USAGE),
        ],
    )
    agent = Agent(script.model)

    @agent.tool_plain
    def lookup(code: str) -> str:
        return f"found {code}"

    result = await agent.run("look both up")

    assert result.output == "both found"
    assert [response.usage for response in responses(result)] == [TOOL_USAGE, FINAL_USAGE]
    calls = parts_of(responses(result)[0], ToolCallPart)
    assert [(call.tool_name, call.args, call.tool_call_id) for call in calls] == [
        ("lookup", {"code": "A1"}, "call-1-1"),
        ("lookup", {"code": "B2"}, "call-1-2"),
    ]
    assert [len(request.messages) for request in script.requests] == [1, 3]
    returns = parts_of(script.requests[1].messages[-1], ToolReturnPart)
    assert [(part.tool_call_id, part.content) for part in returns] == [
        ("call-1-1", "found A1"),
        ("call-1-2", "found B2"),
    ]
    script.assert_consumed()


async def test_a_structured_reply_becomes_the_agent_output() -> None:
    script = Script("scripted-structured", [structured({"title": "weekly", "score": 7}, usage=TEXT_USAGE)])

    result = await Agent(script.model, output_type=Report).run("rate the week")

    assert result.output == Report(title="weekly", score=7)
    (response,) = responses(result)
    assert response.usage == TEXT_USAGE
    assert [(call.args, call.tool_call_id) for call in parts_of(response, ToolCallPart)] == [
        ({"title": "weekly", "score": 7}, "call-1-1")
    ]
    script.assert_consumed()


class Summary(BaseModel):
    text: str


@pytest.mark.parametrize(("output_type", "count"), [(str, 0), ([Report, Summary], 2)])
async def test_a_structured_reply_needs_exactly_one_output_tool_or_it_is_recorded_as_a_script_error(
    output_type: Any, count: int
) -> None:
    script = Script("scripted-structured", [structured({"title": "weekly", "score": 7}, usage=TEXT_USAGE)])
    message = f"request 1 has a structured reply, which needs 1 output tool, not {count}"

    with pytest.raises(ScriptError, match=message):
        await Agent(script.model, output_type=output_type).run("rate the week")

    with pytest.raises(AssertionError, match=message):
        script.assert_consumed()


async def test_the_settings_of_each_request_are_recorded() -> None:
    script = Script("scripted-settings", [text("one", usage=TEXT_USAGE), text("two", usage=TEXT_USAGE)])
    agent = Agent(script.model, model_settings={"temperature": 0.2})

    await agent.run("first")
    await agent.run("second", model_settings={"max_tokens": 64})

    assert [request.settings for request in script.requests] == [
        {"temperature": 0.2},
        {"temperature": 0.2, "max_tokens": 64},
    ]


async def test_recorded_settings_stay_as_sent_when_the_caller_changes_its_settings_afterwards() -> None:
    script = Script("scripted-settings", [text("one", usage=TEXT_USAGE)])
    settings: ModelSettings = {"temperature": 0.2}
    await Agent(script.model, model_settings=settings).run("first")

    settings["temperature"] = 0.9

    assert script.requests[0].settings == {"temperature": 0.2}


async def test_a_request_past_the_script_raises_and_stays_recorded_when_caught() -> None:
    script = Script("scripted-short", [text("only reply", usage=TEXT_USAGE)])
    agent = Agent(script.model)
    await agent.run("first")

    with pytest.raises(ScriptError, match="request 2 ran past the 1 scripted replies"):
        await agent.run("second")

    assert len(script.requests) == 2
    with pytest.raises(AssertionError, match="request 2 ran past the 1 scripted replies"):
        script.assert_consumed()


async def test_an_unused_reply_fails_assert_consumed() -> None:
    script = Script("scripted-long", [text("first", usage=TEXT_USAGE), text("never sent", usage=TEXT_USAGE)])
    await Agent(script.model).run("only one request")

    with pytest.raises(AssertionError, match="1 requests for 2 scripted replies"):
        script.assert_consumed()


def test_each_reply_copies_its_usage_so_a_shared_usage_object_stays_unchanged() -> None:
    usage = RequestUsage(input_tokens=10, output_tokens=5)
    reply = text("hi", usage=usage)
    usage.input_tokens = 99

    assert reply.usage == RequestUsage(input_tokens=10, output_tokens=5)


@pytest.mark.parametrize("usage", [RequestUsage(), RequestUsage(input_tokens=0, output_tokens=0)])
def test_a_reply_without_usage_is_rejected_because_function_model_would_estimate_it(usage: RequestUsage) -> None:
    with pytest.raises(ValueError, match="non-zero usage"):
        text("hi", usage=usage)


def test_a_tool_calls_reply_needs_at_least_one_call() -> None:
    with pytest.raises(ValueError, match="at least one tool call"):
        tool_calls(usage=TEXT_USAGE)


@pytest.mark.parametrize("name", ["", "Scripted", "openai:gpt-4o", "provider/model"])
def test_a_model_name_must_be_lowercase_without_a_provider_separator(name: str) -> None:
    with pytest.raises(ValueError, match="lowercase"):
        Script(name, [text("hi", usage=TEXT_USAGE)])


@pytest.mark.parametrize("status", [503, 400])
async def test_an_http_error_reply_fails_that_call_only_and_is_recorded(status: int) -> None:
    script = Script(
        "scripted-flaky",
        [text("before", usage=TEXT_USAGE), http_error(status), text("after", usage=FINAL_USAGE)],
    )
    agent = Agent(script.model)

    first = await agent.run("one")
    with pytest.raises(ModelHTTPError) as raised:
        await agent.run("two")
    third = await agent.run("three")

    assert (first.output, third.output) == ("before", "after")
    assert (raised.value.status_code, raised.value.model_name) == (status, "scripted-flaky")
    assert raised.value.body == {"error": f"scripted {status} on request 2"}
    assert [parts_of(request.messages[-1], UserPromptPart)[0].content for request in script.requests] == [
        "one",
        "two",
        "three",
    ]
    script.assert_consumed()


async def test_a_fallback_model_answers_from_the_backup_after_a_scripted_503() -> None:
    primary = Script("scripted-primary", [http_error(503)])
    backup = Script("scripted-backup", [text("from the backup", usage=TEXT_USAGE)])

    result = await Agent(FallbackModel(primary.model, backup.model)).run("answer")

    assert result.output == "from the backup"
    assert [(response.model_name, response.usage) for response in responses(result)] == [
        ("scripted-backup", TEXT_USAGE)
    ]
    assert (len(primary.requests), len(backup.requests)) == (1, 1)
    primary.assert_consumed()
    backup.assert_consumed()


@pytest.mark.parametrize("status", [200, 429, 500, 502])
def test_an_http_error_reply_is_a_503_or_a_400(status: int) -> None:
    with pytest.raises(ValueError, match="503 or 400"):
        http_error(status)


async def stream_once(script: Script, prompt: str) -> tuple[list[str], str, ModelResponse]:
    async with Agent(script.model).run_stream(prompt) as stream:
        deltas = [delta async for delta in stream.stream_text(delta=True, debounce_by=None)]
        output = await stream.get_output()
        response = [message for message in stream.all_messages() if isinstance(message, ModelResponse)][-1]
    return deltas, output, response


async def test_a_streamed_reply_streams_its_chunks_with_a_deterministic_estimated_usage() -> None:
    script = Script("scripted-stream", [streamed("Hel", "lo ", "world"), streamed("Hel", "lo ", "world")])

    first = await stream_once(script, "greet me")
    second = await stream_once(script, "greet me")

    deltas, output, response = first
    assert deltas == ["Hel", "lo ", "world"]
    assert "".join(deltas) == output == "Hello world"
    assert response.model_name == "scripted-stream"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert second[2].usage == response.usage
    assert [parts_of(request.messages[-1], UserPromptPart)[0].content for request in script.requests] == [
        "greet me",
        "greet me",
    ]
    script.assert_consumed()


@pytest.mark.usage_call_break
async def test_the_final_stream_chunk_prices_the_usage_the_response_reports() -> None:
    script = Script("scripted-stream", [streamed("a priced ", "streamed reply")])
    calculator = CostCalculator(pricing_overrides={"scripted-stream": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}})
    agent = FastroAgent(agent=Agent(script.model), cost_calculator=calculator)

    chunks = [chunk async for chunk in agent.run_stream("price this")]

    final = chunks[-1]
    assert final.is_final
    assert final.usage_data is not None
    assert len(script.requests) == 1
    assert final.usage_data.content == "a priced streamed reply"
    assert final.usage_data.model == "scripted-stream"
    assert final.usage_data.input_tokens > 0
    assert final.usage_data.output_tokens > 0
    assert final.usage_data.cost_microcents == calculator.calculate_cost(
        "scripted-stream", final.usage_data.input_tokens, final.usage_data.output_tokens
    )
    assert final.usage_data.cost_microcents > 0
    script.assert_consumed()


async def test_a_streamed_request_can_get_a_scripted_http_error() -> None:
    script = Script("scripted-stream", [http_error(503)])

    with pytest.raises(ModelHTTPError) as raised:
        await stream_once(script, "fail")

    assert raised.value.status_code == 503
    assert len(script.requests) == 1
    script.assert_consumed()


async def test_a_streamed_request_for_a_plain_reply_is_recorded_as_a_script_error() -> None:
    script = Script("scripted-stream", [text("not a stream", usage=TEXT_USAGE)])

    with pytest.raises(ScriptError, match="request 1 is streamed but its scripted reply is not"):
        await stream_once(script, "stream it")

    with pytest.raises(AssertionError, match="request 1 is streamed but its scripted reply is not"):
        script.assert_consumed()


async def test_a_plain_request_for_a_streamed_reply_is_recorded_as_a_script_error() -> None:
    script = Script("scripted-stream", [streamed("only ", "streamed")])

    with pytest.raises(ScriptError, match="request 1 is not streamed but its scripted reply is"):
        await Agent(script.model).run("run it")

    with pytest.raises(AssertionError, match="request 1 is not streamed but its scripted reply is"):
        script.assert_consumed()


def test_a_streamed_reply_needs_at_least_one_chunk() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        streamed()


def test_expected_microcents_is_whole_dollar_prices_times_tokens() -> None:
    usage = RequestUsage(input_tokens=1_000, output_tokens=50, cache_read_tokens=200, cache_write_tokens=100)

    cost = expected_microcents("scripted-small", usage)

    assert PRICES["scripted-small"] == Price(input=3, output=15, cache_read=1, cache_write=4)
    assert cost == 700 * 3 + 50 * 15 + 200 * 1 + 100 * 4 == 3_450
    assert type(cost) is int


def test_expected_microcents_takes_the_run_usage_fastroagent_reads() -> None:
    usage = RunUsage(requests=2, input_tokens=300, output_tokens=20, cache_read_tokens=100)

    assert expected_microcents("scripted-large", usage) == 200 * 5 + 20 * 25 + 100 * 2 == 1_700


SHAPES = [
    RequestUsage(input_tokens=120, output_tokens=30),
    RequestUsage(input_tokens=120, output_tokens=0),
    RequestUsage(input_tokens=200, output_tokens=40, cache_read_tokens=50),
    RequestUsage(input_tokens=200, output_tokens=40, cache_write_tokens=60),
    RequestUsage(input_tokens=200, output_tokens=40, cache_read_tokens=50, cache_write_tokens=60),
    RequestUsage(input_tokens=110, output_tokens=7, cache_read_tokens=110),
    RequestUsage(input_tokens=1_234_567, output_tokens=765_431, cache_read_tokens=333_333, cache_write_tokens=1),
]


@pytest.mark.parametrize("model", sorted(PRICES))
@pytest.mark.parametrize("usage", SHAPES)
def test_the_pricebook_calculator_agrees_with_expected_microcents(model: str, usage: RequestUsage) -> None:
    assert calculator().calculate_cost(
        model,
        usage.input_tokens,
        usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
    ) == expected_microcents(model, usage)


def test_every_pricebook_rate_is_distinct_within_and_across_models() -> None:
    rates = [(price.input, price.output, price.cache_read, price.cache_write) for price in PRICES.values()]

    assert len(PRICES) >= 2
    assert len({rate for model_rates in rates for rate in model_rates}) == 4 * len(rates)


def test_calculator_gives_a_fresh_costcalculator_each_time() -> None:
    first = calculator()
    first.add_pricing_override("scripted-small", input_per_mtok=100.0, output_per_mtok=100.0)

    assert calculator().calculate_cost("scripted-small", 10, 0) == 30


@pytest.mark.usage_call_break
async def test_a_scripted_run_costs_exactly_expected_microcents() -> None:
    usage = RequestUsage(input_tokens=400, output_tokens=25, cache_read_tokens=150, cache_write_tokens=50)
    script = Script("scripted-large", [text("cached", usage=usage)])
    agent = FastroAgent(agent=Agent(script.model), cost_calculator=calculator())

    response = await agent.run("price this")

    assert (
        response.cost_microcents == expected_microcents("scripted-large", usage) == 200 * 5 + 25 * 25 + 150 * 2 + 50 * 7
    )
    script.assert_consumed()


@pytest.mark.parametrize(
    ("rates", "message"),
    [
        ({"input": 0}, "input"),
        ({"output": -1}, "output"),
        ({"cache_read": 0}, "cache_read"),
        ({"cache_write": 0}, "cache_write"),
        ({"input": 2.5}, "input"),
        ({"output": True}, "output"),
    ],
)
def test_a_price_is_a_positive_whole_number_of_dollars_per_mtok(rates: dict[str, Any], message: str) -> None:
    fields: dict[str, Any] = {"input": 3, "output": 15, "cache_read": 1, "cache_write": 4} | rates

    with pytest.raises(ValueError, match=f"^the {message} price must be a positive whole number of dollars per Mtok"):
        Price(**fields)


@pytest.mark.parametrize(
    ("model", "usage", "message"),
    [
        ("scripted-unpriced", RequestUsage(input_tokens=1), "no pricebook price for 'scripted-unpriced'"),
        ("openai:scripted-small", RequestUsage(input_tokens=1), "no pricebook price for 'openai:scripted-small'"),
        (
            "scripted-small",
            RequestUsage(input_tokens=10, cache_read_tokens=8, cache_write_tokens=3),
            "cache tokens 11 exceed input tokens 10",
        ),
        ("scripted-small", RequestUsage(input_tokens=10, input_audio_tokens=2), "the pricebook has no audio prices"),
        ("scripted-small", RequestUsage(output_tokens=10, output_audio_tokens=2), "the pricebook has no audio prices"),
        (
            "scripted-small",
            RequestUsage(input_tokens=10, cache_read_tokens=4, cache_audio_read_tokens=1),
            "the pricebook has no audio prices",
        ),
    ],
)
def test_expected_microcents_refuses_usage_it_cannot_price(model: str, usage: RequestUsage, message: str) -> None:
    with pytest.raises(ValueError, match=f"^{message}"):
        expected_microcents(model, usage)
