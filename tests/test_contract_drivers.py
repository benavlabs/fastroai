"""Self-tests for the contract drivers, run through the v0 driver."""

from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.usage import RequestUsage

from tests.contract.drivers.base import AgentStep, Driver, FeatureMissing, FunctionStep, Outcome, StepView, Usage
from tests.contract.drivers.v0 import V0Driver
from tests.contract.pricebook import expected_microcents
from tests.contract.scripted import Script, text
from tests.test_conftest import use_the_repo_pytest_config

SMALL_USAGE = RequestUsage(input_tokens=120, output_tokens=30, cache_read_tokens=20)
LARGE_USAGE = RequestUsage(input_tokens=200, output_tokens=40, cache_write_tokens=50)


def driver() -> Driver:
    return V0Driver()


def prompts(script: Script) -> list[list[Any]]:
    return [
        [
            part.content
            for message in request.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        for request in script.requests
    ]


async def test_function_steps_read_pipeline_inputs_and_their_dependencies() -> None:
    async def load(view: StepView) -> str:
        return view.input("document").upper()

    async def count(view: StepView) -> int:
        return len(view.dependency("load"))

    contract = driver()
    pipeline = contract.pipeline({"load": FunctionStep(load), "count": FunctionStep(count)}, {"count": ["load"]})

    outcome = await contract.run(pipeline, {"document": "a short text"})

    assert outcome == Outcome(
        output=12, step_outputs={"load": "A SHORT TEXT", "count": 12}, step_usage={}, total_cost_microcents=0
    )


@pytest.mark.usage_call_break
async def test_an_agent_step_sends_its_prompt_and_reports_exact_usage_and_cost() -> None:
    script = Script("scripted-small", [text("a summary", usage=SMALL_USAGE)])
    contract = driver()
    pipeline = contract.pipeline(
        {"summarize": AgentStep(contract.agent(script), lambda view: f"Summarize: {view.input('document')}")}, {}
    )

    outcome = await contract.run(pipeline, {"document": "a short text"})

    cost = expected_microcents("scripted-small", SMALL_USAGE)
    assert outcome == Outcome(
        output="a summary",
        step_outputs={"summarize": "a summary"},
        step_usage={
            "summarize": Usage(
                model="scripted-small",
                input_tokens=120,
                output_tokens=30,
                cache_read_tokens=20,
                cache_write_tokens=0,
                cost_microcents=cost,
            )
        },
        total_cost_microcents=cost,
    )
    assert cost == 100 * 3 + 30 * 15 + 20 * 1
    assert prompts(script) == [["Summarize: a short text"]]
    script.assert_consumed()


@pytest.mark.usage_call_break
async def test_a_pipeline_mixes_function_and_agent_steps_and_totals_the_agent_costs() -> None:
    small = Script("scripted-small", [text("short", usage=SMALL_USAGE)])
    large = Script("scripted-large", [text("long", usage=LARGE_USAGE)])

    async def load(view: StepView) -> str:
        return view.input("document")

    async def join(view: StepView) -> str:
        return f"{view.dependency('brief')}+{view.dependency('full')}"

    contract = driver()
    pipeline = contract.pipeline(
        {
            "load": FunctionStep(load),
            "brief": AgentStep(contract.agent(small), lambda view: f"Brief: {view.dependency('load')}"),
            "full": AgentStep(contract.agent(large), lambda view: f"Full: {view.dependency('load')}"),
            "join": FunctionStep(join),
        },
        {"brief": ["load"], "full": ["load"], "join": ["brief", "full"]},
    )

    outcome = await contract.run(pipeline, {"document": "notes"})

    assert outcome.output == "short+long"
    assert outcome.step_outputs == {"load": "notes", "brief": "short", "full": "long", "join": "short+long"}
    assert set(outcome.step_usage) == {"brief", "full"}
    assert outcome.step_usage["brief"].cost_microcents == expected_microcents("scripted-small", SMALL_USAGE)
    assert outcome.step_usage["full"].model == "scripted-large"
    assert outcome.step_usage["full"].cache_write_tokens == 50
    assert outcome.step_usage["full"].cost_microcents == expected_microcents("scripted-large", LARGE_USAGE)
    assert outcome.total_cost_microcents == expected_microcents("scripted-small", SMALL_USAGE) + expected_microcents(
        "scripted-large", LARGE_USAGE
    )
    assert (prompts(small), prompts(large)) == ([["Brief: notes"]], [["Full: notes"]])
    small.assert_consumed()
    large.assert_consumed()


async def test_the_output_step_picks_the_pipeline_output() -> None:
    async def first(view: StepView) -> str:
        return "first"

    async def second(view: StepView) -> str:
        return "second"

    contract = driver()
    pipeline = contract.pipeline({"a": FunctionStep(first), "b": FunctionStep(second)}, {"b": ["a"]}, output_step="a")

    outcome = await contract.run(pipeline, {})

    assert (outcome.output, outcome.step_outputs) == ("first", {"a": "first", "b": "second"})


async def test_reading_an_input_the_pipeline_was_not_given_fails_instead_of_reading_none() -> None:
    async def read(view: StepView) -> Any:
        return view.input("missing")

    contract = driver()
    pipeline = contract.pipeline({"read": FunctionStep(read)}, {})

    with pytest.raises(Exception, match="no pipeline input 'missing'"):
        await contract.run(pipeline, {"document": "present"})


async def test_reading_a_step_on_the_same_level_fails_instead_of_reading_none() -> None:
    async def first(view: StepView) -> str:
        return "first"

    async def second(view: StepView) -> Any:
        return view.dependency("a")

    contract = driver()
    pipeline = contract.pipeline({"a": FunctionStep(first), "b": FunctionStep(second)}, {}, output_step="b")

    with pytest.raises(Exception, match="'a' not a dependency of 'b'"):
        await contract.run(pipeline, {})


async def test_a_step_input_given_as_none_is_read_as_none() -> None:
    async def read(view: StepView) -> Any:
        return view.input("maybe")

    contract = driver()
    outcome = await contract.run(contract.pipeline({"read": FunctionStep(read)}, {}), {"maybe": None})

    assert outcome.output is None


def test_an_agent_needs_a_script_whose_model_the_pricebook_prices() -> None:
    script = Script("scripted-unpriced", [text("hi", usage=SMALL_USAGE)])

    with pytest.raises(ValueError, match="^scripted model 'scripted-unpriced' has no pricebook price"):
        driver().agent(script)


def test_feature_missing_is_an_exception_of_its_own() -> None:
    assert issubclass(FeatureMissing, Exception)
    assert not issubclass(FeatureMissing, (AssertionError, NotImplementedError))


def test_the_driver_fixture_yields_the_v0_driver(pytester: pytest.Pytester) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest("from tests.contract.conftest import driver\n")
    pytester.makepyfile(
        "from tests.contract.drivers.v0 import V0Driver\n\n"
        "def test_driver(driver):\n"
        "    assert type(driver) is V0Driver\n"
    )

    result = pytester.runpytest("-p", "no:asyncio", "-v")

    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*::test_driver[[]v0[]] PASSED*"])
