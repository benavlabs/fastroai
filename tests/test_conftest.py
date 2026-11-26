"""Tests for the suite-wide pytest hooks in tests/conftest.py."""

import json
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic_ai import models

from fastroai import FastroAgent
from tests import conftest as suite_conftest

USAGE_CALL_BREAK = SpecifierSet(">=2.0,<2.64")
ROOT = Path(__file__).resolve().parent.parent
INNER_RUN_SKIPS = {"testpaths", "asyncio_mode"}

MARKED_TESTS = """
import pytest

@pytest.mark.usage_call_break
def test_raises_type_error():
    raise TypeError("'RunUsage' object is not callable")

@pytest.mark.usage_call_break
def test_raises_other_error():
    raise ValueError("not the known break")

@pytest.mark.usage_call_break
def test_passes():
    pass
"""


def use_the_repo_pytest_config(pytester: pytest.Pytester) -> None:
    options = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    lines = ["[tool.pytest.ini_options]"]
    for key, value in options.items():
        if key not in INNER_RUN_SKIPS:
            lines.append(f"{key} = {str(value).lower() if isinstance(value, bool) else json.dumps(value)}")
    pytester.makepyprojecttoml("\n".join(lines) + "\n")


def test_usage_call_break_marker_xfails_only_the_type_error_on_the_broken_range(pytester: pytest.Pytester) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest("from tests.conftest import pytest_collection_modifyitems\n")
    pytester.makepyfile(MARKED_TESTS)

    result = pytester.runpytest("-p", "no:asyncio", "-rxXf")

    if Version(version("pydantic-ai-slim")) in USAGE_CALL_BREAK:
        result.assert_outcomes(xfailed=1, failed=2)
        result.stdout.fnmatch_lines(
            ["XFAIL *test_raises_type_error*", "FAILED *test_raises_other_error*", "FAILED *test_passes*"]
        )
    else:
        result.assert_outcomes(passed=1, failed=2)
        result.stdout.fnmatch_lines(["FAILED *test_raises_type_error*", "FAILED *test_raises_other_error*"])


WRAPPED_TESTS = """
import pytest
from fastroai import StepExecutionError

@pytest.mark.usage_call_break
def test_wrapped_break():
    raise StepExecutionError("summarize", TypeError("'RunUsage' object is not callable"))

@pytest.mark.usage_call_break
def test_wrapped_other_type_error():
    raise StepExecutionError("summarize", TypeError("unsupported operand type(s) for +: 'RunUsage' and 'int'"))

@pytest.mark.usage_call_break
def test_wrapped_other_error():
    raise StepExecutionError("summarize", ValueError("'RunUsage' object is not callable"))

@pytest.mark.usage_call_break
def test_wrapped_other_not_callable():
    raise StepExecutionError("summarize", TypeError("'NoneType' object is not callable"))

def test_unmarked_wrapped_break():
    raise StepExecutionError("summarize", TypeError("'RunUsage' object is not callable"))
"""

SUITE_HOOKS = "from tests.conftest import pytest_collection_modifyitems, pytest_runtest_call\n"

STEP_ERROR = "fastroai.pipelines.executor.StepExecutionError: Step 'summarize' failed:"


def test_usage_call_break_unwraps_only_the_known_break_from_a_step_error_on_the_broken_range(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite_conftest, "USAGE_CALL_BREAK", SpecifierSet(">=0"))
    monkeypatch.setenv("COLUMNS", "500")
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(SUITE_HOOKS)
    pytester.makepyfile(WRAPPED_TESTS)

    result = pytester.runpytest("-p", "no:asyncio", "-rfx")

    result.assert_outcomes(xfailed=1, failed=4)
    result.stdout.fnmatch_lines(
        [
            f"FAILED *::test_wrapped_other_type_error - {STEP_ERROR} unsupported operand type(s) for +: 'RunUsage' and 'int'",
            f"FAILED *::test_wrapped_other_error - {STEP_ERROR} 'RunUsage' object is not callable",
            f"FAILED *::test_wrapped_other_not_callable - {STEP_ERROR} 'NoneType' object is not callable",
            f"FAILED *::test_unmarked_wrapped_break - {STEP_ERROR} 'RunUsage' object is not callable",
            "XFAIL *::test_wrapped_break - result.usage() raises TypeError on pydantic-ai >=2.0,<2.64",
        ]
    )


def test_usage_call_break_unwraps_nothing_outside_the_broken_range(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite_conftest, "USAGE_CALL_BREAK", SpecifierSet("<0"))
    monkeypatch.setenv("COLUMNS", "500")
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(SUITE_HOOKS)
    pytester.makepyfile(WRAPPED_TESTS)

    result = pytester.runpytest("-p", "no:asyncio", "-rf")

    result.assert_outcomes(failed=5)
    result.stdout.fnmatch_lines(
        [
            f"FAILED *::test_wrapped_break - {STEP_ERROR} 'RunUsage' object is not callable",
            f"FAILED *::test_wrapped_other_type_error - {STEP_ERROR} unsupported operand type(s) for +: 'RunUsage' and 'int'",
            f"FAILED *::test_wrapped_other_error - {STEP_ERROR} 'RunUsage' object is not callable",
            f"FAILED *::test_wrapped_other_not_callable - {STEP_ERROR} 'NoneType' object is not callable",
            f"FAILED *::test_unmarked_wrapped_break - {STEP_ERROR} 'RunUsage' object is not callable",
        ]
    )


def test_a_marked_pipeline_running_a_real_agent_xfails_only_on_the_broken_range(pytester: pytest.Pytester) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(SUITE_HOOKS)
    pytester.makepyfile(
        """
import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from fastroai import FastroAgent, Pipeline, step


@pytest.mark.usage_call_break
def test_pipeline_with_an_agent_step():
    agent = FastroAgent(
        agent=Agent(
            FunctionModel(
                lambda messages, info: ModelResponse(
                    parts=[TextPart("done")], usage=RequestUsage(input_tokens=10, output_tokens=2)
                )
            )
        )
    )

    @step
    async def summarize(ctx):
        return (await ctx.run(agent, "go")).output

    result = asyncio.run(Pipeline(name="p", steps={"summarize": summarize}).execute({}, None))

    assert result.output == "done"
"""
    )

    result = pytester.runpytest("-p", "no:asyncio")

    if Version(version("pydantic-ai-slim")) in USAGE_CALL_BREAK:
        result.assert_outcomes(xfailed=1)
    else:
        result.assert_outcomes(passed=1)


def test_every_contract_marker_is_registered(pytester: pytest.Pytester) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makepyfile(
        """
import pytest

@pytest.mark.keep
def test_keep():
    pass

@pytest.mark.fix("F0", release="0.6.2", raises=AssertionError)
def test_fix():
    pass

@pytest.mark.new("N0", since="1.0.0")
def test_new():
    pass

@pytest.mark.usage_call_break
def test_usage_call_break():
    pass
"""
    )

    result = pytester.runpytest_subprocess("-p", "no:asyncio")

    result.assert_outcomes(passed=4, warnings=0)


def test_a_misspelled_marker_fails_collection(pytester: pytest.Pytester) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makepyfile(
        """
import pytest

@pytest.mark.kepp
def test_misspelled():
    pass
"""
    )

    result = pytester.runpytest_subprocess("-p", "no:asyncio")

    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(["*'kepp' not found in `markers` configuration option*"])


def test_usage_call_break_covers_pydantic_ai_2_0_through_2_63() -> None:
    from tests import conftest

    assert conftest.USAGE_CALL_BREAK == USAGE_CALL_BREAK


async def test_an_unscripted_model_request_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    agent: FastroAgent[str] = FastroAgent()

    with pytest.raises(RuntimeError, match="ALLOW_MODEL_REQUESTS is False"):
        await agent.run("This request must never leave the test run.")


@pytest.fixture(scope="module")
def allow_model_requests_seen_by_a_module_fixture() -> bool:
    return models.ALLOW_MODEL_REQUESTS


def test_model_requests_are_refused_before_module_fixtures_run(
    allow_model_requests_seen_by_a_module_fixture: bool,
) -> None:
    assert allow_model_requests_seen_by_a_module_fixture is False
