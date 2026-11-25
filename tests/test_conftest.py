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
