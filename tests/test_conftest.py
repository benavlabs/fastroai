"""Tests for the suite-wide pytest hooks in tests/conftest.py."""

from importlib.metadata import version

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic_ai import models

from fastroai import FastroAgent

USAGE_CALL_BREAK = SpecifierSet(">=2.0,<2.64")

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


def test_usage_call_break_marker_xfails_only_the_type_error_on_the_broken_range(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("from tests.conftest import pytest_collection_modifyitems, pytest_configure\n")
    pytester.makepyfile(MARKED_TESTS)

    result = pytester.runpytest("-p", "no:asyncio", "--strict-markers", "-rxXf")

    if Version(version("pydantic-ai-slim")) in USAGE_CALL_BREAK:
        result.assert_outcomes(xfailed=1, failed=2)
        result.stdout.fnmatch_lines(
            ["XFAIL *test_raises_type_error*", "FAILED *test_raises_other_error*", "FAILED *test_passes*"]
        )
    else:
        result.assert_outcomes(passed=1, failed=2)
        result.stdout.fnmatch_lines(["FAILED *test_raises_type_error*", "FAILED *test_raises_other_error*"])


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
