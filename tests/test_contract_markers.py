"""Tests for the contract suite's marker logic in tests/contract/conftest.py."""

import ast
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

from tests import conftest as suite_conftest
from tests.contract import conftest as contract_conftest
from tests.contract.fixed import FIXED
from tests.test_conftest import use_the_repo_pytest_config

CONTRACT = Path(__file__).resolve().parent / "contract"
CONTRACT_HOOK = "from tests.contract.conftest import pytest_collection_modifyitems\n"
SUITE_HOOK = "from tests.conftest import pytest_collection_modifyitems\n"

FIX_TESTS = """
import pytest

@pytest.mark.fix("F-FAILS", release="0.6.2", raises=AssertionError)
def test_unlisted_fails():
    assert 1 == 2

@pytest.mark.fix("F-PASSES", release="0.6.2", raises=AssertionError)
def test_unlisted_passes():
    pass

@pytest.mark.fix("F-OTHER", release="0.6.2", raises=AssertionError)
def test_unlisted_raises_another_exception():
    raise KeyError("not the declared exception")

@pytest.mark.fix("F-LISTED-PASSES", release="0.6.2", raises=AssertionError)
def test_listed_passes():
    pass

@pytest.mark.fix("F-LISTED-FAILS", release="0.6.2", raises=AssertionError)
def test_listed_fails():
    assert 1 == 2
"""

COMBINED_TESTS = """
import pytest

@pytest.mark.usage_call_break
@pytest.mark.fix("F-BREAK", release="0.6.2", raises=AssertionError)
def test_hits_the_break():
    raise TypeError("'RunUsage' object is not callable")

@pytest.mark.usage_call_break
@pytest.mark.fix("F-VALUE", release="0.6.2", raises=AssertionError)
def test_fails_on_a_wrong_value():
    assert 1 == 2
"""


def test_a_fix_test_is_xfail_strict_on_its_declared_exception_until_its_id_is_fixed(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract_conftest, "fixed_ids", lambda: frozenset({"F-LISTED-PASSES", "F-LISTED-FAILS"}))
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(CONTRACT_HOOK)
    pytester.makepyfile(FIX_TESTS)

    result = pytester.runpytest("-p", "no:asyncio", "-rA")

    result.assert_outcomes(passed=1, failed=3, xfailed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ test_unlisted_passes _*",
            "[[]XPASS(strict)[]] F-PASSES",
            "*_ test_unlisted_raises_another_exception _*",
            "E       KeyError: 'not the declared exception'",
            "*_ test_listed_fails _*",
            "E       assert 1 == 2",
            "PASSED *::test_listed_passes",
            "XFAIL *::test_unlisted_fails - F-FAILS",
            "FAILED *::test_unlisted_passes",
            "FAILED *::test_unlisted_raises_another_exception",
            "FAILED *::test_listed_fails",
        ]
    )


@pytest.mark.parametrize(
    "marker",
    [
        'fix(release="0.6.2", raises=AssertionError)',
        'fix("", release="0.6.2", raises=AssertionError)',
        'fix(1, release="0.6.2", raises=AssertionError)',
        'fix("F0", raises=AssertionError)',
        'fix("F0", release="", raises=AssertionError)',
        'fix("F0", release=6.2, raises=AssertionError)',
        'fix("F0", "0.6.2", raises=AssertionError)',
        'fix("F0", "F1", release="0.6.2", raises=AssertionError)',
        'fix("F0", release="0.6.2")',
        'fix("F0", release="0.6.2", raises="AssertionError")',
        'fix("F0", release="0.6.2", raises=AssertionError())',
        'fix("F0", release="0.6.2", raises=dict)',
        'fix("F0", release="0.6.2", raises=AssertionError, reason="extra")',
    ],
)
def test_a_fix_marker_without_an_id_release_and_exception_stops_the_run(pytester: pytest.Pytester, marker: str) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(CONTRACT_HOOK)
    pytester.makepyfile(f"import pytest\n\n@pytest.mark.{marker}\ndef test_bad_marker():\n    assert 1 == 2\n")

    result = pytester.runpytest("-p", "no:asyncio")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*::test_bad_marker: use @pytest.mark.fix("<id>", release=*, raises=*'])


@pytest.mark.parametrize(
    "source",
    [
        """
import pytest

@pytest.mark.fix("F-A", release="0.6.2", raises=KeyError)
@pytest.mark.fix("F-B", release="0.6.2", raises=AssertionError)
def test_two_bugs():
    assert 1 == 2
""",
        """
import pytest

pytestmark = pytest.mark.fix("F-A", release="0.6.2", raises=KeyError)

@pytest.mark.fix("F-B", release="0.6.2", raises=AssertionError)
def test_two_bugs():
    assert 1 == 2
""",
    ],
)
def test_a_test_with_two_fix_markers_stops_the_run(pytester: pytest.Pytester, source: str) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(CONTRACT_HOOK)
    pytester.makepyfile(source)

    result = pytester.runpytest("-p", "no:asyncio")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*::test_two_bugs: carries 2 fix markers, give each bug its own test"])


@pytest.mark.parametrize(
    ("break_range", "xfailed", "failed"),
    [
        (SpecifierSet(">=0"), "test_hits_the_break", "test_fails_on_a_wrong_value"),
        (SpecifierSet("<0"), "test_fails_on_a_wrong_value", "test_hits_the_break"),
    ],
)
def test_usage_call_break_wins_over_fix_on_the_broken_range(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    break_range: SpecifierSet,
    xfailed: str,
    failed: str,
) -> None:
    monkeypatch.setattr(suite_conftest, "USAGE_CALL_BREAK", break_range)
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(SUITE_HOOK)
    pytester.mkdir("contract")
    (pytester.path / "contract" / "conftest.py").write_text(CONTRACT_HOOK)
    (pytester.path / "contract" / "test_combined.py").write_text(COMBINED_TESTS)

    result = pytester.runpytest("-p", "no:asyncio", "-rA")

    result.assert_outcomes(failed=1, xfailed=1)
    result.stdout.fnmatch_lines([f"XFAIL *::{xfailed}*", f"FAILED *::{failed}*"], consecutive=False)


def fix_ids(source: str) -> set[str]:
    ids = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fix"):
            continue
        owner = node.func.value
        on_mark = (isinstance(owner, ast.Attribute) and owner.attr == "mark") or (
            isinstance(owner, ast.Name) and owner.id == "mark"
        )
        if on_mark and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            ids.add(node.args[0].value)
    return ids


def test_fix_ids_reads_every_way_a_fix_marker_is_written() -> None:
    source = """
import pytest
from pytest import mark

pytestmark = [pytest.mark.keep, pytest.mark.fix("F-MODULE", release="0.6.2", raises=AssertionError)]

@pytest.mark.fix("F-CLASS", release="0.6.2", raises=AssertionError)
class TestGrouped:
    @mark.fix("F-METHOD", release="0.7.0", raises=KeyError)
    def test_method(self):
        pass

@pytest.mark.new("N-NEW", since="1.0.0")
@pytest.mark.fix("F-FUNCTION", release="0.6.2", raises=AssertionError)
def test_function():
    fix("F-NOT-A-MARKER")
    other.fix("F-NOT-ON-MARK")
"""
    assert fix_ids(source) == {"F-MODULE", "F-CLASS", "F-METHOD", "F-FUNCTION"}


def test_fixed_lists_only_ids_that_contract_tests_use() -> None:
    paths = sorted(CONTRACT.rglob("*.py"))
    assert CONTRACT / "fixed.py" in paths
    used = set().union(*(fix_ids(path.read_text()) for path in paths))
    assert used >= FIXED


def test_the_hook_reads_fixed_from_fixed_py() -> None:
    assert contract_conftest.fixed_ids() is FIXED
