"""Tests for the contract suite's marker logic in tests/contract/conftest.py."""

import ast
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

from tests import conftest as suite_conftest
from tests.contract import conftest as contract_conftest
from tests.contract.drivers.v0 import V0Driver
from tests.contract.fixed import FIXED
from tests.test_conftest import use_the_repo_pytest_config

CONTRACT = Path(__file__).resolve().parent / "contract"
CONTRACT_HOOK = "from tests.contract.conftest import pytest_collection_modifyitems\n"
SUITE_HOOK = "from tests.conftest import pytest_collection_modifyitems\n"
CONTRACT_HOOK_AND_DRIVER = "from tests.contract.conftest import driver, pytest_collection_modifyitems\n"

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


NEW_TESTS = """
import pytest
from tests.contract.drivers.base import FeatureMissing

@pytest.mark.new("N-MISSING", since="1.0.0")
def test_lacking_raises_feature_missing(driver):
    raise FeatureMissing("N-MISSING")

@pytest.mark.new("N-OTHER", since="1.0.0")
def test_lacking_raises_another_exception(driver):
    raise KeyError("not FeatureMissing")

@pytest.mark.new("N-PASSES", since="1.0.0")
def test_lacking_passes(driver):
    pass

@pytest.mark.new("N-HAS-PASSES", since="1.0.0")
def test_has_passes(driver):
    pass

@pytest.mark.new("N-HAS-MISSING", since="1.0.0")
def test_has_raises_feature_missing(driver):
    raise FeatureMissing("N-HAS-MISSING")
"""


def test_a_new_test_is_xfail_strict_through_feature_missing_on_a_driver_that_lacks_its_feature(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(V0Driver, "features", frozenset({"N-HAS-PASSES", "N-HAS-MISSING"}))
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(CONTRACT_HOOK_AND_DRIVER)
    pytester.makepyfile(NEW_TESTS)

    result = pytester.runpytest("-p", "no:asyncio", "-rA")

    result.assert_outcomes(passed=1, failed=3, xfailed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ test_lacking_raises_another_exception?v0? _*",
            "E       KeyError: 'not FeatureMissing'",
            "*_ test_lacking_passes?v0? _*",
            "[[]XPASS(strict)[]] N-PASSES",
            "*_ test_has_raises_feature_missing?v0? _*",
            "E       *FeatureMissing: N-HAS-MISSING",
            "PASSED *::test_has_passes?v0?",
            "XFAIL *::test_lacking_raises_feature_missing?v0? - N-MISSING",
            "FAILED *::test_lacking_raises_another_exception?v0?*",
            "FAILED *::test_lacking_passes?v0?*",
            "FAILED *::test_has_raises_feature_missing?v0?*",
        ]
    )


def test_the_v0_driver_has_no_features() -> None:
    assert V0Driver.features == frozenset()


@pytest.mark.parametrize(
    "marker",
    [
        'new(since="1.0.0")',
        'new("", since="1.0.0")',
        'new(1, since="1.0.0")',
        'new("N0")',
        'new("N0", since="")',
        'new("N0", since=1.0)',
        'new("N0", "1.0.0")',
        'new("N0", "N1", since="1.0.0")',
        'new("N0", since="1.0.0", reason="extra")',
    ],
)
def test_a_new_marker_without_an_id_and_a_since_version_stops_the_run(pytester: pytest.Pytester, marker: str) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(CONTRACT_HOOK_AND_DRIVER)
    pytester.makepyfile(f"import pytest\n\n@pytest.mark.{marker}\ndef test_bad_marker(driver):\n    pass\n")

    result = pytester.runpytest("-p", "no:asyncio")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*::test_bad_marker?v0?: use @pytest.mark.new("<id>", since="<version>"), got *'])


def test_a_test_with_two_new_markers_stops_the_run(pytester: pytest.Pytester) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(CONTRACT_HOOK_AND_DRIVER)
    pytester.makepyfile(
        """
import pytest

pytestmark = pytest.mark.new("N-A", since="1.0.0")

@pytest.mark.new("N-B", since="1.0.0")
def test_two_features(driver):
    pass
"""
    )

    result = pytester.runpytest("-p", "no:asyncio")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*::test_two_features?v0?: carries 2 new markers, give each feature its own test"])


@pytest.mark.parametrize(
    ("decorators", "arguments", "nodeid"),
    [
        ("", "", "test_no_driver"),
        ('@pytest.mark.parametrize("x", [1])\n', "x", "test_no_driver[[]1[]]"),
    ],
)
def test_a_new_test_without_the_driver_fixture_stops_the_run(
    pytester: pytest.Pytester, decorators: str, arguments: str, nodeid: str
) -> None:
    use_the_repo_pytest_config(pytester)
    pytester.makeconftest(CONTRACT_HOOK_AND_DRIVER)
    pytester.makepyfile(
        f'import pytest\n\n{decorators}@pytest.mark.new("N0", since="1.0.0")\ndef test_no_driver({arguments}):\n    pass\n'
    )

    result = pytester.runpytest("-p", "no:asyncio")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"*::{nodeid}: a new test must take the driver fixture"])


def contract_markers(source: str) -> dict[str, list[str]]:
    tree = ast.parse(source)
    module_marks = [
        mark
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
        for mark in marks_in(node.value)
    ]
    found: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
            found[node.name] = module_marks + decorator_marks(node)
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for method in node.body:
                if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef) and method.name.startswith("test"):
                    found[f"{node.name}.{method.name}"] = module_marks + decorator_marks(node) + decorator_marks(method)
    return found


def decorator_marks(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    return [mark for decorator in node.decorator_list for mark in marks_in(decorator)]


def marks_in(node: ast.expr) -> list[str]:
    if isinstance(node, ast.List | ast.Tuple):
        return [mark for element in node.elts for mark in marks_in(element)]
    target = node.func if isinstance(node, ast.Call) else node
    if not isinstance(target, ast.Attribute) or target.attr not in {"keep", "fix", "new"}:
        return []
    owner = target.value
    on_mark = (isinstance(owner, ast.Attribute) and owner.attr == "mark") or (
        isinstance(owner, ast.Name) and owner.id == "mark"
    )
    return [target.attr] if on_mark else []


def test_contract_markers_reads_every_way_a_contract_marker_is_attached() -> None:
    source = """
import pytest
from pytest import mark

pytestmark = [pytest.mark.keep, pytest.mark.usage_call_break]
LATER = pytest.mark.new("N9", since="1.0.0")

def test_module_only():
    pass

@pytest.mark.fix("F0", release="0.6.2", raises=AssertionError)
async def test_module_and_fix():
    pass

@mark.new("N0", since="1.0.0")
class TestGrouped:
    def test_class_mark(self):
        pass

    @pytest.mark.keep
    def test_class_and_method(self):
        pass

    def helper(self):
        pass

def helper():
    pass

@other.keep
@pytest.mark.parametrize("x", [1])
def test_not_a_contract_mark(x):
    pass
"""
    assert contract_markers(source) == {
        "test_module_only": ["keep"],
        "test_module_and_fix": ["keep", "fix"],
        "TestGrouped.test_class_mark": ["keep", "new"],
        "TestGrouped.test_class_and_method": ["keep", "new", "keep"],
        "test_not_a_contract_mark": ["keep"],
    }


@pytest.mark.parametrize("module_mark", ["pytest.mark.keep", "(pytest.mark.keep,)"])
def test_contract_markers_reads_a_single_module_mark(module_mark: str) -> None:
    source = f"import pytest\npytestmark = {module_mark}\n\ndef test_one():\n    pass\n"
    assert contract_markers(source) == {"test_one": ["keep"]}


def wrong_marker_counts(root: Path) -> dict[str, list[str]]:
    found = {
        f"{path.relative_to(root)}::{name}": marks
        for path in sorted(root.rglob("test_*.py"))
        for name, marks in contract_markers(path.read_text()).items()
    }
    return {test: marks for test, marks in found.items() if len(marks) != 1}


def test_wrong_marker_counts_names_every_test_without_exactly_one_contract_marker(tmp_path: Path) -> None:
    (tmp_path / "test_top.py").write_text(
        "import pytest\n\n"
        "def test_none():\n    pass\n\n"
        "@pytest.mark.keep\ndef test_one():\n    pass\n\n"
        '@pytest.mark.keep\n@pytest.mark.new("N0", since="1.0.0")\ndef test_two():\n    pass\n'
    )
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "test_deep.py").write_text("def test_deep_none():\n    pass\n")
    (tmp_path / "helpers.py").write_text("def test_not_in_a_test_file():\n    pass\n")

    assert wrong_marker_counts(tmp_path) == {
        "nested/test_deep.py::test_deep_none": [],
        "test_top.py::test_none": [],
        "test_top.py::test_two": ["keep", "new"],
    }


def test_every_contract_test_carries_exactly_one_of_keep_fix_or_new() -> None:
    assert contract_markers((CONTRACT / "test_no_mock.py").read_text()) == {
        "test_nothing_under_tests_contract_uses_unittest_mock": ["keep"]
    }
    assert wrong_marker_counts(CONTRACT) == {}
