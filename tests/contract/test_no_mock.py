"""Nothing under tests/contract reaches unittest.mock."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.keep

CONTRACT = Path(__file__).resolve().parent


def mock_uses(source: str) -> list[int]:
    tree = ast.parse(source)
    unittest_names = {"unittest"} | {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "unittest" and alias.asname
    }
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom | ast.Attribute) and reaches_mock(node, unittest_names)
    )


def reaches_mock(node: ast.Import | ast.ImportFrom | ast.Attribute, unittest_names: set[str]) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name == "unittest.mock" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module == "unittest.mock" or (
            node.module == "unittest" and any(alias.name == "mock" for alias in node.names)
        )
    return node.attr == "mock" and isinstance(node.value, ast.Name) and node.value.id in unittest_names


def test_nothing_under_tests_contract_uses_unittest_mock() -> None:
    paths = sorted(CONTRACT.rglob("*.py"))
    assert Path(__file__).resolve() in paths
    uses = {str(path.relative_to(CONTRACT)): mock_uses(path.read_text()) for path in paths}
    assert {path: lines for path, lines in uses.items() if lines} == {}
