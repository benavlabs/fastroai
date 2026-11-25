"""Suite-wide pytest hooks."""

from collections.abc import Iterator
from importlib.metadata import version

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic_ai import models

USAGE_CALL_BREAK = SpecifierSet(">=2.0,<2.64")
USAGE_CALL_BREAK_REASON = "result.usage() raises TypeError on pydantic-ai >=2.0,<2.64"


@pytest.fixture(scope="session", autouse=True)
def refuse_model_requests() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
        yield


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"usage_call_break: runs a real agent, so it is xfail(strict, TypeError) where {USAGE_CALL_BREAK_REASON}",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if Version(version("pydantic-ai-slim")) not in USAGE_CALL_BREAK:
        return
    for item in items:
        if item.get_closest_marker("usage_call_break"):
            item.add_marker(pytest.mark.xfail(strict=True, raises=TypeError, reason=USAGE_CALL_BREAK_REASON))
