"""Contract suite hooks."""

import pytest

from tests.contract.drivers.base import Driver
from tests.contract.drivers.v0 import V0Driver
from tests.contract.fixed import FIXED

FIX_USAGE = 'use @pytest.mark.fix("<id>", release="<version>", raises=<exception class>)'


def fixed_ids() -> frozenset[str]:
    return FIXED


def fix_marker_arguments(item: pytest.Item, marker: pytest.Mark) -> tuple[str, type[BaseException]]:
    kwargs = marker.kwargs
    raises = kwargs.get("raises")
    if (
        len(marker.args) == 1
        and isinstance(marker.args[0], str)
        and marker.args[0]
        and set(kwargs) == {"release", "raises"}
        and isinstance(kwargs["release"], str)
        and kwargs["release"]
        and isinstance(raises, type)
        and issubclass(raises, BaseException)
    ):
        return marker.args[0], raises
    raise pytest.UsageError(f"{item.nodeid}: {FIX_USAGE}, got {marker.args} {marker.kwargs}")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    fixed = fixed_ids()
    for item in items:
        markers = list(item.iter_markers("fix"))
        if not markers:
            continue
        if len(markers) > 1:
            raise pytest.UsageError(f"{item.nodeid}: carries {len(markers)} fix markers, give each bug its own test")
        fix_id, raises = fix_marker_arguments(item, markers[0])
        if fix_id not in fixed:
            item.add_marker(pytest.mark.xfail(strict=True, raises=raises, reason=fix_id))


@pytest.fixture(params=[V0Driver], ids=["v0"])
def driver(request: pytest.FixtureRequest) -> Driver:
    return request.param()
