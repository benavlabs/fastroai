"""Self-tests for the contract suite's helpers."""

import pytest

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
