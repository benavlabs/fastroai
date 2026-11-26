"""Tests that the package metadata states the Python versions CI tests."""

import tomllib
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "Programming Language :: Python :: "


def project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def classified_versions() -> set[Version]:
    return {
        Version(classifier.removeprefix(PREFIX))
        for classifier in project()["classifiers"]
        if classifier.startswith(PREFIX) and classifier.removeprefix(PREFIX).count(".") == 1
    }


def ci_python_versions() -> set[Version]:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
    return {Version(str(version)) for version in workflow["jobs"]["tests"]["strategy"]["matrix"]["python-version"]}


def test_every_python_ci_tests_has_a_classifier() -> None:
    tested = ci_python_versions()
    assert tested
    assert tested <= classified_versions()


def test_every_python_classifier_is_within_what_ci_tests_and_requires_python() -> None:
    tested = ci_python_versions()
    requires = SpecifierSet(project()["requires-python"])
    for version in classified_versions():
        assert min(tested) <= version <= max(tested), version
        assert version in requires, version
