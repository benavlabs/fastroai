"""Tests that CONTRIBUTING.md states what CI actually runs."""

import re
import shlex
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
EXPECTED_JOBS = {
    ("linting.yml", "lint"),
    ("type-checking.yml", "type-check"),
    ("tests-type-checking.yml", "type-check"),
    ("tests.yml", "tests"),
    ("tests.yml", "compat"),
}
COMMAND_SEPARATORS = {";", "&&", "||", "|", "&"}
UV_COMMANDS = {"uv", "uvx"}
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\n`]*)\n(.*?)^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE)
NON_SHELL_FENCES = {"python", "py", "toml", "yaml", "yml", "json", "text", "txt", "output"}
UV_WORD = re.compile(r"(?<![\w-])uvx?(?![\w-])")
INLINE_CODE = re.compile(r"`((?:[^`\n]|\n(?!\s*\n))+)`")


def workflow_jobs() -> dict[tuple[str, str], list[dict[str, Any]]]:
    paths = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    assert paths
    jobs = {}
    for path in paths:
        workflow = yaml.safe_load(path.read_text())
        assert isinstance(workflow, dict) and isinstance(workflow.get("jobs"), dict) and workflow["jobs"], path.name
        for name, job in workflow["jobs"].items():
            assert isinstance(job.get("steps"), list) and job["steps"], f"{path.name}:{name}"
            jobs[(path.name, name)] = job["steps"]
    return jobs


def shell_commands(step: dict[str, Any]) -> list[list[str]]:
    commands: list[list[str]] = []
    heredoc_end = None
    for line in str(step.get("run", "")).replace("\\\n", " ").splitlines():
        if heredoc_end is not None:
            if line.strip() == heredoc_end:
                heredoc_end = None
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
        command: list[str] = []
        for index, token in enumerate(tokens):
            if token in COMMAND_SEPARATORS:
                commands.append(command)
                command = []
            else:
                command.append(token)
            if token == "<<" and index + 1 < len(tokens):
                heredoc_end = tokens[index + 1].lstrip("-")
        commands.append(command)
    return [command for command in commands if command]


def runs_uv(step: dict[str, Any]) -> bool:
    return any(UV_COMMANDS & set(command) for command in shell_commands(step))


def is_setup_uv(step: dict[str, Any]) -> bool:
    return str(step.get("uses", "")).startswith("astral-sh/setup-uv@")


def ci_uv_pins() -> list[str]:
    pins = []
    for steps in workflow_jobs().values():
        for step in filter(is_setup_uv, steps):
            settings = step.get("with") or {}
            assert settings.get("version"), step
            pins.append(str(settings["version"]))
    return pins


def test_every_expected_ci_job_is_present() -> None:
    assert workflow_jobs().keys() >= EXPECTED_JOBS


def test_every_job_that_runs_uv_installs_it_first_with_one_setup_uv_step() -> None:
    for job, steps in workflow_jobs().items():
        uv_steps = [index for index, step in enumerate(steps) if runs_uv(step)]
        if uv_steps:
            setup_steps = [index for index, step in enumerate(steps) if is_setup_uv(step)]
            assert len(setup_steps) == 1, job
            assert setup_steps[0] < uv_steps[0], job


def test_every_setup_uv_step_pins_the_same_uv_version() -> None:
    assert len(set(ci_uv_pins())) == 1


def is_uv_sync(command: list[str]) -> bool:
    return "uv" in command and "sync" in command[command.index("uv") + 1 :]


def option_value(command: list[str], *options: str) -> str | None:
    for index, token in enumerate(command):
        if token in options and index + 1 < len(command):
            return command[index + 1]
        for option in options:
            if token.startswith(f"{option}="):
                return token.split("=", 1)[1]
    return None


def test_every_uv_sync_in_ci_checks_the_lock_file() -> None:
    syncs = [
        command
        for steps in workflow_jobs().values()
        for step in steps
        for command in shell_commands(step)
        if is_uv_sync(command)
    ]
    assert syncs
    assert all("--locked" in command for command in syncs)


def ci_checks() -> set[tuple[str, ...]]:
    return {
        tuple(token for token in command if token != "--no-sync")
        for steps in workflow_jobs().values()
        for step in steps
        for command in shell_commands(step)
        if command[:2] == ["uv", "run"] and "python" not in command
    }


def shell_blocks(text: str) -> list[str]:
    return [
        re.sub(r"^[ \t]*\$[ \t]+", "", body, flags=re.MULTILINE)
        for _, info, body in FENCE.findall(text)
        if (info.split() or [""])[0].lower() not in NON_SHELL_FENCES
    ]


def contributing_snippets() -> list[str]:
    text = (ROOT / "CONTRIBUTING.md").read_text()
    spans = [" ".join(span.split()) for span in INLINE_CODE.findall(FENCE.sub("", text))]
    return shell_blocks(text) + spans


def uv_lines(snippet: str) -> list[str]:
    return [line for line in snippet.replace("\\\n", " ").splitlines() if UV_WORD.search(line)]


def doc_commands(snippet: str) -> list[list[str]]:
    return [command for line in uv_lines(snippet) for command in shell_commands({"run": line})]


def uv_run_commands(snippet: str) -> list[list[str]]:
    return [command for command in doc_commands(snippet) if command[:2] == ["uv", "run"]]


def is_fixer(command: list[str]) -> bool:
    return "--fix" in command or ("format" in command and "--check" not in command)


def test_contributing_quick_setup_runs_the_checks_ci_runs() -> None:
    quick_setup = (ROOT / "CONTRIBUTING.md").read_text().split("## Quick Setup", 1)[1].split("\n## ", 1)[0]
    (block,) = shell_blocks(quick_setup)
    assert {tuple(command) for command in uv_run_commands(block)} == ci_checks()
    (sync,) = [command for command in doc_commands(block) if is_uv_sync(command)]
    tested = yaml.safe_load((WORKFLOWS / "tests.yml").read_text())["jobs"]["tests"]["strategy"]["matrix"]
    assert option_value(sync, "--python", "-p") in tested["python-version"]


def test_every_chain_of_checks_in_contributing_runs_the_checks_ci_runs() -> None:
    chains = [
        commands
        for snippet in contributing_snippets()
        for line in uv_lines(snippet)
        if len(commands := uv_run_commands(line)) > 1 and not any(map(is_fixer, commands))
    ]
    assert len(chains) >= 2
    assert all({tuple(command) for command in chain} == ci_checks() for chain in chains)


def test_every_uv_sync_in_contributing_checks_the_lock_file() -> None:
    syncs = [command for snippet in contributing_snippets() for command in doc_commands(snippet) if is_uv_sync(command)]
    assert syncs
    assert all("--locked" in command for command in syncs)


def test_contributing_states_the_uv_version_ci_pins() -> None:
    (pin,) = set(ci_uv_pins())
    stated = re.findall(r"\buv\s+(?:version\s+)?v?(\d[\w.+-]*\w|\d)", (ROOT / "CONTRIBUTING.md").read_text())
    assert set(stated) == {pin}
