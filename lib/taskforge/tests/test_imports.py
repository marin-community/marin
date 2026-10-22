# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The import order of Taskforge's packages.

The test parses every module under ``src/taskforge`` and collects each ``taskforge.<package>``
import, in any form and at any depth. A stage package ``stageNN_<name>`` may import stages with a
lower number and the shared modules in ``SHARED``. Two packages may share a stage number when they
are alternative ways of running that stage; ``BUILDS_ON`` names the same-number packages each one may
import. A shared module may import other shared modules but no stage. A package in ``TOP``
runs the whole pipeline: ``stage_coordinator`` runs the stages, and ``agentic`` replaces them with
one agent. It may import every stage and shared module, and no package may import it.

The build stage has one package, ``stage40_build_with_program``, and the name states the
distinction that decides where build code goes: it is about whether a build *program* produces the
task draft at all, not about how the program's writer runs. The SDK, the standard template, the
build services, the program runner and the agentic author (an agent session that writes and
iterates a build program) all belong there. A ``stage40_build_with_agent`` package is reserved for a
build in which an agent produces the draft directly, with no program; nothing that writes a program
goes there, however it is driven. When that package exists, ``BUILDS_ON`` lets it import the program
build and never the reverse.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import taskforge

SHARED = ("capability", "child_process", "content_hash", "ledger", "llm", "sandbox", "spec")
TOP = ("agentic", "stage_coordinator")
BUILDS_ON: dict[str, tuple[str, ...]] = {}
STAGE = re.compile(r"stage(\d+)_[a-z0-9_]+")
ROOT = Path(taskforge.__file__).parent


def package_of(module: Path) -> str:
    return module.relative_to(ROOT).parts[0].removesuffix(".py")


def imported_packages(module: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line, package)`` for each ``taskforge.<package>`` that ``module`` imports."""
    for node in ast.walk(ast.parse(module.read_text(), filename=str(module))):
        if isinstance(node, ast.ImportFrom) and node.level:
            raise AssertionError(f"{module}:{node.lineno} uses a relative import; import taskforge.<package> by name")
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module == "taskforge":
            names = [f"taskforge.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "taskforge" and len(parts) > 1:
                yield node.lineno, parts[1]


def stage_number(package: str) -> int | None:
    match = STAGE.fullmatch(package)
    return int(match.group(1)) if match else None


def import_allowed(importer: str, imported: str) -> bool:
    if imported in TOP:
        return False
    if imported in SHARED:
        return True
    if importer in TOP:
        return stage_number(imported) is not None
    if imported in BUILDS_ON.get(importer, ()):
        return True
    imported_number = stage_number(imported)
    importer_number = stage_number(importer)
    return imported_number is not None and importer_number is not None and imported_number < importer_number


def modules() -> list[Path]:
    return [module for module in sorted(ROOT.rglob("*.py")) if module.relative_to(ROOT) != Path("__init__.py")]


def test_every_package_is_shared_top_or_a_numbered_stage():
    unplaced = {package_of(module) for module in modules()} - set(SHARED) - set(TOP)
    assert all(stage_number(package) is not None for package in unplaced), sorted(unplaced)


def test_stages_import_only_lower_stages_and_shared_modules():
    wrong = [
        f"{module.relative_to(ROOT)}:{line}: {package_of(module)} imports {imported}"
        for module in modules()
        for line, imported in imported_packages(module)
        if imported != package_of(module) and not import_allowed(package_of(module), imported)
    ]
    assert not wrong, "\n".join(wrong)


def test_every_import_form_counts_and_an_upward_import_is_refused(tmp_path):
    module = tmp_path / "author.py"
    module.write_text(
        "from taskforge import stage40_build_with_program, llm\n"
        "import taskforge.content_hash\n"
        "from taskforge.stage20_triage.preflight import preflight\n"
    )

    assert sorted(imported_packages(module)) == [
        (1, "llm"),
        (1, "stage40_build_with_program"),
        (2, "content_hash"),
        (3, "stage20_triage"),
    ]
    assert not import_allowed("stage30_author", "stage40_build_with_program")
    assert import_allowed("stage40_build_with_program", "stage30_author")
    assert not import_allowed("llm", "stage30_author")
    assert import_allowed("stage_coordinator", "stage60_review")
    assert not import_allowed("stage60_review", "stage_coordinator")
