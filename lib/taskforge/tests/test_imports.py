# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The import order of Taskforge's packages.

The test parses every module under ``src/taskforge`` and collects each ``taskforge.<package>``
import, in any form and at any depth. A stage package ``stageNN_<name>`` may import stages with a
lower number and the shared modules in ``SHARED``. A shared module may import other shared modules
but no stage.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import taskforge

SHARED = ("content_hash", "ledger", "llm")
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
    if imported in SHARED:
        return True
    imported_number = stage_number(imported)
    importer_number = stage_number(importer)
    return imported_number is not None and importer_number is not None and imported_number < importer_number


def modules() -> list[Path]:
    return [module for module in sorted(ROOT.rglob("*.py")) if module.relative_to(ROOT) != Path("__init__.py")]


def test_every_package_is_shared_or_a_numbered_stage():
    unplaced = {package_of(module) for module in modules()} - set(SHARED)
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
        "from taskforge import stage40_build, llm\n"
        "import taskforge.content_hash\n"
        "from taskforge.stage20_triage.verdict import accept\n"
    )

    assert sorted(imported_packages(module)) == [
        (1, "llm"),
        (1, "stage40_build"),
        (2, "content_hash"),
        (3, "stage20_triage"),
    ]
    assert not import_allowed("stage30_author", "stage40_build")
    assert import_allowed("stage40_build", "stage30_author")
    assert not import_allowed("llm", "stage30_author")
