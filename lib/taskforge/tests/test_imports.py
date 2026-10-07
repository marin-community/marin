# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The seam rule: Taskforge's packages are totally ordered and a package imports only from its left.

Every module under ``src/taskforge`` is parsed, and every ``taskforge.<package>`` import it makes,
in any form (``import taskforge.x``, ``from taskforge.x import y``, ``from taskforge import x``) and
including imports inside functions, is an edge that must point left in ``ORDER``.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import taskforge

ORDER = (
    "content_hash",
    "atomic_file",
    "ledger",
    "spec",
    "sandbox",
    "llm",
    "proposal",
    "triage",
    "build",
    "validate",
    "review",
    "loop",
    "queue",
)
ROOT = Path(taskforge.__file__).parent


def package_of(module: Path) -> str:
    return module.relative_to(ROOT).parts[0].removesuffix(".py")


def imported_packages(module: Path) -> Iterator[tuple[int, str]]:
    """``(line, package)`` for each ``taskforge.<package>`` the module imports."""
    for node in ast.walk(ast.parse(module.read_text(), filename=str(module))):
        if isinstance(node, ast.ImportFrom) and node.level:
            raise AssertionError(f"{module}:{node.lineno} uses a relative import; import taskforge.<package> by name")
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom) and node.module == "taskforge":
            names = [f"taskforge.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "taskforge" and len(parts) > 1:
                yield node.lineno, parts[1]


def edges() -> Iterator[tuple[str, str, str]]:
    """``(location, importer, imported)`` for every cross-package import under ``src/taskforge``."""
    for module in sorted(ROOT.rglob("*.py")):
        if module.parent == ROOT and module.name == "__init__.py":
            continue
        importer = package_of(module)
        for line, imported in imported_packages(module):
            if imported != importer:
                yield f"{module.relative_to(ROOT)}:{line}", importer, imported


def test_every_package_has_a_place_in_the_order():
    packages = {package_of(module) for module in ROOT.rglob("*.py") if module.relative_to(ROOT) != Path("__init__.py")}
    assert packages <= set(ORDER), f"add {sorted(packages - set(ORDER))} to ORDER and the README"


def test_imports_point_only_downstream_in_the_package_order():
    rank = {package: index for index, package in enumerate(ORDER)}
    wrong = [f"{at}: {src} imports {dst}" for at, src, dst in edges() if rank[dst] > rank[src]]
    assert not wrong, "\n".join(wrong)


def test_every_import_form_counts_as_an_edge(tmp_path):
    module = tmp_path / "decision.py"
    module.write_text(
        "from taskforge import loop, spec\nimport taskforge.queue\nfrom taskforge.ledger.jsonl import JsonlLedger\n"
    )
    assert sorted(imported_packages(module)) == [(1, "loop"), (1, "spec"), (2, "queue"), (3, "ledger")]
