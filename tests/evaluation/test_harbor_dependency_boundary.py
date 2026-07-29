# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Marin environment must not contain or import Harbor."""

import ast
import tomllib
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_EXTERNAL_DRIVER = _ROOT / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"
_EXTERNAL_MODULES = frozenset({"daytona", "harbor", "harbor_config", "upath"})


def _external_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    return imports & _EXTERNAL_MODULES


def test_only_isolated_driver_imports_harbor_runtime_modules():
    violations = {
        str(path.relative_to(_ROOT)): sorted(imports)
        for source_root in (_ROOT / "lib/marin/src/marin", _ROOT / "experiments")
        for path in source_root.rglob("*.py")
        if path != _EXTERNAL_DRIVER
        if (imports := _external_imports(path))
    }

    assert violations == {}


def test_root_workspace_does_not_resolve_harbor():
    marin_project = tomllib.loads((_ROOT / "lib/marin/pyproject.toml").read_text())
    root_project = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    root_lock = tomllib.loads((_ROOT / "uv.lock").read_text())

    assert "harbor" not in marin_project["project"]["optional-dependencies"]
    assert "harbor" not in root_project["tool"]["uv"]["sources"]
    assert "harbor" not in {package["name"] for package in root_lock["package"]}
