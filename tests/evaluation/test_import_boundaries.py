# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Marin package must not import experiment definitions."""

import ast
from pathlib import Path


def test_marin_library_does_not_import_experiments():
    root = Path(__file__).parents[2] / "lib" / "marin" / "src" / "marin"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            if any(name == "experiments" or name.startswith("experiments.") for name in imported):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")

    assert violations == []
