# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import sys
from pathlib import Path


def test_recipe_schema_imports_only_its_runtime_dependencies():
    source = Path(__file__).resolve().parents[1] / "src/skyrlrecipe"
    for path in source.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names | {"pydantic"}, path
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split(".")[0] in sys.stdlib_module_names | {"pydantic"}, path
