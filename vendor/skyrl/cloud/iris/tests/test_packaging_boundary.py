"""Import-boundary tests for the CPU-only launcher installation."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
    "module,blocked",
    [
        ("skyrl_train.hf_export", ("flash_attn", "ray", "torch", "vllm")),
        ("cloud.iris.launch", ("flash_attn", "ray", "skyrl_train.objective", "skyrl_train.trainer", "torch", "vllm")),
        (
            "skyrl_train.entrypoints.main_base",
            ("flash_attn", "skyrl_train.objective", "skyrl_train.trainer", "torch", "vllm"),
        ),
    ],
)
def test_importing_entrypoints_does_not_import_training_stacks(module, blocked) -> None:
    program = f"""
import json
import sys

import {module}

blocked = {blocked!r}
print(json.dumps(sorted(name for name in blocked if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_recipe_schema_imports_only_its_runtime_dependencies():
    source = REPOSITORY_ROOT / "marinskyrl/recipe_schema"
    for path in source.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names | {"pydantic"}, path
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split(".")[0] in sys.stdlib_module_names | {"pydantic"}, path
