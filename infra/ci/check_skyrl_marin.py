# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check SkyRL's Marin library imports inside its independently resolved environment."""

import argparse
import ast
import importlib
import inspect
import os
import subprocess
from pathlib import Path

MARIN_PACKAGES = {"rolloutengine", "shellbox", "taskcompendium", "verifyit"}
PYTHON_ENVIRONMENT_KEYS = {"PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"}


def check_project(source: Path, *, marin_source: Path | None = None) -> None:
    """Check frozen packages, optionally selecting Marin imports from a checkout."""
    environment = {key: value for key, value in os.environ.items() if key not in PYTHON_ENVIRONMENT_KEYS}
    if marin_source is not None:
        source_roots = [
            (marin_source / "lib" / package / "src").resolve(strict=True) for package in sorted(MARIN_PACKAGES)
        ]
        for root in source_roots:
            if not root.is_dir():
                raise ValueError(f"Marin source root is not a directory: {root}")
        environment["PYTHONPATH"] = os.pathsep.join(str(root) for root in source_roots)
    subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--project",
            str(source.resolve()),
            "--frozen",
            "--extra",
            "cpu",
            "--no-group",
            "dev",
            "python",
            str(Path(__file__).resolve()),
            "imports",
            "--source",
            str(source.resolve()),
        ],
        env=environment,
        check=True,
    )


def check_imports(source: Path) -> None:
    """Check imported symbols and explicit constructor keywords without loading CUDA."""
    for directory in ("cloud", "marinskyrl", "skyrl-train/skyrl_train", "skyrl-gym/skyrl_gym"):
        for path in (source / directory).rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            imported = {}
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in MARIN_PACKAGES:
                            importlib.import_module(alias.name)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    assert node.module is not None
                    if node.module.split(".")[0] in MARIN_PACKAGES:
                        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), imported)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in imported:
                    continue
                keywords = {item.arg for item in node.keywords if item.arg is not None}
                if not keywords:
                    continue
                parameters = inspect.signature(imported[node.func.id]).parameters
                if any(value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()):
                    continue
                unknown = keywords - parameters.keys()
                if unknown:
                    raise TypeError(f"{path}:{node.lineno}: {node.func.id} does not accept {sorted(unknown)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("project", "imports"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--marin-source", type=Path, help="Check checkout sources instead of locked Marin packages")
    args = parser.parse_args()
    if args.command == "project":
        check_project(args.source, marin_source=args.marin_source)
    else:
        check_imports(args.source)


if __name__ == "__main__":
    main()
