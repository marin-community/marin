# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check SkyRL's Marin library imports inside its independently resolved environment."""

import argparse
import ast
import importlib
import inspect
from pathlib import Path

MARIN_PACKAGES = {"rolloutengine", "shellbox", "taskcompendium", "verifyit"}


def check_imports(source: Path) -> None:
    """Check imported symbols and explicit constructor keywords without loading CUDA."""
    for directory in ("cloud", "marinskyrl", "skyrl-train/skyrl_train"):
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
    parser.add_argument("--source", type=Path, required=True)
    check_imports(parser.parse_args().source)


if __name__ == "__main__":
    main()
