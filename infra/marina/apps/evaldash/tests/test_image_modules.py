# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The image copies Marin modules instead of installing ``marin-core``.

That keeps JAX and the training stack out of the Marina image, but it means a new ``marin.*`` import
in this app is invisible until the deployed container fails to start. This walks the imports for real
and checks the Dockerfile copies every Marin module they reach.
"""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
REPO_ROOT = APP.parents[3]
MARIN_SRC = REPO_ROOT / "lib" / "marin" / "src"
DOCKERFILE = REPO_ROOT / "infra" / "marina" / "Dockerfile"


def _marin_imports(source: Path) -> set[str]:
    """Dotted ``marin.*`` module names one file imports, at any statement depth."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.startswith("marin."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("marin."):
            names.add(node.module)
    return names


def _module_path(dotted: str) -> Path | None:
    """The file backing a dotted module name, or None when it is a package or absent."""
    candidate = MARIN_SRC / Path(*dotted.split(".")).with_suffix(".py")
    return candidate if candidate.exists() else None


def _required_modules() -> set[str]:
    """Every Marin module the served app reaches, following imports transitively."""
    pending = {name for source in APP.glob("*.py") for name in _marin_imports(source)}
    reached: set[str] = set()
    while pending:
        dotted = pending.pop()
        if dotted in reached:
            continue
        reached.add(dotted)
        path = _module_path(dotted)
        if path is not None:
            pending |= _marin_imports(path)
    return {dotted for dotted in reached if _module_path(dotted) is not None}


@pytest.mark.parametrize("dotted", sorted(_required_modules()))
def test_the_image_copies_every_marin_module_the_app_imports(dotted: str):
    relative = Path("lib/marin/src") / Path(*dotted.split(".")).with_suffix(".py")
    assert f"COPY {relative.as_posix()} " in DOCKERFILE.read_text(encoding="utf-8"), (
        f"{dotted} is imported by evaldash but not copied into the image; " f"the deployed container would fail to start"
    )
