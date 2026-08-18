# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaldash image vendors individual Marin modules instead of installing ``marin-core``.

That keeps the image small, but it means a new import in the server tree is invisible until the
deployed container fails to start. This walks the imports for real and checks the Dockerfile copies
every Marin module they reach.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVALDASH_SRC = _REPO_ROOT / "infra" / "evaldash" / "src"
_MARIN_SRC = _REPO_ROOT / "lib" / "marin" / "src"
_DOCKERFILE = _REPO_ROOT / "infra" / "evaldash" / "Dockerfile"


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
    candidate = _MARIN_SRC / Path(*dotted.split(".")).with_suffix(".py")
    return candidate if candidate.exists() else None


def _required_modules() -> set[str]:
    """Every Marin module the evaldash server tree reaches, following imports transitively."""
    pending = {name for source in _EVALDASH_SRC.glob("*.py") for name in _marin_imports(source)}
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
def test_dockerfile_copies_every_marin_module_the_server_imports(dotted: str):
    relative = Path("lib/marin/src") / Path(*dotted.split(".")).with_suffix(".py")
    assert f"COPY {relative.as_posix()} " in _DOCKERFILE.read_text(encoding="utf-8"), (
        f"{dotted} is imported by the evaldash server but not copied into the image; "
        f"the deployed container would fail to start"
    )
