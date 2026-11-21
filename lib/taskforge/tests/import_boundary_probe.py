# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import every ``taskforge`` module while refusing any top-level module no declared distribution provides.

Run as its own process by ``test_import_boundary``. A module is declared when it is built in, comes from the standard
library, is a file one of ``marin-taskforge``'s transitive requirements installed, or lies under the source root an
editable install of one of them puts on the path. Zephyr is refused first, as the control that the boundary holds
in an environment that has it.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata as metadata
import pkgutil
import sys
import sysconfig
from importlib.machinery import ModuleSpec
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = "marin-taskforge"
UNPLACED = (None, "built-in", "frozen")


def closure(root: str) -> list[metadata.Distribution]:
    """Every distribution ``root`` requires, transitively, with the extras each requirement names."""
    seen: set[tuple[str, frozenset[str]]] = set()
    stack, found = [(root, frozenset[str]())], []
    while stack:
        name, extras = stack.pop()
        if (key := (canonicalize_name(name), extras)) in seen:
            continue
        seen.add(key)
        found.append(distribution := metadata.distribution(name))
        for line in distribution.requires or ():
            requirement = Requirement(line)
            marker = requirement.marker
            if marker is None or any(marker.evaluate({"extra": extra}) for extra in extras or {""}):
                stack.append((requirement.name, frozenset(requirement.extras)))
    return found


def homes(distributions: list[metadata.Distribution]) -> tuple[set[Path], set[Path]]:
    """The files the distributions installed, and the source roots their editable ``.pth`` files add."""
    files, roots = set(), set()
    for distribution in distributions:
        for file in distribution.files or ():
            path = Path(str(distribution.locate_file(file))).resolve()
            files.add(path)
            if path.suffix == ".pth" and path.exists():
                roots.update(Path(line).resolve() for line in path.read_text().splitlines() if line.startswith("/"))
    return files, roots


FILES, ROOTS = homes(closure(ROOT))
HOMES = (*ROOTS, *(Path(sysconfig.get_paths()[key]).resolve() for key in ("stdlib", "platstdlib")))


def declared(spec: ModuleSpec) -> bool:
    places = [] if spec.origin in UNPLACED else [spec.origin]
    places += list(spec.submodule_search_locations or ())
    if not places:
        return True
    for place in (Path(str(p)).resolve() for p in places):
        if place in FILES or any(place.is_relative_to(home) for home in HOMES):
            return True
        if any(file.is_relative_to(place) for file in FILES):  # a package directory an install populated
            return True
    return False


class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if "." in fullname or fullname in sys.builtin_module_names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname)
        if spec is not None and not declared(spec):
            raise ModuleNotFoundError(f"{fullname} ({spec.origin}) is from no distribution {ROOT} declares")
        return None


def main() -> None:
    sys.meta_path.insert(0, Boundary())
    try:
        importlib.import_module("zephyr")
    except ModuleNotFoundError:
        pass
    else:
        raise SystemExit("the boundary let zephyr through")
    taskforge = importlib.import_module("taskforge")
    for module in pkgutil.walk_packages(taskforge.__path__, "taskforge."):
        importlib.import_module(module.name)


if __name__ == "__main__":
    main()
